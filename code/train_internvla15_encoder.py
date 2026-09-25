#!/usr/bin/env python3
"""Vision-only LoRA tuning for the official InternVLA-A1.5 LIBERO model.

The 24-layer Qwen3.5-VL visual tower and ``visual.merger`` receive rank-32 LoRA
adapters in the released reference configuration. The language trunk, action
expert, flow head, and all other policy parameters remain frozen.
"""

from __future__ import annotations

import vla_paths

import argparse
import io
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from safetensors.torch import load_model, save_file
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


ROOT = Path(vla_paths.resolve(''))
REPO = ROOT / "third_party/InternVLA-A-series"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ROOT / "code"))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig  # noqa: E402
from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import (  # noqa: E402
    UnifyInternVLAA15InputsTransformFn,
)
from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import (  # noqa: E402
    InternVLAA15Policy,
)
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (  # noqa: E402
    FASTInternVLAA15ActionTokenizerTransformFn,
    InternVLAA15ChatProcessorTransformFn,
)
from lerobot.transforms.core import (  # noqa: E402
    ComposeFieldsTransform,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ReorderStateActionTransform,
    ResizeImagesWithPadFn,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR  # noqa: E402

import internvla15_encoder_mask as mask  # noqa: E402


EXPECTED_IGNORED_WEIGHTS = {
    "model._wan_grid_sizes",
    "model.learnable_to_wan_proj.bias",
    "model.learnable_to_wan_proj.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "models/InternVLA-A1.5-Libero")
    parser.add_argument("--qwen", type=Path, default=ROOT / "models/Qwen3.5-2B")
    parser.add_argument("--fast", type=Path, default=ROOT / "models/physical-intelligence-fast")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(vla_paths.resolve('data/lerobot/local/libero_plus_mix')),
    )
    parser.add_argument("--suite-map", type=Path, default=ROOT / "config/libero_task_suite_map.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/internvla15-encoder-r32-b32-s12000")
    parser.add_argument("--max-steps", type=int, default=12_000)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--decay-learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="auto", help="auto | none | checkpoint directory")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--stop-after-steps", type=int, default=0, help="Testing gate; 0 disables it")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize_task(text: str) -> str:
    return " ".join(text.strip().lower().split())


class LiberoPlusInternVLAIterable(IterableDataset):
    """Episode-local stream with current stereo images/state and eight actions."""

    def __init__(self, root: Path, horizon: int = 8, seed: int = 42):
        self.root = root
        self.horizon = horizon
        self.seed = seed
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text()) if info_path.is_file() else {}
        # Target-only datasets are lightweight metadata views.  Their episode
        # indices deliberately retain the ids of the underlying mixed corpus,
        # and ``root_path`` identifies that physical parquet store.
        self.data_root = Path(info.get("root_path", root))
        self.chunks_size = int(info.get("chunks_size", 1000))
        self.episodes: list[tuple[int, int, str]] = []
        valid = 0
        frames = 0
        episode_ids: set[int] = set()
        for episode in read_jsonl(root / "meta/episodes.jsonl"):
            episode_index = int(episode["episode_index"])
            if episode_index in episode_ids:
                raise ValueError(f"duplicate episode_index {episode_index}")
            episode_ids.add(episode_index)
            length = int(episode["length"])
            if length >= horizon:
                self.episodes.append((episode_index, length, str(episode["tasks"][0])))
                valid += length - horizon + 1
            frames += length
        self.valid_count = valid
        print(
            f"[data] episodes={len(self.episodes):,} frames={frames:,} valid_h{horizon}={valid:,} "
            f"data_root={self.data_root}",
            flush=True,
        )

    def __len__(self) -> int:
        return self.valid_count

    @staticmethod
    def decode_image(value) -> torch.Tensor:
        if isinstance(value, Image.Image):
            image = value.convert("RGB")
        elif isinstance(value, dict) and value.get("bytes") is not None:
            image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        elif isinstance(value, dict) and value.get("path"):
            image = Image.open(value["path"]).convert("RGB")
        else:
            raise TypeError(f"unsupported LeRobot image value: {type(value)}")
        array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)

    def __iter__(self):
        import pandas as pd

        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        epoch = 0
        while True:
            order_rng = np.random.default_rng(self.seed + epoch * 1_000_003)
            order = order_rng.permutation(len(self.episodes))[worker_id::worker_count]
            rng = np.random.default_rng(self.seed + worker_id + epoch * 1_000_003)
            for slot in order:
                episode_index, length, task = self.episodes[int(slot)]
                path = (
                    self.data_root / "data"
                    / f"chunk-{episode_index // self.chunks_size:03d}"
                    / f"episode_{episode_index:06d}.parquet"
                )
                frame = pd.read_parquet(
                    path, columns=["image", "wrist_image", "state", "actions"]
                )
                if len(frame) != length:
                    raise ValueError(f"metadata/parquet length mismatch: {path}")
                starts = rng.permutation(length - self.horizon + 1)
                images = frame["image"].to_numpy()
                wrists = frame["wrist_image"].to_numpy()
                states = frame["state"].to_numpy()
                actions = frame["actions"].to_numpy()
                for start in starts:
                    start = int(start)
                    yield {
                        "image": self.decode_image(images[start]),
                        "wrist_image": self.decode_image(wrists[start]),
                        "state": torch.from_numpy(
                            np.asarray(states[start], dtype=np.float32).copy()
                        ),
                        "actions": torch.from_numpy(
                            np.stack(actions[start : start + self.horizon]).astype(
                                np.float32, copy=False
                            )
                        ),
                        "task": task,
                    }
            epoch += 1


class SuiteNormalizer:
    """Apply the immutable released InternVLA per-suite mean/std contract."""

    def __init__(self, stats_path: Path, suite_map_path: Path):
        self.stats_path = stats_path
        self.stats = json.loads(stats_path.read_text())
        suite_map = json.loads(suite_map_path.read_text())
        self.task_to_suite: dict[str, str] = {}
        self.params: dict[str, tuple[torch.Tensor, ...]] = {}
        for mapped_suite, tasks in suite_map.items():
            suite = mapped_suite.removesuffix("_no_noops")
            if suite not in self.stats:
                raise KeyError(f"{suite} absent from {stats_path}")
            state = self.stats[suite]["observation.state"]
            action = self.stats[suite]["action"]
            tensors = tuple(
                torch.tensor(values, dtype=torch.float32)
                for values in (state["mean"], state["std"], action["mean"], action["std"])
            )
            if tuple(x.numel() for x in tensors) != (8, 8, 7, 7):
                raise ValueError(f"unexpected state/action stats shapes for {suite}")
            if not (0.0 <= action["min"][-1] <= action["max"][-1] <= 1.0):
                raise ValueError(f"unexpected InternVLA [0,1] gripper stats for {suite}")
            self.params[suite] = tensors
            for task in tasks:
                key = normalize_task(task)
                if key in self.task_to_suite:
                    raise ValueError(f"task occurs in multiple suites: {task}")
                self.task_to_suite[key] = suite
        if len(self.task_to_suite) != 40:
            raise ValueError(f"expected 40 mapped tasks, found {len(self.task_to_suite)}")

    def __call__(self, state: torch.Tensor, actions: torch.Tensor, task: str):
        suite = self.task_to_suite.get(normalize_task(task))
        if suite is None:
            raise KeyError(f"no suite mapping for {task!r}")
        grip = actions[..., 6]
        binary = torch.isclose(grip, torch.ones_like(grip), atol=1e-5) | torch.isclose(
            grip, -torch.ones_like(grip), atol=1e-5
        )
        if not bool(binary.all()):
            raise ValueError("Plus gripper must be a binary LIBERO command in {-1,+1}")
        state_mean, state_std, action_mean, action_std = self.params[suite]
        converted = actions.clone()
        converted[..., 6] = 0.5 * (1.0 - converted[..., 6])
        normalized_state = (state - state_mean) / (state_std + 1e-6)
        normalized_action = (converted - action_mean) / (action_std + 1e-6)
        reconstructed = normalized_action * (action_std + 1e-6) + action_mean
        if not torch.allclose(reconstructed, converted, atol=2e-6, rtol=2e-6):
            raise RuntimeError("action normalization round-trip failed")
        return normalized_state, normalized_action, suite


class InternVLAPlusCollator:
    def __init__(self, qwen: Path, fast: Path, stats: Path, suite_map: Path):
        self.normalizer = SuiteNormalizer(stats, suite_map)
        self.resize = ResizeImagesWithPadFn(
            height=224, width=224, mapping={"image": "", "wrist_image": ""}
        )
        self.remap = RemapImageKeyTransformFn(
            mapping={
                "image": f"{OBS_IMAGES}.image0",
                "wrist_image": f"{OBS_IMAGES}.image1",
            }
        )
        self.compose = ComposeFieldsTransform(
            mapping={OBS_STATE: ["state"], ACTION: ["actions"]}
        )
        self.fast = FASTInternVLAA15ActionTokenizerTransformFn(
            action_tokenizer_name=str(fast), qwen35_model_name=str(qwen),
            chunk_size=8, max_action_dim=32,
        )
        self.chat = InternVLAA15ChatProcessorTransformFn(
            pretrained_model_name_or_path=str(qwen), max_length=650, num_views=3,
            tokenize_state=True, max_state_dim=32, use_fast_action_tokens=True,
            mode="train", action_mode="end_effector",
        )
        self.pad = PadStateAndActionTransformFn(max_state_dim=32, max_action_dim=32)
        self.reorder = ReorderStateActionTransform()
        self.unify = UnifyInternVLAA15InputsTransformFn(
            num_video_frames=0, video_height=224, video_width=224
        )

    def transform_one(self, example: dict) -> dict:
        state, actions, _ = self.normalizer(
            example["state"], example["actions"], example["task"]
        )
        data = dict(example)
        data["state"] = state
        data["actions"] = actions
        data = self.resize(data)
        data = self.remap(data)
        data = self.compose(data)
        data = self.fast(data)
        data = self.chat(data)
        data = self.pad(data)
        data = self.reorder(data)
        return self.unify(data)

    @staticmethod
    def collate(samples: list[dict]) -> dict[str, torch.Tensor]:
        from torch.utils.data._utils.collate import default_collate

        visual = {f"{OBS_STR}.pixel_values", f"{OBS_STR}.image_grid_thw"}
        output = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            output[key] = torch.cat(values, dim=0) if key in visual else default_collate(values)
        # The released transform pads every sequence to 650 tokens. Plus prompts
        # use only ~330-360; removing the all-masked right tail is mathematically
        # identical and avoids nearly half of language/logit compute.
        attention_key = f"{OBS_STR}.attention_mask"
        attention = output[attention_key]
        positions = torch.arange(1, attention.shape[1] + 1, dtype=attention.dtype)
        valid_length = int((attention * positions).amax().item())
        for key in (
            f"{OBS_STR}.input_ids", attention_key, f"{OBS_STR}.fast_token_mask", "VQA.labels"
        ):
            output[key] = output[key][:, :valid_length]
        return output

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        return self.collate([self.transform_one(example) for example in examples])


def latest_checkpoint(output: Path) -> Path | None:
    complete = []
    for path in output.glob("ckpt-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        if all(
            file.is_file()
            for file in (
                path / "state.json",
                path / "training_state.pt",
                path / "adapter/adapter_config.json",
                path / "vision_projector.safetensors",
            )
        ):
            complete.append((step, path))
    return max(complete)[1] if complete else None


def resolve_resume(args: argparse.Namespace) -> Path | None:
    if args.resume.lower() == "none":
        return None
    if args.resume.lower() == "auto":
        return latest_checkpoint(args.output)
    path = Path(args.resume)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def load_stock_policy(args: argparse.Namespace) -> InternVLAA15Policy:
    config = PreTrainedConfig.from_pretrained(args.base)
    config.vlm_model_name_or_path = str(args.qwen)
    config.device = "cuda:0"
    config.action_loss_only = True
    config.inference_backend = "standard"
    config.gradient_checkpointing = args.gradient_checkpointing
    config.enable_vqa_loss = True
    config.lambda_vqa = 1.0
    config.tokenize_state = True
    config.optimizer_lr = args.learning_rate
    config.optimizer_betas = (0.9, 0.95)
    config.optimizer_eps = 1e-8
    config.optimizer_weight_decay = args.weight_decay
    config.optimizer_grad_clip_norm = args.max_grad_norm
    config.scheduler_warmup_steps = args.warmup_steps
    config.scheduler_decay_steps = args.max_steps
    config.scheduler_decay_lr = args.decay_learning_rate
    config.push_to_hub = False
    config.repo_id = None
    policy = InternVLAA15Policy(config)
    missing, unexpected = load_model(
        policy, str(args.base / "model.safetensors"), strict=False, device="cuda:0"
    )
    if missing:
        raise RuntimeError(f"base checkpoint missing policy weights: {sorted(missing)}")
    if set(unexpected) != EXPECTED_IGNORED_WEIGHTS:
        raise RuntimeError(
            f"unexpected base weights changed: expected={sorted(EXPECTED_IGNORED_WEIGHTS)} "
            f"actual={sorted(unexpected)}"
        )
    return policy


def configure_policy(args: argparse.Namespace, resume: Path | None):
    import unified_lora_contract as contract
    dual_lora = contract.projector_mode() == "lora"
    contract.check_resume(resume)
    policy = load_stock_policy(args)
    mask.summarize(policy, "stock")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    visual = policy.model.qwen3_5_with_expert.qwen3_5.visual
    targets = mask.relative_linear_targets(visual)
    if len(targets) != 96:
        raise RuntimeError(f"expected 96 visual tower Linear targets, found {len(targets)}")
    if dual_lora:
        targets += [n for n, m in visual.named_modules() if n.startswith("merger.") and isinstance(m, torch.nn.Linear)]
        if len(targets) <= 96:
            raise RuntimeError("No projector Linear targets found")
    if resume is None:
        lora = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            bias="none", target_modules=mask.target_regex(targets),
            init_lora_weights="gaussian",
        )
        visual = get_peft_model(visual, lora)
    else:
        visual = PeftModel.from_pretrained(
            visual, resume / "adapter", is_trainable=True, local_files_only=True
        )
    # Qwen3.5 exposes ``visual`` through a proxy; replace the actual registered
    # child so the PEFT wrapper remains reachable for adapter-only saves.
    policy.model.qwen3_5_with_expert.qwen3_5.model.visual = visual
    for name, parameter in policy.named_parameters():
        if mask.is_projector(name) and not dual_lora:
            parameter.requires_grad_(True)
    if resume is not None:
        projector = {
            name: value
            for name, value in __import__("safetensors.torch", fromlist=["load_file"])
            .load_file(str(resume / "vision_projector.safetensors"), device="cpu")
            .items()
        }
        named = dict(policy.named_parameters())
        if set(projector) != {name for name in named if mask.is_projector(name)}:
            raise RuntimeError("resume projector tensor set does not match current model")
        with torch.no_grad():
            for name, value in projector.items():
                named[name].copy_(value.to(named[name]))
    trainable = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    leaks = [name for name, _ in trainable if not mask.is_vision(name)]
    invalid_vision = [
        name for name, _ in trainable
        if not ("lora_" in name or mask.is_projector(name))
    ]
    if leaks or invalid_vision:
        raise RuntimeError(
            f"trainable scope violation: outside={leaks[:5]} invalid_visual={invalid_vision[:5]}"
        )
    trainable_count = sum(p.numel() for _, p in trainable)
    if dual_lora:
        contract.audit(policy, mask.is_vision, mask.is_projector)
    total_count = sum(p.numel() for p in policy.parameters())
    print(
        f"[scope] targets={len(targets)} rank={args.lora_rank} alpha={args.lora_alpha} "
        f"trainable={trainable_count:,}/{total_count:,} "
        f"({100*trainable_count/total_count:.4f}%) tensors={len(trainable)}",
        flush=True,
    )
    print(f"[scope] first={targets[0]} last={targets[-1]}", flush=True)
    return policy, [p for _, p in trainable]


def save_checkpoint(policy, optimizer, scheduler, args, step: int) -> Path:
    destination = args.output / f"ckpt-{step}"
    if destination.exists():
        print(f"[save] keeping existing checkpoint {destination}", flush=True)
        return destination
    temporary = args.output / f".ckpt-{step}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    visual = policy.model.qwen3_5_with_expert.qwen3_5.model.visual
    visual.save_pretrained(temporary / "adapter", safe_serialization=True)
    projector = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in policy.named_parameters()
        if mask.is_projector(name)
    }
    expected_count = 10 if __import__("unified_lora_contract").projector_mode() == "lora" else 6
    if len(projector) != expected_count:
        raise RuntimeError(f"expected 6 merger tensors, found {len(projector)}")
    save_file(projector, str(temporary / "vision_projector.safetensors"))
    training_state = {
        "completed_optimizer_steps": step,
        "samples_seen": step * args.effective_batch_size,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    torch.save(training_state, temporary / "training_state.pt")
    state = {
        "format": "internvla15-vision-lora-v1",
        "completed_optimizer_steps": step,
        "samples_seen": step * args.effective_batch_size,
        "effective_batch_size": args.effective_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.effective_batch_size // args.micro_batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "decay_learning_rate": args.decay_learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "projector_update": __import__("unified_lora_contract").projector_mode(),
        "trainable_scope": "Qwen3.5-VL tower LoRA + merger " + __import__("unified_lora_contract").projector_mode(),
        "normalization": "released four-suite InternVLA mean/std",
        "gripper_contract": "Plus +1=close/-1=open -> InternVLA 0=close/1=open",
        "action_alignment": "state/image[t] -> delta action targets[t:t+8]",
    }
    (temporary / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    shutil.copy2(args.base / "stats.json", temporary / "stats.json")
    os.rename(temporary, destination)
    print(f"[save] {destination}", flush=True)
    return destination


def restore_training_state(path: Path, optimizer, scheduler) -> int:
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["completed_optimizer_steps"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.to("cuda:0", non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def main() -> None:
    args = parse_args()
    if args.effective_batch_size % args.micro_batch_size:
        raise ValueError("effective batch must be divisible by micro batch")
    if args.save_every <= 0 or args.max_steps <= 0:
        raise ValueError("max steps and save interval must be positive")
    for path in (args.base, args.qwen, args.fast, args.dataset):
        if not path.is_dir():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    resume = resolve_resume(args)
    if resume:
        print(f"[resume] {resume}", flush=True)
    elif any(args.output.iterdir()):
        allowed = {"run_manifest.json"}
        unexpected = [p for p in args.output.iterdir() if p.name not in allowed]
        if unexpected:
            raise FileExistsError(f"non-empty fresh output: {unexpected[:5]}")

    set_seed(args.seed)
    collator = InternVLAPlusCollator(
        args.qwen, args.fast, args.base / "stats.json", args.suite_map
    )
    dataset = LiberoPlusInternVLAIterable(args.dataset, horizon=8, seed=args.seed)
    loader = DataLoader(
        dataset, batch_size=args.micro_batch_size, num_workers=args.num_workers,
        collate_fn=collator, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    policy, trainable = configure_policy(args, resume)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=args.warmup_steps, num_decay_steps=args.max_steps,
        peak_lr=args.learning_rate, decay_lr=args.decay_learning_rate,
    ).build(optimizer, args.max_steps)
    step = restore_training_state(resume, optimizer, scheduler) if resume else 0
    initial_step = step
    accumulation = args.effective_batch_size // args.micro_batch_size
    run_manifest = {
        "base": str(args.base), "qwen": str(args.qwen), "fast": str(args.fast),
        "dataset": str(args.dataset), "max_steps": args.max_steps,
        "effective_batch_size": args.effective_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accumulation,
        "optimizer": "AdamW(lr=5e-5, betas=(0.9,0.95), eps=1e-8, wd=0.01)",
        "scheduler": "native linear warmup 2000 + cosine to 5e-6 over 12000",
        "projector_update": __import__("unified_lora_contract").projector_mode(),
        "trainable_scope": "vision tower r32/a16/dropout0 LoRA + merger " + __import__("unified_lora_contract").projector_mode(),
        "wan_auxiliary_branch": False,
        "fast_action_token_loss": True,
        "flow_matching_action_loss": True,
        "resume": str(resume) if resume else None,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    started = time.monotonic()
    rolling: dict[str, float] = {}
    rolling_steps = 0
    gate_done = step > 0
    stop_step = step + args.stop_after_steps if args.stop_after_steps else args.max_steps
    while step < args.max_steps and step < stop_step:
        for _ in range(accumulation):
            batch = to_device(next(iterator))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}: {loss}")
            (loss / accumulation).backward()
            for key, value in metrics.items():
                rolling[key] = rolling.get(key, 0.0) + float(value) / accumulation
        if not gate_done:
            absent = [name for name, p in policy.named_parameters() if p.requires_grad and p.grad is None]
            leaks = [name for name, p in policy.named_parameters() if not p.requires_grad and p.grad is not None]
            if absent or leaks:
                raise RuntimeError(f"gradient gate failed: absent={absent[:5]} leaks={leaks[:5]}")
            print("[gate] trainable gradients present and frozen gradients absent", flush=True)
            gate_done = True
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        rolling_steps += 1
        if step == 1 or step % args.log_every == 0:
            elapsed = time.monotonic() - started
            samples = (step - initial_step) * args.effective_batch_size
            rate = samples / max(elapsed, 1e-6)
            summary = " ".join(
                f"{key}={value/max(rolling_steps, 1):.5f}"
                for key, value in sorted(rolling.items())
            )
            print(
                f"[train] step={step}/{args.max_steps} lr={scheduler.get_last_lr()[0]:.3e} "
                f"grad={float(grad_norm):.4f} samples_s={rate:.2f} "
                f"peak_gib={torch.cuda.max_memory_allocated()/2**30:.2f} {summary}",
                flush=True,
            )
            rolling.clear()
            rolling_steps = 0
        if step % args.save_every == 0 or step == args.max_steps or step == stop_step:
            save_checkpoint(policy, optimizer, scheduler, args, step)

    print(f"[done] completed_optimizer_steps={step}", flush=True)


if __name__ == "__main__":
    main()
