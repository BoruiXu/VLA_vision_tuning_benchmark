#!/usr/bin/env python3
"""Vision-only LoRA tuning for the official NORA-1.5 LIBERO checkpoint.

The released entry point selects LoRA for both the Qwen2.5-VL visual tower and
its visual merger/projector. The language trunk, FAST head, and flow-matching action expert
remain frozen. Checkpoints contain independently mergeable artifacts plus
optimizer/scheduler/RNG state for continuation.
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
import tensorflow as tf
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import get_scheduler


ROOT = Path(vla_paths.resolve(''))
NORA_REPO = ROOT / "third_party/nora-1.5"
sys.path.insert(0, str(NORA_REPO))
sys.path.insert(0, str(ROOT / "code"))

# NORA uses TensorFlow only for the released dlimp image resize. Keep that CPU-only
# so it neither compiles H200 kernels nor competes with PyTorch for accelerator memory.
tf.config.set_visible_devices([], "GPU")

from inference.modelling_expert import VLAWithExpert, resize_image as nora_resize_image  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

import nora15_encoder_mask as mask  # noqa: E402


ACTION_TOKEN_MIN = 151665
ACTION_TOKEN_MAX = 153712


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "models/nora-1.5-libero")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(vla_paths.resolve('data/lerobot/local/libero_plus_mix')),
    )
    parser.add_argument("--suite-map", type=Path, default=ROOT / "config/libero_task_suite_map.json")
    parser.add_argument(
        "--base-stats", type=Path,
        default=Path(vla_paths.resolve('models/openvla-oft/dataset_statistics.json')),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs/nora15-encoder-r32-b32-s12000")
    parser.add_argument("--max-steps", type=int, default=12_000)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="auto", help="auto | none | checkpoint directory")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


class LiberoPlusHorizonDataset(IterableDataset):
    """Episode-streamed Plus frames with five in-episode delta actions per sample.

    The image bytes live inside one parquet per episode. Shuffling arbitrary global
    frame indices would reopen a different parquet for almost every sample, while
    converting 2.2M embedded images into another Arrow cache would duplicate a very
    large dataset. We therefore shuffle episodes and then frame starts within each
    episode; each worker holds only its current episode in memory.
    """

    def __init__(self, root: Path, horizon: int = 5, seed: int = 42):
        self.root = root
        self.horizon = horizon
        self.seed = seed
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text()) if info_path.is_file() else {}
        # Camera-only and Noise-only are metadata views over the mixed parquet
        # store.  Preserve their original, non-contiguous episode ids and use
        # the physical data root recorded by the audited split builder.
        self.data_root = Path(info.get("root_path", root))
        self.chunks_size = int(info.get("chunks_size", 1000))
        self.episodes = []
        episodes = _read_jsonl(root / "meta/episodes.jsonl")
        valid_count = 0
        frame_count = 0
        episode_ids: set[int] = set()
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            if episode_index in episode_ids:
                raise ValueError(f"duplicate episode_index {episode_index}")
            episode_ids.add(episode_index)
            length = int(episode["length"])
            if length >= horizon:
                instruction = str(episode["tasks"][0])
                self.episodes.append((episode_index, length, instruction))
                valid_count += length - horizon + 1
            frame_count += length
        self.valid_count = valid_count
        print(
            f"[data] episodes={len(episodes):,} frames={frame_count:,} "
            f"valid_h{horizon}={valid_count:,} data_root={self.data_root}",
            flush=True,
        )

    def __len__(self) -> int:
        return self.valid_count

    @staticmethod
    def _decode_image(value) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict) and value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if isinstance(value, dict) and value.get("path"):
            return Image.open(value["path"]).convert("RGB")
        raise TypeError(f"unsupported LeRobot image value: {type(value)}")

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
            for episode_slot in order:
                episode_index, length, instruction = self.episodes[int(episode_slot)]
                path = (
                    self.data_root / "data"
                    / f"chunk-{episode_index // self.chunks_size:03d}"
                    / f"episode_{episode_index:06d}.parquet"
                )
                frame = pd.read_parquet(path, columns=["image", "actions"])
                if len(frame) != length:
                    raise ValueError(f"metadata/parquet length mismatch for {path}")
                starts = rng.permutation(length - self.horizon + 1)
                actions_column = frame["actions"].to_numpy()
                images_column = frame["image"].to_numpy()
                for start in starts:
                    start = int(start)
                    yield {
                        "image": self._decode_image(images_column[start]),
                        "actions": np.stack(
                            actions_column[start : start + self.horizon]
                        ).astype(np.float32, copy=False),
                        "instruction": instruction,
                    }
            epoch += 1


class NoraCollator:
    def __init__(self, processor, fast_tokenizer, stats_path: Path, suite_map_path: Path):
        self.processor = processor
        self.fast_tokenizer = fast_tokenizer
        self.processor.tokenizer.padding_side = "left"
        stats = json.loads(stats_path.read_text())
        suite_map = json.loads(suite_map_path.read_text())
        self.task_to_suite = {}
        self.action_stats = {}
        for suite, tasks in suite_map.items():
            if suite not in stats:
                raise KeyError(f"{suite} missing from {stats_path}")
            action = stats[suite]["action"]
            action_mask = action.get("mask")
            if not np.isclose(action["q01"][-1], 0.0) or not np.isclose(action["q99"][-1], 1.0):
                raise ValueError(f"{suite} does not use the expected NORA/OXE [0,1] gripper contract")
            if action_mask is not None and bool(action_mask[-1]):
                raise ValueError(f"{suite} unexpectedly normalizes the gripper dimension")
            self.action_stats[suite] = (
                np.asarray(action["q01"], dtype=np.float32),
                np.asarray(action["q99"], dtype=np.float32),
            )
            for task in tasks:
                if task in self.task_to_suite:
                    raise ValueError(f"task occurs in multiple suites: {task}")
                self.task_to_suite[task] = suite
        if len(self.task_to_suite) != 40:
            raise ValueError(f"expected 40 base tasks, found {len(self.task_to_suite)}")

    def normalize(self, actions: np.ndarray, instruction: str) -> np.ndarray:
        suite = self.task_to_suite.get(instruction)
        if suite is None:
            raise KeyError(f"no LIBERO suite mapping for {instruction!r}")
        low, high = self.action_stats[suite]
        gripper = actions[..., 6]
        if not np.all(np.isclose(gripper, -1.0) | np.isclose(gripper, 1.0)):
            raise ValueError("Plus gripper must be a binary LIBERO environment command in {-1,+1}")
        normalized = actions.copy()
        # NORA's released LIBERO action expert consumes per-suite q01/q99-normalized
        # delta commands in the first six dimensions.  Its unmasked gripper target
        # follows the OpenVLA/OXE convention: 0=close, 1=open.  The converted Plus
        # dataset stores the LIBERO environment command instead (+1=close,
        # -1=open), so this conversion is part of the model contract.
        normalized[..., :6] = 2.0 * (actions[..., :6] - low[:6]) / (high[:6] - low[:6]) - 1.0
        normalized[..., 6] = 0.5 * (1.0 - actions[..., 6])
        return normalized

    @staticmethod
    def _vlm_action(tokens) -> str:
        return "".join(f"<robot_action_{token}>" for token in tokens)

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        messages = []
        normalized_actions = []
        for example in examples:
            actions = self.normalize(example["actions"], example["instruction"])
            normalized_actions.append(actions)
            fast_tokens = self.fast_tokenizer(actions)[0]
            image = nora_resize_image(example["image"])
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image", "image": image,
                                "resized_height": 224, "resized_width": 224,
                            },
                            {"type": "text", "text": example["instruction"]},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": self._vlm_action(fast_tokens)}],
                    },
                ]
            )
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        batch = self.processor(
            text=text, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        expert_attention = batch["attention_mask"].clone()
        for index, sequence in enumerate(labels):
            action_mask = (sequence >= ACTION_TOKEN_MIN) & (sequence <= ACTION_TOKEN_MAX)
            locations = torch.nonzero(action_mask, as_tuple=False)
            if locations.numel() == 0:
                raise ValueError("FAST tokenizer produced no NORA action tokens")
            first = int(locations[0])
            sequence[:first] = -100
            expert_attention[index, first:] = 0
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        batch["expert_attention"] = expert_attention
        batch["actions"] = torch.from_numpy(np.stack(normalized_actions))
        return batch


def latest_checkpoint(output: Path) -> Path | None:
    checkpoints = []
    for path in output.glob("ckpt-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        if (
            (path / "state.json").is_file()
            and (path / "training_state.pt").is_file()
            and (path / "lora_adapter/adapter_config.json").is_file()
            and (path / "vision_projector.pt").is_file()
        ):
            checkpoints.append((step, path))
    return max(checkpoints)[1] if checkpoints else None


def configure_model(base: Path, resume: Path | None, rank: int, alpha: int):
    import unified_lora_contract as contract
    dual_lora = contract.projector_mode() == "lora"
    contract.check_resume(resume)
    model = VLAWithExpert.from_pretrained(base)
    mask.summarize(model, "stock")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = mask.vision_linear_modules(model, include_projector=dual_lora)
    if not targets:
        raise RuntimeError("NORA visual Linear target set is empty")
    if resume is None:
        config = LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
            target_modules=mask.target_modules_regex(targets),
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, config)
    else:
        model = PeftModel.from_pretrained(model, resume / "lora_adapter", is_trainable=True)
    for name, parameter in model.named_parameters():
        if mask.is_projector(name) and not dual_lora:
            parameter.requires_grad_(True)
    if resume is not None:
        projector = torch.load(resume / "vision_projector.pt", map_location="cpu", weights_only=True)
        named = dict(model.named_parameters())
        missing = sorted(set(projector) - set(named))
        if missing:
            raise KeyError(f"projector names absent after resume: {missing[:5]}")
        with torch.no_grad():
            for name, value in projector.items():
                named[name].copy_(value.to(dtype=named[name].dtype))
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    leaked = [name for name, _ in trainable if not mask.is_vision(name)]
    if leaked:
        raise RuntimeError(f"trainable tensors leaked outside vlm.model.visual: {leaked[:10]}")
    if not trainable:
        raise RuntimeError("zero trainable parameters")
    if dual_lora:
        contract.audit(model, mask.is_vision, mask.is_projector)
    total = sum(parameter.numel() for parameter in model.parameters())
    count = sum(parameter.numel() for _, parameter in trainable)
    print(
        f"[scope] LoRA targets={len(targets)} rank={rank} alpha={alpha}; "
        f"trainable={count:,}/{total:,} ({100*count/total:.4f}%) tensors={len(trainable)}",
        flush=True,
    )
    print(f"[scope] first={targets[0]} last={targets[-1]}", flush=True)
    return model, [parameter for _, parameter in trainable]


def save_checkpoint(
    model, optimizer, scheduler, output: Path, step: int, samples_seen: int, args: argparse.Namespace
) -> Path:
    destination = output / f"ckpt-{step}"
    if destination.exists():
        print(f"[save] keeping existing complete checkpoint {destination}", flush=True)
        return destination
    temporary = output / f".ckpt-{step}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary / "lora_adapter", safe_serialization=True)
    projector = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if mask.is_projector(name)
    }
    if not projector:
        raise RuntimeError("no visual projector tensors found while saving")
    torch.save(projector, temporary / "vision_projector.pt")
    state = {
        "completed_optimizer_steps": step,
        "samples_seen": samples_seen,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    torch.save(state, temporary / "training_state.pt")
    manifest = {
        "format": "nora15-vision-lora-v2-gripper-aligned",
        "base": str(args.base.resolve()),
        "dataset": str(args.dataset.resolve()),
        "completed_optimizer_steps": step,
        "samples_seen": samples_seen,
        "effective_batch_size": args.effective_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "projector_update": __import__("unified_lora_contract").projector_mode(),
        "trainable_scope": "Qwen2.5-VL tower LoRA + merger " + __import__("unified_lora_contract").projector_mode(),
        "normalization": (
            "immutable four-suite LIBERO base q01/q99; no extra delta transform; "
            "gripper env {-1,+1} -> NORA {1,0} (0=close,1=open)"
        ),
    }
    (temporary / "state.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.rename(temporary, destination)
    print(f"[save] step={step} samples={samples_seen:,} -> {destination}", flush=True)
    return destination


def restore_training_state(checkpoint: Path, optimizer, scheduler) -> tuple[int, int]:
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["completed_optimizer_steps"]), int(state["samples_seen"])


def main() -> None:
    args = parse_args()
    if args.effective_batch_size % args.micro_batch_size:
        raise ValueError("effective batch size must be divisible by micro batch size")
    accumulation = args.effective_batch_size // args.micro_batch_size
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.base / "model.safetensors").is_file():
        raise FileNotFoundError(f"NORA-1.5 LIBERO checkpoint is incomplete: {args.base}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    resume = None
    if args.resume == "auto":
        resume = latest_checkpoint(args.output)
    elif args.resume not in ("", "none", "0"):
        resume = Path(args.resume)
    print(
        f"[cfg] batch={args.micro_batch_size}x{accumulation}={args.effective_batch_size} "
        f"steps={args.max_steps} lr={args.learning_rate} warmup={args.warmup_steps} "
        f"clip={args.max_grad_norm} resume={resume}",
        flush=True,
    )

    model, trainable = configure_model(args.base, resume, args.lora_rank, args.lora_alpha)
    dataset = LiberoPlusHorizonDataset(args.dataset, horizon=5, seed=args.seed)
    collator = NoraCollator(model.base_model.model.processor, model.base_model.model.fast_tokenizer,
                            args.base_stats, args.suite_map)
    loader = DataLoader(
        dataset, batch_size=args.micro_batch_size, collate_fn=collator,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    # Start CPU/TF image workers before creating a CUDA context in the parent process.
    iterator = iter(loader)

    device = torch.device("cuda:0")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, eps=1e-8,
    )
    scheduler = get_scheduler(
        "cosine", optimizer=optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    completed_steps = samples_seen = 0
    if resume is not None:
        completed_steps, samples_seen = restore_training_state(resume, optimizer, scheduler)
        print(f"[resume] step={completed_steps} samples={samples_seen:,} from {resume}", flush=True)
    if completed_steps >= args.max_steps:
        print(f"[done] checkpoint already reached requested step {args.max_steps}", flush=True)
        return

    first_audit = completed_steps == 0
    started = time.time()
    while completed_steps < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        loss_values = {"combined": 0.0, "expert": 0.0, "vlm": 0.0}
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            actions = batch.pop("actions").to(device, non_blocking=True)
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(vlm_inputs=inputs, actions=actions, alpha=10.0)
                scaled_loss = output["combined_loss"] / accumulation
            scaled_loss.backward()
            loss_values["combined"] += float(output["combined_loss"].detach()) / accumulation
            loss_values["expert"] += float(output["expert_loss"].detach()) / accumulation
            loss_values["vlm"] += float(output["vlm_loss"].detach()) / accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if first_audit:
            first_audit = False
            reached = sum(
                parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
                for parameter in trainable
            )
            print(f"[audit] nonzero gradients reached {reached}/{len(trainable)} trainable tensors", flush=True)
            if reached == 0:
                raise RuntimeError("no gradient reached NORA visual adapters")
        optimizer.step()
        scheduler.step()
        completed_steps += 1
        samples_seen += args.effective_batch_size
        if completed_steps % args.log_every == 0 or completed_steps == 1:
            elapsed = time.time() - started
            print(
                f"[train] step={completed_steps}/{args.max_steps} samples={samples_seen:,} "
                f"loss={loss_values['combined']:.6f} expert={loss_values['expert']:.6f} "
                f"vlm={loss_values['vlm']:.6f} grad_preclip={float(grad_norm):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} sec_per_step={elapsed/max(1, completed_steps):.3f}",
                flush=True,
            )
        if completed_steps % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, args.output, completed_steps, samples_seen, args)

    if not (args.output / f"ckpt-{completed_steps}").exists():
        save_checkpoint(model, optimizer, scheduler, args.output, completed_steps, samples_seen, args)
    print(f"[done] {completed_steps} optimizer steps, {samples_seen:,} samples", flush=True)


if __name__ == "__main__":
    main()
