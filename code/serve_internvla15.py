#!/usr/bin/env python3
"""HTTP inference server for merged InternVLA-A1.5 LIBERO checkpoints."""

from __future__ import annotations

import vla_paths

import argparse
import base64
import json
import random
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_model  # noqa: E402


ROOT = Path(vla_paths.resolve(''))
REPO = ROOT / "third_party/InternVLA-A-series"
sys.path.insert(0, str(REPO / "src"))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import (  # noqa: E402
    InternVLAA15Policy,
)
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (  # noqa: E402
    InternVLAA15ChatProcessorTransformFn,
)
from lerobot.transforms.core import (  # noqa: E402
    ComposeFieldsTransform,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_PREFIX, OBS_STATE, OBS_STR  # noqa: E402


EXPECTED_STOCK_EXTRAS = {
    "model._wan_grid_sizes",
    "model.learnable_to_wan_proj.bias",
    "model.learnable_to_wan_proj.weight",
}


def decode_json_numpy(value):
    """Decode the compact ndarray representation emitted by json_numpy.patch()."""
    if isinstance(value, dict):
        if {"__numpy__", "dtype", "shape"}.issubset(value):
            raw = base64.b64decode(value["__numpy__"])
            array = np.frombuffer(raw, dtype=np.dtype(value["dtype"]))
            return array.reshape(tuple(value["shape"]))
        return {key: decode_json_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_json_numpy(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, default=ROOT / "models/Qwen3.5-2B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8995)
    return parser.parse_args()


def image_tensor(value) -> torch.Tensor:
    array = np.asarray(value, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {array.shape}")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)


class InternVLAInference:
    def __init__(self, checkpoint: Path, qwen: Path):
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint / "model.safetensors")
        stats_path = checkpoint / "stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(stats_path)
        self.stats = json.loads(stats_path.read_text())
        expected = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
        if set(self.stats) != expected:
            raise RuntimeError(f"expected four LIBERO stats keys, got {sorted(self.stats)}")

        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.vlm_model_name_or_path = str(qwen)
        config.device = "cuda:0"
        config.action_loss_only = True
        config.inference_backend = "standard"
        config.gradient_checkpointing = False
        self.policy = InternVLAA15Policy(config)
        missing, unexpected = load_model(
            self.policy,
            str(checkpoint / "model.safetensors"),
            strict=False,
            device="cuda:0",
        )
        if missing:
            raise RuntimeError(f"checkpoint is missing policy weights: {sorted(missing)}")
        unexpected = set(unexpected)
        if unexpected not in (set(), EXPECTED_STOCK_EXTRAS):
            raise RuntimeError(
                "unexpected checkpoint weights: "
                f"expected [] or {sorted(EXPECTED_STOCK_EXTRAS)}, got {sorted(unexpected)}"
            )
        print(f"[serve] validated checkpoint extras={sorted(unexpected)}", flush=True)
        self.policy.to("cuda:0")
        self.policy.eval()
        if any("lora_" in name for name, _ in self.policy.named_parameters()):
            raise RuntimeError("merged checkpoint still contains LoRA parameters")

        self.resize = ResizeImagesWithPadFn(
            height=224, width=224, mapping={"image": "", "wrist_image": ""}
        )
        self.remap = RemapImageKeyTransformFn(
            mapping={
                "image": f"{OBS_IMAGES}.image0",
                "wrist_image": f"{OBS_IMAGES}.image1",
            }
        )
        self.compose = ComposeFieldsTransform(mapping={OBS_STATE: ["state"]})
        self.chat = InternVLAA15ChatProcessorTransformFn(
            pretrained_model_name_or_path=str(qwen), max_length=650, num_views=3,
            tokenize_state=True, max_state_dim=32, use_fast_action_tokens=False,
            mode="eval", action_mode="end_effector",
        )
        self.lock = threading.Lock()

    @staticmethod
    def suite_key(value: str) -> str:
        key = value.removesuffix("_no_noops")
        if key not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
            raise ValueError(f"invalid InternVLA stats key {value!r}")
        return key

    def preprocess(self, payload: dict) -> tuple[dict[str, torch.Tensor], str]:
        suite = self.suite_key(str(payload["unnorm_key"]))
        state = torch.tensor(np.asarray(payload["state"], dtype=np.float32))
        if state.numel() != 8:
            raise ValueError(f"expected 8-d LIBERO state, got {tuple(state.shape)}")
        state_stats = self.stats[suite]["observation.state"]
        mean = torch.tensor(state_stats["mean"], dtype=torch.float32)
        std = torch.tensor(state_stats["std"], dtype=torch.float32)
        state = (state - mean) / (std + 1e-6)
        data = {
            "image": image_tensor(payload["full_image"]),
            "wrist_image": image_tensor(payload["wrist_image"]),
            "state": state,
            "task": str(payload["instruction"]),
        }
        data = self.resize(data)
        data = self.remap(data)
        data = self.compose(data)
        data = self.chat(data)
        data[OBS_STATE] = F.pad(data[OBS_STATE], (0, 32 - data[OBS_STATE].shape[-1]))
        batch = {
            OBS_STATE: data[OBS_STATE].unsqueeze(0),
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": data[f"{OBS_STR}.input_ids"].unsqueeze(0),
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"].unsqueeze(0),
            f"{OBS_STR}.fast_token_mask": data[f"{OBS_STR}.fast_token_mask"].unsqueeze(0),
        }
        batch = {key: value.to("cuda:0", non_blocking=True) for key, value in batch.items()}
        return batch, suite

    def act(self, payload: dict) -> np.ndarray:
        seed = int(payload.get("request_seed", 7))
        with self.lock, torch.random.fork_rng(devices=[0]):
            # The Qwen processor and CUDA policy are shared by every request.
            # Keep the complete prepare/predict path serialized.
            batch, suite = self.preprocess(payload)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                normalized = self.policy.predict_action_chunk(batch)[0, :, :7]
        normalized = normalized.float().cpu().numpy()
        action_stats = self.stats[suite][ACTION]
        mean = np.asarray(action_stats["mean"], dtype=np.float32)
        std = np.asarray(action_stats["std"], dtype=np.float32)
        actions = normalized * (std + 1e-6) + mean
        # Match the released InternVLA LIBERO backend: denormalize first, then
        # clip every action dimension to the immutable per-suite action space.
        low = np.asarray(action_stats["min"], dtype=np.float32)
        high = np.asarray(action_stats["max"], dtype=np.float32)
        actions = np.clip(actions, low, high)
        # InternVLA/OXE: 0=close, 1=open. LIBERO env: +1=close, -1=open.
        actions[..., 6] = np.where(actions[..., 6] < 0.5, 1.0, -1.0)
        return actions.astype(np.float32)


def main() -> None:
    args = parse_args()
    print(f"[serve] loading checkpoint={args.checkpoint}", flush=True)
    inference = InternVLAInference(args.checkpoint, args.qwen)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, value) -> None:
            encoded = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_json(200, {"ok": True, "chunk_size": 8})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/act":
                self.send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = decode_json_numpy(json.loads(self.rfile.read(length)))
                actions = inference.act(payload)
                self.send_json(200, actions.tolist())
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *_args) -> None:
            return

    print("[serve] ready action_chunk=8 flow_steps=10", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
