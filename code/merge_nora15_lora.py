#!/usr/bin/env python3
"""Merge a NORA-1.5 visual LoRA checkpoint into a self-contained model directory."""

from __future__ import annotations

import vla_paths

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import tensorflow as tf
from peft import PeftModel


ROOT = Path(vla_paths.resolve(''))
NORA_REPO = ROOT / "third_party/nora-1.5"
sys.path.insert(0, str(NORA_REPO))
sys.path.insert(0, str(ROOT / "code"))

tf.config.set_visible_devices([], "GPU")

from inference.modelling_expert import VLAWithExpert  # noqa: E402

import nora15_encoder_mask as mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "models/nora-1.5-libero")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-stats", type=Path,
        default=Path(vla_paths.resolve('models/openvla-oft/dataset_statistics.json')),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = args.checkpoint / "lora_adapter"
    projector_file = args.checkpoint / "vision_projector.pt"
    state_file = args.checkpoint / "state.json"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"missing adapter: {adapter}")
    if not projector_file.is_file():
        raise FileNotFoundError(f"missing visual projector: {projector_file}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[merge] base={args.base}", flush=True)
    base = VLAWithExpert.from_pretrained(args.base)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    projector = torch.load(projector_file, map_location="cpu", weights_only=True)
    named = dict(model.named_parameters())
    missing = sorted(set(projector) - set(named))
    if missing:
        raise KeyError(f"projector tensors absent from PEFT model: {missing[:10]}")
    with torch.no_grad():
        for name, value in projector.items():
            parameter = named[name]
            if tuple(parameter.shape) != tuple(value.shape):
                raise ValueError(f"shape mismatch for {name}: {parameter.shape} != {value.shape}")
            parameter.copy_(value.to(dtype=parameter.dtype))
    merged = model.merge_and_unload()
    if any("lora_" in name for name, _ in merged.named_parameters()):
        raise RuntimeError("LoRA parameters remain after merge_and_unload")
    merged.save_pretrained(args.output)
    shutil.copy2(args.base_stats, args.output / "dataset_statistics.json")
    manifest = {
        "format": "nora15-merged-vision-lora-v1",
        "base": str(args.base.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output": str(args.output.resolve()),
        "projector_update": json.loads(state_file.read_text()).get("projector_update", "full"),
        "trainable_scope": json.loads(state_file.read_text()).get("trainable_scope"),
        "normalization": (
            "immutable four-suite LIBERO base q01/q99; gripper uses NORA/OXE "
            "0=close,1=open and is converted to LIBERO +1=close,-1=open at eval"
        ),
        "projector_tensors": sorted(projector),
        "checkpoint_state": json.loads(state_file.read_text()) if state_file.is_file() else None,
    }
    (args.output / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model_file = args.output / "model.safetensors"
    if not model_file.is_file() or model_file.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"merged NORA model file is missing or too small: {model_file}")
    state = torch.load(args.checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    print(
        f"[merge] complete step={state['completed_optimizer_steps']} "
        f"projector_tensors={len(projector)} size={model_file.stat().st_size:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
