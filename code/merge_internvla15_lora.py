#!/usr/bin/env python3
"""Merge an InternVLA-A1.5 visual LoRA checkpoint into a full policy."""

from __future__ import annotations

import vla_paths

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, load_model


ROOT = Path(vla_paths.resolve(''))
REPO = ROOT / "third_party/InternVLA-A-series"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ROOT / "code"))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import (  # noqa: E402
    InternVLAA15Policy,
)

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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output {args.output}")
    for file in (
        args.checkpoint / "adapter/adapter_config.json",
        args.checkpoint / "vision_projector.safetensors",
        args.checkpoint / "state.json",
    ):
        if not file.is_file():
            raise FileNotFoundError(file)
    config = PreTrainedConfig.from_pretrained(args.base)
    config.vlm_model_name_or_path = str(args.qwen)
    config.device = "cpu"
    config.action_loss_only = True
    config.inference_backend = "standard"
    config.gradient_checkpointing = False
    config.push_to_hub = False
    config.repo_id = None
    policy = InternVLAA15Policy(config)
    missing, unexpected = load_model(
        policy, str(args.base / "model.safetensors"), strict=False, device="cpu"
    )
    if missing or set(unexpected) != EXPECTED_IGNORED_WEIGHTS:
        raise RuntimeError(f"base load mismatch: missing={missing} unexpected={unexpected}")
    visual = PeftModel.from_pretrained(
        policy.model.qwen3_5_with_expert.qwen3_5.model.visual,
        args.checkpoint / "adapter", is_trainable=False, local_files_only=True,
    )
    policy.model.qwen3_5_with_expert.qwen3_5.model.visual = visual
    projector = load_file(str(args.checkpoint / "vision_projector.safetensors"), device="cpu")
    named = dict(policy.named_parameters())
    expected = {name for name in named if mask.is_projector(name)}
    if set(projector) != expected:
        raise RuntimeError("projector sidecar does not match PEFT-wrapped model")
    with torch.no_grad():
        for name, value in projector.items():
            named[name].copy_(value.to(named[name]))
    merged_visual = visual.merge_and_unload(safe_merge=True)
    if any("lora_" in name for name, _ in merged_visual.named_parameters()):
        raise RuntimeError("LoRA tensors remain after merge")
    policy.model.qwen3_5_with_expert.qwen3_5.model.visual = merged_visual
    args.output.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(args.output)
    shutil.copy2(args.base / "stats.json", args.output / "stats.json")
    state = json.loads((args.checkpoint / "state.json").read_text())
    manifest = {
        "format": "internvla15-merged-vision-lora-v1",
        "base": str(args.base.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "projector_update": state.get("projector_update", "full"),
        "trainable_scope": state.get("trainable_scope"),
        "checkpoint_state": state,
    }
    (args.output / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model_file = args.output / "model.safetensors"
    if not model_file.is_file() or model_file.stat().st_size < 5_000_000_000:
        raise RuntimeError(f"merged model missing or too small: {model_file}")
    print(
        f"[merge] complete step={state['completed_optimizer_steps']} "
        f"size={model_file.stat().st_size:,}", flush=True,
    )


if __name__ == "__main__":
    main()
