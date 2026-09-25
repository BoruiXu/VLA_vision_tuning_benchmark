#!/usr/bin/env python3
"""Merge a resumable pi0.5 visual-LoRA checkpoint into a vanilla policy checkpoint."""

from __future__ import annotations

import vla_paths

import argparse
import json
import shutil
from pathlib import Path

import safetensors.torch
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as training_config
import register_pi05_config
register_pi05_config.install()

import pi05_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--base", type=Path, default=Path(vla_paths.resolve('models/pi05-libero-pytorch')))
    parser.add_argument("--config-name", default="pi05_visonly")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = (args.output or Path(str(checkpoint) + "--merged")).resolve()
    manifest_path = checkpoint / "lora_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing LoRA manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output}")
        shutil.rmtree(output)
    temporary = Path(str(output) + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    cfg = training_config.get_config(args.config_name)
    object.__setattr__(cfg.model, "dtype", cfg.pytorch_training_precision)
    device = torch.device(args.device)
    model = PI0Pytorch(cfg.model).to(device)
    safetensors.torch.load_model(model, args.base / "model.safetensors", strict=True, device=str(device))
    injected = pi05_lora.inject_lora(
        model,
        rank=int(manifest["rank"]),
        alpha=int(manifest["alpha"]),
        target_regex=str(manifest["target_regex"]),
        adapter_seed=int(manifest["adapter_seed"]),
    )
    if injected["target_modules"] != manifest["target_modules"]:
        raise RuntimeError("merge target modules differ from the training manifest")
    safetensors.torch.load_model(model, checkpoint / "model.safetensors", strict=True, device=str(device))
    merged_names = pi05_lora.merge_lora_(model)
    safetensors.torch.save_model(model, temporary / "model.safetensors")

    for name in ("metadata.pt",):
        if (checkpoint / name).is_file():
            shutil.copy2(checkpoint / name, temporary / name)
    if (checkpoint / "assets").is_dir():
        shutil.copytree(checkpoint / "assets", temporary / "assets")
    merge_manifest = {
        "format": "pi05-merged-vision-lora-v1",
        "base": str(args.base.resolve()),
        "checkpoint": str(checkpoint),
        "completed_optimizer_steps": int(checkpoint.name),
        "merged_module_count": len(merged_names),
        "source_lora_manifest": manifest,
    }
    (temporary / "merge_manifest.json").write_text(json.dumps(merge_manifest, indent=2) + "\n")
    temporary.rename(output)
    print(f"[merge] complete: {len(merged_names)} modules -> {output}", flush=True)


if __name__ == "__main__":
    main()
