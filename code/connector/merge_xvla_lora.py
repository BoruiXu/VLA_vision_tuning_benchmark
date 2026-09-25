#!/usr/bin/env python3
"""Merge an X-VLA vision LoRA checkpoint and its full-rank projector into a serveable model."""

from __future__ import annotations

import vla_paths

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel

REPO = os.environ.get("XVLA_REPO", vla_paths.resolve('third_party/X-VLA'))
sys.path.insert(0, REPO)

from models.modeling_xvla import XVLA  # noqa: E402
from models.processing_xvla import XVLAProcessor  # noqa: E402


CUSTOM_CODE = (
    "action_hub.py",
    "configuration_florence2.py",
    "configuration_xvla.py",
    "modeling_florence2.py",
    "modeling_xvla.py",
    "processing_xvla.py",
    "transformer.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=vla_paths.resolve('models/xvla-libero'))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    adapter = checkpoint / "lora_adapter"
    projector_file = checkpoint / "vision_projector.pt"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"missing LoRA adapter: {adapter}")
    if not projector_file.is_file():
        raise FileNotFoundError(f"missing projector sidecar: {projector_file}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    print(f"[merge] loading base {args.base}", flush=True)
    base = XVLA.from_pretrained(args.base)
    adapter_manifest = checkpoint / "adapter_manifest.json"
    protocol = json.loads(adapter_manifest.read_text()) if adapter_manifest.is_file() else {}
    if protocol.get("projector_update") == "lora":
        import xvla_projector_lora
        xvla_projector_lora.install(base.vlm, protocol["lora_rank"], protocol["lora_alpha"])
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)

    try:
        projector = torch.load(projector_file, map_location="cpu", weights_only=True)
    except TypeError:
        projector = torch.load(projector_file, map_location="cpu")
    named = dict(model.named_parameters())
    missing = sorted(set(projector) - set(named))
    if missing:
        raise KeyError(f"projector tensor names not found in PEFT model: {missing}")
    with torch.no_grad():
        for name, value in projector.items():
            parameter = named[name]
            if tuple(parameter.shape) != tuple(value.shape):
                raise ValueError(f"shape mismatch for {name}: {parameter.shape} != {value.shape}")
            parameter.copy_(value.to(dtype=parameter.dtype, device=parameter.device))
    print(f"[merge] restored {len(projector)} projector tensors", flush=True)

    merged = model.merge_and_unload()
    if protocol.get("projector_update") == "lora":
        xvla_projector_lora.merge(merged.vlm)
    if any("lora_" in name for name, _ in merged.named_parameters()):
        raise RuntimeError("LoRA parameters remain after merge_and_unload")
    merged.save_pretrained(output, safe_serialization=True)
    XVLAProcessor.from_pretrained(args.base).save_pretrained(output)
    for filename in CUSTOM_CODE:
        source = Path(args.base) / filename
        if not source.is_file():
            raise FileNotFoundError(f"missing custom model code: {source}")
        shutil.copy2(source, output / filename)

    adapter_manifest = checkpoint / "adapter_manifest.json"
    manifest = {
        "format": "xvla-merged-vision-lora-v1",
        "base": str(Path(args.base).resolve()),
        "checkpoint": str(checkpoint),
        "output": str(output),
        "projector_tensors": sorted(projector),
        "adapter_manifest": json.loads(adapter_manifest.read_text())
        if adapter_manifest.is_file() else None,
    }
    (output / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # A real reload catches missing config/custom-code or malformed weight files before evaluation.
    reloaded = XVLA.from_pretrained(output)
    n_params = sum(parameter.numel() for parameter in reloaded.parameters())
    print(f"[merge] reload PASS: {n_params:,} params -> {output}", flush=True)


if __name__ == "__main__":
    main()
