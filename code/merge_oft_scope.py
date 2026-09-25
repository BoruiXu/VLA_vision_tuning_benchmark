#!/usr/bin/env python3
"""Merge/materialize any OpenVLA-OFT visual-scope ablation checkpoint."""

from __future__ import annotations

import vla_paths

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor


ROOT = Path(vla_paths.resolve(''))
REPO = Path(vla_paths.resolve('third_party/openvla-oft'))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "code"))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import (  # noqa: E402
    PrismaticImageProcessor,
    PrismaticProcessor,
)

import encoder_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--base", type=Path, default=Path(vla_paths.resolve('models/openvla-oft')))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def copy_eval_sidecars(checkpoint: Path, output: Path) -> list[str]:
    copied = []
    for source in checkpoint.iterdir():
        if source.is_file() and (
            source.name.startswith("action_head--")
            or source.name.startswith("proprio_projector--")
        ):
            shutil.copy2(source, output / source.name)
            copied.append(source.name)

    stats_source = checkpoint / "dataset_statistics.json"
    if not stats_source.is_file():
        raise FileNotFoundError(f"missing dataset statistics: {stats_source}")
    stats = json.loads(stats_source.read_text())
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    missing = [key + "_no_noops" for key in suites if key + "_no_noops" not in stats]
    if missing:
        raise KeyError(f"expected suite-specific base statistics: {missing}")
    for key in suites:
        stats[key] = stats[key + "_no_noops"]
    (output / "dataset_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    copied.append("dataset_statistics.json (with suite aliases)")
    return copied


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    base = args.base.resolve()
    output = (args.output or Path(str(checkpoint) + "--merged")).resolve()
    state_file = checkpoint / "training_state.pt"
    if not state_file.is_file():
        raise FileNotFoundError(f"missing training state: {state_file}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    training_state = torch.load(state_file, map_location="cpu", weights_only=False)
    scope = training_state.get("scope")
    if scope not in {"vision_lora", "tower_lora", "projector_only", "full_rank_vision"}:
        raise ValueError(f"checkpoint has missing/unsupported scope: {scope!r}")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    model = AutoModelForVision2Seq.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    parameterization = "lora"
    if scope in {"vision_lora", "tower_lora"}:
        adapter = checkpoint / "lora_adapter"
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        if any("lora_" in name for name, _ in model.named_parameters()):
            raise RuntimeError("LoRA parameters remain after merge")
    else:
        parameterization = "full_rank"
        visual_file = checkpoint / "vision_scope_state.pt"
        visual_state = torch.load(visual_file, map_location="cpu", weights_only=True)
        named = dict(model.named_parameters())
        expected = sorted(name for name in named if encoder_mask.in_scope(name, scope))
        if sorted(visual_state) != expected:
            raise KeyError(
                f"visual state mismatch: saved={len(visual_state)}, expected={len(expected)}"
            )
        with torch.no_grad():
            for name, value in visual_state.items():
                parameter = named[name]
                if tuple(parameter.shape) != tuple(value.shape):
                    raise ValueError(f"shape mismatch for {name}")
                parameter.copy_(value.to(dtype=parameter.dtype))

    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(base, trust_remote_code=True).save_pretrained(output)
    copied = copy_eval_sidecars(checkpoint, output)
    manifest = {
        "format": "openvla-oft-merged-visual-scope-v1",
        "base": str(base),
        "checkpoint": str(checkpoint),
        "output": str(output),
        "scope": scope,
        "parameterization": parameterization,
        "completed_optimizer_steps": int(training_state["completed_optimizer_steps"]),
        "normalization": "immutable OFT_m suite stats, suite-conditioned",
        "copied_sidecars": sorted(copied),
    }
    (output / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[merge] complete: scope={scope}, step={manifest['completed_optimizer_steps']} -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
