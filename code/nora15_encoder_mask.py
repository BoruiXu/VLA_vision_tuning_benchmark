"""Auditable definition of NORA-1.5's visual encoder and projector boundary."""

from __future__ import annotations

import re


# Qwen2.5-VL is nested as VLAWithExpert.vlm.model.visual in the pinned
# Transformers build. Keep the leading boundary flexible so these checks also
# work after PEFT prepends ``base_model.model``.
VISION_RE = re.compile(r"(^|\.)vlm\.model\.visual\.")
PROJECTOR_RE = re.compile(r"(^|\.)vlm\.model\.visual\.merger\.")


def is_vision(name: str) -> bool:
    return VISION_RE.search(name) is not None


def is_projector(name: str) -> bool:
    return PROJECTOR_RE.search(name) is not None


def vision_linear_modules(model, include_projector=False) -> list[str]:
    """Return exact visual-tower Linear names, excluding the full-rank merger."""
    import torch.nn as nn

    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and is_vision(name) and (include_projector or not is_projector(name))
    ]


def target_modules_regex(names: list[str]) -> str:
    if not names:
        raise ValueError("empty NORA visual LoRA target set")
    return "|".join(re.escape(name) for name in names)


def summarize(model, tag: str = "") -> dict[str, int]:
    counts = {"vision": 0, "projector": 0, "other": 0, "total": 0}
    for name, parameter in model.named_parameters():
        size = parameter.numel()
        counts["total"] += size
        if is_projector(name):
            counts["projector"] += size
        elif is_vision(name):
            counts["vision"] += size
        else:
            counts["other"] += size
    print(
        f"[mask]{' ' + tag if tag else ''} visual_tower={counts['vision']/1e6:.2f}M "
        f"projector={counts['projector']/1e6:.2f}M other={counts['other']/1e6:.2f}M "
        f"total={counts['total']/1e6:.2f}M",
        flush=True,
    )
    return counts
