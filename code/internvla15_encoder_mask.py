"""Auditable InternVLA-A1.5 visual-tower and projector boundary."""

from __future__ import annotations

import re


VISION_RE = re.compile(r"^model\.qwen3_5_with_expert\.qwen3_5\.(?:model\.)?visual\.")
PROJECTOR_RE = re.compile(
    r"^model\.qwen3_5_with_expert\.qwen3_5\.(?:model\.)?visual\."
    r"(?:base_model\.model\.)?merger\."
)


def is_vision(name: str) -> bool:
    return VISION_RE.search(name) is not None


def is_projector(name: str) -> bool:
    return PROJECTOR_RE.search(name) is not None


def relative_linear_targets(visual) -> list[str]:
    """Return exact tower Linear names relative to Qwen3.5VisionModel."""
    import torch.nn as nn

    return [
        name
        for name, module in visual.named_modules()
        if isinstance(module, nn.Linear) and not name.startswith("merger.")
    ]


def target_regex(names: list[str]) -> str:
    if not names:
        raise ValueError("empty InternVLA visual LoRA target set")
    return "^(?:" + "|".join(re.escape(name) for name in names) + ")$"


def summarize(policy, tag: str = "") -> dict[str, int]:
    counts = {"tower": 0, "projector": 0, "other": 0, "total": 0}
    for name, parameter in policy.named_parameters():
        size = parameter.numel()
        counts["total"] += size
        if is_projector(name):
            counts["projector"] += size
        elif is_vision(name):
            counts["tower"] += size
        else:
            counts["other"] += size
    print(
        f"[mask]{' ' + tag if tag else ''} tower={counts['tower']/1e6:.2f}M "
        f"projector={counts['projector']/1e6:.2f}M other={counts['other']/1e6:.2f}M "
        f"total={counts['total']/1e6:.2f}M",
        flush=True,
    )
    return counts
