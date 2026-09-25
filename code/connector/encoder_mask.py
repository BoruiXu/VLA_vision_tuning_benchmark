"""Auditable visual-scope masks for OpenVLA-OFT.

The tower/projector split is intentionally explicit because the scope ablation must not adapt
the Llama trunk or either continuous-control head.  Patterns also match PEFT's
``base_model.model.`` prefix.
"""

from __future__ import annotations

import re


TOWER_RE = r"(^|\.)vision_backbone\."
PROJECTOR_RE = r"(^|\.)projector\."
VISION_RE = r"(^|\.)(vision_backbone|projector)\."
TRUNK_RE = r"(^|\.)language_model\."

_tower = re.compile(TOWER_RE)
_projector = re.compile(PROJECTOR_RE)
_vision = re.compile(VISION_RE)
_trunk = re.compile(TRUNK_RE)


def is_tower(name: str) -> bool:
    return _tower.search(name) is not None


def is_projector(name: str) -> bool:
    return _projector.search(name) is not None


def is_vision(name: str) -> bool:
    return _vision.search(name) is not None


def is_trunk(name: str) -> bool:
    return _trunk.search(name) is not None


def in_scope(name: str, scope: str) -> bool:
    if scope == "tower_lora":
        return is_tower(name)
    if scope in ("projector_only", "projector_lora"):
        return is_projector(name)
    if scope in ("vision_lora", "full_rank_vision"):
        return is_vision(name)
    raise ValueError(f"unsupported OpenVLA-OFT visual scope: {scope}")


def linear_modules(model, scope: str = "vision_lora") -> list[str]:
    """Return exact full names of Linear modules in a declared visual scope."""
    import torch.nn as nn

    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and in_scope(name, scope)
    ]


def vision_linear_modules(model) -> list[str]:
    """Backward-compatible name used by the completed combined-vision runs."""
    return linear_modules(model, "vision_lora")


def target_modules_regex(names) -> str:
    """Exact alternation; suffix matching could leak into the language model."""
    return "|".join(re.escape(name) for name in names)


def summarize(model, tag: str = "") -> dict[str, int]:
    tower = projector = trunk = other = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if is_tower(name):
            tower += count
        elif is_projector(name):
            projector += count
        elif is_trunk(name):
            trunk += count
        else:
            other += count
    vision = tower + projector
    total = vision + trunk + other
    print(
        f"[mask]{' ' + tag if tag else ''} tower {tower/1e6:.1f}M  "
        f"projector {projector/1e6:.1f}M  vision {vision/1e6:.1f}M "
        f"({100*vision/total:.2f}%)  trunk {trunk/1e6:.1f}M  other {other/1e6:.1f}M  "
        f"total {total/1e6:.1f}M  trunk/vision {trunk/max(vision, 1):.2f}",
        flush=True,
    )
    return {
        "tower": tower,
        "projector": projector,
        "vision": vision,
        "trunk": trunk,
        "other": other,
        "total": total,
    }
