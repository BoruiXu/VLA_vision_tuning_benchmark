"""Auditable tower/projector masks for the X-VLA visual-scope ablation."""

from __future__ import annotations

import re


TOWER_RE = r"(^|\.)vlm\.(vision_tower|image_pos_embed|visual_temporal_embed)(\.|$)"
PROJ_RE = r"(^|\.)vlm\.(image_projection|image_proj_norm)(\.|$)"
VISION_RE = (
    r"(^|\.)vlm\.(vision_tower|image_proj_norm|image_projection|"
    r"image_pos_embed|visual_temporal_embed)(\.|$)"
)
TRUNK_RE = r"(^|\.)vlm\.language_model\."

_tower = re.compile(TOWER_RE)
_projector = re.compile(PROJ_RE)
_vision = re.compile(VISION_RE)
_trunk = re.compile(TRUNK_RE)


def is_tower(name: str) -> bool:
    return _tower.search(name) is not None


def is_projection(name: str) -> bool:
    return _projector.search(name.replace(".parametrizations.image_projection.", ".image_projection.")) is not None


def is_vision(name: str) -> bool:
    return _vision.search(name.replace(".parametrizations.image_projection.", ".image_projection.")) is not None


def is_trunk(name: str) -> bool:
    return _trunk.search(name) is not None


def in_scope(name: str, scope: str) -> bool:
    if scope == "tower_lora":
        return is_tower(name)
    if scope == "projector_only":
        return is_projection(name)
    if scope in ("vision_lora", "full_rank_vision"):
        return is_vision(name)
    raise ValueError(f"unsupported X-VLA visual scope: {scope}")


def linear_modules(model, scope: str = "vision_lora") -> list[str]:
    import torch.nn as nn

    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and in_scope(name, scope)
    ]


def vision_linear_modules(model) -> list[str]:
    return linear_modules(model, "vision_lora")


def target_modules_regex(names) -> str:
    return "|".join(re.escape(name) for name in names)


def full_rank_regex(scope: str) -> str:
    if scope == "projector_only":
        return r"^vlm\.(image_projection|image_proj_norm)(\.|$)"
    if scope == "full_rank_vision":
        return (
            r"^vlm\.(vision_tower|image_proj_norm|image_projection|"
            r"image_pos_embed|visual_temporal_embed)(\.|$)"
        )
    raise ValueError(f"scope {scope!r} is not full-rank")


def summarize(model, tag: str = "") -> dict[str, int]:
    tower = projector = trunk = other = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if is_tower(name):
            tower += count
        elif is_projection(name):
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
