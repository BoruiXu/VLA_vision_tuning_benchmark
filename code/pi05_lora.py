"""Minimal, auditable LoRA support for the PyTorch pi0.5 visual encoder.

The stock OpenPI trainer has a full-rank regex mask but no PEFT/LoRA path.  This
module wraps only ``nn.Linear`` modules in the SigLIP tower and its multimodal
projector.  The base layer is frozen; only the two low-rank matrices train.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_TARGET_RE = (
    r"^paligemma_with_expert\.paligemma\.model\."
    r"(vision_tower|multi_modal_projector)(\.|$)"
)


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable low-rank residual."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int, seed: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank)

        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

        # Use a private CPU generator so adapter initialisation cannot perturb
        # DataLoader/flow-noise RNG streams.  This mirrors PEFT's Gaussian
        # initialisation: A ~ N(0, 1/r), B = 0, hence exact stock output at step 0.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        a = torch.empty(rank, base_layer.in_features, dtype=torch.float32, device="cpu")
        a.normal_(mean=0.0, std=1.0 / rank, generator=generator)
        self.lora_A = nn.Parameter(a.to(base_layer.weight.device))
        self.lora_B = nn.Parameter(
            torch.zeros(base_layer.out_features, rank, dtype=torch.float32, device=base_layer.weight.device)
        )

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base_layer.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        low_rank = F.linear(F.linear(inputs.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base + (low_rank * self.scaling).to(base.dtype)


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 32,
    alpha: int = 16,
    target_regex: str = DEFAULT_TARGET_RE,
    adapter_seed: int = 420032,
) -> dict:
    """Freeze the model and replace exactly the target visual Linear modules."""

    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("LoRA has already been injected")
    pattern = re.compile(target_regex)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and pattern.search(name)
    ]
    if not targets:
        raise RuntimeError(f"pi0.5 LoRA target regex matched no Linear modules: {target_regex}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for index, name in enumerate(targets):
        layer = model.get_submodule(name)
        _set_submodule(model, name, LoRALinear(layer, rank, alpha, adapter_seed + index))

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    outside = [name for name, _ in trainable if ".lora_A" not in name and ".lora_B" not in name]
    if outside:
        raise RuntimeError(f"trainable tensors escaped the LoRA adapters: {outside[:8]}")
    if len(trainable) != 2 * len(targets):
        raise RuntimeError(f"expected {2 * len(targets)} trainable tensors, found {len(trainable)}")
    if any(torch.count_nonzero(module.lora_B).item() for module in model.modules() if isinstance(module, LoRALinear)):
        raise RuntimeError("LoRA B must be exactly zero at initialisation")

    manifest = {
        "format": "pi05-vision-lora-v1",
        "rank": int(rank),
        "alpha": int(alpha),
        "scaling": float(alpha) / float(rank),
        "dropout": 0.0,
        "adapter_seed": int(adapter_seed),
        "target_regex": target_regex,
        "target_modules": targets,
        "target_module_count": len(targets),
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": sum(p.numel() for _, p in trainable),
        "trainable_scope": "SigLIP vision tower + multimodal projector Linear LoRA; trunk/action expert frozen",
        "zero_init_contract": "Gaussian A with private RNG; exactly-zero B; stock-equivalent output at step 0",
    }
    return manifest


def summarize(model: nn.Module) -> dict:
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    count = sum(p.numel() for _, p in trainable)
    if not targets or not trainable:
        raise RuntimeError("pi0.5 LoRA audit found an empty target/trainable set")
    return {
        "target_module_count": len(targets),
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": count,
        "total_parameter_count": total,
        "trainable_fraction": count / total,
        "first_target": targets[0][0],
        "last_target": targets[-1][0],
    }


def audit_gradients(model: nn.Module) -> dict:
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    reached = [(name, p) for name, p in trainable if p.grad is not None]
    nonzero = [(name, p) for name, p in reached if torch.count_nonzero(p.grad).item()]
    if not nonzero:
        raise RuntimeError("no non-zero gradient reached any pi0.5 LoRA tensor")
    return {
        "trainable": len(trainable),
        "with_gradient": len(reached),
        "nonzero_gradient": len(nonzero),
    }


def write_checkpoint_manifest(directory: str | Path, manifest: dict, gradient_audit: dict | None) -> None:
    payload = dict(manifest)
    if gradient_audit is not None:
        payload["first_step_gradient_audit"] = gradient_audit
    Path(directory, "lora_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


@torch.no_grad()
def merge_lora_(model: nn.Module) -> list[str]:
    """Merge every LoRA residual and restore the vanilla ``nn.Linear`` tree."""

    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    if not targets:
        raise RuntimeError("cannot merge: model contains no pi0.5 LoRA layers")
    names = []
    for name, module in targets:
        delta = torch.matmul(module.lora_B, module.lora_A) * module.scaling
        module.base_layer.weight.add_(delta.to(module.base_layer.weight.dtype))
        _set_submodule(model, name, module.base_layer)
        names.append(name)
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("LoRA layers remain after merge")
    return names
