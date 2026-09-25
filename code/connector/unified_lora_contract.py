"""Shared, fail-closed contract for the 2026-09-15 dual-LoRA main experiment."""
import json
import os


def projector_mode():
    mode = os.environ.get("VLA_PROJECTOR_UPDATE", "lora")
    if mode not in ("full", "lora"):
        raise ValueError(mode)
    return mode


def check_resume(checkpoint, filename="state.json"):
    if checkpoint is None:
        return
    state = json.loads((checkpoint / filename).read_text())
    if state.get("projector_update", "full") != projector_mode():
        raise RuntimeError("Cannot resume a different projector-update protocol")


def audit(model, is_vision, is_projector):
    groups = {"tower": [], "projector": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not is_vision(name) or not any(k in name for k in ("lora_A", "lora_B", "low_rank_A", "low_rank_B")):
            raise RuntimeError("Non-LoRA trainable parameter: " + name)
        groups["projector" if is_projector(name) else "tower"].append((name, p))
    connector_only = os.environ.get('XVLA_SCOPE') == 'projector_lora' or os.environ.get('OFT_SCOPE') == 'projector_lora'
    if connector_only and (groups['tower'] or not groups['projector']):
        raise RuntimeError('Connector-only must have projector LoRA and zero trainable tower parameters')
    if not connector_only and not all(groups.values()):
        raise RuntimeError("Both tower and projector must contain trainable LoRA factors")
    print("[dual-lora-audit] " + json.dumps({k: {"tensors": len(v), "parameters": sum(p.numel() for _, p in v),
          "names": [n for n, _ in v]} for k, v in groups.items()}), flush=True)
    # Zero-initialized B makes A's first-step gradient zero by design. Require
    # a nonzero B gradient in each region, not nonzero gradients in every A.
    reached = set()
    for region, pairs in groups.items():
        for name, p in pairs:
            if "lora_B" not in name and "low_rank_B" not in name:
                continue
            def hook(grad, region=region):
                if region not in reached and grad.detach().isfinite().all() and grad.detach().abs().sum() > 0:
                    reached.add(region)
                    print("[dual-lora-gradient] nonzero finite gradient: " + region, flush=True)
                return grad
            p.register_hook(hook)
    return groups
