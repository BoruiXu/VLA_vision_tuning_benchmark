"""Launcher for the X-VLA arm of the cross-architecture encoder study.

Runs the X-VLA repo's `train.py` with LoRA r=32 on the visual tower and a low-rank
parametrization of the vision-to-language projection in the released reference setup.
The action pathway and shared trunk stay frozen. Historical full-rank branches remain in
this shared implementation, but are not selected by the public paper launchers.

THE THREE PATCHES

  (1) LORA SCOPE  *the independent variable*. `XVLA.from_pretrained` is wrapped so the model it
                  returns is already PEFT-wrapped, with `target_modules` set to the 96 nn.Linear
                  modules inside `vlm.vision_tower` (see xvla_encoder_mask.py) and nothing else.
                  The repo also ships `peft_train.py`, but that one hardcodes r=8,
                  `target_modules="all-linear"` and `modules_to_save=[soft_prompt_hub,
                  action_encoder, action_decoder]` -- i.e. it adapts the whole model and fully
                  trains the action pathway, which is the opposite of this experiment.

  (2) TRAIN SAVE  Save the LoRA adapter and the full-rank projector as separate training
                  artifacts.  Merging is deliberately performed only after training by
                  `merge_xvla_lora.py`; optimizer/RNG resume state is handled by repo/train.py.

  (3) GRAD AUDIT  One-shot count of non-zero gradients at the first optimizer step. The vision
                  encoder sits upstream of the entire frozen stack, so backward has to traverse
                  the frozen Florence-2 trunk and the frozen action decoder to reach it; any
                  no_grad on that path leaves the adapters at their initialisation while the loss
                  still falls.

WHAT IS NOT CHANGED
    The action space. X-VLA emits ABSOLUTE end-effector pose (`action_mode='ee6d'`,
    `num_actions=30`), and the dataloader is built from `model.action_mode` inside the repo's own
    `main()`. Nothing here touches it -- which is the point: the action space is the architectural
    variable H3 is about, so it must stay exactly as the pretrained model defines it.

Env knobs (the shell script is the single source of truth for values):
    XVLA_LORA_RANK      default 32
    XVLA_LORA_ALPHA     default 16
    XVLA_LORA_DROPOUT   default 0.0
    XVLA_TRAIN_PROJ     0|1, default 1 -- also train `image_projection` (a bare nn.Parameter that
                        LoRA cannot wrap) and `image_proj_norm` full-rank. See xvla_encoder_mask.
"""

import vla_paths

import copy
import json
import os
import sys

REPO = os.environ.get("XVLA_REPO", vla_paths.resolve('third_party/X-VLA'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LORA_RANK = int(os.environ.get("XVLA_LORA_RANK", "32"))
LORA_ALPHA = int(os.environ.get("XVLA_LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("XVLA_LORA_DROPOUT", "0.0"))
SCOPE = os.environ.get("XVLA_SCOPE", "vision_lora").strip().lower()
if SCOPE not in {"vision_lora", "tower_lora"}:
    raise ValueError(f"LoRA launcher only accepts vision_lora|tower_lora, got {SCOPE!r}")
TRAIN_PROJ = os.environ.get("XVLA_TRAIN_PROJ", "1") == "1"
import unified_lora_contract as contract
PROJECTOR_UPDATE = contract.projector_mode()
DELTA_LAMBDA = float(os.environ.get("XVLA_DELTA_LAMBDA", "0"))
DELTA_ATTENTION = os.environ.get("XVLA_DELTA_ATTENTION", "taylor").strip().lower()
DELTA_TEMPERATURE = float(os.environ.get("XVLA_DELTA_TEMPERATURE", "1"))
if DELTA_LAMBDA < 0:
    raise ValueError(f"XVLA_DELTA_LAMBDA must be non-negative, got {DELTA_LAMBDA}")
if DELTA_ATTENTION not in {"uniform", "taylor"}:
    raise ValueError(
        f"XVLA_DELTA_ATTENTION must be uniform|taylor, got {DELTA_ATTENTION!r}"
    )
if DELTA_TEMPERATURE <= 0:
    raise ValueError(
        f"XVLA_DELTA_TEMPERATURE must be positive, got {DELTA_TEMPERATURE}"
    )

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

import train as xt  # noqa: E402  (the X-VLA repo's own vla training script)
import xvla_encoder_mask as mask  # noqa: E402


# ---- (2) adapter + projector save --------------------------------------------------------------
def _install_training_save(peft_model):
    """Save unmerged trainable artifacts; merging is a separate, auditable final step."""

    def save_pretrained(save_dir, *a, **kw):
        os.makedirs(save_dir, exist_ok=True)
        adapter_dir = os.path.join(save_dir, "lora_adapter")
        # Call through the class because the instance attribute below shadows this method.
        type(peft_model).save_pretrained(
            peft_model, adapter_dir, safe_serialization=True
        )
        projector = (
            {
                name: parameter.detach().cpu()
                for name, parameter in peft_model.named_parameters()
                if mask.is_projection(name)
            }
            if TRAIN_PROJ
            else {}
        )
        if TRAIN_PROJ and not projector:
            raise RuntimeError("projector is trainable but no projector tensors were found at save time")
        torch.save(projector, os.path.join(save_dir, "vision_projector.pt"))
        manifest = {
            "format": "xvla-vision-lora-v1",
            "scope": SCOPE,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "train_projector": TRAIN_PROJ,
            "projector_update": PROJECTOR_UPDATE,
            "projector_tensors": sorted(projector),
            "merge_script": vla_paths.resolve('code/merge_xvla_lora.py'),
            "feature_preservation": {
                "enabled": DELTA_LAMBDA > 0,
                "reference": "frozen stock X-VLA vision encoder",
                "tap": "post-projector image tokens (vlm.image_proj_norm output)",
                "lambda": DELTA_LAMBDA,
                "attention": DELTA_ATTENTION,
                "attention_temperature": DELTA_TEMPERATURE,
                "method_note": (
                    "DELTA-style behavior regularization; Taylor channel importance is a "
                    "scalable first-order approximation to DELTA's exhaustive supervised "
                    "channel-ablation attention"
                ),
            },
        }
        with open(os.path.join(save_dir, "adapter_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[patch 2] saved unmerged LoRA adapter + {len(projector)} projector tensors "
              f"to {save_dir}", flush=True)

    peft_model.save_pretrained = save_pretrained
    return peft_model


# ---- (1) LoRA scope ----------------------------------------------------------------------------
_orig_from_pretrained = xt.XVLA.from_pretrained


def _install_feature_preservation(peft_model, teacher_vlm):
    """Add DELTA-style behavior regularization without changing inference.

    DELTA defines transfer regularization on layer outputs rather than weights.
    Its original supervised attention measures the loss change from ablating
    every CNN channel, which would require hundreds of extra VLA forwards per
    batch here.  ``taylor`` uses the standard first-order loss sensitivity
    |activation * gradient| as a scalable supervised channel-importance proxy.
    ``uniform`` keeps DELTA's unweighted feature-matching ablation.
    """

    if DELTA_LAMBDA == 0:
        return peft_model

    base_model = peft_model.get_base_model()
    captured: dict[str, torch.Tensor] = {}

    def capture_projected_tokens(_module, _inputs, output):
        if "student" in captured:
            raise RuntimeError("vision feature hook fired more than once in one X-VLA forward")
        captured["student"] = output

    base_model.vlm.image_proj_norm.register_forward_hook(capture_projected_tokens)
    original_forward = peft_model.forward
    state = {"calls": 0, "teacher_device": None}

    def forward_with_feature_preservation(*args, **kwargs):
        captured.clear()
        # DaViT uses stochastic depth in train mode.  Save the RNG state before
        # the student vision pass so the frozen teacher can replay the same
        # masks.  Restore the post-student state afterwards so the regularizer
        # does not perturb diffusion-time/noise sampling in future steps.
        cpu_rng_before = torch.get_rng_state()
        cuda_device = None
        cuda_rng_before = None
        image_for_device = kwargs.get("image_input")
        if image_for_device is not None and image_for_device.is_cuda:
            cuda_device = image_for_device.device
            cuda_rng_before = torch.cuda.get_rng_state(cuda_device)
        loss_dict = original_forward(*args, **kwargs)
        cpu_rng_after = torch.get_rng_state()
        cuda_rng_after = (
            torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None
        )
        if "student" not in captured:
            raise RuntimeError("X-VLA vision feature hook did not fire")
        student = captured.pop("student")

        try:
            image_input = kwargs["image_input"]
            image_mask = kwargs["image_mask"]
        except KeyError as exc:
            raise RuntimeError(
                "feature-preserving X-VLA expects keyword image_input and image_mask"
            ) from exc
        flat_mask = image_mask.view(-1).to(torch.bool)
        valid_images = image_input.flatten(0, 1)[flat_mask]
        if valid_images.shape[0] != student.shape[0]:
            raise RuntimeError(
                f"student/teacher image count mismatch: {student.shape[0]} vs "
                f"{valid_images.shape[0]}"
            )

        if state["teacher_device"] != valid_images.device:
            teacher_vlm.to(device=valid_images.device)
            state["teacher_device"] = valid_images.device
            print(f"[delta] moved frozen stock vision teacher to {valid_images.device}", flush=True)
        teacher_vlm.train(base_model.training)
        try:
            torch.set_rng_state(cpu_rng_before)
            if cuda_device is not None and cuda_rng_before is not None:
                torch.cuda.set_rng_state(cuda_rng_before, cuda_device)
            with torch.no_grad():
                teacher = teacher_vlm._encode_image(valid_images).detach()
        finally:
            torch.set_rng_state(cpu_rng_after)
            if cuda_device is not None and cuda_rng_after is not None:
                torch.cuda.set_rng_state(cuda_rng_after, cuda_device)
        if student.shape != teacher.shape:
            raise RuntimeError(
                f"student/teacher feature shape mismatch: {student.shape} vs {teacher.shape}"
            )

        error_by_channel = (student.float() - teacher.float()).square().mean(dim=(0, 1))
        task_loss = sum(loss_dict.values())
        if DELTA_ATTENTION == "taylor":
            gradient = torch.autograd.grad(
                task_loss, student, retain_graph=True, create_graph=False
            )[0].detach()
            importance = (student.detach().float() * gradient.float()).abs().mean(dim=(0, 1))
            scaled = importance / importance.mean().clamp_min(1e-12)
            weights = torch.softmax(scaled / DELTA_TEMPERATURE, dim=0) * scaled.numel()
        else:
            importance = torch.ones_like(error_by_channel)
            weights = importance
        feature_raw = (weights.detach() * error_by_channel).mean()
        feature_loss = feature_raw * DELTA_LAMBDA
        loss_dict["delta_feature"] = feature_loss

        state["calls"] += 1
        if state["calls"] == 1 or state["calls"] % 50 == 0:
            print(
                f"[delta] call={state['calls']} task={task_loss.detach().float().item():.6f} "
                f"feature_raw={feature_raw.detach().float().item():.6f} "
                f"weighted={feature_loss.detach().float().item():.6f} "
                f"attention={DELTA_ATTENTION}",
                flush=True,
            )
        return loss_dict

    peft_model.forward = forward_with_feature_preservation
    print(
        "[delta] enabled post-projector feature preservation: "
        f"lambda={DELTA_LAMBDA:g} attention={DELTA_ATTENTION} "
        f"temperature={DELTA_TEMPERATURE:g}",
        flush=True,
    )
    return peft_model


def _from_pretrained_lora(*a, **kw):
    model = _orig_from_pretrained(*a, **kw)
    print(f"[patch 1] loaded X-VLA: action_mode={model.action_mode}, "
          f"num_actions={model.num_actions}", flush=True)
    mask.summarize(model, "stock")

    names = mask.linear_modules(model, SCOPE)
    if not names:
        raise ValueError("vision target set is empty -- xvla_encoder_mask.VISION_RE matched nothing")
    leaked = [n for n in names if mask.is_trunk(n)]
    if leaked:
        raise ValueError(f"vision target set leaked into the language trunk: {leaked[:5]}")

    cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=mask.target_modules_regex(names),
        init_lora_weights="gaussian",
    )
    print(f"[patch 1] LoRA scope={SCOPE}: {len(names)} nn.Linear modules "
          f"(r={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT})", flush=True)
    print(f"[patch 1] e.g. {names[0]}  ...  {names[-1]}", flush=True)

    # Copy the untouched visual path before PEFT modifies ``model`` in place.
    # The language encoder is deleted because the regularizer only calls
    # ``_encode_image``.  Keeping the teacher outside the student module tree
    # prevents it from entering optimizer state or adapter checkpoints.
    teacher_vlm = None
    if DELTA_LAMBDA > 0:
        teacher_vlm = copy.deepcopy(model.vlm)
        if hasattr(teacher_vlm, "language_model"):
            del teacher_vlm.language_model
        teacher_vlm.requires_grad_(False)
        teacher_vlm.eval()

    peft_model = get_peft_model(model, cfg)

    if TRAIN_PROJ and PROJECTOR_UPDATE == "lora":
        import xvla_projector_lora
        xvla_projector_lora.install(peft_model.get_base_model().vlm, LORA_RANK, LORA_ALPHA)
        contract.audit(peft_model, mask.is_vision, mask.is_projection)
    elif TRAIN_PROJ:
        # image_projection is a bare nn.Parameter, so LoRA cannot wrap it; unfreeze full-rank.
        n_proj = 0
        for name, p in peft_model.named_parameters():
            if mask.is_projection(name):
                p.requires_grad_(True)
                n_proj += p.numel()
        print(f"[patch 1] XVLA_TRAIN_PROJ=1: {n_proj/1e6:.2f}M projection params trainable "
              f"FULL-RANK alongside the adapters", flush=True)

    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    outside = [n for n in trainable if not mask.is_vision(n)]
    if outside:
        raise ValueError(f"{len(outside)} trainable params outside the vision encoder, "
                         f"e.g. {outside[:5]}")
    wrong_visual_region = [n for n in trainable if not mask.in_scope(n, SCOPE)]
    if wrong_visual_region:
        raise ValueError(
            f"{len(wrong_visual_region)} trainable tensors outside {SCOPE}, "
            f"e.g. {wrong_visual_region[:5]}"
        )
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in peft_model.parameters())
    if n_train == 0:
        raise ValueError("zero trainable params after LoRA -- the mask did not apply")
    print(f"[patch 1] trainable tensors {len(trainable)}, params {n_train:,} "
          f"({100 * n_train / n_all:.4f}%)", flush=True)

    if teacher_vlm is not None:
        peft_model = _install_feature_preservation(peft_model, teacher_vlm)
    return _install_training_save(peft_model)


xt.XVLA.from_pretrained = staticmethod(_from_pretrained_lora)


# ---- (3) one-shot gradient audit ----------------------------------------------------------------
_orig_build_optimizer = xt.build_optimizer


def _build_optimizer_audited(model, *a, **kw):
    opt = _orig_build_optimizer(model, *a, **kw)
    for g in opt.param_groups:
        print(f"[opt] group {g['name']:<18} tensors {len(g['params']):>4}  "
              f"params {sum(p.numel() for p in g['params'])/1e6:>8.2f}M  wd {g['weight_decay']}",
              flush=True)

    _orig_step = opt.step
    state = {"audited": False}

    def _step_with_audit(*sa, **sk):
        if not state["audited"]:
            state["audited"] = True
            got = sum(1 for g in opt.param_groups for p in g["params"]
                      if p.grad is not None and p.grad.abs().sum() > 0)
            tot = sum(len(g["params"]) for g in opt.param_groups)
            print(f"[audit] first step: {got}/{tot} trainable tensors have non-zero grad",
                  flush=True)
            if got == 0:
                raise RuntimeError("no gradient reached any trainable parameter")
        return _orig_step(*sa, **sk)

    opt.step = _step_with_audit
    return opt


xt.build_optimizer = _build_optimizer_audited


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("X-VLA encoder-LoRA training", parents=[xt.get_args_parser()])
    args = parser.parse_args()
    if args.trainable_regex:
        raise SystemExit("--trainable_regex is for full-rank masking; this launcher uses LoRA. "
                         "Leave it empty.")
    print(f"[cfg] SCOPE={SCOPE}  LoRA r={LORA_RANK} alpha={LORA_ALPHA} "
          f"dropout={LORA_DROPOUT}  TRAIN_PROJ={TRAIN_PROJ}  repo={REPO}  "
          f"DELTA_LAMBDA={DELTA_LAMBDA:g}  DELTA_ATTENTION={DELTA_ATTENTION}", flush=True)
    xt.main(args)
