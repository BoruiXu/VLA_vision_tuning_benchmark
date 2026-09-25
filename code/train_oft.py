"""Launcher for the OpenVLA-OFT arm of the cross-architecture encoder study.

Runs the OFFICIAL `vla-scripts/finetune.py` with a small set of runtime patches. The official
repo at /opt/vla-retain/third_party/openvla-oft is NOT modified -- every deviation is injected from
this file so that the whole delta can be audited on one screen.

THE EXPERIMENT
    Stage 3 of the plan: on LIBERO-plus, train ONLY the vision encoder and leave everything
    downstream frozen. Concretely, LoRA r=32 attached to the vision tower + its projector, with
    the Llama-2 trunk, the action head and the proprio projector all frozen. The pi0.5 arm of the
    same study used `vision_tower|multi_modal_projector` with everything else frozen, so the two
    masks are the same cut of the architecture.

THE FIVE PATCHES
    (1) DATASET      register `libero_mix` in the three OXE registries. No behavioural change.

    (2) LORA SCOPE   *the independent variable*. Stock OFT passes `target_modules="all-linear"`,
                     which adapts 440 layers including 226 in the language model. We replace the
                     target set with the 200-odd nn.Linear modules inside vision_backbone +
                     projector, enumerated by name from the live module tree (see encoder_mask.py)
                     and asserted to contain zero language-model layers.

    (3) HEADS        `init_module` only loads a checkpoint when `cfg.resume` is set; otherwise the
                     action head and proprio projector are RANDOMLY INITIALISED. That is fine when
                     they are trainable, and fatal here: freezing a random action head would train
                     the encoder against noise. We load both from the OFT_m checkpoint and freeze
                     them. This is the single most dangerous silent failure in this script, so the
                     loaded tensor count and checksum are printed.

    (4) OPTIMIZER    paper D.4 says "AdamW with weight decay 0.1"; stock code takes PyTorch's
                     default 0.01.

    (5) SCHEDULE     paper D.4 says "cosine schedule with warmup"; stock code uses MultiStepLR
                     (one 10x step decay). NOTE a footgun in the official loop: it also has a
                     manual warmup block that overwrites param_group["lr"] every micro-batch and
                     fights the scheduler. We keep `--lr_warmup_steps 0` so that block never
                     fires, and implement warmup inside the cosine lambda instead.

Everything else -- data pipeline, action head architecture, image augmentation, checkpointing,
LoRA rank/alpha/dropout -- is the stock OFT code path.

Config comes from env vars so the shell script stays the single source of truth:
    OFT_TRAINABLE      encoder | all        which modules LoRA attaches to  (default: encoder)
    OFT_HEADS_DIR      dir holding action_head--<step>_checkpoint.pt        (default: --vla_path)
    OFT_HEADS_STEP     step number of those checkpoints                     (default: 300000)
    OFT_FREEZE_HEADS   1 | 0                                                (default: 1 iff encoder)
    OFT_WEIGHT_DECAY   OFT_WARMUP_STEPS   OFT_MAX_STEPS   OFT_MIN_LR_RATIO
    OFT_RESUME_DIR     checkpoint directory containing lora_adapter/training_state.pt
    OFT_RESUME_STEP    exact number of completed optimizer updates in that checkpoint
    OFT_SEED           Python/NumPy/PyTorch/TensorFlow seed (default: 42, matching pi0.5)
    OFT_NORM_STATS     released base checkpoint dataset_statistics.json
    OFT_TASK_SUITE_MAP exact mapping of the 40 train instructions to the four base suites
"""

import vla_paths

import json
import math
import os
import random
import sys
from pathlib import Path

REPO = vla_paths.resolve('third_party/openvla-oft')
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "vla-scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TRAINABLE = os.environ.get("OFT_TRAINABLE", "encoder")
SCOPE = os.environ.get(
    "OFT_SCOPE", "vision_lora" if TRAINABLE == "encoder" else "all_linear_lora"
).strip().lower()
PEFT_SCOPES = {"vision_lora", "tower_lora", "all_linear_lora"}
FULL_RANK_SCOPES = {"projector_only", "full_rank_vision"}
HEADS_DIR = os.environ.get("OFT_HEADS_DIR", "")
HEADS_STEP = int(os.environ.get("OFT_HEADS_STEP", "300000"))
FREEZE_HEADS = os.environ.get("OFT_FREEZE_HEADS", "1" if SCOPE != "all_linear_lora" else "0") == "1"
WEIGHT_DECAY = float(os.environ.get("OFT_WEIGHT_DECAY", "0.1"))
WARMUP_STEPS = int(os.environ.get("OFT_WARMUP_STEPS", "0"))
MAX_STEPS = int(os.environ.get("OFT_MAX_STEPS", "0"))
MIN_LR_RATIO = float(os.environ.get("OFT_MIN_LR_RATIO", "0.0"))
RESUME_DIR = os.environ.get("OFT_RESUME_DIR", "").rstrip("/")
RESUME_STEP = int(os.environ.get("OFT_RESUME_STEP", "0"))
SEED = int(os.environ.get("OFT_SEED", "42"))
NORM_STATS = os.environ.get(
    "OFT_NORM_STATS", vla_paths.resolve('models/openvla-oft/dataset_statistics.json')
)
TASK_SUITE_MAP = os.environ.get(
    "OFT_TASK_SUITE_MAP",
    vla_paths.resolve('config/libero_task_suite_map.json'),
)
DOMAIN_FILTER = os.environ.get("OFT_DOMAIN_FILTER", "").strip().lower()
DELTA_LAMBDA = float(os.environ.get('OFT_DELTA_LAMBDA', '0'))
DELTA_TEMPERATURE = float(os.environ.get('OFT_DELTA_TEMPERATURE', '1'))
assert math.isfinite(DELTA_LAMBDA) and DELTA_LAMBDA >= 0
assert math.isfinite(DELTA_TEMPERATURE) and DELTA_TEMPERATURE > 0
assert DELTA_LAMBDA == 0 or SCOPE == 'vision_lora'
DOMAIN_AUDIT = os.environ.get(
    "OFT_DOMAIN_AUDIT",
    vla_paths.resolve('data/libero_plus_domain_splits/audit.json'),
)

assert TRAINABLE in ("encoder", "all"), f"OFT_TRAINABLE must be encoder|all, got {TRAINABLE!r}"
assert SCOPE in PEFT_SCOPES | FULL_RANK_SCOPES, (
    "OFT_SCOPE must be vision_lora|tower_lora|projector_only|full_rank_vision|"
    f"all_linear_lora, got {SCOPE!r}"
)
assert bool(RESUME_DIR) == (RESUME_STEP > 0), (
    "OFT_RESUME_DIR and a positive OFT_RESUME_STEP must be supplied together"
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict  # noqa: E402

import encoder_mask  # noqa: E402
import openvla_domain_filter  # noqa: E402
import register_libero_plus  # noqa: E402
import stock_normalization  # noqa: E402

register_libero_plus.register()

import oft_finetune_base as ft  # noqa: E402  (experiment-local, audited copy of official finetune.py)
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402

# Optional target-only data intervention.  This operates on raw RLDS trajectory provenance before
# standardization, augmentation, chunking, or shuffle, so Camera-only/Noise-only are exact views
# of the same source episode ids used by X-VLA.
openvla_domain_filter.install(DOMAIN_FILTER, DOMAIN_AUDIT)

# Keep the frozen heads' action/proprio coordinate system fixed.  The one-dataset LIBERO-plus
# export otherwise recomputes a new mixed q01/q99 range, unlike the pi0.5 vision-only arm which
# explicitly reused its released base-model assets.
stock_normalization.install(ft, NORM_STATS, TASK_SUITE_MAP)


# The upstream helper rewrites config/modeling files inside cfg.vla_path on every launch.  The
# pinned base checkpoint has already been synchronized and is treated as immutable for this study.
ft.update_auto_map = lambda path: print(f"[base] immutable checkpoint; skip auto_map rewrite: {path}", flush=True)
ft.check_model_logic_mismatch = lambda path: print(
    f"[base] immutable checkpoint; skip model-file rewrite: {path}", flush=True
)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    import tensorflow as tf  # noqa: E402

    tf.random.set_seed(SEED)
except ImportError:
    tf = None

_RESUME_STATE = None
if RESUME_DIR:
    state_path = os.path.join(RESUME_DIR, "training_state.pt")
    _RESUME_STATE = torch.load(state_path, map_location="cpu", weights_only=False)
    saved_step = int(_RESUME_STATE["completed_optimizer_steps"])
    method = _RESUME_STATE.get('feature_preservation', {})
    assert float(method.get('lambda', 0)) == DELTA_LAMBDA, 'Resume DELTA coefficient mismatch'
    if DELTA_LAMBDA:
        assert method.get('attention') == 'taylor' and method.get('temperature') == DELTA_TEMPERATURE
    if saved_step != RESUME_STEP:
        raise ValueError(f"resume step mismatch: env={RESUME_STEP}, state={saved_step} ({state_path})")
    print(f"[resume] loaded training state metadata from {state_path} @ step {saved_step}", flush=True)


# ---- (2) LoRA scope: attach adapters to the vision encoder only ------------------------------
_orig_get_peft_model = ft.get_peft_model


def _get_peft_model_scoped(model, lora_config, *a, **kw):
    if DELTA_LAMBDA > 0:
        from oft_feature_preservation import install
        install(ft, model, DELTA_LAMBDA, DELTA_TEMPERATURE)
    if SCOPE in FULL_RANK_SCOPES:
        encoder_mask.summarize(model, "pre-full-rank-mask")
        trainable = []
        for name, parameter in model.named_parameters():
            keep = encoder_mask.in_scope(name, SCOPE)
            parameter.requires_grad_(keep)
            if keep:
                trainable.append(name)
        if not trainable:
            raise ValueError(f"full-rank scope {SCOPE!r} matched no parameters")
        leaked = [name for name in trainable if not encoder_mask.is_vision(name)]
        if leaked:
            raise ValueError(f"full-rank scope leaked outside vision: {leaked[:5]}")
        if RESUME_DIR:
            state_path = Path(RESUME_DIR) / "vision_scope_state.pt"
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            named = dict(model.named_parameters())
            missing = sorted(set(state) - set(named))
            expected = sorted(name for name in named if encoder_mask.in_scope(name, SCOPE))
            omitted = sorted(set(expected) - set(state))
            if missing or omitted:
                raise KeyError(
                    f"scope resume mismatch: unknown={missing[:5]} omitted={omitted[:5]}"
                )
            with torch.no_grad():
                for name, value in state.items():
                    parameter = named[name]
                    if tuple(parameter.shape) != tuple(value.shape):
                        raise ValueError(
                            f"resume shape mismatch for {name}: {parameter.shape} != {value.shape}"
                        )
                    parameter.copy_(value.to(dtype=parameter.dtype, device=parameter.device))
            print(f"[resume] restored {len(state)} full-rank tensors from {state_path}", flush=True)
        n_train = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        n_all = sum(parameter.numel() for parameter in model.parameters())
        model._vla_robustness_scope = SCOPE
        # The upstream loop calls this PEFT convenience method unconditionally inside its
        # ``if cfg.use_lora`` block. Full-rank modes deliberately return the raw model while
        # retaining cfg.use_lora only to pass the upstream entry assertion.
        model.print_trainable_parameters = lambda: print(
            f"trainable params: {n_train:,} || all params: {n_all:,} || "
            f"trainable%: {100*n_train/n_all:.6f}",
            flush=True,
        )
        print(
            f"[patch 2] full-rank scope={SCOPE}: tensors={len(trainable)}, "
            f"params={n_train:,} ({100*n_train/n_all:.4f}%); no LoRA adapter installed",
            flush=True,
        )
        print(f"[patch 2] e.g. {trainable[0]}  ...  {trainable[-1]}", flush=True)
        return model

    if SCOPE in ("vision_lora", "tower_lora"):
        names = encoder_mask.linear_modules(model, SCOPE)
        bad = [n for n in names if encoder_mask.is_trunk(n)]
        if bad:
            raise ValueError(f"vision target set leaked into the language trunk: {bad[:5]}")
        if not names:
            raise ValueError(f"visual target set is empty for OFT_SCOPE={SCOPE}")
        lora_config.target_modules = encoder_mask.target_modules_regex(names)
        print(f"[patch 2] LoRA scope={SCOPE}: {len(names)} nn.Linear modules "
              f"(r={lora_config.r}, alpha={lora_config.lora_alpha})", flush=True)
        print(f"[patch 2] e.g. {names[0]}  ...  {names[-1]}", flush=True)
        encoder_mask.summarize(model, "pre-LoRA")
    else:
        print(f"[patch 2] LoRA scope = stock all-linear (OFT_SCOPE={SCOPE})", flush=True)

    peft_model = _orig_get_peft_model(model, lora_config, *a, **kw)

    if RESUME_DIR:
        adapter_dir = os.path.join(RESUME_DIR, "lora_adapter")
        adapter_state = load_peft_weights(adapter_dir, device="cpu")
        result = set_peft_model_state_dict(peft_model, adapter_state, adapter_name="default")
        unexpected = list(getattr(result, "unexpected_keys", []))
        if unexpected:
            raise ValueError(f"unexpected keys while restoring LoRA adapter: {unexpected[:5]}")
        print(f"[resume] restored {len(adapter_state)} LoRA tensors from {adapter_dir}", flush=True)

    # The adapter is worthless if it landed somewhere else, and a mis-scoped run looks perfectly
    # healthy from the loss curve. Verify the trainable set directly.
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if SCOPE in ("vision_lora", "tower_lora"):
        leaked = [n for n in trainable if not encoder_mask.is_vision(n)]
        if leaked:
            raise ValueError(f"{len(leaked)} trainable params outside the vision encoder, "
                             f"e.g. {leaked[:5]}")
        wrong_visual_region = [n for n in trainable if not encoder_mask.in_scope(n, SCOPE)]
        if wrong_visual_region:
            raise ValueError(
                f"{len(wrong_visual_region)} trainable tensors outside {SCOPE}, "
                f"e.g. {wrong_visual_region[:5]}"
            )
        if n_train == 0:
            raise ValueError("zero trainable params after LoRA -- mask did not apply")
    peft_model._vla_robustness_scope = SCOPE
    if DELTA_LAMBDA > 0:
        import unified_lora_contract as contract
        contract.audit(peft_model, encoder_mask.is_vision, encoder_mask.is_projector)
    print(f"[patch 2] trainable tensors {len(trainable)}, params {n_train:,} "
          f"({100 * n_train / sum(p.numel() for p in peft_model.parameters()):.4f}%)", flush=True)
    return peft_model


ft.get_peft_model = _get_peft_model_scoped


# ---- (3) action head / proprio projector: load from OFT_m, then freeze -----------------------
class _FrozenShim(nn.Module):
    """Stand-in for DDP around a module with nothing to train.

    DDP builds its gradient reducer from the parameters that require grad AT CONSTRUCTION TIME
    and raises outright when there are none; freezing *after* wrapping instead makes backward
    fail with "expected to have finished reduction". Neither is worth risking for a module that
    never receives a gradient, so it is simply not wrapped. `.module` -- which the training loop
    reaches for (`action_head.module.predict_action(...)`) -- is preserved, and `state_dict()`
    keeps the same `module.` prefix DDP would have produced, so checkpoints stay loadable by the
    stock `remove_ddp_in_checkpoint`.
    """

    def __init__(self, m):
        super().__init__()
        self.module = m

    def forward(self, *a, **k):
        return self.module(*a, **k)


def _init_module_pretrained(module_class, module_name, cfg, device_id, module_args,
                            to_bf16=False, find_unused_params=False):
    """Replacement for ft.init_module: always load the pretrained weights, optionally freeze."""
    module = module_class(**module_args)
    ft.count_parameters(module, module_name)

    src = HEADS_DIR or cfg.vla_path
    state_dict = ft.load_checkpoint(module_name, src, HEADS_STEP)
    module.load_state_dict(state_dict, strict=True)  # strict: a silent key mismatch here is fatal
    checksum = sum(float(v.float().abs().sum()) for v in state_dict.values() if v.is_floating_point())
    print(f"[patch 3] {module_name}: loaded {len(state_dict)} tensors from {src} "
          f"@step {HEADS_STEP}, |sum| = {checksum:.4e}", flush=True)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    if FREEZE_HEADS:
        for p in module.parameters():
            p.requires_grad_(False)
        print(f"[patch 3] {module_name}: FROZEN "
              f"({sum(p.numel() for p in module.parameters())/1e6:.1f}M params, not DDP-wrapped)",
              flush=True)
        return _FrozenShim(module)

    return ft.wrap_ddp(module, device_id, find_unused_params)


ft.init_module = _init_module_pretrained


# ---- (4) AdamW with the paper's weight decay --------------------------------------------------
_orig_adamw = ft.AdamW
_CURRENT_OPT = []


def _adamw_with_wd(params, lr, **kw):
    kw.setdefault("weight_decay", WEIGHT_DECAY)
    print(f"[patch 4] AdamW(lr={lr}, weight_decay={kw['weight_decay']})", flush=True)
    opt = _orig_adamw(params, lr=lr, **kw)
    if _RESUME_STATE is not None:
        opt.load_state_dict(_RESUME_STATE["optimizer"])
        print(f"[resume] restored AdamW state @ step {RESUME_STEP}", flush=True)
    _CURRENT_OPT.clear()
    _CURRENT_OPT.append(opt)

    # One-shot gradient audit at the first optimizer step. Freezing the trunk does not by itself
    # guarantee that gradient reaches the encoder -- the encoder sits upstream of the whole frozen
    # stack, and any `no_grad` or detach along that path would leave the adapters at their
    # initialisation while the loss still falls (the action head can absorb a lot). Counting
    # non-zero grads once is the cheapest proof that the run is doing what it claims.
    state = {"audited": False}

    def _audit_before_step(optimizer, args, kwargs):
        if not state["audited"]:
            state["audited"] = True
            got = sum(1 for g in optimizer.param_groups for p in g["params"]
                      if p.grad is not None and p.grad.abs().sum() > 0)
            tot = sum(len(g["params"]) for g in optimizer.param_groups)
            print(f"[audit] first step: {got}/{tot} trainable tensors have non-zero grad",
                  flush=True)
            if got == 0:
                raise RuntimeError("no gradient reached any trainable parameter")
        return args, kwargs

    opt.register_step_pre_hook(_audit_before_step)
    return opt


ft.AdamW = _adamw_with_wd


# ---- (5) warmup + cosine in place of the stock MultiStepLR ------------------------------------
_CURRENT_SCHED = []


def _cosine_instead_of_multistep(optimizer, milestones=None, gamma=None):
    """Signature-compatible stand-in for MultiStepLR; ignores milestones/gamma by design."""
    if MAX_STEPS <= 0:
        raise ValueError("OFT_MAX_STEPS must be set so the cosine schedule knows its horizon")

    def lr_lambda(step: int) -> float:
        if WARMUP_STEPS > 0 and step < WARMUP_STEPS:
            # linear 10% -> 100%, matching the ramp shape the official manual block used
            return 0.1 + 0.9 * (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        progress = min(1.0, max(0.0, progress))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * progress))

    print(f"[patch 5] warmup+cosine: warmup={WARMUP_STEPS}, horizon={MAX_STEPS}, "
          f"min_lr_ratio={MIN_LR_RATIO} (MultiStepLR milestones={milestones} ignored)", flush=True)
    sched = LambdaLR(optimizer, lr_lambda)
    if _RESUME_STATE is not None:
        sched.load_state_dict(_RESUME_STATE["scheduler"])
        for group, restored_lr in zip(optimizer.param_groups, sched.get_last_lr()):
            group["lr"] = restored_lr
        random.setstate(_RESUME_STATE["python_rng"])
        np.random.set_state(_RESUME_STATE["numpy_rng"])
        torch.set_rng_state(_RESUME_STATE["torch_rng"])
        if torch.cuda.is_available() and _RESUME_STATE.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(_RESUME_STATE["cuda_rng"])
        print(f"[resume] restored LR scheduler state @ step {RESUME_STEP}", flush=True)
    _CURRENT_SCHED.clear()
    _CURRENT_SCHED.append(sched)
    return sched


ft.MultiStepLR = _cosine_instead_of_multistep


# ---- (6) save optimizer/scheduler/RNG state next to every adapter checkpoint -----------------
_orig_save_training_checkpoint = ft.save_training_checkpoint


def _save_full_rank_scope_checkpoint(*args, **kwargs):
    """Save only the selected raw visual tensors; DDP state handles exact resume separately."""
    cfg = kwargs["cfg"]
    run_dir = kwargs["run_dir"]
    completed_steps = int(kwargs["log_step"])
    vla = kwargs["vla"]
    processor = kwargs["processor"]
    proprio_projector = kwargs["proprio_projector"]
    noisy_action_projector = kwargs["noisy_action_projector"]
    action_head = kwargs["action_head"]
    train_dataset = kwargs["train_dataset"]
    distributed_state = kwargs["distributed_state"]

    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = Path(run_dir)
        suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{completed_steps}_chkpt")
        suffix = f"{completed_steps}_checkpoint.pt"

    if distributed_state.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ft.save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        raw = vla.module
        state = {
            name: parameter.detach().cpu()
            for name, parameter in raw.named_parameters()
            if encoder_mask.in_scope(name, SCOPE)
        }
        expected = [name for name, parameter in raw.named_parameters() if parameter.requires_grad]
        if not state or sorted(state) != sorted(expected):
            raise RuntimeError(
                f"full-rank checkpoint scope mismatch: saved={len(state)} trainable={len(expected)}"
            )
        state_path = checkpoint_dir / "vision_scope_state.pt"
        tmp_path = checkpoint_dir / "vision_scope_state.pt.tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, state_path)

        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{suffix}")
        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{suffix}"
            )
        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{suffix}")

        manifest = {
            "format": "openvla-oft-visual-scope-v1",
            "scope": SCOPE,
            "parameterization": "full_rank",
            "completed_optimizer_steps": completed_steps,
            "tensor_count": len(state),
            "parameter_count": sum(value.numel() for value in state.values()),
            "base": str(Path(cfg.vla_path).resolve()),
            "normalization": "immutable OFT_m suite stats, suite-conditioned",
        }
        (checkpoint_dir / "scope_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"[scope-save] {SCOPE}: {len(state)} tensors -> {state_path}",
            flush=True,
        )
    ft.dist.barrier()


def _save_training_checkpoint_resumable(*args, **kwargs):
    if SCOPE in FULL_RANK_SCOPES:
        _save_full_rank_scope_checkpoint(*args, **kwargs)
    else:
        _orig_save_training_checkpoint(*args, **kwargs)

    cfg = kwargs["cfg"]
    run_dir = kwargs["run_dir"]
    completed_steps = int(kwargs["log_step"])
    distributed_state = kwargs["distributed_state"]
    if not distributed_state.is_main_process:
        return
    if not _CURRENT_OPT or not _CURRENT_SCHED:
        raise RuntimeError("optimizer/scheduler not registered before checkpoint save")

    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
    else:
        checkpoint_dir = type(run_dir)(str(run_dir) + f"--{completed_steps}_chkpt")
    state_path = checkpoint_dir / "training_state.pt"
    tmp_path = checkpoint_dir / "training_state.pt.tmp"
    state = {
        "format_version": 1,
        "scope": SCOPE,
        "completed_optimizer_steps": completed_steps,
        "optimizer": _CURRENT_OPT[0].state_dict(),
        "scheduler": _CURRENT_SCHED[0].state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "seed": SEED,
        "feature_preservation": {'lambda': DELTA_LAMBDA, 'attention': 'taylor',
                                 'temperature': DELTA_TEMPERATURE, 'tap': 'post-projector',
                                 'method': 'DELTA-style Taylor approximation'},
    }
    torch.save(state, tmp_path)
    os.replace(tmp_path, state_path)
    print(f"[resume] saved optimizer/scheduler/RNG state: {state_path}", flush=True)


ft.save_training_checkpoint = _save_training_checkpoint_resumable


# ---- (7) mirror metrics to stdout -------------------------------------------------------------
# The stock loop reports loss ONLY through wandb. On a multi-hour run that means the log file has
# no loss curve at all and no way to tell a healthy run from a diverging one without opening the
# wandb artefacts. Wrap the logger so every metric also lands in the log we already tee.
_orig_log = ft.log_metrics_to_wandb


def _log_and_print(metrics, prefix, step, wandb_entity):
    _orig_log(metrics, prefix, step, wandb_entity)
    if prefix == "VLA Train":
        lr = _CURRENT_SCHED[0].get_last_lr()[0] if _CURRENT_SCHED else None
        body = "  ".join(f"{k}={v:.5f}" for k, v in sorted(metrics.items()))
        print(f"[step {step}] {body}" + (f"  lr={lr:.3e}" if lr is not None else ""), flush=True)


ft.log_metrics_to_wandb = _log_and_print


def _report_peak_mem():
    if torch.cuda.is_available():
        print("[mem] peak allocated %.1f GB, peak reserved %.1f GB"
              % (torch.cuda.max_memory_allocated() / 2**30,
                 torch.cuda.max_memory_reserved() / 2**30), flush=True)


if __name__ == "__main__":
    import atexit

    atexit.register(_report_peak_mem)
    print(f"[cfg] TRAINABLE={TRAINABLE}  SCOPE={SCOPE}  FREEZE_HEADS={FREEZE_HEADS}  "
          f"HEADS_STEP={HEADS_STEP}  wd={WEIGHT_DECAY}  warmup={WARMUP_STEPS}  "
          f"max_steps={MAX_STEPS}  seed={SEED}  resume_step={RESUME_STEP}  "
          f"domain_filter={DOMAIN_FILTER or 'all'}  normalization=base-suite-conditioned", flush=True)
    ft.finetune()
