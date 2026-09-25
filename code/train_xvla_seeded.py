#!/usr/bin/env python3
"""X-VLA vision-LoRA entrypoint with a rank-independent data-worker RNG stream.

The upstream loader derives worker seeds from the process-wide PyTorch RNG. PEFT consumes a
rank-dependent number of random values while initialising LoRA A matrices, so merely passing the
same ``--seed`` gives different trajectory shuffles and ColorJitter draws at different ranks.
Set ``XVLA_WORKER_BASE_SEED`` to the 32-bit worker-0 seed captured from the completed r=32 run to
make a rank ablation reuse its exact data/augmentation stream. With the variable unset, this file
only reports the upstream-derived worker seeds and is useful for auditing the reference run.
"""

from __future__ import annotations

import vla_paths

import argparse
import os
import random
import sys

import numpy as np
import torch

REPO = os.environ.get("XVLA_REPO", vla_paths.resolve('third_party/X-VLA'))
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, CODE)

import datasets as xvla_datasets  # noqa: E402


FIXED_BASE = os.environ.get("XVLA_WORKER_BASE_SEED", "").strip()
PIN_MEMORY = os.environ.get("XVLA_PIN_MEMORY", "1").strip() != "0"
SHARING_STRATEGY = os.environ.get("XVLA_SHARING_STRATEGY", "").strip()
SCOPE = os.environ.get("XVLA_SCOPE", "vision_lora").strip().lower()

if SHARING_STRATEGY:
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def _audited_worker_init(worker_id: int) -> None:
    if FIXED_BASE:
        seed = (int(FIXED_BASE) + worker_id) % (2**32)
        source = "fixed-r32-reference"
    else:
        seed = torch.initial_seed() % (2**32)
        source = "upstream-derived"
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"[data-seed] worker={worker_id} seed={seed} source={source}", flush=True)


# create_dataloader resolves this global when it constructs the DataLoader.
xvla_datasets.worker_init_fn = _audited_worker_init

# The target-only pair runs concurrently.  On this host, two pin-memory
# resource-sharer threads exited immediately after their simultaneous 2k
# checkpoints even though RAM and /dev/shm were almost empty.  This opt-in path
# preserves four persistent workers, worker partitioning and worker seeds, but
# moves the identical CPU tensors synchronously instead of through the fragile
# pin-memory thread.
if not PIN_MEMORY:
    def _create_dataloader(
        batch_size: int,
        metas_path: str,
        num_actions: int,
        training: bool,
        action_mode: str,
    ):
        return torch.utils.data.DataLoader(
            xvla_datasets.InfiniteDataReader(
                metas_path,
                num_actions=num_actions,
                training=training,
                action_mode=action_mode,
            ),
            batch_size=batch_size,
            num_workers=4,
            pin_memory=False,
            worker_init_fn=_audited_worker_init,
            persistent_workers=True,
        )

    xvla_datasets.create_dataloader = _create_dataloader

# LoRA modes install adapter/save/gradient-audit patches. Full-rank modes use the upstream
# trainable-regex path, whose Accelerate state contains the exact selected parameters and optimizer.
if SCOPE in {"vision_lora", "tower_lora"}:
    import train_xvla as wrapped  # noqa: E402
elif SCOPE in {"projector_only", "full_rank_vision"}:
    import train as raw_train  # noqa: E402
    import xvla_encoder_mask as scope_mask  # noqa: E402

    _original_apply_mask = raw_train.apply_trainable_regex

    def _apply_audited_mask(model, pattern):
        scope_mask.summarize(model, "stock")
        result = _original_apply_mask(model, pattern)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        leaked = [name for name in trainable if not scope_mask.in_scope(name, SCOPE)]
        if not trainable or leaked:
            raise RuntimeError(
                f"invalid {SCOPE} mask: trainable={len(trainable)} leaked={leaked[:5]}"
            )
        print(
            f"[scope] {SCOPE}: {len(trainable)} full-rank tensors; "
            f"e.g. {trainable[0]} ... {trainable[-1]}",
            flush=True,
        )
        return result

    raw_train.apply_trainable_regex = _apply_audited_mask
    _original_optimizer = raw_train.build_optimizer

    def _build_audited_optimizer(*args, **kwargs):
        optimizer = _original_optimizer(*args, **kwargs)
        original_step = optimizer.step
        audited = False

        def _step(*step_args, **step_kwargs):
            nonlocal audited
            if not audited:
                audited = True
                params = [p for group in optimizer.param_groups for p in group["params"]]
                nonzero = sum(
                    p.grad is not None and bool(p.grad.detach().abs().sum().item() > 0)
                    for p in params
                )
                print(
                    f"[audit] first step: {nonzero}/{len(params)} trainable tensors "
                    "have non-zero grad",
                    flush=True,
                )
                if nonzero == 0:
                    raise RuntimeError("no gradient reached the selected full-rank visual scope")
            return original_step(*step_args, **step_kwargs)

        optimizer.step = _step
        return optimizer

    raw_train.build_optimizer = _build_audited_optimizer

    class _RawWrapper:
        xt = raw_train

    wrapped = _RawWrapper()
else:
    raise ValueError(f"unsupported XVLA_SCOPE={SCOPE!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "X-VLA seeded encoder-LoRA training", parents=[wrapped.xt.get_args_parser()]
    )
    args = parser.parse_args()
    if os.environ.get("VLA_PROJECTOR_UPDATE") == "lora":
        from pathlib import Path
        import unified_lora_contract as contract
        checkpoint = wrapped.xt.resolve_resume_checkpoint(args.resume_from_checkpoint, Path(args.output_dir))
        contract.check_resume(checkpoint, "adapter_manifest.json")
    if SCOPE in {"vision_lora", "tower_lora"} and args.trainable_regex:
        raise SystemExit(
            "--trainable_regex is for full-rank masking; this launcher uses LoRA. Leave it empty."
        )
    if SCOPE in {"projector_only", "full_rank_vision"}:
        expected = scope_mask.full_rank_regex(SCOPE)
        if args.trainable_regex != expected:
            raise SystemExit(
                f"full-rank scope requires the exact audited regex {expected!r}, "
                f"got {args.trainable_regex!r}"
            )
    print(
        f"[data-seed] scope={SCOPE}; requested_base={FIXED_BASE or 'upstream-derived'}; "
        f"worker i uses base+i modulo 2^32; pin_memory={PIN_MEMORY}; "
        f"sharing_strategy={torch.multiprocessing.get_sharing_strategy()}",
        flush=True,
    )
    wrapped.xt.main(args)
