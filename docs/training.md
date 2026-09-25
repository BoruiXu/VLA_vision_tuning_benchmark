# Training and checkpoint handling

Run training in separate model environments. `scripts/train.py` exposes only the experiment combinations included in the paper. It defaults to a dry-run plan and explicitly selects one GPU when executed. The numerical recipe is in `config/training.json`.

## Model contracts

- **OpenVLA-OFT:** LoRA on `vision_backbone` and `projector`; language parameters, external action head and proprio projector remain frozen. The action/proprio heads must be loaded from the released base checkpoint, not randomly initialized. Batch 16 × accumulation 2 = 32; LR 5e-4, warmup 1,000, cosine decay, AdamW weight decay 0.1. Keep all four suite-specific base normalization statistics.
- **X-VLA:** DaViT visual tower plus a low-rank parametrization of `image_projection`. The connector normalization stays frozen in dual LoRA. Language, soft prompts, action encoder and action decoder remain frozen; frozen soft prompts are still used in the forward path. Actions are absolute EEF poses. Training aligns image[t+1], previous target[t], and future targets[t+1:t+31]. `verify_xvla_data_alignment.py` gates this contract on an actual episode. LR 1e-4, warmup 2,000, cosine to 1e-5, gradient clip 1.0.
- **π0.5:** LoRA targets only PaliGemma `vision_tower` and `multi_modal_projector`. The registered configuration preserves the base stats and the already-relative action convention. Training horizon is 10; evaluation executes 5 actions before replanning. The config registration omits unrelated experimental modules. LR 5e-5 after 1,000 warmup steps.
- **NORA-1.5:** LoRA on Qwen2.5-VL visual tower and visual merger; language, FAST and flow-action modules remain frozen. Five-action chunks; LR 8e-5, warmup 1,000, cosine decay. The implementation takes fixed per-suite normalization from `models/openvla-oft/dataset_statistics.json`, which is byte-identical to the statistics file released with the NORA-1.5 LIBERO checkpoint; do not substitute newly computed subset statistics.
- **InternVLA-A1.5:** LoRA on Qwen3.5 visual tower and merger; language/action modules remain frozen. Eight-action execution; LR 5e-5, warmup 2,000, cosine to 5e-6.

All model-specific implementations retain save/load and trainable-parameter checks. The public default projector mode is explicitly LoRA; the launchers never silently select a full-rank connector.

## Scope and retention strategies

```bash
# Camera-only connector LoRA; same optimizer budget as the reference.
python scripts/train.py --model oft --split camera_only --method connector_lora \
  --base models/openvla-oft --gpu 0 --execute

# Visual backbone LoRA, connector frozen.
python scripts/train.py --model xvla --split camera_only --method backbone_lora \
  --base models/xvla-libero --gpu 0 --execute

# DELTA-style: dual LoRA + frozen-teacher feature preservation, lambda=1, tau=1.
python scripts/train.py --model xvla --split camera_only --method delta \
  --base models/xvla-libero --gpu 0 --execute

# Extra training seeds used by the 560-episode sensitivity experiment.
python scripts/train.py --model xvla --split camera_only --seed 43 \
  --base models/xvla-libero --gpu 0 --execute

# WiSE-FT requires merged base and tuned tensors, not an unmerged adapter.
python code/build_visual_wiseft.py --model xvla \
  --base models/xvla-libero --tuned models/xvla-camera-merged \
  --output models/xvla-camera-wiseft050 --alpha 0.5
```

DELTA-style compares post-connector features with a frozen base visual branch. Taylor channel importance approximates supervised channel-ablation attention; this is not an exact implementation of the original CNN DELTA algorithm. WiSE-FT interpolates only visual weights as `base + alpha * (tuned − base)` and verifies that nonvisual weights are unchanged. Paper weights are 0.25, 0.5 and 0.75.

Connector-only training uses `code/connector/`, not the historical full-rank `projector_only` branch in shared utilities. Pass `--connector-only` to `scripts/merge.py` for this arm. Backbone-only freezes the connector; a historical `projector_update=full` metadata value in that branch does not mean it is trainable.

## Inputs and output locations

Use the layout in [setup](setup.md). For OFT, `--rlds` is the directory containing `libero_mix/1.0.0`; for other models, `--dataset` selects a LeRobot store or a metadata view. π0.5 accesses the full corpus at `data/lerobot/local/libero_plus_mix` and filters it by subset episode IDs; its `--dataset` selects subset metadata, not a replacement corpus name.

X-VLA/NORA/InternVLA save `runs/<name>/ckpt-<step>`. OFT saves `runs/<name>--<step>_chkpt`. π0.5 saves `runs/pi05_visonly/<name>/<step>`. Merging is separate and must use the same base checkpoint, rank, alpha and scope that produced the adapters. Frozen statistics and model sidecars must accompany the merged weights.

## Resume and validation

`--resume auto` finds a compatible checkpoint in the selected run; `--resume none` starts without loading optimizer state. OFT and X-VLA accept explicit checkpoint paths; π0.5's public entry point resumes the latest checkpoint in its run. Do not use a new name to claim continuation of an old run. The portable launcher saves a protocol guard and rejects changed run arguments.

The trainers save optimizer, scheduler and RNG state for resumption. Before a full reproduction, record the base-checkpoint hash and run a short train → save → restart → merge → rollout check, verifying finite nonzero gradients in the intended regions and unchanged frozen modules.
