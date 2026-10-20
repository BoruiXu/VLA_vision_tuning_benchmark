# VLA-Retain

**Benchmarking Cross-Dimension Effects of Vision Tuning in VLAs**

Does adapting the vision encoder to one distribution shift compromise robustness to other shifts?

This artifact contains the training extensions, evaluation adapters, episode-level outcomes, and statistical analysis used in the paper. It covers OpenVLA-OFT, π0.5, InternVLA-A1.5, NORA-1.5, and X-VLA. Only experiments used in the accompanying manuscript are included.

## Start with the results

No GPU, model weights, simulator, or training dataset is required to inspect or recompute the results.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/validate_release.py
python analysis/reproduce.py
python analysis/plot_results.py
```

See [the recomputed summary](results/recomputed/summary.md), [complete metrics and CIs](results/recomputed/metrics.json), and [the run catalog](data/catalog.json). The default analysis uses 20,000 paired task-family bootstrap draws. Numerical figures are written as vector PDFs under `results/figures/`.

## What is included

| Experiment | Models / settings | Runs | Episodes per run |
|---|---|---:|---:|
| Main LIBERO-Plus comparison | Five models × Base, Camera-view, Sensor-noise, Mixed | 20 | 10,030 |
| Camera-view tuning strategies | OFT and X-VLA × connector LoRA, backbone LoRA, three WiSE-FT weights, DELTA-style | 12 | 10,030 |
| Original LIBERO control | Five models × Base, Camera-view, Sensor-noise | 15 | 2,000 |
| Checkpoint curves | OFT and X-VLA × 0, 2k, 4k, 8k, 12k updates | 10 | 560 |
| Additional training seeds | OFT and X-VLA × seeds 43 and 44 | 4 | 560 |
| **Total** | Shared baselines are stored once | **61** | **358,800 total** |

The seed-42 sensitivity result reuses the 12k checkpoint-curve result. The two qualitative cases contain paired rollout videos and are not added to quantitative episode totals.

```text
VLA-retain/
├── code/                  Model-specific training, LoRA masks, merge and data utilities
│   └── connector/         Isolated connector-only LoRA implementation
├── scripts/               Portable train / merge / serve / evaluate entry points
├── evaluation/            Model servers and benchmark wrappers
├── config/                Training recipes, upstream revisions, environment inventories
├── patches/               Required X-VLA and OpenPI source changes
├── data/
│   ├── catalog.json       Run IDs, protocols, provenance, counts and hashes
│   ├── episodes/          Compressed, normalized per-episode outcomes
│   └── manifests/         Evaluation tasks, initial-state hashes, training split IDs
├── analysis/              Paired metrics, confidence intervals and vector figures
├── results/               Recomputed numerical outputs and figures
├── qualitative/           Two paired video examples from the paper
├── docs/                  Setup, protocols, paper-to-artifact map and data schema
└── licenses/              Preserved upstream license texts
```

## Train and merge

Training requires the original model-specific environments, released base checkpoints and LIBERO-Plus training data. Follow [setup](docs/setup.md) first. Commands print a launch plan unless `--execute` is supplied; this makes it possible to check paths and arguments without allocating a GPU.

```bash
# One model, one training GPU; the default is tower + connector LoRA.
python scripts/train.py --model xvla --split camera_only \
  --base models/xvla-libero --gpu 0 --execute

# Resume the same run. --steps is the final total, not additional updates.
python scripts/train.py --model xvla --split camera_only \
  --base models/xvla-libero --gpu 0 --resume auto --execute

# Merge in a separate step, into a new directory.
python scripts/merge.py --model xvla --base models/xvla-libero \
  --checkpoint runs/xvla-camera_only-dual_lora-seed42/ckpt-12000 \
  --output models/xvla-camera-merged --execute
```

Use `--model oft|pi05|internvla15|nora15|xvla` and `--split camera_only|noise_only|mixed`. The shared recipe is rank 32, alpha 16, effective batch 32, 12,000 optimizer updates, and a checkpoint every 2,000 updates. Each model retains its own optimizer and action contract. See [training details](docs/training.md) for scope experiments, DELTA-style, WiSE-FT, and resume limitations.

## Evaluate

Inference and rendering can run on separate machines. Use the model environment for serving and the simulator environment for rendering. The default server batch limit is **8**. For OFT, the compatibility adapter dispatches items sequentially within that batch to preserve its evaluation math.

```bash
# Model host. Use a private network or SSH tunnel if rendering is remote.
python scripts/serve.py --model xvla --checkpoint models/xvla-camera-merged \
  --gpu 0 --batch 8 --address 127.0.0.1:8100 --execute

# Renderer host. EGL indices are machine-specific; verify yours before running.
python scripts/evaluate.py --benchmark plus --server-url ws://127.0.0.1:8100 \
  --libero-repo /path/to/LIBERO-plus --libero-config /path/to/libero-config \
  --gpu 1 --egl-device 0 --shards 16 --output outputs/xvla-camera-plus --execute
```

Use `--benchmark screen560` for checkpoint/seed sensitivity and `--benchmark original` for the four original LIBERO suites with 50 initial states per task. The evaluator rejects nonempty output directories instead of mixing runs. See [evaluation details](docs/evaluation.md).

## Data and statistics

Episode pairing uses `(suite, task_id, episode_idx)`. Overall success pools all episodes; off-target change averages the six non-target dimension changes. Confidence intervals resample the 40 source-task families, retaining their complete perturbation sets and paired base/tuned outcomes. They are pointwise intervals for fixed checkpoints, not training-seed uncertainty.

The release does not contain training images, model weights, optimizer checkpoints, raw machine logs, private job controllers, or experiments absent from the paper. Obtain upstream data and weights under their respective terms. [The paper map](docs/paper_map.md) lists the exact included comparisons; [the schema](docs/data.md) describes all data fields.
