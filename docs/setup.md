# Setup

## Lightweight result reproduction

Python 3.10 or newer, NumPy and Matplotlib are sufficient for `analysis/reproduce.py` and the numerical figures. The README's analysis environment is independent of the GPU/model environments. Optional qualitative plotting additionally needs `imageio` and `imageio-ffmpeg`.

## GPU environments

Use Linux/NVIDIA CUDA for the supplied training/evaluation entry points. The original training used single H200 GPUs; serving and simulator rendering can run separately. Model packages require different transformer/JAX/LeRobot versions, so do not install all models into the same environment.

`config/upstream.json` pins the source revisions read from the experiment repositories. `config/environments/*.json` records installed package names and versions for each model environment. These are dependency inventories, not universal cross-platform lockfiles; CUDA builds and transitive wheels must match your machine. Install each upstream project's dependencies at the pinned revision before using its adapter.

```bash
# Prints the pinned sources first; add --execute to clone/apply patches.
python scripts/setup_sources.py --component xvla --component harness
python scripts/setup_sources.py --component xvla --component harness --execute
```

The setup tool refuses to overwrite an existing checkout. It applies X-VLA's Plus loader/resume patch and OpenPI's minimal visual-training config-field patch. Local adapters register the two needed π0.5 configurations. Unrelated experimental OpenPI modules are excluded.

In each model environment, make the pinned harness importable through its `src` tree. The entry points set `PYTHONPATH`; install its transport dependencies compatible with that environment (`anyio`, `jsonargparse[signatures]`, `msgpack`, `PyYAML`, `websockets`, and `lazyregistry`). Use the pinned repository's dependency metadata together with the recorded environment inventory. Never let a generic dependency upgrade silently replace a model's transformer or simulator stack.

## Base checkpoints

Download the official LIBERO-specific versions under their original licenses. Do not substitute a generic pretrained checkpoint. Expected layout:

| Local directory | Official source |
|---|---|
| `models/openvla-oft` | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |
| `models/xvla-libero` | `2toINF/X-VLA-Libero` |
| `models/nora-1.5-libero` | `declare-lab/nora-1.5-libero` |
| `models/InternVLA-A1.5-Libero` | `InternRobotics/InternVLA-A1.5-Libero` |
| `models/pi05-libero` | Official OpenPI `pi05_libero` assets / JAX release |
| `models/pi05-libero-pytorch` | The corresponding OpenPI PyTorch conversion |
| `models/Qwen3.5-2B` | InternVLA's required Qwen base assets |
| `models/physical-intelligence-fast` | InternVLA's required FAST tokenizer assets |

Follow each pinned repository's checkpoint/conversion instructions. Preserve all processor/tokenizer/config files, external heads and normalization assets. OFT needs both `action_head--300000_checkpoint.pt` and `proprio_projector--300000_checkpoint.pt`. NORA's training and merge use the OFT base's fixed suite-statistics file, which is byte-identical to the file released with the NORA-1.5 LIBERO checkpoint; π0.5 uses its own base normalization assets. Model weights are not redistributed; record the hashes of downloaded weights for a new reproduction.

## Training data

Place the released LIBERO-Plus RLDS corpus under `data/rlds/libero_mix/1.0.0`, and its LeRobot v2.1 conversion under `data/lerobot/local/libero_plus_mix`. Preserve original episode indices, task labels, image bytes, actions, frame order and statistics metadata. The source corpus is not bundled here. If you rebuild the conversion, verify the metadata/episode-order hashes against `data/manifests/training_split_audit.json`; silently substituting a different conversion invalidates the split IDs.

```bash
# Run in an environment with TensorFlow, TFDS, parquet and LeRobot dependencies.
python code/split_libero_plus_domains.py \
  --rlds-builder data/rlds/libero_mix/1.0.0 \
  --lerobot-root data/lerobot/local/libero_plus_mix \
  --output-root data/libero_plus_domain_splits
```

Use `--help` to inspect the exported splitter's accepted flags before invoking it. Its metadata views use your local corpus paths. Those generated paths belong under ignored data directories, not in a public result export.

## Simulator

Set up the original LIBERO and LIBERO-Plus repositories/assets in separate simulator environments as appropriate. The manuscript uses robosuite 1.4.1 and MuJoCo; the original-LIBERO task manifest was created from revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`. The exact expected Plus registry is bundled as `data/manifests/plus_tasks.json`.

Prepare the benchmark's `LIBERO_CONFIG_PATH` for your local assets. Supply it explicitly with `--libero-config`, and give `--libero-repo` as the repository root containing the `libero` package. EGL device numbering can differ from CUDA numbering; verify the renderer device mapping before launching shards. The release does not include private machine-specific EGL maps or container images.

Simulator-side compatibility adjustments used in our evaluation, such as torch 1.11 support in the legacy LIBERO-Plus environment, are implemented in the benchmark wrappers under `evaluation/benchmarks/`. After installing the simulator, run a short rollout on a few tasks and compare it with the bundled episode outcomes before a full evaluation.
