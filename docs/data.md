# Data schema and metrics

## Evaluation data

`data/episodes/*.jsonl.gz` contains one JSON object per episode. Decompress with Python's `gzip` module; no specialized database is needed.

```json
{"suite":"libero_goal","task_id":0,"episode_idx":0,"dimension":"camera","family":"libero_goal/<source_task>","success":true,"steps":100}
```

`steps` is optional because some source formats do not record it. Missing fields are not filled with synthetic values.

- `suite`, `task_id`, `episode_idx`: stable pairing key within the same benchmark and evaluation manifest.
- `dimension`: `camera`, `robot`, `language`, `light`, `background`, `noise`, `layout`, or `original`.
- `family`: source-task cluster; 40 clusters across four suites.
- `success`: the recorded terminal task-success decision, not a score inferred from a video or log.
- `steps`: executed rollout steps when retained in the source aggregate.

`data/catalog.json` records model, training split, method, seed when applicable, benchmark, episode count, success count, normalized-file SHA-256, original source-file checksums, evaluator format, runtime and whether SQLite was cross-checked. The source identifiers are content hashes, not private paths. Runtime/source fields describe the input artifacts and are not altered by normalization.

Per-run UUIDs, hostnames, timestamps, elapsed times, traceback text, user paths and account configuration are not exported. Raw databases remain outside this package. Their outcome content is represented by the allowlisted fields above.

## Training subsets

| Training split | Episodes | Frames | Source tasks |
|---|---:|---:|---:|
| Camera-view | 2,871 | 447,040 | 40 |
| Sensor-noise | 2,852 | 445,454 | 40 |
| Mixed | 14,347 | 2,238,036 | 40 |

The subsets are determined by the original RLDS `episode_metadata.file_path` perturbation label, joined to the corresponding LeRobot episode order and checked against task names. They are not selected using model outcomes or image similarity. Camera and noise have no overlapping episode IDs. Mixed also contains environment/background, language and lighting training variants; the training-domain taxonomy is not identical to the seven evaluation dimensions.

`*_training_ids.json` contains exact episode IDs and ordered-ID hashes. `training_split_audit.json` preserves counts and join checks without machine paths. `code/split_libero_plus_domains.py` rebuilds local metadata views; it does not copy the large image corpus. Compare its ID hashes with the release before training. Training and evaluation use separate perturbation instances, not a random split of the released evaluation episodes.

## Evaluation manifests

Full LIBERO-Plus has 10,030 episodes: Camera 1,599; Robot 1,550; Language 1,537; Lighting 1,142; Background 1,076; Noise 1,601; Layout 1,525. The four suites contain 2,402 / 2,518 / 2,591 / 2,519 episodes.

The fixed 560-episode screen uses two variants per source-task family and dimension. Selection is outcome-blind, stratified by available difficulty labels, and shared across checkpoint/seed runs. It is an exploratory subset of the evaluation set, not an independent validation set. `screen560.json` and `screen560_tasks.json` record exact identities.

Original LIBERO uses 40 tasks × 50 initial states = 2,000 episodes. `original_libero.json` records BDDL/init-file hashes and the 50 initial-state hashes per task. The adapter checks these hashes instead of assuming task IDs alone imply identical resets.

## Statistical definitions

For each dimension, success is successes divided by episodes. `delta_pp = tuned − base` in percentage points. Overall pools episodes and is not the mean of seven dimension rates. Off-target macro change is the mean of six non-target changes; off-target sum is six times that value. Off-target micro instead pools the six dimensions' episodes. Worst off-target change is their minimum.

For the main comparisons, percentile 95% CIs use 20,000 bootstrap draws with seed 20260921. Each draw resamples 40 task families with replacement and retains all episodes within a family. Base and tuned share draws. Recompute nonlinear summaries, including the worst dimension, within each draw. Do not subtract independent CI endpoints or average interval limits. Intervals are pointwise and unadjusted for multiple comparisons; they do not measure variation across training seeds.

Source-adjusted change subtracts each model's original-LIBERO change from its Plus dimension change. Mixed training has no unique target dimension, so the analysis reports per-dimension and overall changes without an off-target label.
