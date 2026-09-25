# Evaluation protocol

The portable entry points use the pinned `vla-evaluation-harness` source and the local adapters under `evaluation/`. Start one model server per checkpoint, then connect simulator shards to its explicit WebSocket URL. No remote hostnames, SSH automation or GPU scheduling policy are embedded in the package.

## Model-specific behavior

| Model | Adapter | Executed actions before replanning |
|---|---|---:|
| OpenVLA-OFT | `oft_curve_compat.py` + suite-dependent stats | 8 |
| π0.5 | `pi0_stable.py` + PyTorch eager batch sampler | 5 |
| InternVLA-A1.5 | `internvla15.py` | 8 |
| NORA-1.5 | `nora15.py` | 5 |
| X-VLA | `xvla_batched.py` + absolute EEF controller | 30 |

The server batches up to eight observations per request. OFT's compatibility implementation processes the observations of a batch sequentially inside `predict_batch`. Keep the native action decoder, image resize/crop, gripper, normalization and execution horizon fixed across checkpoints.

Global evaluation seed is 7. `model_servers/common.py` derives flow-noise seeds from stable episode/request identities rather than request arrival order. Training seeds are separate from evaluation noise seeds. For X-VLA, recurrent proprio uses the previous predicted target after the first chunk; it is not replaced with measured state.

The Plus wrapper retains the lazy-rendering contract for variants where it is applicable. Sensor-noise variants keep the normal observation path, because rendering itself participates in corruption. Do not change this to accelerate an evaluation without a parity test.

## Task identity and initial states

- Full Plus: exactly the released 10,030 task variants, one episode each.
- Screen: the same 560 task identities across checkpoints and training seeds; never sample a new subset per run.
- Original LIBERO: four suites × ten tasks × fifty initial states. The wrapper verifies ordered initial-state hashes before evaluation.

The task-selection wrapper validates both task count and available task names. Compare final keys/counts and errors before calling a run complete. A terminated simulator job is not a task failure; do not turn errors into `success=false`, and do not retry valid failures selectively.

## Cross-host operation and partial results

The default server binds loopback. For remote rendering, use an SSH tunnel or a trusted private interface supplied through `--address`; the protocol is not an authenticated public service. The renderer needs its own checkout of this release and its own installed simulator/assets. It does not need model weights.

`scripts/evaluate.py` launches only its own subprocesses, records shard logs, merges only after all shards exit successfully and preserves partial output on failure. It refuses a nonempty output directory. Automatic cross-host recovery controllers are intentionally not distributed; inspect partial results before a deliberate recovery. For a new run, always choose a fresh output directory.

`scripts/serve.py` and `scripts/evaluate.py` print plans by default. An `--execute` flag is required to start either process. The packaged historical outcomes are not overwritten by these tools.
