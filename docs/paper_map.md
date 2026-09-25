# Paper-to-artifact map

This package follows the accompanying 22-page manuscript. It does not collect every historical experiment in the workspace.

| Paper location | Released evidence / implementation |
|---|---|
| Sections 3–4: definitions, setup, data | `docs/data.md`, `config/training.json`, `data/manifests/`, `evaluation/` |
| Tables 1 and 6: architecture / parameter scales | `data/model_scales.json`; values retain the paper's rounding |
| Table 2: evaluation composition | `data/manifests/plus_tasks.json`; validate dimension and suite counts from episodes |
| Section 5.1, Table 3, Figures 2–3 | `{model}-plus-stock`, `{model}-plus-camera_only`, `{model}-plus-noise_only` |
| Section 5.2, Figure 4, Table 8 | `{model}-original-stock/camera-only/noise-only`, paired with the main Plus changes |
| Section 5.3, Figure 5 | `{model}-plus-mixed` and the main targeted runs |
| Section 5.4, Figure 6, Tables 4–5 and 9 | `oft/xvla-plus-camera-{connector_lora,backbone_lora,wiseft_025,wiseft_050,wiseft_075,delta}` |
| Table 7: optimization / action horizons | `config/training.json`, `scripts/train.py`, model-specific code |
| Appendix suite breakdown, Tables 10–11 | Group main episode outcomes by `suite` and `dimension` |
| Appendix strategy results, Table 12 | The 12 strategy runs, with the main stock and dual-LoRA endpoints |
| Checkpoint sensitivity, Figure 8 | `oft/xvla-screen-step{0,2000,4000,8000,12000}` |
| Training-seed sensitivity, Table 13 | `oft/xvla-screen-seed{43,44}` plus `*-screen-step12000` for seed 42 |
| Appendix CI tables, Tables 14–20 | `analysis/reproduce.py`, `analysis/bootstrap_core.py` |
| Figures 9–10: qualitative trajectories | `qualitative/cases.json`, four videos, `analysis/plot_qualitative.py` |

`data/catalog.json` is the definitive run inventory. Each entry links a public run ID to one normalized outcome file and source checksums. Paths from the private source workspace are not distributed.
