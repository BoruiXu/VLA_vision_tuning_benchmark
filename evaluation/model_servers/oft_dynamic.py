# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "openvla-oft",
#     "torch>=2.2",
#     "transformers>=4.40,<5",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
#     "timm",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../..", editable = true }
# openvla-oft = { git = "https://github.com/moojink/openvla-oft.git", rev = "e4287e94541f459edc4feabc4e181f537cd569a8" }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""OpenVLA-OFT adapter matching the existing four-suite evaluation contract."""

from __future__ import annotations

from common import StableTaskMixin
from vla_eval.model_servers.oft import OFTModelServer


class DynamicStatsOFTServer(StableTaskMixin, OFTModelServer):
    """Select the checkpoint's suite-specific statistics for every session."""

    def __init__(self, *args, global_seed: int = 7, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_stable_tasks(global_seed)

    def get_observation_params(self):
        params = super().get_observation_params()
        # The completed 10,030-task results use env.seed(7), not the upstream
        # OFT reproduction setting env.seed(0).
        params["env_seed"] = 7
        params["model_chunk_size"] = self.chunk_size
        return params

    def predict(self, obs, ctx):
        unnorm_key = f"{self._suite(ctx)}_no_noops"
        if unnorm_key not in self._vla.norm_stats:
            raise KeyError(f"checkpoint lacks normalization statistics {unnorm_key}")
        self._cfg.unnorm_key = unnorm_key
        return super().predict(obs, ctx)


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(DynamicStatsOFTServer)
