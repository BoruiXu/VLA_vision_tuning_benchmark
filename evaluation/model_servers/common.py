"""Shared helpers for the local vla-eval model-server adapters."""

from __future__ import annotations

import zlib
from typing import Any


LIBERO_SUITES = {
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
}


def normalize_suite(value: str) -> str:
    """Return the base LIBERO suite name used by normalization statistics."""
    suite = value.removesuffix("_no_noops")
    if suite not in LIBERO_SUITES:
        raise ValueError(f"unsupported LIBERO suite: {value!r}")
    return suite


class StableTaskMixin:
    """Track task metadata and derive request-order-independent flow seeds.

    The harness batches requests according to arrival time.  A global torch RNG
    would therefore make model outputs depend on which requests happened to
    share a batch.  Seeds derived from the rollout location keep stock/tuned and
    batch-size comparisons paired.
    """

    global_seed: int

    def _init_stable_tasks(self, global_seed: int = 7) -> None:
        self.global_seed = int(global_seed)
        self._task_by_session: dict[str, dict[str, Any]] = {}

    async def on_episode_start(self, config: dict[str, Any], ctx: Any) -> None:
        task = config.get("task") or {}
        self._task_by_session[ctx.session_id] = dict(task)
        await super().on_episode_start(config, ctx)

    async def on_episode_end(self, result: dict[str, Any], ctx: Any) -> None:
        self._task_by_session.pop(ctx.session_id, None)
        await super().on_episode_end(result, ctx)

    def _task(self, ctx: Any) -> dict[str, Any]:
        try:
            return self._task_by_session[ctx.session_id]
        except KeyError as exc:
            raise RuntimeError(f"missing EPISODE_START metadata for session {ctx.session_id}") from exc

    def _suite(self, ctx: Any) -> str:
        return normalize_suite(str(self._task(ctx).get("suite", "")))

    def _request_seed(self, ctx: Any) -> int:
        task = self._task(ctx)
        suite = str(task.get("suite", ""))
        task_id = int(task.get("task_id", 0))
        episode_idx = int(task.get("episode_idx", 0))
        # The legacy evaluator counted the ten initial no-op steps in `t`.
        # vla-eval performs those steps inside benchmark.reset(), so add them
        # back here to preserve the completed NORA/InternVLA request seeds.
        rollout_t = int(ctx.step) + 10
        return int(
            (
                self.global_seed * 1_000_003
                + zlib.crc32(suite.encode("utf-8"))
                + task_id * 10_007
                + episode_idx * 1_009
                + rollout_t
            )
            % (2**31 - 1)
        )
