"""vla-eval adapter for local InternVLA-A1.5 LIBERO checkpoints."""

from __future__ import annotations

import vla_paths

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from common import StableTaskMixin
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, Observation


ROOT = Path(os.environ.get("VLA_ROBUSTNESS_ROOT", vla_paths.resolve('')))
sys.path.insert(0, str(ROOT / "code"))

import serve_internvla15 as legacy  # noqa: E402


class InternVLA15ModelServer(StableTaskMixin, PredictModelServer):
    def __init__(
        self,
        checkpoint: str,
        qwen: str = vla_paths.resolve('models/Qwen3.5-2B'),
        global_seed: int = 7,
        *,
        chunk_size: int = 8,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            action_ensemble=action_ensemble,
            **kwargs,
        )
        self._init_stable_tasks(global_seed)
        self._inference = legacy.InternVLAInference(Path(checkpoint), Path(qwen))

    def get_observation_params(self) -> dict[str, Any]:
        return {
            "send_wrist_image": True,
            "send_state": True,
            "quat_no_antipodal": True,
            "model_chunk_size": self.chunk_size,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "image": IMAGE_RGB,
            "wrist_image": IMAGE_RGB,
            "state": STATE_EEF_POS_AA_GRIP,
            "language": LANGUAGE,
        }

    def _payload(self, obs: Observation, ctx: SessionContext) -> dict[str, Any]:
        images = obs.get("images", {})
        if not isinstance(images, dict) or len(images) < 2:
            raise ValueError("InternVLA-A1.5 requires agent-view and wrist images")
        values = list(images.values())
        state = obs.get("states", obs.get("state"))
        if state is None:
            raise ValueError("InternVLA-A1.5 requires LIBERO proprio state")
        return {
            "full_image": np.asarray(values[0], dtype=np.uint8),
            "wrist_image": np.asarray(values[1], dtype=np.uint8),
            "state": np.asarray(state, dtype=np.float32),
            "instruction": str(obs.get("task_description", "")),
            "unnorm_key": f"{self._suite(ctx)}_no_noops",
        }

    @staticmethod
    def _pad_rows(rows: list[torch.Tensor], value: int | bool) -> torch.Tensor:
        squeezed = [row.squeeze(0) for row in rows]
        return pad_sequence(squeezed, batch_first=True, padding_value=value)

    def _prepare_batch(
        self,
        obs_batch: list[Observation],
        ctx_batch: list[SessionContext],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        samples = [
            self._inference.preprocess(self._payload(obs, ctx))
            for obs, ctx in zip(obs_batch, ctx_batch)
        ]
        batches = [item[0] for item in samples]
        suites = [item[1] for item in samples]
        state_key = legacy.OBS_STATE
        string_prefix = legacy.OBS_STR
        batch = {
            state_key: torch.cat([item[state_key] for item in batches], dim=0),
            f"{string_prefix}.pixel_values": torch.cat(
                [item[f"{string_prefix}.pixel_values"] for item in batches], dim=0
            ),
            f"{string_prefix}.image_grid_thw": torch.cat(
                [item[f"{string_prefix}.image_grid_thw"] for item in batches], dim=0
            ),
            f"{string_prefix}.input_ids": self._pad_rows(
                [item[f"{string_prefix}.input_ids"] for item in batches], 0
            ),
            f"{string_prefix}.attention_mask": self._pad_rows(
                [item[f"{string_prefix}.attention_mask"] for item in batches], 0
            ),
            f"{string_prefix}.fast_token_mask": self._pad_rows(
                [item[f"{string_prefix}.fast_token_mask"] for item in batches], False
            ),
        }
        return batch, suites

    def _noise(self, ctx_batch: list[SessionContext], device: torch.device) -> torch.Tensor:
        model = self._inference.policy.model
        rows = []
        for ctx in ctx_batch:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._request_seed(ctx))
            rows.append(
                torch.randn(
                    (1, model.config.chunk_size, model.config.max_action_dim),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
            )
        return torch.cat(rows, dim=0)

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        return self.predict_batch([obs], [ctx])[0]

    def predict_batch(
        self,
        obs_batch: list[Observation],
        ctx_batch: list[SessionContext],
    ) -> list[Action]:
        if len(obs_batch) != len(ctx_batch):
            raise ValueError("observation/context batch mismatch")
        batch, suites = self._prepare_batch(obs_batch, ctx_batch)
        policy = self._inference.policy
        action_model = policy.model
        fixed_noise = self._noise(ctx_batch, next(policy.parameters()).device)
        original_sample_noise = action_model.sample_noise

        def sample_noise(shape, device):
            if tuple(shape) != tuple(fixed_noise.shape):
                raise RuntimeError(
                    f"InternVLA noise-shape mismatch: requested {tuple(shape)}, "
                    f"prepared {tuple(fixed_noise.shape)}"
                )
            return fixed_noise.to(device=device)

        action_model.sample_noise = sample_noise
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                normalized = policy.predict_action_chunk(batch)[:, :, :7]
        finally:
            action_model.sample_noise = original_sample_noise

        normalized_np = normalized.float().cpu().numpy()
        results = []
        for row, suite in zip(normalized_np, suites):
            stats = self._inference.stats[suite][legacy.ACTION]
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            actions = row * (std + 1e-6) + mean
            actions = np.clip(
                actions,
                np.asarray(stats["min"], dtype=np.float32),
                np.asarray(stats["max"], dtype=np.float32),
            )
            actions[..., 6] = np.where(actions[..., 6] < 0.5, 1.0, -1.0)
            results.append({"actions": actions.astype(np.float32)})
        return results


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(InternVLA15ModelServer)
