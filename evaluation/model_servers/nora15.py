"""vla-eval adapter for the local NORA-1.5 LIBERO checkpoints.

Run this file with the dedicated NORA environment through
``harness/scripts/serve_local_model.sh``.  It intentionally reuses the pinned
local NORA source tree instead of resolving a second copy from PyPI/GitHub.
"""

from __future__ import annotations

import vla_paths

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import StableTaskMixin
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    DimSpec,
)
from vla_eval.types import Action, Observation


ROOT = Path(os.environ.get("VLA_ROBUSTNESS_ROOT", vla_paths.resolve('')))
NORA_REPO = ROOT / "third_party/nora-1.5"
sys.path.insert(0, str(NORA_REPO))

import tensorflow as tf  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

tf.config.set_visible_devices([], "GPU")

import inference.modelling_expert as nora_modeling  # noqa: E402


class NORA15ModelServer(StableTaskMixin, PredictModelServer):
    def __init__(
        self,
        checkpoint: str,
        num_flow_steps: int = 10,
        global_seed: int = 7,
        *,
        chunk_size: int = 5,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            action_ensemble=action_ensemble,
            **kwargs,
        )
        self._init_stable_tasks(global_seed)
        self.checkpoint = Path(checkpoint)
        self.num_flow_steps = int(num_flow_steps)
        if not (self.checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(self.checkpoint / "model.safetensors")

        self._model = nora_modeling.VLAWithExpert.from_pretrained(self.checkpoint)
        stats_file = self.checkpoint / "dataset_statistics.json"
        if stats_file.is_file():
            self._model.norm_stats.update(json.loads(stats_file.read_text()))
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            key = f"{suite}_no_noops"
            if key not in self._model.norm_stats:
                raise KeyError(f"NORA checkpoint lacks normalization statistics {key}")
        self._model.to(torch.device("cuda:0"))
        self._model.eval()

    def get_observation_params(self) -> dict[str, Any]:
        return {"model_chunk_size": self.chunk_size}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "language": LANGUAGE}

    @staticmethod
    def _primary_image(obs: Observation) -> np.ndarray:
        images = obs.get("images", {})
        if not isinstance(images, dict) or not images:
            raise ValueError("NORA requires an agent-view image")
        return np.asarray(next(iter(images.values())), dtype=np.uint8)

    def _prepare_batch(self, obs_batch: list[Observation]) -> dict[str, torch.Tensor]:
        texts = []
        image_inputs = []
        for obs in obs_batch:
            image = Image.fromarray(self._primary_image(obs))
            image = nora_modeling.resize_image(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "resized_height": 224,
                            "resized_width": 224,
                        },
                        {"type": "text", "text": str(obs.get("task_description", ""))},
                    ],
                }
            ]
            texts.append(
                self._model.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            sample_images, sample_videos = nora_modeling.process_vision_info(messages)
            if sample_videos:
                raise ValueError("unexpected video input in NORA LIBERO preprocessing")
            image_inputs.extend(sample_images)

        inputs = self._model.processor(
            text=texts,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        device = self._model.vlm.device
        return {key: value.to(device) for key, value in inputs.items()}

    def _noise(self, ctx_batch: list[SessionContext], device: torch.device) -> torch.Tensor:
        rows = []
        for ctx in ctx_batch:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._request_seed(ctx))
            rows.append(
                torch.randn(
                    (1, self._model.action_chunk_length, 7),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
            )
        return torch.cat(rows, dim=0)

    def _sample_batch(
        self,
        inputs: dict[str, torch.Tensor],
        ctx_batch: list[SessionContext],
    ) -> np.ndarray:
        model = self._model
        batch_size = inputs["input_ids"].shape[0]
        vlm_outputs = model.vlm(**inputs)
        kv_cache = vlm_outputs.past_key_values
        vlm_pad_mask = inputs["attention_mask"].clone()
        actions = self._noise(ctx_batch, model.vlm.device)
        dt = -1.0 / max(1, self.num_flow_steps)
        time = torch.tensor(1.0, dtype=model.vlm.dtype, device=model.vlm.device)
        action_pad_mask = torch.ones(
            batch_size, model.action_chunk_length, device=model.vlm.device, dtype=torch.bool
        )
        action_attn_mask = torch.zeros_like(action_pad_mask)
        concat_pad_mask = torch.cat([vlm_pad_mask, action_pad_mask], dim=1)
        concat_attn_mask = torch.cat([vlm_pad_mask, action_attn_mask], dim=1)
        full_mask = nora_modeling.make_att_2d_masks(concat_pad_mask, concat_attn_mask)
        while time >= -dt / 2:
            velocity = model.denoise_step(
                x_t=actions,
                timestep=time.expand(batch_size),
                vlm_kv_cache=kv_cache,
                full_2d_attn_mask=full_mask,
            )
            actions += dt * velocity
            time += dt
        return actions.float().cpu().numpy()

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        return self.predict_batch([obs], [ctx])[0]

    def predict_batch(
        self,
        obs_batch: list[Observation],
        ctx_batch: list[SessionContext],
    ) -> list[Action]:
        if len(obs_batch) != len(ctx_batch):
            raise ValueError("observation/context batch mismatch")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            normalized = self._sample_batch(self._prepare_batch(obs_batch), ctx_batch)

        results = []
        for row, ctx in zip(normalized, ctx_batch):
            key = f"{self._suite(ctx)}_no_noops"
            actions = self._model.unnormalize_action(row[None], key)[0].astype(np.float32)
            # Released NORA: 0=close, 1=open; LIBERO: +1=close, -1=open.
            actions[..., -1] = np.where(actions[..., -1] > 0.5, -1.0, 1.0)
            results.append({"actions": actions})
        return results


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(NORA15ModelServer)
