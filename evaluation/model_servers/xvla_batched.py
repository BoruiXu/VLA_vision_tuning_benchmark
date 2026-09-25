# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.2",
#     "transformers>=4.44,<=4.51.3",
#     "numpy>=1.24",
#     "pillow>=9.0",
#     "opencv-python-headless",
#     "fastapi",
#     "json-numpy",
#     "uvicorn",
#     "einops",
#     "timm",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""Batched, deterministic X-VLA server for the LIBERO-Plus evaluation."""

from __future__ import annotations

import numpy as np

from common import StableTaskMixin
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_ABSOLUTE,
    ROTATION_AA,
    DimSpec,
)
from vla_eval.model_servers.xvla import (
    XVLAModelServer,
    _convert_ee6d_to_7d,
    _obs_state_array,
    _ordered_images,
    _state_to_xvla_proprio,
)


class BatchedXVLAServer(StableTaskMixin, XVLAModelServer):
    """Preserve X-VLA's closed-loop proprio while batching GPU forwards."""

    def __init__(self, *args, global_seed: int = 7, **kwargs):
        super().__init__(*args, **kwargs)
        if self.benchmark_profile != "libero":
            raise ValueError("BatchedXVLAServer currently supports benchmark_profile='libero' only")
        self._init_stable_tasks(global_seed)

    def predict(self, obs, ctx):
        return self.predict_batch([obs], [ctx])[0]

    def get_observation_params(self):
        params = super().get_observation_params()
        params["model_chunk_size"] = self.chunk_size
        return params

    def get_action_spec(self):
        return {
            "position": POSITION_ABSOLUTE,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self):
        # The wire observation is the benchmark's 8-D EEF/controller state;
        # this adapter converts it to X-VLA's 20-D contiguous-rot6d proprio.
        state = DimSpec(
            "state",
            20,
            "xvla_proprio_20d",
            accepts=frozenset(
                {
                    "eef_pos3_axisangle3_gripper2",
                    "rot6d_contiguous_proprio_20d",
                    "rot6d_interleaved_proprio_20d",
                }
            ),
        )
        return {
            "agentview": IMAGE_RGB,
            "wrist": IMAGE_RGB,
            "state": state,
            "language": LANGUAGE,
        }

    def _proprio(self, obs, ctx, device, dim):
        import torch

        last_actions = self._last_raw_actions.get(ctx.session_id)
        if last_actions is not None:
            value = np.zeros(dim, dtype=np.float32)
            n = min(self._predicted_proprio_dims or dim, dim, last_actions.shape[-1])
            value[:n] = last_actions[-1, :n]
        else:
            raw = _obs_state_array(obs)
            if raw is None:
                value = np.zeros(dim, dtype=np.float32)
            elif len(raw) == dim:
                value = raw.astype(np.float32, copy=False)
            else:
                value = _state_to_xvla_proprio(
                    raw,
                    dim,
                    euler_state=self._euler_state,
                    axisangle_to_rot6d=self._axisangle_to_rot6d,
                    euler_to_rot6d=self._euler_to_rot6d,
                )
        return torch.tensor(value, dtype=torch.float32, device=device)

    def _initial_noise(self, ctx_batch, device, shape):
        import torch

        rows = []
        for ctx in ctx_batch:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._request_seed(ctx))
            rows.append(torch.randn((1, *shape), generator=generator, device=device, dtype=torch.float32))
        return torch.cat(rows, dim=0)

    def _generate(self, inputs, domain_id, proprio, initial_noise):
        import torch

        model = self._model
        enc = model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
        batch_size = inputs["input_ids"].shape[0]
        action = torch.zeros_like(initial_noise)
        steps = max(1, int(self.denoising_steps))
        for index in range(steps, 0, -1):
            time = torch.full(
                (batch_size,), index / steps, device=proprio.device, dtype=proprio.dtype
            )
            noisy = initial_noise * time.view(-1, 1, 1) + action * (1 - time).view(-1, 1, 1)
            proprio_model, noisy_model = model.action_space.preprocess(proprio, noisy)
            action = model.transformer(
                domain_id=domain_id,
                action_with_noise=noisy_model,
                proprio=proprio_model,
                t=time,
                **enc,
            )
        return model.action_space.postprocess(action)

    def predict_batch(self, obs_batch, ctx_batch):
        import torch
        from PIL import Image

        if len(obs_batch) != len(ctx_batch):
            raise ValueError("observation/context batch mismatch")
        image_batch = []
        instructions = []
        for obs, ctx in zip(obs_batch, ctx_batch):
            if obs.get("episode_restart"):
                self._last_raw_actions.pop(ctx.session_id, None)
            images = _ordered_images(obs, self._image_keys)
            if self._unflip_wrist and len(images) >= 2:
                images[1] = images[1][::-1, ::-1].copy()
            image_batch.append([Image.fromarray(image) for image in images])
            instructions.append(str(obs.get("task_description", "")))

        inputs = self._processor(images=image_batch, language_instruction=instructions)
        device = next(self._model.parameters()).device
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        dim = self._model.action_space.dim_action
        proprio = torch.stack(
            [self._proprio(obs, ctx, device, dim) for obs, ctx in zip(obs_batch, ctx_batch)]
        )
        domain_id = torch.full(
            (len(obs_batch),), self.domain_id, dtype=torch.long, device=device
        )
        initial_noise = self._initial_noise(
            ctx_batch, device, (self._model.num_actions, dim)
        )
        with torch.inference_mode():
            actions = self._generate(inputs, domain_id, proprio, initial_noise)
        actions_np = actions.float().cpu().numpy()

        results = []
        for raw_actions, ctx in zip(actions_np, ctx_batch):
            self._last_raw_actions[ctx.session_id] = raw_actions.copy()
            converted = _convert_ee6d_to_7d(
                raw_actions,
                self._gripper_threshold,
                self._gripper_close_above,
                self._rot6d_to_matrix,
            )
            results.append({"actions": converted})
        return results


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(BatchedXVLAServer)
