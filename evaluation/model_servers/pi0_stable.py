# /// script
# requires-python = "~=3.11"
# dependencies = ["vla-eval", "openpi", "numpy>=1.24"]
# ///
"""π0.5 adapter preserving the existing LIBERO-Plus rollout contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import register_pi05_config
register_pi05_config.install()

from common import StableTaskMixin
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.pi0 import Pi0ModelServer
from vla_eval.types import Action, Observation


class StablePi05Server(StableTaskMixin, Pi0ModelServer):
    """Use paired flow noise and preserve the legacy action-state feedback.

    ``code/eval_plus.py`` accumulates every executed delta action into
    ``observation/action_states``.  The standard harness adapter omits that
    optional input.  It is normally ignored by stock π0.5, but retaining it is
    important for checkpoint compatibility and makes the protocol explicit.
    """

    def __init__(self, *args: Any, global_seed: int = 7, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.max_batch_size > 1 and not self._policy._is_pytorch_model:
            raise ValueError("Batched StablePi05Server requires a merged PyTorch checkpoint")
        if self.max_batch_size > 1:
            # The pinned policy defaults to max-autotune compilation. Dynamic
            # request batches otherwise compile many shapes during live rollout.
            # Use the same eager sampler for B1 partial and B>1 full batches.
            sampler = self._policy._sample_actions
            self._policy._sample_actions = getattr(sampler, '_torchdynamo_orig_callable', sampler)
        self._init_stable_tasks(global_seed)
        self._action_states: dict[str, np.ndarray] = {}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        self._action_states[ctx.session_id] = np.zeros(7, dtype=np.float32)
        await super().on_episode_start(config, ctx)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        self._action_states.pop(ctx.session_id, None)
        await super().on_episode_end(result, ctx)

    def _noise(self, ctx: SessionContext) -> np.ndarray:
        model = self._policy._model
        # The JAX BaseModel exposes these as direct attributes, whereas the
        # PyTorch PI0Pytorch used by our merged LoRA checkpoint keeps them in
        # ``model.config``.  Read both layouts so the paired-noise protocol is
        # identical for stock and merged checkpoints.
        config = getattr(model, "config", None)
        horizon = getattr(model, "action_horizon", None)
        action_dim = getattr(model, "action_dim", None)
        if horizon is None and config is not None:
            horizon = getattr(config, "action_horizon", None)
        if action_dim is None and config is not None:
            action_dim = getattr(config, "action_dim", None)
        if horizon is None or action_dim is None:
            raise AttributeError(
                "unable to resolve action_horizon/action_dim from policy model or model.config"
            )
        return np.random.default_rng(self._request_seed(ctx)).standard_normal(
            (int(horizon), int(action_dim)), dtype=np.float32
        )

    def get_observation_params(self) -> dict[str, Any]:
        params = super().get_observation_params()
        params.update({"quat_no_antipodal": True, "model_chunk_size": self.chunk_size})
        return params

    def _prepare_observation(self, obs: Observation, ctx: SessionContext) -> dict[str, Any]:
        images = obs.get("images", {})
        values = list(images.values()) if isinstance(images, dict) else []
        primary = (
            np.asarray(values[0], dtype=np.uint8)
            if values
            else np.zeros((256, 256, 3), dtype=np.uint8)
        )
        primary = self._maybe_resize(primary)
        openpi_obs: dict[str, Any] = {
            self.image_key: primary,
            "prompt": str(obs.get("task_description", "")),
            "observation/action_states": self._action_states.setdefault(
                ctx.session_id, np.zeros(7, dtype=np.float32)
            ).copy(),
        }
        if self.wrist_image_key:
            wrist = np.asarray(values[1], dtype=np.uint8) if len(values) > 1 else np.zeros_like(primary)
            openpi_obs[self.wrist_image_key] = self._maybe_resize(wrist)
        if self.state_key:
            state = obs.get("states", obs.get("state"))
            openpi_obs[self.state_key] = (
                np.asarray(state, dtype=np.float64)
                if state is not None
                else np.zeros(self.state_dim, dtype=np.float64)
            )

        return openpi_obs

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        result = self._policy.infer(self._prepare_observation(obs, ctx), noise=self._noise(ctx))
        actions = np.asarray(result["actions"], dtype=np.float32)
        # The next policy query happens after exactly chunk_size actions have
        # been emitted by PredictModelServer's per-session chunk buffer.
        self._action_states[ctx.session_id] += actions[: self.chunk_size].sum(axis=0)
        return {"actions": actions}

    def _infer_batch(self, observations: list[dict[str, Any]], noise: np.ndarray) -> list[dict[str, Any]]:
        """Batch the pinned OpenPI Policy's transformed inputs, not raw observations.

        Input and output transforms run per sample exactly as in Policy.infer;
        only sample_actions receives a batch. Keep the old single-item path.
        """
        import jax
        import torch
        from openpi.models import model as openpi_model

        policy = self._policy
        if not policy._is_pytorch_model:
            raise ValueError("Batch inference supports merged PyTorch checkpoints only")
        prepared = [policy._input_transform(jax.tree.map(lambda x: x, obs)) for obs in observations]
        inputs = jax.tree.map(
            lambda *rows: torch.from_numpy(np.stack(rows)).to(policy._pytorch_device), *prepared
        )
        kwargs = dict(policy._sample_kwargs)
        kwargs['noise'] = torch.from_numpy(noise).to(policy._pytorch_device)
        # sample_actions has the same no_grad decorator used by Policy.infer.
        # Do not add inference_mode here: that changes the compiled graph mode.
        actions = policy._sample_actions(
            policy._pytorch_device, openpi_model.Observation.from_dict(inputs), **kwargs
        )
        if actions.shape[0] != len(observations):
            raise RuntimeError("OpenPI returned the wrong batch size")
        outputs = {'state': inputs['state'], 'actions': actions}
        return [policy._output_transform(jax.tree.map(
            lambda x: np.asarray(x[i].detach().cpu()), outputs
        )) for i in range(len(observations))]

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        if len(obs_batch) != len(ctx_batch):
            raise ValueError("observation/context batch mismatch")
        if not obs_batch:
            return []
        if len({ctx.session_id for ctx in ctx_batch}) != len(ctx_batch):
            raise ValueError("A batch must contain distinct sessions")
        if len(obs_batch) == 1:
            return [self.predict(obs_batch[0], ctx_batch[0])]
        prepared = [self._prepare_observation(obs, ctx) for obs, ctx in zip(obs_batch, ctx_batch)]
        noise = np.stack([self._noise(ctx) for ctx in ctx_batch])
        outputs = self._infer_batch(prepared, noise)
        actions = [np.asarray(row['actions'], dtype=np.float32) for row in outputs]
        if any(a.ndim != 2 or a.shape[1] != 7 or a.shape[0] < self.chunk_size
               or not np.isfinite(a).all() for a in actions):
            raise RuntimeError("Invalid batched OpenPI actions")
        for ctx, row in zip(ctx_batch, actions):
            self._action_states[ctx.session_id] += row[: self.chunk_size].sum(axis=0)
        return [{'actions': row} for row in actions]


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(StablePi05Server)
