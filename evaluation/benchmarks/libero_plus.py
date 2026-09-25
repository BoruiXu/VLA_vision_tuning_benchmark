"""LIBERO-Plus adapter for the already-validated Python 3.8 environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.benchmarks.libero_plus.benchmark import LIBEROPlusBenchmark
from vla_eval.specs import GRIPPER_CLOSE_POS, POSITION_ABSOLUTE, POSITION_DELTA, ROTATION_AA


class LegacyEnvLIBEROPlusBenchmark(LIBEROPlusBenchmark):
    """Make harness v0.5 compatible with the legacy environment's torch 1.11.

    The official v0.5 benchmark passes ``weights_only=False`` to torch.load,
    which is correct for modern PyTorch.  Torch 1.11 predates that keyword.
    The existing 10,030-task evaluator already strips it; this wrapper does the
    same before delegating to the otherwise unmodified harness benchmark.
    """

    def __init__(
        self,
        *,
        suite: str = "libero_spatial",
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
        absolute_action: bool = False,
        max_steps: int | None = None,
        env_seed: int | None = None,
        quat_no_antipodal: bool = False,
        category: str | None = None,
        difficulty_level: int | None = None,
        model_chunk_size: int = 1,
        lazy_render: bool = True,
        **kwargs: Any,
    ) -> None:
        # Keep inherited auto-configurable parameters explicit. The harness
        # inspects this signature before construction; hiding these behind
        # **kwargs silently prevents model-server observation parameters from
        # reaching LIBEROBenchmark (notably X-VLA absolute-action mode).
        super().__init__(
            suite=suite,
            seed=seed,
            num_steps_wait=num_steps_wait,
            send_wrist_image=send_wrist_image,
            send_state=send_state,
            absolute_action=absolute_action,
            max_steps=max_steps,
            env_seed=env_seed,
            quat_no_antipodal=quat_no_antipodal,
            category=category,
            difficulty_level=difficulty_level,
            **kwargs,
        )
        self.model_chunk_size = max(1, int(model_chunk_size))
        self.lazy_render = bool(lazy_render)
        self._rollout_step = 0
        self._lazy_active = False
        self._last_raw_images: dict[str, np.ndarray] = {}

    def _init_libero(self) -> None:
        if self._task_suite is not None:
            return
        import torch

        original = torch.load

        def compat_load(*args: Any, **kwargs: Any):
            kwargs.pop("weights_only", None)
            return original(*args, **kwargs)

        torch.load = compat_load
        super()._init_libero()

    def reset(self, task):
        raw_obs = super().reset(task)
        self._rollout_step = 0
        self._last_raw_images = {}
        # Noise variants post-process the observation images, so they must keep
        # the normal observable path.  Other variants can render only when the
        # model's chunk buffer is empty, exactly like eval_xvla_plus.py.
        self._lazy_active = self.lazy_render and getattr(self._env, "noise", 0) == 0
        if self._lazy_active:
            for name in ("agentview_image", "robot0_eye_in_hand_image"):
                try:
                    self._env.env.modify_observable(name, "enabled", False)
                    self._env.env.modify_observable(name, "active", False)
                except Exception:
                    self._lazy_active = False
                    break
        return raw_obs

    def step(self, action):
        result = super().step(action)
        self._rollout_step += 1
        return result

    def get_action_spec(self):
        """Declare the controller mode actually selected in ``reset``."""
        return {
            "position": POSITION_ABSOLUTE if self.absolute_action else POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def _supply_images(self, raw_obs):
        raw_obs = dict(raw_obs)
        keys = ("agentview_image", "robot0_eye_in_hand_image")
        for key in keys:
            if key in raw_obs:
                self._last_raw_images[key] = np.asarray(raw_obs[key])
        if self._lazy_active and self._rollout_step % self.model_chunk_size == 0:
            sim = self._env.env.sim
            self._last_raw_images["agentview_image"] = sim.render(
                camera_name="agentview", width=256, height=256
            )
            self._last_raw_images["robot0_eye_in_hand_image"] = sim.render(
                camera_name="robot0_eye_in_hand", width=256, height=256
            )
        for key in keys:
            if key not in raw_obs:
                try:
                    raw_obs[key] = self._last_raw_images[key]
                except KeyError as exc:
                    raise RuntimeError(f"missing cached LIBERO image {key} at step {self._rollout_step}") from exc
        return raw_obs

    def make_obs(self, raw_obs, task):
        obs = super().make_obs(self._supply_images(raw_obs), task)
        # A few noise variants produce grayscale frames.  All five VLAs expect
        # HWC RGB, matching the guard in the completed X-VLA evaluator.
        for key, image in list(obs["images"].items()):
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            elif array.ndim == 3 and array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            obs["images"][key] = np.ascontiguousarray(array)
        return obs
