# ------------------------------------------------------------------------------
# LIBERO-plus (LeRobot v2.1) handler for X-VLA.
#
# The corpus stores relative EEF deltas; X-VLA's action head speaks absolute
# [xyz, rot6d, gripper]. The conversion is pure controller arithmetic:
#
#     goal_pos = ee_pos + scale(a[:3])
#     goal_ori = mat(axisangle2quat(scale(a[3:6]))) @ (obs_mat @ RZ)
#
# RZ is the fixed observation->controller frame offset (robosuite reports the eef body
# quaternion in the obs dict but builds its goal from the grip site). Validated against
# replayed ground truth on standard LIBERO: max 0.98 mm and 0.875 deg, medians 0.52 mm
# and 0.03 deg -- 2-3% of the 0.05 m / 0.5 rad per-step action scale.
#
# Reading parquet directly avoids materialising an 880 GB HDF5 copy of the images.
#
# X-VLA's official LIBERO loader drops the first frame.  Its recurrent inference contract is:
# observation[t+1] + previous controller target[t] -> targets[t+1:t+31].  Therefore only the two
# image streams are shifted by one here; abs_trajectory still starts at target[t].
# ------------------------------------------------------------------------------
from __future__ import annotations

import io
import random
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.interpolate import interp1d

try:
    from .base import DomainHandler
except ImportError:  # allow standalone import for offline checks
    DomainHandler = object

RZ = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
OUTPUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])


def _quat2mat(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def _axisangle2mat(aa):
    theta = np.linalg.norm(aa)
    if theta < 1e-9:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


class LiberoPlusLeRobotHandler(DomainHandler):
    """LIBERO-plus in LeRobot v2.1 parquet form, converted to absolute EEF-6D on the fly."""

    dataset_name = "panda"

    def _episode_path(self, traj_idx: int) -> str:
        ep = self.meta["datalist"][traj_idx]["episode_index"]
        return "%s/data/chunk-%03d/episode_%06d.parquet" % (
            self.meta["root_path"], ep // int(self.meta.get("chunks_size", 1000)), ep)

    @staticmethod
    def _to_abs(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """[pos3, rot6d, grip] in the controller frame the model was trained in."""
        ee_pos, aa = state[:3], state[3:6]
        ctl_mat = _axisangle2mat(aa) @ RZ
        s = np.clip(action[:6], -1.0, 1.0) * OUTPUT_MAX
        goal_pos = ee_pos + s[:3]
        goal_ori = _axisangle2mat(s[3:6]) @ ctl_mat
        return np.concatenate([goal_pos, goal_ori[:, 0], goal_ori[:, 1], action[6:7]])

    def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool,
                     image_aug, lang_aug_map: dict | None, **kwargs) -> Iterable[dict]:
        df = pd.read_parquet(self._episode_path(traj_idx))
        ins = self.meta["datalist"][traj_idx]["tasks"][0]

        states = np.stack(df["state"].to_numpy())
        actions = np.stack(df["actions"].to_numpy())
        left = np.stack([self._to_abs(states[t], actions[t]) for t in range(len(df))])
        left[:, 9] = (left[:, 9] > 0.0).astype(np.float32)     # binarise gripper, as their handler does
        right = np.zeros_like(left)

        # freq/qdur chosen so one query window spans exactly num_actions consecutive control
        # steps of THIS corpus -- what matters is index spacing, not wall-clock rate.
        freq, qdur = float(num_actions), 1.0
        t = np.arange(left.shape[0], dtype=np.float64) / freq
        L = interp1d(t, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(t, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))

        idxs = list(range(0, max(0, left.shape[0] - num_actions)))
        if training:
            random.shuffle(idxs)

        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:2] = True
        img_col = df["image"].to_numpy()
        wrist_col = df["wrist_image"].to_numpy()

        for idx in idxs:
            cur = t[idx]
            q = np.linspace(cur, min(cur + qdur, float(t.max())), num_actions + 1, dtype=np.float32)
            lseq, rseq = torch.tensor(L(q)), torch.tensor(R(q))
            if (lseq[1] - lseq[0]).abs().max() < 1e-5:
                continue
            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])

            imgs = [image_aug(Image.open(io.BytesIO(img_col[idx + 1]["bytes"])).convert("RGB")),
                    image_aug(Image.open(io.BytesIO(wrist_col[idx + 1]["bytes"])).convert("RGB"))]
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            yield {
                "language_instruction": ins,
                "image_input": torch.stack(imgs, dim=0),
                "image_mask": image_mask,
                "abs_trajectory": torch.cat([lseq, rseq], -1).float(),
            }
