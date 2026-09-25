#!/usr/bin/env python3
"""Gate the LIBERO-plus temporal pairing used by X-VLA on one real episode."""

from __future__ import annotations

import vla_paths

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = vla_paths.resolve('third_party/X-VLA')
sys.path.insert(0, REPO)

from datasets.domain_handler.libero_plus import LiberoPlusLeRobotHandler  # noqa: E402
from datasets.utils import action_slice  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta",
        default=vla_paths.resolve('data/lerobot/local/libero_plus_mix/meta/info.json'),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text())
    meta["datalist"] = [json.loads(line) for line in
                        (meta_path.parent / "episodes.jsonl").read_text().splitlines()]
    meta.setdefault("root_path", str(meta_path.parent.parent))
    handler = LiberoPlusLeRobotHandler(meta=meta, num_views=2)
    frame = pd.read_parquet(handler._episode_path(0))
    states = np.stack(frame["state"].to_numpy())
    actions = np.stack(frame["actions"].to_numpy())
    targets = np.stack([handler._to_abs(states[i], actions[i]) for i in range(len(frame))])
    targets[:, 9] = (targets[:, 9] > 0).astype(np.float32)
    horizon = 30
    valid = [i for i in range(len(frame) - horizon)
             if np.max(np.abs(targets[i + 1] - targets[i])) >= 1e-5]
    if not valid:
        raise RuntimeError("episode has no valid training sample")
    expected_idx = valid[0]

    def to_tensor(image):
        return torch.from_numpy(np.array(image, copy=True))

    sample = next(handler.iter_episode(
        0, num_actions=horizon, training=False, image_aug=to_tensor, lang_aug_map=None
    ))
    sliced = action_slice(sample["abs_trajectory"])
    np.testing.assert_allclose(sliced["proprio"][:10], targets[expected_idx], atol=2e-6)
    np.testing.assert_allclose(sliced["action"][0, :10], targets[expected_idx + 1], atol=2e-6)
    np.testing.assert_allclose(sliced["action"][-1, :10], targets[expected_idx + horizon], atol=2e-6)

    encoded = frame["image"].to_numpy()[expected_idx + 1]["bytes"]
    expected_image = np.array(Image.open(io.BytesIO(encoded)).convert("RGB"))
    np.testing.assert_array_equal(sample["image_input"][0].numpy(), expected_image)

    last_idx = len(frame) - horizon - 1
    assert last_idx + 1 < len(frame)
    assert last_idx + horizon == len(frame) - 1
    print(
        "PASS: image[t+1] + proprio=target[t] -> labels=target[t+1:t+31]; "
        f"T={len(frame)}, first_valid_t={expected_idx}, last_valid_t={last_idx}",
        flush=True,
    )


if __name__ == "__main__":
    main()
