"""Filter original LeRobot frame indices without changing chunks, IDs or stats."""
import hashlib
import json
import os
from pathlib import Path


def install(data_module):
    split = Path(os.environ["PI05_DOMAIN_META"])
    info = json.loads((split / "info.json").read_text())
    rows = [json.loads(line) for line in (split / "episodes.jsonl").read_text().splitlines()]
    ids = [int(row["episode_index"]) for row in rows]
    if len(ids) != len(set(ids)) or len(ids) != info["total_episodes"]:
        raise RuntimeError("Invalid domain episode selection")
    original = data_module.create_torch_dataset

    def create(data_config, action_horizon, model_config):
        import numpy as np
        from torch.utils.data import Subset
        if data_config.repo_id != "local/libero_plus_mix":
            raise RuntimeError("Domain filter requires the original mixed LeRobot store")
        dataset = original(data_config, action_horizon, model_config)
        raw = dataset
        while hasattr(raw, "_dataset"):
            raw = raw._dataset
        bounds = raw.episode_data_index
        indices = np.concatenate([np.arange(int(bounds["from"][i]), int(bounds["to"][i]), dtype=np.int64) for i in ids])
        if len(indices) != info["total_frames"]:
            raise RuntimeError("Selected frame count mismatch")
        print(f"[domain-gate] {split.parent.name}: episodes={len(ids)} frames={len(indices)} original IDs and action boundaries retained", flush=True)
        # Hugging Face datasets rejects numpy.int64 keys; preserve Python ints.
        return Subset(dataset, indices.tolist())

    data_module.create_torch_dataset = create
    return {"metadata": str(split), "domain_filter": info["domain_filter"],
            "episodes": len(ids), "frames": info["total_frames"],
            "episodes_file_sha256": hashlib.sha256((split / "episodes.jsonl").read_bytes()).hexdigest()}
