"""Runtime RLDS filter for target-only OpenVLA-OFT training.

The perturbation label is provenance carried by the original RLDS trajectory, not a feature
reconstructed from images or task text.  Install this patch before ``RLDSDataset`` is built and
set ``OFT_DOMAIN_FILTER`` to ``camera_view`` or ``noise``.

Normalization is intentionally outside this module.  The experiment launcher continues to use
the released, suite-conditioned OFT_m base statistics.  The per-split loader-statistics file only
provides correct trajectory/transition counts and avoids computing unused target-only action
statistics.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import dlimp as dl
import tensorflow as tf


VALID_DOMAINS = {"camera_view": "camera_only", "noise": "noise_only"}


def install(domain: str, audit_path: str | Path) -> dict[str, Any] | None:
    """Filter raw ``libero_mix`` trajectories to one audited perturbation domain."""

    domain = domain.strip().lower()
    if not domain:
        print("[domain] filter disabled: using the full libero_mix corpus", flush=True)
        return None
    if domain not in VALID_DOMAINS:
        raise ValueError(
            f"OFT_DOMAIN_FILTER must be empty or one of {sorted(VALID_DOMAINS)}, got {domain!r}"
        )

    audit_path = Path(audit_path).resolve()
    audit = json.loads(audit_path.read_text())
    split_name = VALID_DOMAINS[domain]
    split = audit["splits"][split_name]
    if split["source_domain"] != domain:
        raise ValueError(f"domain audit mismatch for {split_name}: {split}")
    if audit["camera_noise_overlap_episodes"] != 0:
        raise ValueError("Camera-only and Noise-only are not disjoint according to the audit")

    loader_statistics_path = Path(split["rlds_loader_statistics"])
    loader_statistics = json.loads(loader_statistics_path.read_text())
    if int(loader_statistics["num_trajectories"]) != int(split["episodes"]):
        raise ValueError("loader statistics trajectory count differs from split audit")
    if int(loader_statistics["num_transitions"]) != int(split["frames"]):
        raise ValueError("loader statistics transition count differs from split audit")

    original_from_rlds = dl.DLataset.from_rlds

    @tf.autograph.experimental.do_not_convert
    def keep_domain(traj: dict[str, Any]) -> tf.Tensor:
        source_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        return tf.strings.regex_full_match(source_path, rf".*/pro_data/{domain}/.*")

    def filtered_from_rlds(
        builder: Any,
        split: str = "train",
        shuffle: bool = True,
        num_parallel_reads: int = tf.data.AUTOTUNE,
    ) -> dl.DLataset:
        dataset = original_from_rlds(
            builder,
            split=split,
            shuffle=shuffle,
            num_parallel_reads=num_parallel_reads,
        )
        builder_name = getattr(builder, "name", getattr(builder.info, "name", ""))
        if builder_name == "libero_mix":
            dataset = dataset.filter(keep_domain)
        return dataset

    # from_rlds is declared staticmethod upstream.  Preserve that descriptor contract so Python
    # does not inject DLataset as a first positional argument.
    dl.DLataset.from_rlds = staticmethod(filtered_from_rlds)

    # make_interleaved_dataset invokes this module global twice: once to get size/statistics and
    # once to build the repeating dataset.  Supplying an audited stats-shaped record both times
    # gives the right dataset length without recomputing target-only normalization statistics.
    from prismatic.vla.datasets.rlds import dataset as rlds_dataset

    original_make_dataset = rlds_dataset.make_dataset_from_rlds

    def make_dataset_with_counts(*args: Any, **kwargs: Any):
        name = kwargs.get("name", args[0] if args else None)
        if name == "libero_mix" and kwargs.get("dataset_statistics") is None:
            kwargs["dataset_statistics"] = copy.deepcopy(loader_statistics)
        return original_make_dataset(*args, **kwargs)

    rlds_dataset.make_dataset_from_rlds = make_dataset_with_counts
    print(
        f"[domain] filter={domain} ({split_name}): {split['episodes']:,} trajectories, "
        f"{split['frames']:,} transitions, 40 tasks, ids={split['episode_ids_sha256']}; "
        f"audit={audit_path}",
        flush=True,
    )
    return {
        "domain": domain,
        "split_name": split_name,
        "episodes": int(split["episodes"]),
        "frames": int(split["frames"]),
        "episode_ids_sha256": split["episode_ids_sha256"],
    }
