#!/usr/bin/env python3
"""Build audited Camera-only and Noise-only views of LIBERO-plus training data.

The original RLDS export records the perturbation family in
``episode_metadata.file_path``.  The LeRobot conversion kept the same deterministic episode
order but did not copy that field into ``episodes.jsonl``.  This script joins the two formats,
checks every row against the task encoded in the source filename, and emits lightweight
metadata views that continue to point at the original parquet corpus.

No image or trajectory data is copied.  The resulting LeRobot views are directly consumable by
X-VLA's ``LiberoPlusLeRobotHandler``, whose loader honours ``info.json[root_path]`` and the
original, possibly non-contiguous ``episode_index`` values.
"""

from __future__ import annotations

import vla_paths

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import tensorflow as tf
import tensorflow_datasets as tfds


DEFAULT_RLDS = Path(
    vla_paths.resolve('data/rlds/libero_mix/1.0.0')
)
DEFAULT_LEROBOT = Path(vla_paths.resolve('data/lerobot/local/libero_plus_mix'))
DEFAULT_OUTPUT = Path(vla_paths.resolve('data/libero_plus_domain_splits'))

DOMAIN_TO_SPLIT = {
    "camera_view": "camera_only",
    "noise": "noise_only",
}
KNOWN_DOMAINS = {"env", "light", "noise", "language", "camera_view"}
DOMAIN_RE = re.compile(r"/pro_data/([^/]+)/")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _id_digest(ids: Iterable[int]) -> str:
    payload = "\n".join(str(index) for index in ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _source_paths(rlds_builder: Path) -> list[str]:
    builder = tfds.builder_from_directory(str(rlds_builder))
    dataset = builder.as_dataset(split="train", shuffle_files=False)

    @tf.autograph.experimental.do_not_convert
    def select_path(episode: dict[str, Any]) -> tf.Tensor:
        return episode["episode_metadata"]["file_path"]

    # Selecting before iteration lets tf.data prune the large image/step tensors from the graph.
    dataset = dataset.map(select_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return [value.decode("utf-8") for value in dataset.as_numpy_iterator()]


def _validate_join(episodes: list[dict[str, Any]], paths: list[str]) -> list[dict[str, Any]]:
    if len(episodes) != len(paths):
        raise ValueError(f"RLDS/LeRobot episode count mismatch: {len(paths)} != {len(episodes)}")

    joined = []
    for expected_index, (episode, source_path) in enumerate(zip(episodes, paths)):
        episode_index = int(episode["episode_index"])
        if episode_index != expected_index:
            raise ValueError(
                f"LeRobot episode order is not contiguous at row {expected_index}: {episode_index}"
            )
        match = DOMAIN_RE.search(source_path)
        if match is None:
            raise ValueError(f"source path has no /pro_data/<domain>/ component: {source_path}")
        domain = match.group(1)
        if domain not in KNOWN_DOMAINS:
            raise ValueError(f"unknown perturbation domain {domain!r}: {source_path}")

        tasks = episode.get("tasks", [])
        if len(tasks) != 1:
            raise ValueError(f"episode {episode_index} has {len(tasks)} task strings, expected one")
        task_slug = _slug(tasks[0])
        source_slug = _slug(Path(source_path).stem)
        if task_slug not in source_slug:
            raise ValueError(
                f"RLDS/LeRobot task mismatch at episode {episode_index}: "
                f"task={tasks[0]!r}, source={source_path!r}"
            )

        joined.append(
            {
                "episode_index": episode_index,
                "domain": domain,
                "source_file_path": source_path,
                "task": tasks[0],
                "length": int(episode["length"]),
            }
        )
    return joined


def _build_view(
    split_name: str,
    domain: str,
    selected: list[dict[str, Any]],
    episodes_by_id: dict[int, dict[str, Any]],
    stats_by_id: dict[int, dict[str, Any]],
    all_tasks: list[dict[str, Any]],
    base_info: dict[str, Any],
    mixed_loader_stats: dict[str, Any],
    lerobot_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    selected_ids = [row["episode_index"] for row in selected]
    selected_id_set = set(selected_ids)
    selected_episodes = [episodes_by_id[index] for index in selected_ids]
    selected_stats = [stats_by_id[index] for index in selected_ids]
    selected_task_names = {row["task"] for row in selected}
    selected_tasks = [row for row in all_tasks if row["task"] in selected_task_names]

    if len(selected_id_set) != len(selected_ids):
        raise ValueError(f"{split_name}: duplicate episode ids")
    if set(stats_by_id).issuperset(selected_id_set) is False:
        raise ValueError(f"{split_name}: episode statistics are incomplete")
    if {row["task"] for row in selected_tasks} != selected_task_names:
        raise ValueError(f"{split_name}: tasks.jsonl is incomplete")

    info = dict(base_info)
    info.update(
        {
            "root_path": str(lerobot_root.resolve()),
            "total_episodes": len(selected),
            "total_frames": sum(row["length"] for row in selected),
            "total_tasks": len(selected_task_names),
            "total_chunks": len(
                {index // int(base_info.get("chunks_size", 1000)) for index in selected_ids}
            ),
            # This range indexes rows in the metadata view, not physical source episode ids.
            "splits": {"train": f"0:{len(selected)}"},
            "domain_filter": {
                "source_field": "episode_metadata.file_path",
                "source_path_component": f"/pro_data/{domain}/",
                "domain": domain,
                "split_name": split_name,
                "episode_ids_sha256": _id_digest(selected_ids),
            },
        }
    )

    meta = output_root / split_name / "meta"
    _write_json(meta / "info.json", info)
    _write_jsonl(meta / "episodes.jsonl", selected_episodes)
    _write_jsonl(meta / "episodes_stats.jsonl", selected_stats)
    _write_jsonl(meta / "tasks.jsonl", selected_tasks)
    _write_jsonl(output_root / split_name / "source_episodes.jsonl", selected)
    loader_stats = deepcopy(mixed_loader_stats)
    loader_stats["num_trajectories"] = len(selected)
    loader_stats["num_transitions"] = sum(row["length"] for row in selected)
    _write_json(output_root / split_name / "rlds_loader_statistics.json", loader_stats)

    # Check the exact physical files addressed by the first/last and every selected source id.
    chunks_size = int(base_info.get("chunks_size", 1000))
    missing = []
    for index in selected_ids:
        path = lerobot_root / "data" / f"chunk-{index // chunks_size:03d}" / f"episode_{index:06d}.parquet"
        if not path.is_file():
            missing.append(str(path))
            if len(missing) == 10:
                break
    if missing:
        raise FileNotFoundError(f"{split_name}: selected parquet files are missing: {missing}")

    return {
        "source_domain": domain,
        "episodes": len(selected),
        "frames": sum(row["length"] for row in selected),
        "tasks": len(selected_task_names),
        "episode_ids_sha256": _id_digest(selected_ids),
        "first_episode_id": selected_ids[0],
        "last_episode_id": selected_ids[-1],
        "metadata_info": str(meta / "info.json"),
        "rlds_loader_statistics": str(
            output_root / split_name / "rlds_loader_statistics.json"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlds-builder", type=Path, default=DEFAULT_RLDS)
    parser.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()

    rlds_builder = args.rlds_builder.resolve()
    lerobot_root = args.lerobot_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"output exists; inspect it or rerun with --force: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    meta_root = lerobot_root / "meta"
    info_path = meta_root / "info.json"
    episodes_path = meta_root / "episodes.jsonl"
    stats_path = meta_root / "episodes_stats.jsonl"
    tasks_path = meta_root / "tasks.jsonl"
    base_info = json.loads(info_path.read_text())
    episodes = _read_jsonl(episodes_path)
    episode_stats = _read_jsonl(stats_path)
    tasks = _read_jsonl(tasks_path)

    print(f"[scan] reading RLDS source labels from {rlds_builder}", flush=True)
    paths = _source_paths(rlds_builder)
    print(f"[scan] joining and validating {len(paths):,} episodes", flush=True)
    joined = _validate_join(episodes, paths)

    if len(joined) != int(base_info["total_episodes"]):
        raise ValueError("joined count differs from info.json total_episodes")
    if sum(row["length"] for row in joined) != int(base_info["total_frames"]):
        raise ValueError("joined frame count differs from info.json total_frames")

    # OpenVLA needs a statistics-shaped record for loader bookkeeping.  The actual normalization
    # remains the released four-suite OFT_m statistics; these mixed action/proprio arrays are only
    # used to avoid recomputing irrelevant target-only statistics, with counts corrected below.
    candidates = sorted(rlds_builder.glob("dataset_statistics_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no cached RLDS loader statistics under {rlds_builder}")
    mixed_loader_stats = json.loads(candidates[-1].read_text())
    if (
        int(mixed_loader_stats["num_trajectories"]) != len(joined)
        or int(mixed_loader_stats["num_transitions"])
        != sum(row["length"] for row in joined)
    ):
        raise ValueError(f"cached loader statistics do not match source totals: {candidates[-1]}")

    episodes_by_id = {int(row["episode_index"]): row for row in episodes}
    stats_by_id = {int(row["episode_index"]): row for row in episode_stats}
    if len(episodes_by_id) != len(episodes) or len(stats_by_id) != len(episode_stats):
        raise ValueError("duplicate episode indices in source LeRobot metadata")

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_domain[row["domain"]].append(row)
    if set(by_domain) != KNOWN_DOMAINS:
        raise ValueError(f"domain set mismatch: found {sorted(by_domain)}, expected {sorted(KNOWN_DOMAINS)}")

    _write_jsonl(output_root / "episode_domains.jsonl", joined)
    split_summaries = {}
    for domain, split_name in DOMAIN_TO_SPLIT.items():
        if not by_domain[domain]:
            raise ValueError(f"domain {domain!r} is empty")
        split_summaries[split_name] = _build_view(
            split_name,
            domain,
            by_domain[domain],
            episodes_by_id,
            stats_by_id,
            tasks,
            base_info,
            mixed_loader_stats,
            lerobot_root,
            output_root,
        )

    camera_ids = {row["episode_index"] for row in by_domain["camera_view"]}
    noise_ids = {row["episode_index"] for row in by_domain["noise"]}
    overlap = camera_ids & noise_ids
    if overlap:
        raise ValueError(f"Camera-only/Noise-only overlap: {sorted(overlap)[:10]}")

    domain_summary = {
        domain: {
            "episodes": len(rows),
            "frames": sum(row["length"] for row in rows),
            "tasks": len({row["task"] for row in rows}),
        }
        for domain, rows in sorted(by_domain.items())
    }
    audit = {
        "schema_version": 1,
        "creation_command": " ".join(sys.argv),
        "rlds_builder": str(rlds_builder),
        "lerobot_root": str(lerobot_root),
        "output_root": str(output_root),
        "source_metadata_sha256": {
            name: _sha256(path)
            for name, path in {
                "info.json": info_path,
                "episodes.jsonl": episodes_path,
                "episodes_stats.jsonl": stats_path,
                "tasks.jsonl": tasks_path,
            }.items()
        },
        "source_totals": {
            "episodes": len(joined),
            "frames": sum(row["length"] for row in joined),
            "tasks": len({row["task"] for row in joined}),
        },
        "domains": domain_summary,
        "splits": split_summaries,
        "camera_noise_overlap_episodes": 0,
        "join_validation": {
            "rlds_and_lerobot_counts_equal": True,
            "episode_indices_contiguous_and_ordered": True,
            "all_task_slugs_match_source_filenames": True,
            "all_selected_parquet_files_exist": True,
        },
    }
    _write_json(output_root / "audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    # Keep label extraction off GPU even when this is launched from a training shell.
    tf.config.set_visible_devices([], "GPU")
    main()
