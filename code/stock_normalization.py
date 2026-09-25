"""Keep OpenVLA-OFT's released LIBERO normalization fixed during plus tuning.

The LIBERO-plus RLDS export is flattened into one dataset named ``libero_mix``.  Letting the
stock loader compute one new set of mixed q01/q99 statistics would change the action/proprio
coordinate system while this experiment is supposed to change only the visual encoder.  The
released OFT_m base instead has one statistics entry for each of the four LIBERO suites.

This module patches the trajectory normalization hook to choose that released entry from the
trajectory's language instruction.  It also makes the dataset expose those same four entries to
checkpoint saving, so evaluation uses exactly the coordinate systems seen by the frozen heads.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import tensorflow as tf


EXPECTED_KEYS = (
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
)


def _load_and_validate(stats_path: Path, task_map_path: Path):
    stats = json.loads(stats_path.read_text())
    task_groups = json.loads(task_map_path.read_text())
    if tuple(task_groups) != EXPECTED_KEYS:
        raise ValueError(
            f"task map keys/order must be {EXPECTED_KEYS}, found {tuple(task_groups)}"
        )
    missing_stats = [key for key in EXPECTED_KEYS if key not in stats]
    if missing_stats:
        raise KeyError(f"base checkpoint lacks normalization entries: {missing_stats}")

    task_to_index = {}
    for index, suite_key in enumerate(EXPECTED_KEYS):
        tasks = task_groups[suite_key]
        if len(tasks) != 10 or len(set(tasks)) != 10:
            raise ValueError(f"{suite_key}: expected 10 unique tasks, found {len(set(tasks))}")
        for task in tasks:
            normalized = " ".join(task.lower().split())
            if normalized in task_to_index:
                raise ValueError(f"task appears in multiple suites: {normalized!r}")
            task_to_index[normalized] = index
    if len(task_to_index) != 40:
        raise ValueError(f"expected 40 unique LIBERO tasks, found {len(task_to_index)}")

    # Fail before touching the dataset if an entry is malformed.  The frozen OFT action head is
    # 7-D and its proprio projector is 8-D; the gripper action must remain unnormalized.
    for suite_key in EXPECTED_KEYS:
        record = stats[suite_key]
        for field, dim in (("action", 7), ("proprio", 8)):
            for stat_name in ("mean", "std", "min", "max", "q01", "q99"):
                if len(record[field][stat_name]) != dim:
                    raise ValueError(
                        f"{suite_key}/{field}/{stat_name}: expected {dim} values"
                    )
        if record["action"].get("mask") != [True] * 6 + [False]:
            raise ValueError(f"{suite_key}: expected gripper-excluding action mask")
    return stats, task_to_index


def install(ft_module: Any, stats_path: str | Path, task_map_path: str | Path) -> dict:
    """Install suite-conditioned base normalization into an imported finetune module.

    ``ft_module`` is the experiment-local copy of the official finetune launcher.  It imported
    ``RLDSDataset`` by name, so replacing that reference is enough to keep the upstream repo
    immutable.
    """

    stats_path = Path(stats_path).resolve()
    task_map_path = Path(task_map_path).resolve()
    stats, task_to_index = _load_and_validate(stats_path, task_map_path)

    # make_dataset_from_rlds imported the function directly into this module's globals.
    from prismatic.vla.datasets.rlds import dataset as rlds_dataset

    original_normalize = rlds_dataset.normalize_action_and_proprio
    keys = tf.constant(list(task_to_index), dtype=tf.string)
    values = tf.constant(list(task_to_index.values()), dtype=tf.int64)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), default_value=-1
    )

    # Stack the four released entries once. tf.gather then selects a complete metadata record in
    # the trajectory graph without Python callbacks or nondeterministic mutable state.
    stacked = {}
    for field in ("action", "proprio"):
        stacked[field] = {}
        stat_names = set().union(*(stats[key][field].keys() for key in EXPECTED_KEYS))
        for stat_name in stat_names:
            arrays = [stats[key][field].get(stat_name) for key in EXPECTED_KEYS]
            if any(value is None for value in arrays):
                continue
            dtype = tf.bool if stat_name == "mask" else tf.float32
            stacked[field][stat_name] = tf.constant(arrays, dtype=dtype)

    def suite_conditioned_normalize(traj, metadata, normalization_type):
        del metadata  # the mixed-dataset statistics are deliberately not part of this experiment
        language = tf.strings.lower(tf.strings.strip(traj["task"]["language_instruction"][0]))
        suite_index = table.lookup(language)
        assertion = tf.debugging.assert_greater_equal(
            suite_index,
            tf.constant(0, tf.int64),
            message="unknown LIBERO task; refusing to choose the wrong normalization suite",
        )
        with tf.control_dependencies([assertion]):
            selected = {
                field: {
                    stat_name: tf.gather(tensor, suite_index)
                    for stat_name, tensor in field_stats.items()
                }
                for field, field_stats in stacked.items()
            }
            return original_normalize(traj, selected, normalization_type)

    rlds_dataset.normalize_action_and_proprio = suite_conditioned_normalize

    original_dataset_class = ft_module.RLDSDataset

    class BaseNormalizedRLDSDataset(original_dataset_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # save_dataset_statistics mutates NumPy arrays into lists, hence a private copy.
            self.dataset_statistics = copy.deepcopy(
                {key: stats[key] for key in EXPECTED_KEYS}
            )

    ft_module.RLDSDataset = BaseNormalizedRLDSDataset
    print(
        f"[patch 1b] normalization = immutable OFT_m base stats, "
        f"suite-conditioned over {len(task_to_index)} tasks; source={stats_path}",
        flush=True,
    )
    return {"stats": stats, "task_to_index": task_to_index}
