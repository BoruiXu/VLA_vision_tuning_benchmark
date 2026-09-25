"""Pinned original LIBERO tasks with the established harness rollout adapter.

The parent provides legacy torch compatibility, lazy rendering, absolute X-VLA
actions, and the same observation processing as our LIBERO-Plus evaluations.
Only task discovery is replaced by the original LIBERO task registry.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from benchmarks.libero_plus import LegacyEnvLIBEROPlusBenchmark
from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark


class OriginalLIBEROBenchmark(LegacyEnvLIBEROPlusBenchmark):
    def get_tasks(self):
        import libero.libero.benchmark as registry
        expected_root = Path(os.environ['ORIGINAL_LIBERO_REPO']).resolve()
        assert expected_root in Path(registry.__file__).resolve().parents, registry.__file__
        tasks = LIBEROBenchmark.get_tasks(self)
        assert len(tasks) == 10, 'Original LIBERO must have 10 tasks per suite'
        manifest_path = Path(os.environ['ORIGINAL_LIBERO_MANIFEST'])
        manifest = json.loads(manifest_path.read_text())
        spec = manifest['suites'][self.suite]
        for task, entry in zip(tasks, spec):
            assert task['task_id'] == entry['task_id']
            assert task['task_obj'].name == entry['name']
            states = self._task_suite.get_task_init_states(task['task_id'])
            assert len(states) >= 50
            hashes = [hashlib.sha256(np.asarray(s).tobytes()).hexdigest() for s in states[:50]]
            assert hashes == entry['initial_states_sha256'], 'Initial-state drift'
        return tasks

    def reset(self, task):
        assert 0 <= task['episode_idx'] < 50
        return super().reset(task)
