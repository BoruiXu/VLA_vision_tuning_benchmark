"""Restrict task discovery to a released outcome-blind manifest."""
import json
import os
from pathlib import Path
from benchmarks.libero_plus import LegacyEnvLIBEROPlusBenchmark

class SelectedLIBEROPlusBenchmark(LegacyEnvLIBEROPlusBenchmark):
    def get_tasks(self):
        tasks=super().get_tasks()
        source=json.loads(Path(os.environ['VLA_TASK_SELECTION']).read_text())
        expected={int(t['i']):t for t in source[self.suite]}
        selected=[t for t in tasks if int(t['task_id']) in expected]
        assert len(selected)==len(expected),'Task manifest differs from installed benchmark'
        for task in selected:
            # Check names as well as numerical IDs to prevent silent registry drift.
            obj=task.get('task_obj')
            name=getattr(obj,'name',None)
            if name is not None:assert name==expected[int(task['task_id'])]['name']
        return selected
