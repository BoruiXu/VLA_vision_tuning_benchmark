"""Portable paths. Set VLA_RETAIN_ROOT to the release/work directory."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('VLA_RETAIN_ROOT', Path(__file__).resolve().parents[1])).resolve()

def resolve(relative=''):
    return str(ROOT / relative)
