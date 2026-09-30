"""Read-only link to the source experiment ``bert_clip_partition_shap_500``.

The source ``src/`` directory is put on ``sys.path`` so that the models, scorers,
whole-word masker, deletion count and record validation are the exact code that
produced the SHAP values. Bytecode writing is disabled first so that importing
those modules never adds or updates files inside the source folder.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
OUTPUT_DIR = ROOT / "outputs"


def load_own_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


SOURCE_ROOT = (ROOT / load_own_config()["source_experiment"]).resolve()
SOURCE_SRC = SOURCE_ROOT / "src"
if not (SOURCE_SRC / "faithfulness.py").exists():
    raise FileNotFoundError(f"Source experiment not found at {SOURCE_ROOT}")
if str(SOURCE_SRC) not in sys.path:
    sys.path.insert(0, str(SOURCE_SRC))


def source_file_hashes() -> dict[str, str]:
    """SHA-256 of every source-experiment file (bytecode caches excluded)."""
    hashes = {}
    for path in sorted(SOURCE_ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts:
            hashes[path.relative_to(SOURCE_ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes
