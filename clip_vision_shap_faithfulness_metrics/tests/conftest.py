"""Shared fixtures: the saved outputs of run_vision_metrics.py."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from source_link import OUTPUT_DIR  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="session")
def checks():
    return json.loads((OUTPUT_DIR / "checks.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def per_image():
    return read_csv(OUTPUT_DIR / "per_image_metrics.csv")


@pytest.fixture(scope="session")
def curve_points():
    return read_csv(OUTPUT_DIR / "curve_points.csv")


@pytest.fixture(scope="session")
def summary_table():
    return {row["Metric"]: row["CLIP Vision"] for row in read_csv(OUTPUT_DIR / "summary_table.csv")}
