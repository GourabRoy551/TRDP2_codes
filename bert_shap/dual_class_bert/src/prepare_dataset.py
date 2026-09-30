"""Prepare a human-reviewed 20-sentence dataset for dual-class BERT SHAP.

The existing qualitative S1-S10 set is copied without modification. Additional
examples are curated from the local labeled SST-2 development split so that the
final set contains ten NEG and ten POS examples. The curated file contains
clean punctuation and capitalization, while retaining every SST-2 source row.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"
VALID_LABELS = {"NEG", "POS"}
SST2_LABELS = {"0": "NEG", "1": "POS"}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else EXPERIMENT_DIR / path


def read_existing(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: list[dict[str, str]] = []
    for row in rows:
        label = row["gold_label"].strip().upper()
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label in {path}: {label!r}")
        result.append(
            {
                "sentence_id": row["sentence_id"].strip(),
                "text": row["text"].strip(),
                "gold_label": label,
                "source": "qualitative_s1_s10",
                "source_row_id": row["sentence_id"].strip(),
            }
        )
    return result


def read_curated(path: Path) -> list[dict[str, str]]:
    """Read the human-reviewed SST-2 additions without altering their text."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sentence_id", "text", "gold_label", "source", "source_row_id"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain {sorted(required)}.")

    result: list[dict[str, str]] = []
    for row in rows:
        label = row["gold_label"].strip().upper()
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid curated label in {path}: {label!r}")
        result.append(
            {
                "sentence_id": row["sentence_id"].strip(),
                "text": row["text"].strip(),
                "gold_label": label,
                "source": row["source"].strip(),
                "source_row_id": row["source_row_id"].strip(),
            }
        )
    return result


def read_sst2(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"sentence", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must contain sentence and label columns.")
        result: list[dict[str, str]] = []
        for source_row_id, row in enumerate(reader):
            text = (row.get("sentence") or "").strip()
            raw_label = (row.get("label") or "").strip()
            if text and raw_label in SST2_LABELS:
                result.append(
                    {
                        "text": text,
                        "gold_label": SST2_LABELS[raw_label],
                        "source": "sst2_dev",
                        "source_row_id": str(source_row_id),
                    }
                )
    return result


def validate_curated_sources(
    curated: list[dict[str, str]], candidates: list[dict[str, str]]
) -> None:
    """Check curated labels and row identifiers against the source SST-2 file."""
    source_lookup = {row["source_row_id"]: row for row in candidates}
    seen_ids: set[str] = set()
    for row in curated:
        source_id = row["source_row_id"]
        if source_id in seen_ids:
            raise ValueError(f"Duplicate curated SST-2 source row: {source_id}")
        seen_ids.add(source_id)
        if source_id not in source_lookup:
            raise ValueError(f"Unknown curated SST-2 source row: {source_id}")
        source_label = source_lookup[source_id]["gold_label"]
        if row["gold_label"] != source_label:
            raise ValueError(
                f"Curated label mismatch for SST-2 row {source_id}: "
                f"{row['gold_label']} != {source_label}"
            )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sentence_id", "text", "gold_label", "source", "source_row_id"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(config_path: Path, output_override: Path | None = None) -> list[dict[str, str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    existing_path = project_path(config["existing_sentences_path"])
    sst2_path = project_path(config["sst2_dev_path"])
    curated_path = project_path(config["curated_additions_path"])
    output_path = output_override or project_path(config["input_path"])
    existing = read_existing(existing_path)
    curated = read_curated(curated_path)
    validate_curated_sources(curated, read_sst2(sst2_path))
    rows = [*existing, *curated]
    if len(rows) != int(config["total_sentences"]):
        raise RuntimeError(
            f"Expected {config['total_sentences']} rows, found {len(rows)}."
        )
    counts = Counter(row["gold_label"] for row in rows)
    if counts != Counter({"NEG": 10, "POS": 10}):
        raise RuntimeError(f"Unexpected final label balance: {dict(counts)}")
    write_csv(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path.resolve()}")
    print(f"Label counts: {dict(sorted(counts.items()))}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    prepare(args.config.resolve(), args.output.resolve() if args.output else None)


if __name__ == "__main__":
    main()
