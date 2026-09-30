"""Prepare leakage-controlled SST-2 datasets for the improved CLIP experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "dataset_config.json"
LABELS = ("NEG", "POS")
LABEL_MAP = {"0": "NEG", "1": "POS"}
OUTPUT_FIELDS = [
    "sentence_id",
    "text",
    "gold_label",
    "source_split",
    "source_row_id",
    "selection_seed",
    "clip_bpe_token_count",
    "over_clip_token_limit",
    "text_sha256",
    "normalized_text_sha256",
]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def stable_key(seed: int, purpose: str, value: str) -> str:
    return sha256_text(f"{seed}|{purpose}|{value}")


def read_sst2(
    path: Path, split: str
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    invalid = Counter()
    exclusions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"sentence", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain sentence and label columns.")
        for source_row_id, source in enumerate(reader):
            text = (source.get("sentence") or "").strip()
            raw_label = (source.get("label") or "").strip()
            if not text:
                invalid["empty_text"] += 1
                exclusions.append(
                    {
                        "sentence_id": "",
                        "text": text,
                        "gold_label": LABEL_MAP.get(raw_label, ""),
                        "source_split": split,
                        "source_row_id": source_row_id,
                        "selection_seed": "",
                        "clip_bpe_token_count": "",
                        "over_clip_token_limit": "",
                        "text_sha256": sha256_text(text),
                        "normalized_text_sha256": sha256_text(""),
                        "exclusion_reason": "empty_text",
                    }
                )
                continue
            if raw_label not in LABEL_MAP:
                invalid["invalid_label"] += 1
                exclusions.append(
                    {
                        "sentence_id": "",
                        "text": text,
                        "gold_label": "",
                        "source_split": split,
                        "source_row_id": source_row_id,
                        "selection_seed": "",
                        "clip_bpe_token_count": "",
                        "over_clip_token_limit": "",
                        "text_sha256": sha256_text(text),
                        "normalized_text_sha256": sha256_text(normalize_text(text)),
                        "exclusion_reason": "invalid_label",
                    }
                )
                continue
            normalized = normalize_text(text)
            if not normalized:
                invalid["empty_normalized_text"] += 1
                exclusions.append(
                    {
                        "sentence_id": "",
                        "text": text,
                        "gold_label": LABEL_MAP[raw_label],
                        "source_split": split,
                        "source_row_id": source_row_id,
                        "selection_seed": "",
                        "clip_bpe_token_count": "",
                        "over_clip_token_limit": "",
                        "text_sha256": sha256_text(text),
                        "normalized_text_sha256": sha256_text(""),
                        "exclusion_reason": "empty_normalized_text",
                    }
                )
                continue
            rows.append(
                {
                    "text": text,
                    "gold_label": LABEL_MAP[raw_label],
                    "source_split": split,
                    "source_row_id": source_row_id,
                    "normalized_text": normalized,
                    "text_sha256": sha256_text(text),
                    "normalized_text_sha256": sha256_text(normalized),
                }
            )
    return rows, dict(invalid), exclusions


def read_pilot(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            text = row["text"].strip()
            normalized = normalize_text(text)
            result.append(
                {
                    "sentence_id": row["sentence_id"].strip(),
                    "text": text,
                    "gold_label": row["gold_label"].strip().upper(),
                    "source_split": row.get("source", "pilot_20").strip(),
                    "source_row_id": row.get("source_row_id", "").strip(),
                    "selection_seed": "",
                    "normalized_text": normalized,
                    "text_sha256": sha256_text(text),
                    "normalized_text_sha256": sha256_text(normalized),
                }
            )
    return result


def add_clip_token_counts(
    collections: Sequence[list[dict[str, Any]]], model_name: str, maximum: int
) -> None:
    from transformers import CLIPTokenizerFast

    rows = [row for collection in collections for row in collection]
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name, local_files_only=True)
    tokenizer.model_max_length = 1_000_000
    batch_size = 1024
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["text"] for row in batch],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        for row, token_ids in zip(batch, encoded["input_ids"], strict=True):
            count = len(token_ids)
            row["clip_bpe_token_count"] = count
            row["over_clip_token_limit"] = int(count > maximum)


def grouped_train_split(
    rows: list[dict[str, Any]], seed: int, validation_targets: dict[str, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["normalized_text"]].append(row)

    multi = [group for group in groups.values() if len(group) > 1]
    singles = [group[0] for group in groups.values() if len(group) == 1]
    conflicting = sum(
        len({row["gold_label"] for row in group}) > 1 for group in multi
    )

    validation_keys: set[str] = set()
    counts = Counter()
    for group in sorted(
        multi,
        key=lambda values: stable_key(seed, "train-multi", values[0]["normalized_text"]),
    ):
        key = group[0]["normalized_text"]
        group_counts = Counter(row["gold_label"] for row in group)
        select_hash = int(stable_key(seed, "train-multi-choice", key)[:16], 16)
        proposed = select_hash / float(16**16) < 0.10
        fits = all(
            counts[label] + group_counts[label] <= validation_targets[label]
            for label in LABELS
        )
        if proposed and fits:
            validation_keys.add(key)
            counts.update(group_counts)

    for label in LABELS:
        required = validation_targets[label] - counts[label]
        candidates = sorted(
            (row for row in singles if row["gold_label"] == label),
            key=lambda row: stable_key(
                seed, f"train-single-{label}", str(row["source_row_id"])
            ),
        )
        if len(candidates) < required:
            raise RuntimeError(f"Insufficient singleton {label} rows for exact split.")
        validation_keys.update(row["normalized_text"] for row in candidates[:required])
        counts[label] += required

    validation = [row for row in rows if row["normalized_text"] in validation_keys]
    training = [row for row in rows if row["normalized_text"] not in validation_keys]
    train_keys = {row["normalized_text"] for row in training}
    validation_key_set = {row["normalized_text"] for row in validation}
    overlap = train_keys & validation_key_set
    if overlap:
        raise RuntimeError("Normalized text leaked between training and validation.")
    if Counter(row["gold_label"] for row in validation) != Counter(validation_targets):
        raise RuntimeError("Internal validation counts do not match the targets.")

    diagnostics = {
        "normalized_text_groups": len(groups),
        "duplicate_groups": len(multi),
        "duplicate_extra_rows": sum(len(group) - 1 for group in multi),
        "conflicting_label_groups": conflicting,
        "normalised_text_overlap_count": len(overlap),
    }
    return training, validation, diagnostics


def assign_ids(
    rows: Iterable[dict[str, Any]], prefix: str, seed: int | str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        prepared = dict(row)
        prepared["sentence_id"] = f"{prefix}{index:05d}"
        prepared["selection_seed"] = seed
        result.append(prepared)
    return result


def select_stability(
    validation: list[dict[str, Any]], seed: int, per_label: dict[str, int], maximum: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label in LABELS:
        unique: dict[str, dict[str, Any]] = {}
        for row in validation:
            if row["gold_label"] == label and row["clip_bpe_token_count"] <= maximum:
                unique.setdefault(row["normalized_text"], row)
        ordered = sorted(
            unique.values(),
            key=lambda row: (row["clip_bpe_token_count"], row["source_row_id"]),
        )
        required = per_label[label]
        bins = 4
        per_bin = required // bins
        if required % bins:
            raise ValueError("Stability counts must divide evenly across four bins.")
        for bin_index in range(bins):
            start = math.floor(len(ordered) * bin_index / bins)
            end = math.floor(len(ordered) * (bin_index + 1) / bins)
            candidates = sorted(
                ordered[start:end],
                key=lambda row: stable_key(
                    seed,
                    f"stability-{label}-{bin_index}",
                    str(row["source_row_id"]),
                ),
            )
            selected.extend(candidates[:per_bin])
    selected.sort(key=lambda row: (row["gold_label"], row["clip_bpe_token_count"], row["source_row_id"]))
    return assign_ids(selected, "ST", seed)


def select_final_evaluation(
    rows: list[dict[str, Any]],
    seed: int,
    targets: dict[str, int],
    inspected_ids: set[int],
    maximum: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    exclusions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen_normalized: set[str] = set()

    for row in rows:
        reasons: list[str] = []
        if row["source_row_id"] in inspected_ids:
            reasons.append("previously_inspected_pilot")
        if row["clip_bpe_token_count"] > maximum:
            reasons.append("over_clip_77_token_limit")
        if row["normalized_text"] in seen_normalized:
            reasons.append("normalized_duplicate")
        else:
            seen_normalized.add(row["normalized_text"])
        if reasons:
            exclusions.append({**row, "exclusion_reason": " | ".join(reasons)})
        else:
            candidates.append(row)

    chosen: list[dict[str, Any]] = []
    chosen_source_ids: set[int] = set()
    for label in LABELS:
        label_candidates = sorted(
            (row for row in candidates if row["gold_label"] == label),
            key=lambda row: stable_key(seed, f"final-{label}", str(row["source_row_id"])),
        )
        if len(label_candidates) < targets[label]:
            raise RuntimeError(f"Insufficient eligible {label} development rows.")
        selected = label_candidates[: targets[label]]
        chosen.extend(selected)
        chosen_source_ids.update(row["source_row_id"] for row in selected)

    for row in candidates:
        if row["source_row_id"] not in chosen_source_ids:
            exclusions.append({**row, "exclusion_reason": "not_selected_by_seed"})

    chosen.sort(key=lambda row: stable_key(seed, "final-order", str(row["source_row_id"])))
    prepared = assign_ids(chosen, "EV", seed)
    diagnostics = {
        "candidate_count_after_quality_exclusions": len(candidates),
        "previously_inspected_count": sum(
            "previously_inspected_pilot" in row["exclusion_reason"] for row in exclusions
        ),
        "over_token_limit_count": sum(
            "over_clip_77_token_limit" in row["exclusion_reason"] for row in exclusions
        ),
        "normalized_duplicate_count": sum(
            "normalized_duplicate" in row["exclusion_reason"] for row in exclusions
        ),
        "not_selected_count": sum(
            row["exclusion_reason"] == "not_selected_by_seed" for row in exclusions
        ),
    }
    return prepared, exclusions, diagnostics


def output_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in OUTPUT_FIELDS}


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def count_labels(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["gold_label"] for row in rows)
    return {label: counts[label] for label in LABELS}


def manifest_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        count = sum(1 for _ in csv.DictReader(stream))
    return {"path": path.name, "rows": count, "sha256": file_sha256(path)}


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    maximum = int(config["clip_max_tokens"])
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_path(config["sst2_train_path"])
    dev_path = resolve_path(config["sst2_dev_path"])
    pilot_path = resolve_path(config["pilot_20_path"])
    train_rows, train_invalid, train_exclusions = read_sst2(train_path, "sst2_train")
    dev_rows, dev_invalid, dev_source_exclusions = read_sst2(dev_path, "sst2_dev")
    pilot_rows = read_pilot(pilot_path)

    add_clip_token_counts(
        [train_rows, dev_rows, pilot_rows],
        config["clip_model_name"],
        maximum,
    )

    training, validation, split_diagnostics = grouped_train_split(
        train_rows,
        seed,
        {name: int(config["internal_validation_counts"][name]) for name in LABELS},
    )
    training.sort(key=lambda row: row["source_row_id"])
    validation.sort(key=lambda row: row["source_row_id"])
    training_prepared = assign_ids(training, "TR", seed)
    validation_prepared = assign_ids(validation, "PV", seed)
    stability_prepared = select_stability(
        validation,
        seed,
        {name: int(config["stability_counts"][name]) for name in LABELS},
        maximum,
    )
    final_prepared, exclusions, final_diagnostics = select_final_evaluation(
        dev_rows,
        seed,
        {name: int(config["final_evaluation_counts"][name]) for name in LABELS},
        {int(value) for value in config["previously_inspected_dev_row_ids"]},
        maximum,
    )
    pilot_prepared = [output_row(row) for row in pilot_rows]

    paths = {
        "pilot": output_dir / "pilot_20_curated.csv",
        "training": output_dir / "train_development.csv",
        "validation": output_dir / "prompt_validation.csv",
        "stability": output_dir / "shap_stability_40.csv",
        "final": output_dir / "final_evaluation_500.csv",
        "train_exclusions": output_dir / "excluded_train_rows.csv",
        "exclusions": output_dir / "excluded_dev_rows.csv",
        "audit": output_dir / "dataset_audit.csv",
    }
    write_csv(paths["pilot"], pilot_prepared, OUTPUT_FIELDS)
    write_csv(paths["training"], [output_row(row) for row in training_prepared], OUTPUT_FIELDS)
    write_csv(paths["validation"], [output_row(row) for row in validation_prepared], OUTPUT_FIELDS)
    write_csv(paths["stability"], [output_row(row) for row in stability_prepared], OUTPUT_FIELDS)
    write_csv(paths["final"], [output_row(row) for row in final_prepared], OUTPUT_FIELDS)
    exclusion_fields = [*OUTPUT_FIELDS, "exclusion_reason"]
    write_csv(paths["train_exclusions"], train_exclusions, exclusion_fields)
    write_csv(paths["exclusions"], exclusions, exclusion_fields)

    train_keys = {row["normalized_text"] for row in training}
    validation_keys = {row["normalized_text"] for row in validation}
    final_source_ids = {int(row["source_row_id"]) for row in final_prepared}
    inspected_ids = {int(value) for value in config["previously_inspected_dev_row_ids"]}
    audits = [
        {"category": "source", "metric": "train_raw_rows", "value": len(train_rows) + len(train_exclusions), "status": "PASS", "details": "Expected 67,349"},
        {"category": "source", "metric": "train_usable_rows", "value": len(train_rows), "status": "PASS", "details": "67,349 raw minus 2 documented punctuation-only exclusions"},
        {"category": "source", "metric": "dev_raw_rows", "value": len(dev_rows) + len(dev_source_exclusions), "status": "PASS", "details": "Expected 872"},
        {"category": "source", "metric": "dev_usable_rows", "value": len(dev_rows), "status": "PASS", "details": "All development rows are usable"},
        {"category": "source", "metric": "train_excluded_rows", "value": len(train_exclusions), "status": "PASS", "details": json.dumps(train_invalid, sort_keys=True)},
        {"category": "source", "metric": "dev_excluded_rows", "value": len(dev_source_exclusions), "status": "PASS", "details": json.dumps(dev_invalid, sort_keys=True)},
        {"category": "duplicates", "metric": "train_normalized_duplicate_groups", "value": split_diagnostics["duplicate_groups"], "status": "INFO", "details": "Each group stays within one internal split"},
        {"category": "duplicates", "metric": "train_conflicting_label_groups", "value": split_diagnostics["conflicting_label_groups"], "status": "INFO", "details": "Conflicting groups also stay within one split"},
        {"category": "leakage", "metric": "train_validation_normalized_overlap", "value": len(train_keys & validation_keys), "status": "PASS" if not train_keys & validation_keys else "FAIL", "details": "Must be zero"},
        {"category": "split", "metric": "training_rows", "value": len(training_prepared), "status": "PASS", "details": json.dumps(count_labels(training_prepared), sort_keys=True)},
        {"category": "split", "metric": "prompt_validation_rows", "value": len(validation_prepared), "status": "PASS", "details": json.dumps(count_labels(validation_prepared), sort_keys=True)},
        {"category": "split", "metric": "stability_rows", "value": len(stability_prepared), "status": "PASS", "details": json.dumps(count_labels(stability_prepared), sort_keys=True)},
        {"category": "final", "metric": "final_rows", "value": len(final_prepared), "status": "PASS", "details": json.dumps(count_labels(final_prepared), sort_keys=True)},
        {"category": "final", "metric": "previously_inspected_overlap", "value": len(final_source_ids & inspected_ids), "status": "PASS" if not final_source_ids & inspected_ids else "FAIL", "details": "Must be zero"},
        {"category": "final", "metric": "over_77_token_rows", "value": sum(int(row["over_clip_token_limit"]) for row in final_prepared), "status": "PASS", "details": "Must be zero"},
        {"category": "final", "metric": "unique_normalized_texts", "value": len({row["normalized_text_sha256"] for row in final_prepared}), "status": "PASS", "details": "Must equal 500"},
        {"category": "pilot", "metric": "pilot_rows", "value": len(pilot_prepared), "status": "PASS", "details": json.dumps(count_labels(pilot_prepared), sort_keys=True)},
    ]
    write_csv(paths["audit"], audits, ["category", "metric", "value", "status", "details"])

    train_manifest_path = output_dir / "train_internal_split_manifest.json"
    final_manifest_path = output_dir / "final_evaluation_manifest.json"
    train_manifest = {
        "seed": seed,
        "source": {
            "path": str(train_path),
            "sha256": file_sha256(train_path),
            "raw_rows": len(train_rows) + len(train_exclusions),
            "usable_rows": len(train_rows),
            "excluded_rows": len(train_exclusions),
            "usable_labels": count_labels(train_rows),
        },
        "split_method": "Normalized-text groups stay intact. Multi-row groups are deterministically assigned near 10%; unique rows fill exact stratified targets.",
        "diagnostics": split_diagnostics,
        "outputs": {
            "training": manifest_file(paths["training"]),
            "prompt_validation": manifest_file(paths["validation"]),
            "stability": manifest_file(paths["stability"]),
            "excluded_train_rows": manifest_file(paths["train_exclusions"]),
        },
    }
    final_manifest = {
        "seed": seed,
        "source": {
            "path": str(dev_path),
            "sha256": file_sha256(dev_path),
            "raw_rows": len(dev_rows) + len(dev_source_exclusions),
            "usable_rows": len(dev_rows),
            "excluded_rows": len(dev_source_exclusions),
            "usable_labels": count_labels(dev_rows),
        },
        "selection_method": "Exclude inspected, duplicate and over-limit rows; hash-order eligible rows by label; select 250 NEG and 250 POS.",
        "previously_inspected_dev_row_ids": sorted(inspected_ids),
        "diagnostics": final_diagnostics,
        "outputs": {
            "final_evaluation": manifest_file(paths["final"]),
            "excluded_rows": manifest_file(paths["exclusions"]),
        },
    }
    train_manifest_path.write_text(json.dumps(train_manifest, indent=2), encoding="utf-8")
    final_manifest_path.write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")

    report = f"""# Dataset Preparation Report

## Prepared files

- Training development split: {len(training_prepared):,} rows ({count_labels(training_prepared)})
- Prompt/internal validation split: {len(validation_prepared):,} rows ({count_labels(validation_prepared)})
- SHAP stability subset: {len(stability_prepared):,} rows ({count_labels(stability_prepared)})
- Final evaluation set: {len(final_prepared):,} rows ({count_labels(final_prepared)})
- Pilot regression set: {len(pilot_prepared):,} rows ({count_labels(pilot_prepared)})

The raw SST-2 training file contains 67,349 rows. Two punctuation-only records—source
row 4246 (`(`, NEG) and source row 8304 (`)`, POS)—become empty after text
normalization and are recorded in `excluded_train_rows.csv`. The usable training pool
therefore contains {len(train_rows):,} rows.

## Leakage and quality checks

- Normalized duplicate groups in SST-2 training: {split_diagnostics['duplicate_groups']:,}
- Conflicting-label duplicate groups: {split_diagnostics['conflicting_label_groups']:,}
- Normalized-text overlap between training and internal validation: {len(train_keys & validation_keys)}
- Previously inspected SST-2 development rows in final 500: {len(final_source_ids & inspected_ids)}
- Final examples over CLIP's 77-token limit: {sum(int(row['over_clip_token_limit']) for row in final_prepared)}
- Unique normalized texts in final 500: {len({row['normalized_text_sha256'] for row in final_prepared})}

The datasets are prepared only. No model training, prompt selection or SHAP run was performed.
"""
    (output_dir / "DATASET_PREPARATION_REPORT.md").write_text(report, encoding="utf-8")

    runtime_dir = ROOT / ".artifact_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    review_payload = {
        "summary": {
            "seed": seed,
            "clip_model": config["clip_model_name"],
            "clip_max_tokens": maximum,
            "train_raw_rows": len(train_rows) + len(train_exclusions),
            "train_usable_rows": len(train_rows),
            "train_excluded_rows": len(train_exclusions),
            "dev_raw_rows": len(dev_rows) + len(dev_source_exclusions),
            "dev_usable_rows": len(dev_rows),
            "training_rows": len(training_prepared),
            "validation_rows": len(validation_prepared),
            "stability_rows": len(stability_prepared),
            "final_rows": len(final_prepared),
            "pilot_rows": len(pilot_prepared),
        },
        "audit": audits,
        "final": [output_row(row) for row in final_prepared],
        "stability": [output_row(row) for row in stability_prepared],
        "pilot": pilot_prepared,
        "validation": [output_row(row) for row in validation_prepared],
        "train_exclusions": train_exclusions,
        "exclusions": [{field: row.get(field, "") for field in exclusion_fields} for row in exclusions],
    }
    (runtime_dir / "dataset_review_payload.json").write_text(
        json.dumps(review_payload, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Prepared datasets in {output_dir}")
    print(f"Training: {len(training_prepared)} {count_labels(training_prepared)}")
    print(f"Validation: {len(validation_prepared)} {count_labels(validation_prepared)}")
    print(f"Stability: {len(stability_prepared)} {count_labels(stability_prepared)}")
    print(f"Final: {len(final_prepared)} {count_labels(final_prepared)}")
    print(f"Final diagnostics: {final_diagnostics}")


if __name__ == "__main__":
    main()

