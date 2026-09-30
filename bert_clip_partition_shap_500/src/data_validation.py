"""Copy, validate and fingerprint the frozen 500-sentence evaluation set.

The dataset is copied byte-for-byte from ``clip_text_shap_improved``. Nothing is
resampled, edited, reordered or relabelled; this module only reads and checks it.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from collections import Counter
from typing import Any

from io_utils import (
    load_json,
    project_path,
    read_csv,
    save_json,
    sha256_file,
    sha256_text,
    utc_now,
)


ALLOWED_LABELS = ("NEG", "POS")


def normalize_text(text: str) -> str:
    """Duplicate-detection normalization (identical to the source dataset preparation)."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def order_fingerprint(sentence_ids: list[str]) -> str:
    """Hash of the ordered ID sequence; both models must reproduce it exactly."""
    return sha256_text("\n".join(sentence_ids))


def copy_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Copy the source file only if the destination is missing; never overwrite a different file."""
    source = project_path(config["dataset"]["source_path"])
    destination = project_path(config["dataset"]["evaluation_path"])
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    source_hash = sha256_file(source)
    if destination.exists():
        copied_now = False
        if sha256_file(destination) != source_hash:
            raise RuntimeError(
                "Existing evaluation_500.csv differs from the source dataset; refusing to overwrite."
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied_now = True
    return {"source": source, "destination": destination, "source_sha256": source_hash, "copied_now": copied_now}


def validate_rows(rows: list[dict[str, str]], config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    id_column, text_column, label_column = (
        dataset["id_column"],
        dataset["text_column"],
        dataset["label_column"],
    )
    ids = [row.get(id_column, "") for row in rows]
    texts = [row.get(text_column, "") for row in rows]
    labels = [row.get(label_column, "") for row in rows]
    normalized = [normalize_text(text) for text in texts]
    label_counts = Counter(labels)
    duplicate_ids = sorted(value for value, count in Counter(ids).items() if count > 1)
    duplicate_texts = sorted(value for value, count in Counter(normalized).items() if count > 1)
    missing_text = [ids[index] for index, text in enumerate(texts) if not str(text).strip()]
    missing_ids = [index for index, value in enumerate(ids) if not str(value).strip()]
    invalid_labels = sorted(set(labels) - set(ALLOWED_LABELS))
    # Cross-check the per-row fingerprints stored by the original preparation script.
    text_hash_mismatches = [
        ids[index]
        for index, row in enumerate(rows)
        if "text_sha256" in row and row["text_sha256"] != sha256_text(texts[index])
    ]
    normalized_hash_mismatches = [
        ids[index]
        for index, row in enumerate(rows)
        if "normalized_text_sha256" in row
        and row["normalized_text_sha256"] != sha256_text(normalized[index])
    ]
    expected_counts = dict(dataset["expected_class_counts"])
    checks = {
        "exactly_500_rows": len(rows) == int(dataset["expected_rows"]),
        "exactly_250_neg": label_counts.get("NEG", 0) == expected_counts["NEG"],
        "exactly_250_pos": label_counts.get("POS", 0) == expected_counts["POS"],
        "unique_sentence_ids": not duplicate_ids and not missing_ids,
        "no_duplicate_normalized_text": not duplicate_texts,
        "no_missing_text": not missing_text,
        "only_neg_pos_labels": not invalid_labels,
        "row_text_sha256_consistent": not text_hash_mismatches,
        "row_normalized_text_sha256_consistent": not normalized_hash_mismatches,
    }
    return {
        "rows": len(rows),
        "class_counts": dict(sorted(label_counts.items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_normalized_texts": duplicate_texts,
        "missing_text_ids": missing_text,
        "invalid_labels": invalid_labels,
        "text_hash_mismatches": text_hash_mismatches,
        "normalized_hash_mismatches": normalized_hash_mismatches,
        "first_sentence_id": ids[0] if ids else "",
        "last_sentence_id": ids[-1] if ids else "",
        "sentence_order_sha256": order_fingerprint(ids),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Copy (if needed), validate, compare hashes and write ``data/dataset_manifest.json``."""
    copy_info = copy_dataset(config)
    destination = copy_info["destination"]
    copied_hash = sha256_file(destination)
    source_manifest_path = project_path(config["dataset"]["source_manifest_path"])
    source_manifest = load_json(source_manifest_path)
    manifest_hash = source_manifest["outputs"]["final_evaluation"]["sha256"]
    rows = read_csv(destination)
    validation = validate_rows(rows, config)
    hash_checks = {
        "copied_equals_source_file": copied_hash == copy_info["source_sha256"],
        "copied_equals_source_manifest": copied_hash == manifest_hash,
        "source_manifest_row_count_matches": int(source_manifest["outputs"]["final_evaluation"]["rows"])
        == validation["rows"],
    }
    manifest = {
        "dataset": "SST-2 development subset, balanced 250 NEG / 250 POS",
        "evaluation_path": str(destination.relative_to(project_path("."))).replace("\\", "/"),
        "source_path": str(copy_info["source"]),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_selection_method": source_manifest.get("selection_method", ""),
        "source_manifest_seed": source_manifest.get("seed"),
        "original_sst2_source": source_manifest.get("source", {}),
        "sha256_copied": copied_hash,
        "sha256_source_file": copy_info["source_sha256"],
        "sha256_source_manifest": manifest_hash,
        "copied_during_this_validation": copy_info["copied_now"],
        "hash_checks": hash_checks,
        "validation": validation,
        "normalization_rule": "NFKC -> casefold -> non-word runs to single space -> strip (same rule as source preparation)",
        "modifications": "none (byte-identical copy; no resampling, editing, reordering or relabelling)",
        "all_checks_passed": validation["all_checks_passed"] and all(hash_checks.values()),
        "validated_at_utc": utc_now(),
    }
    save_json(project_path(config["dataset"]["manifest_path"]), manifest)
    if not manifest["all_checks_passed"]:
        raise RuntimeError(f"Dataset validation failed: {validation['checks']} {hash_checks}")
    return manifest


def load_evaluation_rows(config: dict[str, Any], verify_hash: bool = True) -> list[dict[str, str]]:
    """Load the validated rows in their fixed order (used by both models)."""
    path = project_path(config["dataset"]["evaluation_path"])
    if verify_hash:
        manifest = load_json(project_path(config["dataset"]["manifest_path"]))
        if sha256_file(path) != manifest["sha256_copied"]:
            raise RuntimeError("evaluation_500.csv no longer matches data/dataset_manifest.json.")
    return read_csv(path)


def dataset_hash(config: dict[str, Any]) -> str:
    return sha256_file(project_path(config["dataset"]["evaluation_path"]))
