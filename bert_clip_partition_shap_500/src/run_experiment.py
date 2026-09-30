"""BERT versus CLIP whole-word Partition SHAP on the frozen 500-sentence SST-2 set.

Stages (each gated by the previous one):
  prepare  copy + validate the dataset, write data/dataset_manifest.json
  score    load frozen models, label/prompt/parity checks, score all 500, pick case studies
  smoke    whole-word masking audit on all 500 + 5-sentence SHAP smoke test for both models
  shap     full resumable Partition SHAP run for one model (--model bert|clip)
  metrics  all CSV outputs, five primary metrics and 1,000-iteration bootstrap CIs
  plots    five aggregate and five case-study figures (PNG + PDF)
  wordbars NEG | POS | POS-NEG word bar charts for four report sentences (both CLIP experiments)
  report   FINAL_REPORT.md, RESULTS_SUMMARY.md and LaTeX PDF
  all      every stage in order

No model is trained or fine-tuned in any stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# In the rfem environment, importing pandas/pyarrow (pulled in by shap) after CUDA has
# been initialised makes later directory listings fail with WinError 6714. Importing
# shap first, before any CUDA call, avoids this.
import shap  # noqa: F401,E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_validation import dataset_hash, load_evaluation_rows, order_fingerprint, prepare_dataset  # noqa: E402
from io_utils import (  # noqa: E402
    CHECKPOINT_DIR,
    METRICS_DIR,
    MODEL_KEYS,
    VALUES_DIR,
    choose_device,
    config_sha256,
    configure_determinism,
    environment_versions,
    load_config,
    load_json,
    read_csv,
    save_json,
    sha256_text,
    set_offline_environment,
    utc_now,
    write_csv,
)
from model_outputs import OUTPUT_KEYS, OUTPUT_NAMES, parity_check, predict_label, score_in_batches, verify_frozen  # noqa: E402
from whole_word_masker import MASKER_VERSION, map_tokens_to_units, split_word_units  # noqa: E402


MODEL_SCORES = VALUES_DIR / "model_scores.csv"
MODEL_CHECKS = CHECKPOINT_DIR / "model_checks.json"
SMOKE_RESULT = CHECKPOINT_DIR / "smoke_test.json"
SELECTION = VALUES_DIR / "representative_selection.csv"
METRIC_CHECKS = METRICS_DIR / "acceptance_checks.json"
WARMUP_SENTENCE = "this warm-up sentence is not part of the evaluation set ."
POSTHOC_MARGIN_THRESHOLD = 1e-3


def records_path(model_key: str) -> Path:
    return CHECKPOINT_DIR / f"{model_key}_shap_records.jsonl"


def progress_path(model_key: str) -> Path:
    return CHECKPOINT_DIR / f"{model_key}_progress.json"


# --------------------------------------------------------------------- set-up

def setup(config: dict[str, Any]) -> Any:
    set_offline_environment(bool(config["local_files_only"]))
    configure_determinism(int(config["random_seed"]))
    return choose_device(str(config["device"]))


def build_scorer(model_key: str, config: dict[str, Any], device: Any) -> tuple[Any, dict[str, Any]]:
    """Load one frozen model and return its three-output scorer plus verification info."""
    if model_key == "bert":
        from bert_backend import BertThreeOutputScorer, load_bert, resolve_label_mapping, verify_label_mapping

        model, tokenizer = load_bert(config, device)
        mapping = resolve_label_mapping(model, config)
        scorer = BertThreeOutputScorer(model, tokenizer, device, config["models"]["bert"]["max_length"], mapping)
        verification = verify_label_mapping(scorer, config)
        info = {"label_mapping": mapping, "label_mapping_verification": verification, "semantic_check_passed": verification["passed"]}
    elif model_key == "clip":
        from clip_backend import create_clip_scorer, verify_prompt_source

        scorer, info = create_clip_scorer(config, device)
        model = scorer.model
        info["prompt_source_verification"] = verify_prompt_source(config)
        info["semantic_check_passed"] = info["prompt_source_verification"]["passed"]
    else:
        raise ValueError(f"Unknown model {model_key!r}")
    info["frozen"] = verify_frozen(model)
    info["model_name"] = config["models"][model_key]["name_or_path"]
    return scorer, info


def settings_signature(model_key: str, config: dict[str, Any]) -> str:
    """Records are reusable only if every setting that changes their content is unchanged."""
    settings = {
        "model": config["models"][model_key]["name_or_path"],
        "max_length": config["models"][model_key]["max_length"],
        "prompts": config["models"][model_key].get("prompts"),
        "max_evals": config["shap"]["max_evals"],
        "fractions": config["faithfulness"]["fractions"],
        "random_repeats": config["faithfulness"]["random_repeats"],
        "epsilon": config["faithfulness"]["normalization_epsilon"],
        "seed": config["random_seed"],
        "masker": MASKER_VERSION,
    }
    return sha256_text(json.dumps(settings, sort_keys=True))


def load_model_scores() -> dict[str, dict[str, np.ndarray]]:
    scores: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for row in read_csv(MODEL_SCORES):
        scores[row["model"]][row["sentence_id"]] = np.asarray(
            [float(row["neg_score"]), float(row["pos_score"]), float(row["margin"])], dtype=np.float64
        )
    return scores


def require_gate(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"{description} has not been run ({path.name} missing).")
    result = load_json(path)
    if not result.get("gate_passed"):
        raise RuntimeError(f"{description} gate failed; later stages are blocked.")
    return result


# ------------------------------------------------------------------ stage: score

def stage_score(config: dict[str, Any]) -> None:
    from representative_selection import select_representatives

    device = setup(config)
    rows = load_evaluation_rows(config)
    texts = [row["text"] for row in rows]
    ids = [row["sentence_id"] for row in rows]
    parity_count = int(config["parity_check_sentences"])
    checks: dict[str, Any] = {"device": str(device), "environment": environment_versions(), "models": {}}
    score_rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, str]] = {}
    multi_piece: dict[str, dict[str, int]] = {}
    for model_key in MODEL_KEYS:
        print(f"[score] loading {model_key} on {device}")
        scorer, info = build_scorer(model_key, config, device)
        tolerance = float(config["tolerances"]["batch_parity"][model_key])
        batch_size = int(config["batch_sizes"]["score"][model_key])
        parity = parity_check(scorer, texts[:parity_count], batch_size=8, tolerance=tolerance)
        started = time.perf_counter()
        scores = score_in_batches(scorer, texts, batch_size)
        scoring_seconds = time.perf_counter() - started
        max_tokens = int(config["models"][model_key]["max_length"])
        predictions[model_key], multi_piece[model_key] = {}, {}
        truncated = 0
        for index, row in enumerate(rows):
            units = split_word_units(row["text"])
            mapping = map_tokens_to_units(scorer.tokenizer, row["text"], units)
            token_count = scorer.token_count(row["text"])
            truncated += int(mapping["token_count"] + 2 > max_tokens)
            prediction = predict_label(scores[index, 2])
            predictions[model_key][row["sentence_id"]] = prediction
            multi_piece[model_key][row["sentence_id"]] = mapping["multi_piece_units"]
            score_rows.append(
                {
                    "model": model_key,
                    "row_index": index,
                    "sentence_id": row["sentence_id"],
                    "gold_label": row["gold_label"],
                    "neg_score": float(scores[index, 0]),
                    "pos_score": float(scores[index, 1]),
                    "margin": float(scores[index, 2]),
                    "prediction": prediction,
                    "correct": int(prediction == row["gold_label"]),
                    "word_units": len(units),
                    "content_words": sum(unit.is_content for unit in units),
                    "internal_token_count": token_count,
                    "internal_tokens_without_special": mapping["token_count"],
                    "multi_piece_words": mapping["multi_piece_units"],
                    "unassigned_tokens": mapping["unassigned_tokens"],
                    "tokens_spanning_multiple_units": mapping["tokens_spanning_multiple_units"],
                }
            )
        model_rows = [row for row in score_rows if row["model"] == model_key]
        accuracy = float(np.mean([row["correct"] for row in model_rows]))
        checks["models"][model_key] = {
            **info,
            "parity": parity,
            "scored_sentences": len(model_rows),
            "sentence_order_sha256": order_fingerprint([row["sentence_id"] for row in model_rows]),
            "scoring_seconds": scoring_seconds,
            "accuracy_on_500": accuracy,
            "truncated_sentences": truncated,
            "unassigned_tokens_total": sum(row["unassigned_tokens"] for row in model_rows),
            "tokens_spanning_units_total": sum(row["tokens_spanning_multiple_units"] for row in model_rows),
        }
        print(f"[score] {model_key}: parity={parity['passed']} (max diff {parity['max_batched_vs_individual_abs_difference']:.2e}), "
              f"semantic={info['semantic_check_passed']}, frozen={info['frozen']['frozen']}, accuracy={accuracy:.4f}")
        del scorer
        _release_memory()

    write_csv(MODEL_SCORES, score_rows)
    dataset_order = order_fingerprint(ids)
    tests = {
        f"{model_key}_{name}": value
        for model_key in MODEL_KEYS
        for name, value in {
            "parity_passed": checks["models"][model_key]["parity"]["passed"],
            "semantic_check_passed": checks["models"][model_key]["semantic_check_passed"],
            "frozen": checks["models"][model_key]["frozen"]["frozen"],
            "500_scored": checks["models"][model_key]["scored_sentences"] == 500,
            "same_order_as_dataset": checks["models"][model_key]["sentence_order_sha256"] == dataset_order,
            "no_truncation": checks["models"][model_key]["truncated_sentences"] == 0,
            "every_token_maps_to_one_unit": checks["models"][model_key]["unassigned_tokens_total"] == 0
            and checks["models"][model_key]["tokens_spanning_units_total"] == 0,
        }.items()
    }
    checks.update({"dataset_order_sha256": dataset_order, "tests": tests, "gate_passed": all(tests.values()), "completed_at_utc": utc_now()})
    save_json(MODEL_CHECKS, checks)
    if not checks["gate_passed"]:
        raise RuntimeError(f"Model checks failed: {[name for name, ok in tests.items() if not ok]}")

    # Case-study selection happens here: after scoring, before any SHAP value exists.
    candidates = []
    content_counts = {row["sentence_id"]: row["content_words"] for row in score_rows if row["model"] == "bert"}
    for row in rows:
        sentence_id = row["sentence_id"]
        candidates.append(
            {
                "sentence_id": sentence_id,
                "text": row["text"],
                "gold_label": row["gold_label"],
                "bert_prediction": predictions["bert"][sentence_id],
                "clip_prediction": predictions["clip"][sentence_id],
                "content_words": content_counts[sentence_id],
                "bert_multi_piece_words": multi_piece["bert"][sentence_id],
                "clip_multi_piece_words": multi_piece["clip"][sentence_id],
            }
        )
    selected = select_representatives(candidates, config["representative_selection"])
    if SELECTION.exists() and [row["sentence_id"] for row in read_csv(SELECTION)] == [row["sentence_id"] for row in selected]:
        print("[score] representative selection unchanged; original file (and timestamp) kept")
    else:
        if any(records_path(model_key).exists() for model_key in MODEL_KEYS):
            raise RuntimeError("SHAP records already exist; refusing to change the pre-SHAP case-study selection.")
        write_csv(SELECTION, selected)
    print("[score] gate passed; case studies:", ", ".join(f"{row['role']}={row['sentence_id']}" for row in selected))


def _release_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ stage: smoke

def build_record(
    model_key: str,
    row_index: int,
    row: dict[str, str],
    scores: np.ndarray,
    explainer: Any,
    scorer: Any,
    masker: Any,
    config: dict[str, Any],
    signature: str,
) -> dict[str, Any]:
    from faithfulness import compute_faithfulness
    from partition_shap import explain_sentence

    text = row["text"]
    units = split_word_units(text)
    result = explain_sentence(
        explainer, scorer, text, units, scores, config["shap"]["max_evals"], config["batch_sizes"]["shap"][model_key]
    )
    faith_rows, faith_summary = compute_faithfulness(
        scorer,
        masker,
        text,
        units,
        result["word_values"][:, 2],
        float(scores[2]),
        config["faithfulness"]["fractions"],
        int(config["faithfulness"]["random_repeats"]),
        int(config["random_seed"]),
        row_index,
        float(config["faithfulness"]["normalization_epsilon"]),
    )
    mapping = map_tokens_to_units(scorer.tokenizer, text, units)
    prediction = predict_label(scores[2])
    return {
        "model": model_key,
        "row_index": row_index,
        "sentence_id": row["sentence_id"],
        "text_sha256": sha256_text(text),
        "settings_signature": signature,
        "gold_label": row["gold_label"],
        "scores": scores.tolist(),
        "prediction": prediction,
        "correct": int(prediction == row["gold_label"]),
        "words": [unit.text for unit in units],
        "is_content": [int(unit.is_content) for unit in units],
        "unit_kinds": [unit.kind for unit in units],
        "tokens_per_unit": mapping["tokens_per_unit"],
        "internal_token_count": scorer.token_count(text),
        "word_values": result["word_values"].tolist(),
        "base_values": result["base_values"].tolist(),
        "reconstructed": result["reconstructed"].tolist(),
        "additivity_residual": result["additivity_residual"].tolist(),
        "max_additivity_residual": result["max_additivity_residual"],
        "max_word_margin_linearity_error": result["max_word_margin_linearity_error"],
        "base_margin_linearity_error": result["base_margin_linearity_error"],
        "shap_runtime_seconds": result["shap_runtime_seconds"],
        "shap_model_evaluations": result["shap_model_evaluations"],
        "faithfulness_rows": faith_rows,
        "faithfulness_summary": faith_summary,
        "device": str(scorer.device),
        "completed_at_utc": utc_now(),
    }


def record_checks(record: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    tolerances = config["tolerances"]
    values = np.asarray(record["word_values"], dtype=np.float64)
    return {
        "three_outputs": values.ndim == 2 and values.shape[1] == 3 and len(record["base_values"]) == 3,
        "one_row_per_word": values.shape[0] == len(record["words"]),
        "finite": bool(np.all(np.isfinite(values))) and bool(np.all(np.isfinite(record["base_values"]))),
        "additivity": record["max_additivity_residual"] <= float(tolerances["additivity"][record["model"]]),
        "margin_linearity": record["max_word_margin_linearity_error"] <= float(tolerances["margin_linearity"]),
        "base_margin_linearity": record["base_margin_linearity_error"] <= float(tolerances["base_margin_linearity"]),
        "within_eval_budget": record["shap_model_evaluations"] <= int(config["shap"]["max_evals"]) + int(
            config["batch_sizes"]["shap"][record["model"]]
        ),
    }


def stage_smoke(config: dict[str, Any]) -> None:
    from partition_shap import create_explainer
    from whole_word_masker import audit_sentence_masking, create_word_masker

    require_gate(MODEL_CHECKS, "Model scoring/parity (Part 2)")
    device = setup(config)
    rows = load_evaluation_rows(config)
    all_scores = load_model_scores()
    masker = create_word_masker()
    scorers = {}
    for model_key in MODEL_KEYS:
        scorers[model_key], _ = build_scorer(model_key, config, device)

    print("[smoke] auditing whole-word masking on all 500 sentences for both tokenizers")
    tokenizers = {model_key: scorers[model_key].tokenizer for model_key in MODEL_KEYS}
    rng = np.random.default_rng(int(config["random_seed"]))
    audits = [
        {"sentence_id": row["sentence_id"], **audit_sentence_masking(masker, tokenizers, row["text"], rng, int(config["masking_audit_random_coalitions"]))}
        for row in rows
    ]
    audit_summary = {
        "sentences": len(audits),
        "coalitions_checked": sum(item["coalitions_checked"] for item in audits),
        "bert_partial_word_violations": sum(item["bert_partial_word_violations"] for item in audits),
        "clip_partial_word_violations": sum(item["clip_partial_word_violations"] for item in audits),
        "full_coalition_reproduces_text": all(item["full_coalition_reproduces_text"] for item in audits),
        "empty_coalition_is_empty_string": all(item["empty_coalition_is_empty_string"] for item in audits),
    }
    audit_summary["passed"] = (
        audit_summary["bert_partial_word_violations"] == 0
        and audit_summary["clip_partial_word_violations"] == 0
        and audit_summary["full_coalition_reproduces_text"]
        and audit_summary["empty_coalition_is_empty_string"]
    )
    write_csv(CHECKPOINT_DIR / "masking_audit.csv", audits)
    print(f"[smoke] masking audit: {audit_summary}")

    smoke: dict[str, Any] = {"device": str(device), "masking_audit": audit_summary, "models": {}}
    count = int(config["smoke_test_sentences"])
    for model_key in MODEL_KEYS:
        scorer = scorers[model_key]
        explainer = create_explainer(scorer, masker)
        signature = settings_signature(model_key, config)
        details = []
        for index, row in enumerate(rows[:count]):
            record = build_record(model_key, index, row, all_scores[model_key][row["sentence_id"]], explainer, scorer, masker, config, signature)
            details.append(
                {
                    "sentence_id": row["sentence_id"],
                    "words": len(record["words"]),
                    "max_additivity_residual": record["max_additivity_residual"],
                    "max_word_margin_linearity_error": record["max_word_margin_linearity_error"],
                    "base_margin_linearity_error": record["base_margin_linearity_error"],
                    "shap_runtime_seconds": record["shap_runtime_seconds"],
                    "shap_model_evaluations": record["shap_model_evaluations"],
                    "aopc_shap_abs": record["faithfulness_summary"]["aopc_shap_abs"],
                    "aopc_random": record["faithfulness_summary"]["aopc_random"],
                    "checks": record_checks(record, config),
                }
            )
        # Determinism: explaining the same sentence again must reproduce the values.
        first = build_record(model_key, 0, rows[0], all_scores[model_key][rows[0]["sentence_id"]], explainer, scorer, masker, config, signature)
        repeat = build_record(model_key, 0, rows[0], all_scores[model_key][rows[0]["sentence_id"]], explainer, scorer, masker, config, signature)
        repeat_difference = float(np.max(np.abs(np.asarray(first["word_values"]) - np.asarray(repeat["word_values"]))))
        passed = all(all(item["checks"].values()) for item in details) and repeat_difference <= float(
            config["tolerances"]["additivity"][model_key]
        )
        smoke["models"][model_key] = {
            "sentences": details,
            "repeat_max_abs_difference": repeat_difference,
            "mean_seconds_per_sentence": float(np.mean([item["shap_runtime_seconds"] for item in details])),
            "passed": passed,
        }
        print(f"[smoke] {model_key}: passed={passed}, repeat diff={repeat_difference:.2e}, "
              f"max residual={max(item['max_additivity_residual'] for item in details):.2e}")
    smoke["gate_passed"] = audit_summary["passed"] and all(item["passed"] for item in smoke["models"].values())
    smoke["completed_at_utc"] = utc_now()
    save_json(SMOKE_RESULT, smoke)
    if not smoke["gate_passed"]:
        raise RuntimeError("Smoke test failed; the full 500-sentence runs are blocked.")
    print("[smoke] gate passed")


# ------------------------------------------------------------------- stage: shap

def validate_record(
    record: dict[str, Any], model_key: str, row_index: int, row: dict[str, str], scores: np.ndarray, signature: str, config: dict[str, Any]
) -> str:
    """Return '' for a complete, consistent record, otherwise the reason it is rejected."""
    required = {
        "model", "row_index", "sentence_id", "text_sha256", "settings_signature", "scores", "words", "is_content",
        "word_values", "base_values", "additivity_residual", "faithfulness_rows", "faithfulness_summary", "tokens_per_unit",
    }
    missing = required - set(record)
    if missing:
        return f"missing fields {sorted(missing)}"
    if record["model"] != model_key or record["sentence_id"] != row["sentence_id"] or int(record["row_index"]) != row_index:
        return "identity mismatch"
    if record["text_sha256"] != sha256_text(row["text"]):
        return "text hash mismatch"
    if record["settings_signature"] != signature:
        return "settings changed"
    if record["words"] != [unit.text for unit in split_word_units(row["text"])]:
        return "word units changed"
    if not np.array_equal(np.asarray(record["scores"], dtype=np.float64), scores):
        return "original scores differ from model_scores.csv"
    values = np.asarray(record["word_values"], dtype=np.float64)
    if values.shape != (len(record["words"]), 3) or not np.all(np.isfinite(values)):
        return "SHAP value matrix incomplete"
    expected_faith = len(config["faithfulness"]["fractions"]) * (2 + int(config["faithfulness"]["random_repeats"]))
    if len(record["faithfulness_rows"]) != expected_faith:
        return "faithfulness rows incomplete"
    return ""


def load_valid_records(model_key: str, rows: list[dict[str, str]], scores: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    path = records_path(model_key)
    valid: dict[str, dict[str, Any]] = {}
    rejected: list[str] = []
    if not path.exists():
        return valid, rejected
    signature = settings_signature(model_key, config)
    index_of = {row["sentence_id"]: index for index, row in enumerate(rows)}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rejected.append(f"line {line_number}: truncated JSON (interrupted write)")
                continue
            sentence_id = record.get("sentence_id")
            if sentence_id not in index_of or sentence_id in valid:
                rejected.append(f"line {line_number}: unknown or duplicate sentence {sentence_id}")
                continue
            index = index_of[sentence_id]
            reason = validate_record(record, model_key, index, rows[index], scores[sentence_id], signature, config)
            if reason:
                rejected.append(f"line {line_number} ({sentence_id}): {reason}")
            else:
                valid[sentence_id] = record
    return valid, rejected


def _append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_progress(model_key: str, rows: list[dict[str, str]], records: dict[str, dict[str, Any]], extra: dict[str, Any]) -> None:
    done = [row["sentence_id"] for row in rows if row["sentence_id"] in records]
    save_json(
        progress_path(model_key),
        {
            "model": model_key,
            "completed": len(done),
            "total": len(rows),
            "complete": len(done) == len(rows),
            "last_completed_sentence_id": done[-1] if done else "",
            "max_additivity_residual": max((records[i]["max_additivity_residual"] for i in done), default=0.0),
            "max_word_margin_linearity_error": max((records[i]["max_word_margin_linearity_error"] for i in done), default=0.0),
            "sum_shap_runtime_seconds": sum(records[i]["shap_runtime_seconds"] for i in done),
            "updated_at_utc": utc_now(),
            **extra,
        },
    )
    write_csv(
        CHECKPOINT_DIR / f"{model_key}_partial_sentence_results.csv",
        [
            {
                "sentence_id": i,
                "prediction": records[i]["prediction"],
                "correct": records[i]["correct"],
                "margin": records[i]["scores"][2],
                "max_additivity_residual": records[i]["max_additivity_residual"],
                "max_word_margin_linearity_error": records[i]["max_word_margin_linearity_error"],
                "shap_runtime_seconds": records[i]["shap_runtime_seconds"],
                "shap_model_evaluations": records[i]["shap_model_evaluations"],
                "aopc_shap_abs": records[i]["faithfulness_summary"]["aopc_shap_abs"],
                "aopc_random": records[i]["faithfulness_summary"]["aopc_random"],
            }
            for i in done
        ],
    )


def stage_shap(config: dict[str, Any], model_key: str, limit: int | None) -> None:
    from partition_shap import create_explainer
    from whole_word_masker import create_word_masker

    require_gate(SMOKE_RESULT, "Smoke test (Part 3)")
    rows = load_evaluation_rows(config)
    scores = load_model_scores()[model_key]
    records, rejected = load_valid_records(model_key, rows, scores, config)
    path = records_path(model_key)
    if rejected:
        # Keep every valid record; only unusable lines are dropped, and they are logged.
        print(f"[shap:{model_key}] rejected {len(rejected)} checkpoint line(s): {rejected[:3]}")
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                if row["sentence_id"] in records:
                    stream.write(json.dumps(records[row["sentence_id"]], ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    pending = [(index, row) for index, row in enumerate(rows) if row["sentence_id"] not in records]
    print(f"[shap:{model_key}] {len(records)} valid checkpointed sentences, {len(pending)} pending")
    extra = {"rejected_checkpoint_lines": rejected}
    if not pending:
        _write_progress(model_key, rows, records, extra)
        return
    device = setup(config)
    scorer, _ = build_scorer(model_key, config, device)
    masker = create_word_masker()
    explainer = create_explainer(scorer, masker)
    # Untimed warm-up on a fixed non-dataset sentence so CUDA/numba start-up cost is
    # not charged to whichever evaluation sentence happens to be first.
    explainer([WARMUP_SENTENCE], max_evals=int(config["shap"]["max_evals"]), batch_size=int(config["batch_sizes"]["shap"][model_key]), silent=True)
    signature = settings_signature(model_key, config)
    interval = int(config["checkpoint_interval"])
    started = time.perf_counter()
    processed = 0
    for row_index, row in pending:
        if limit is not None and processed >= limit:
            break
        record = build_record(model_key, row_index, row, scores[row["sentence_id"]], explainer, scorer, masker, config, signature)
        _append_record(path, record)
        records[row["sentence_id"]] = record
        processed += 1
        if processed % interval == 0 or len(records) == len(rows):
            _write_progress(model_key, rows, records, extra)
            elapsed = time.perf_counter() - started
            print(f"[shap:{model_key}] {len(records)}/{len(rows)} done | {elapsed / processed:.2f}s/sentence | "
                  f"max residual {max(r['max_additivity_residual'] for r in records.values()):.2e}")
    _write_progress(model_key, rows, records, extra)
    print(f"[shap:{model_key}] finished this invocation: {processed} new, {len(records)}/{len(rows)} total")


# ---------------------------------------------------------------- stage: metrics

def load_complete_records(config: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, list[dict[str, Any]]]]:
    rows = load_evaluation_rows(config)
    scores = load_model_scores()
    ordered: dict[str, list[dict[str, Any]]] = {}
    for model_key in MODEL_KEYS:
        records, rejected = load_valid_records(model_key, rows, scores[model_key], config)
        if len(records) != len(rows) or rejected:
            raise RuntimeError(f"{model_key}: {len(records)}/{len(rows)} valid records ({len(rejected)} rejected); run the shap stage first.")
        ordered[model_key] = [records[row["sentence_id"]] for row in rows]
    return rows, ordered


def length_group(content_words: int, config: dict[str, Any]) -> str:
    groups = config["length_groups"]
    if content_words <= int(groups["short_max_content_words"]):
        return "short"
    if content_words <= int(groups["medium_max_content_words"]):
        return "medium"
    return "long"


def stage_metrics(config: dict[str, Any]) -> None:
    import evaluation_metrics as em

    rows, records = load_complete_records(config)
    tolerances = config["tolerances"]
    seed = int(config["random_seed"])
    iterations = int(config["bootstrap"]["iterations"])
    level = float(config["bootstrap"]["confidence_level"])
    top_k = int(config["top_k"])
    text_of = {row["sentence_id"]: row["text"] for row in rows}

    # ---- per-sentence, per-word and faithfulness tables
    sentence_rows, word_rows, faith_rows = [], [], []
    for model_key in MODEL_KEYS:
        for record in records[model_key]:
            scores, bases = record["scores"], record["base_values"]
            residual, reconstructed = record["additivity_residual"], record["reconstructed"]
            content = int(sum(record["is_content"]))
            summary = record["faithfulness_summary"]
            sentence_rows.append(
                {
                    "model": model_key,
                    "model_name": config["models"][model_key]["name_or_path"],
                    "row_index": record["row_index"],
                    "sentence_id": record["sentence_id"],
                    "text": text_of[record["sentence_id"]],
                    "gold_label": record["gold_label"],
                    "prediction": record["prediction"],
                    "correct": record["correct"],
                    **{f"{key}_score": scores[index] for index, key in enumerate(("neg", "pos"))},
                    "margin": scores[2],
                    **{f"{key}_base_value": bases[index] for index, key in enumerate(OUTPUT_KEYS)},
                    **{f"{key}_reconstructed": reconstructed[index] for index, key in enumerate(OUTPUT_KEYS)},
                    **{f"{key}_additivity_residual": residual[index] for index, key in enumerate(OUTPUT_KEYS)},
                    "max_additivity_residual": record["max_additivity_residual"],
                    "additivity_tolerance": tolerances["additivity"][model_key],
                    "additivity_pass": int(record["max_additivity_residual"] <= tolerances["additivity"][model_key]),
                    "margin_linearity_error": record["max_word_margin_linearity_error"],
                    "base_margin_linearity_error": record["base_margin_linearity_error"],
                    "margin_linearity_pass": int(
                        record["max_word_margin_linearity_error"] <= tolerances["margin_linearity"]
                        and record["base_margin_linearity_error"] <= tolerances["base_margin_linearity"]
                    ),
                    "shap_runtime_seconds": record["shap_runtime_seconds"],
                    "shap_model_evaluations": record["shap_model_evaluations"],
                    "word_count": len(record["words"]),
                    "content_word_count": content,
                    "length_group": length_group(content, config),
                    "internal_token_count": record["internal_token_count"],
                    "multi_piece_words": sum(len(pieces) > 1 for pieces in record["tokens_per_unit"]),
                    **{key: summary[key] for key in summary},
                    "device": record["device"],
                }
            )
            values = np.asarray(record["word_values"], dtype=np.float64)
            content_mask = np.asarray(record["is_content"], dtype=bool)
            ranks = np.full(len(values), np.nan)
            content_positions = np.flatnonzero(content_mask)
            order = content_positions[np.argsort(-np.abs(values[content_mask, 2]), kind="stable")]
            ranks[order] = np.arange(1, len(order) + 1)
            for index, word in enumerate(record["words"]):
                word_rows.append(
                    {
                        "model": model_key,
                        "sentence_id": record["sentence_id"],
                        "word_index": index,
                        "word": word,
                        "is_content": record["is_content"][index],
                        "unit_kind": record["unit_kinds"][index],
                        "neg_shap": values[index, 0],
                        "pos_shap": values[index, 1],
                        "margin_shap": values[index, 2],
                        "abs_margin_rank_among_content": "" if np.isnan(ranks[index]) else int(ranks[index]),
                        "internal_token_count": len(record["tokens_per_unit"][index]),
                        "internal_tokens": " ".join(record["tokens_per_unit"][index]),
                    }
                )
            for item in record["faithfulness_rows"]:
                faith_rows.append({"model": model_key, "sentence_id": record["sentence_id"], "gold_label": record["gold_label"],
                                   "prediction": record["prediction"], **item})
    write_csv(VALUES_DIR / "sentence_results.csv", sentence_rows)
    write_csv(VALUES_DIR / "word_shap_values.csv", word_rows)
    write_csv(VALUES_DIR / "faithfulness_by_fraction.csv", faith_rows)
    by_model = {model_key: [row for row in sentence_rows if row["model"] == model_key] for model_key in MODEL_KEYS}

    # ---- metric 1: classification
    gold = np.asarray([row["gold_label"] for row in rows])
    predictions = {model_key: np.asarray([row["prediction"] for row in by_model[model_key]]) for model_key in MODEL_KEYS}
    classification = []
    for model_key in MODEL_KEYS:
        classification.append({"model": model_key, "model_name": config["models"][model_key]["name_or_path"],
                               **em.classification_metrics(gold, predictions[model_key])})
    write_csv(METRICS_DIR / "classification_metrics.csv", classification)

    # ---- metric 2: additivity + margin linearity
    additivity_rows = []
    for model_key in MODEL_KEYS:
        selected = by_model[model_key]
        tolerance = float(tolerances["additivity"][model_key])
        for key, name in zip(OUTPUT_KEYS, OUTPUT_NAMES, strict=True):
            additivity_rows.append(em.residual_summary(model_key, name, [row[f"{key}_additivity_residual"] for row in selected], tolerance, "additivity_residual"))
        additivity_rows.append(em.residual_summary(model_key, "max_over_outputs", [row["max_additivity_residual"] for row in selected], tolerance, "additivity_residual"))
        additivity_rows.append(em.residual_summary(model_key, "word_phi_margin_vs_phi_pos_minus_phi_neg", [row["margin_linearity_error"] for row in selected], float(tolerances["margin_linearity"]), "margin_linearity"))
        additivity_rows.append(em.residual_summary(model_key, "base_margin_vs_base_pos_minus_base_neg", [row["base_margin_linearity_error"] for row in selected], float(tolerances["base_margin_linearity"]), "margin_linearity"))
    write_csv(METRICS_DIR / "additivity_summary.csv", additivity_rows)

    # ---- supporting: how the class-specific explanations relate to the margin (scale-free, within model)
    structure_rows = []
    for model_key in MODEL_KEYS:
        per_sentence_r, ratios, pooled = [], [], []
        for record in records[model_key]:
            values = np.asarray(record["word_values"], dtype=np.float64)[np.asarray(record["is_content"], dtype=bool)]
            pooled.append(values)
            if len(values) >= 3 and np.ptp(values[:, 0]) > 0 and np.ptp(values[:, 1]) > 0:
                per_sentence_r.append(float(np.corrcoef(values[:, 0], values[:, 1])[0, 1]))
            class_mass = 0.5 * (np.abs(values[:, 0]).sum() + np.abs(values[:, 1]).sum())
            if class_mass > 0:
                ratios.append(float(np.abs(values[:, 2]).sum() / class_mass))
        stacked = np.concatenate(pooled)
        structure_rows.append(
            {
                "model": model_key,
                "content_word_rows": len(stacked),
                "pooled_pearson_phi_neg_vs_phi_pos": float(np.corrcoef(stacked[:, 0], stacked[:, 1])[0, 1]),
                "median_sentence_pearson_phi_neg_vs_phi_pos": float(np.median(per_sentence_r)),
                "sentences_with_defined_pearson": len(per_sentence_r),
                "median_margin_to_class_mass_ratio": float(np.median(ratios)),
                "definition": "ratio = sum|phi_margin| / mean(sum|phi_NEG|, sum|phi_POS|) over content words; ~2 when NEG and POS "
                              "move in opposite directions, ~0 when they move together (common-mode)",
            }
        )
    write_csv(METRICS_DIR / "shap_output_structure.csv", structure_rows)

    # ---- metrics 4 + 5: pairwise explanation agreement on identical word units
    pair_rows = []
    for bert_record, clip_record in zip(records["bert"], records["clip"], strict=True):
        if bert_record["words"] != clip_record["words"] or bert_record["sentence_id"] != clip_record["sentence_id"]:
            raise RuntimeError(f"Word units differ between models for {bert_record['sentence_id']}.")
        bert_ok = bert_record["prediction"] == bert_record["gold_label"]
        clip_ok = clip_record["prediction"] == clip_record["gold_label"]
        group = {(True, True): "both_correct", (True, False): "bert_only_correct", (False, True): "clip_only_correct", (False, False): "both_incorrect"}[(bert_ok, clip_ok)]
        comparison = em.compare_explanations(
            np.asarray(bert_record["word_values"])[:, 2], np.asarray(clip_record["word_values"])[:, 2],
            np.asarray(bert_record["is_content"], dtype=bool), top_k,
        )
        pair_rows.append(
            {
                "sentence_id": bert_record["sentence_id"],
                "gold_label": bert_record["gold_label"],
                "bert_prediction": bert_record["prediction"],
                "clip_prediction": clip_record["prediction"],
                "prediction_agreement": int(bert_record["prediction"] == clip_record["prediction"]),
                "correctness_group": group,
                "both_correct": int(group == "both_correct"),
                "length_group": length_group(comparison["content_words"], config),
                "bert_top_k_words": " | ".join(bert_record["words"][int(i)] for i in comparison["bert_top_k_word_indices"].split()),
                "clip_top_k_words": " | ".join(clip_record["words"][int(i)] for i in comparison["clip_top_k_word_indices"].split()),
                **comparison,
            }
        )
    write_csv(METRICS_DIR / "pairwise_explanation_comparison.csv", pair_rows)

    # ---- bootstrap (paired: identical resamples for every metric)
    samples = em.stratified_bootstrap_indices(gold, iterations, seed)
    ci_rows: list[dict[str, Any]] = []
    classification_boot = {}
    for model_key, metrics in zip(MODEL_KEYS, classification, strict=True):
        classification_boot[model_key] = em.bootstrap_classification(gold, predictions[model_key], samples)
        for name in ("accuracy", "macro_f1", "balanced_accuracy"):
            ci_rows.append(em.confidence_row(model_key, name, metrics[name], classification_boot[model_key][name], iterations, seed, level))
        for field in ("aopc_shap_abs", "aopc_random", "aopc_shap_minus_random", "aopc_shap_signed", "flip_rate_shap_abs", "flip_rate_random", "flip_rate_shap_signed"):
            values = [row[field] for row in by_model[model_key]]
            ci_rows.append(em.confidence_row(model_key, f"mean_{field}", float(np.mean(values)), em.bootstrap_mean(values, samples), iterations, seed, level))
        flip_diff = [row["flip_rate_shap_abs"] - row["flip_rate_random"] for row in by_model[model_key]]
        ci_rows.append(em.confidence_row(model_key, "mean_flip_rate_shap_minus_random", float(np.mean(flip_diff)), em.bootstrap_mean(flip_diff, samples), iterations, seed, level))
        # Robust companions to the normalized mean (heavy-tailed when |margin| is near zero):
        # medians of the normalized AOPC, and raw AOPC in the model's own units (valid within a model only).
        for field in ("aopc_shap_abs", "aopc_random", "aopc_shap_minus_random"):
            values = [row[field] for row in by_model[model_key]]
            ci_rows.append(em.confidence_row(model_key, f"median_{field}", float(np.median(values)), em.bootstrap_median(values, samples), iterations, seed, level))
        raw = {
            "raw_aopc_shap_abs": [row["raw_aopc_shap_abs"] for row in by_model[model_key]],
            "raw_aopc_random": [row["raw_aopc_random"] for row in by_model[model_key]],
            "raw_aopc_shap_minus_random": [row["raw_aopc_shap_abs"] - row["raw_aopc_random"] for row in by_model[model_key]],
        }
        for field, values in raw.items():
            ci_rows.append(em.confidence_row(model_key, f"mean_{field}", float(np.mean(values)), em.bootstrap_mean(values, samples), iterations, seed, level))
    for name in ("accuracy", "macro_f1", "balanced_accuracy"):
        point = classification[0][name] - classification[1][name]
        ci_rows.append(em.confidence_row("bert_minus_clip", name, point, classification_boot["bert"][name] - classification_boot["clip"][name], iterations, seed, level))
    subsets = {
        "all": np.ones(len(pair_rows), dtype=bool),
        "gold_NEG": np.asarray([row["gold_label"] == "NEG" for row in pair_rows]),
        "gold_POS": np.asarray([row["gold_label"] == "POS" for row in pair_rows]),
        "both_correct": np.asarray([row["both_correct"] == 1 for row in pair_rows]),
        "not_both_correct": np.asarray([row["both_correct"] == 0 for row in pair_rows]),
        "more_than_top_k_content_words": np.asarray([row["top_k_is_trivial"] == 0 for row in pair_rows]),
    }
    pair_metrics = ("spearman_abs_margin", "top_k_overlap_rate", "top_k_jaccard", "sign_agreement", "signed_margin_spearman", "prediction_agreement")
    chance_of = {"top_k_overlap_rate": "top_k_overlap_chance", "top_k_jaccard": "top_k_jaccard_chance", "sign_agreement": "sign_agreement_chance"}
    summary_rows = []
    for subset, mask in subsets.items():
        for metric in pair_metrics:
            values = np.asarray([row[metric] for row in pair_rows], dtype=np.float64)
            masked = np.where(mask, values, np.nan)
            stats = em.describe(masked)
            replicates = np.nanmean(masked[samples], axis=1)
            ci = em.confidence_row(f"bert_vs_clip[{subset}]", metric, stats["mean"], replicates, iterations, seed, level)
            ci_rows.append(ci)
            chance = chance_of.get(metric)
            summary_rows.append(
                {
                    "metric": metric,
                    "subset": subset,
                    "n_defined": stats["n"],
                    "n_undefined": int(mask.sum() - stats["n"]),
                    "mean": stats["mean"],
                    "median": stats["median"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "ci_lower": ci["ci_lower"],
                    "ci_upper": ci["ci_upper"],
                    "chance_level_mean": float(np.nanmean(np.where(mask, [row[chance] for row in pair_rows], np.nan))) if chance else (0.0 if "spearman" in metric else np.nan),
                    "primary": int(metric in {"spearman_abs_margin", "top_k_overlap_rate", "top_k_jaccard"}),
                }
            )
    mcnemar = em.mcnemar_exact(predictions["bert"] == gold, predictions["clip"] == gold)
    summary_rows.append({"metric": "exact_mcnemar_bert_vs_clip_correctness", "subset": "all", "n_defined": len(pair_rows),
                         "mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
                         "ci_lower": np.nan, "ci_upper": np.nan, "chance_level_mean": np.nan, "primary": 0,
                         "p_value": mcnemar["exact_mcnemar_p"], "bert_only_correct": mcnemar["left_only_correct"],
                         "clip_only_correct": mcnemar["right_only_correct"]})
    write_csv(METRICS_DIR / "pairwise_explanation_summary.csv", summary_rows)
    write_csv(METRICS_DIR / "bootstrap_confidence_intervals.csv", ci_rows)

    # ---- metric 3 detail: deletion curves, AOPC tests
    faith_summary = []
    for model_key in MODEL_KEYS:
        model_faith = [row for row in faith_rows if row["model"] == model_key]
        for ranking in ("shap_abs", "shap_signed", "random"):
            for fraction in config["faithfulness"]["fractions"]:
                selected = [row for row in model_faith if row["ranking"] == ranking and row["fraction"] == float(fraction)]
                faith_summary.append(
                    {
                        "model": model_key, "ranking": ranking, "fraction": float(fraction), "rows": len(selected),
                        "mean_normalized_drop": float(np.mean([row["normalized_drop"] for row in selected])),
                        "median_normalized_drop": float(np.median([row["normalized_drop"] for row in selected])),
                        "mean_raw_score_drop": float(np.mean([row["score_drop"] for row in selected])),
                        "prediction_flip_rate": float(np.mean([row["prediction_flip"] for row in selected])),
                    }
                )
        shap_aopc = [row["aopc_shap_abs"] for row in by_model[model_key]]
        random_aopc = [row["aopc_random"] for row in by_model[model_key]]
        test = em.paired_wilcoxon(shap_aopc, random_aopc)
        faith_summary.append(
            {
                "model": model_key, "ranking": "shap_abs_vs_random_AOPC", "fraction": "all", "rows": len(shap_aopc),
                "mean_normalized_drop": test["mean_difference"], "median_normalized_drop": test["median_difference"],
                "mean_raw_score_drop": float(np.mean([row["raw_aopc_shap_abs"] - row["raw_aopc_random"] for row in by_model[model_key]])),
                "prediction_flip_rate": float(np.mean([row["flip_rate_shap_abs"] - row["flip_rate_random"] for row in by_model[model_key]])),
                "fraction_sentences_shap_greater": test["fraction_left_greater"], "wilcoxon_p": test["wilcoxon_p"],
                "median_aopc_shap_abs": float(np.median(shap_aopc)), "median_aopc_random": float(np.median(random_aopc)),
            }
        )
        # POST-HOC sensitivity (added after observing heavy tails in CLIP's normalized AOPC):
        # restrict to sentences whose |original margin| >= 1e-3 in the model's own units.
        kept = [row for row in by_model[model_key] if abs(row["margin"]) >= POSTHOC_MARGIN_THRESHOLD]
        faith_summary.append(
            {
                "model": model_key, "ranking": f"POSTHOC_abs_margin_ge_{POSTHOC_MARGIN_THRESHOLD:g}", "fraction": "all", "rows": len(kept),
                "mean_normalized_drop": float(np.mean([row["aopc_shap_minus_random"] for row in kept])),
                "median_normalized_drop": float(np.median([row["aopc_shap_minus_random"] for row in kept])),
                "mean_aopc_shap_abs": float(np.mean([row["aopc_shap_abs"] for row in kept])),
                "mean_aopc_random": float(np.mean([row["aopc_random"] for row in kept])),
                "fraction_sentences_shap_greater": float(np.mean([row["aopc_shap_minus_random"] > 0 for row in kept])),
                "wilcoxon_p": em.paired_wilcoxon([row["aopc_shap_abs"] for row in kept], [row["aopc_random"] for row in kept])["wilcoxon_p"],
            }
        )
    write_csv(METRICS_DIR / "faithfulness_summary.csv", faith_summary)

    # ---- subgroups
    fields = ("correct", "aopc_shap_abs", "aopc_random", "aopc_shap_minus_random", "flip_rate_shap_abs", "flip_rate_random",
              "max_additivity_residual", "shap_runtime_seconds", "shap_model_evaluations")
    subgroup_rows = []
    for model_key in MODEL_KEYS:
        subgroup_rows += em.subgroup_table(model_key, by_model[model_key], {
            "gold_label": lambda row: row["gold_label"],
            "correctness": lambda row: "correct" if row["correct"] else "incorrect",
            "length_group": lambda row: row["length_group"],
        }, fields)
    subgroup_rows += em.subgroup_table("bert_vs_clip", pair_rows, {
        "gold_label": lambda row: row["gold_label"],
        "correctness_group": lambda row: row["correctness_group"],
        "both_correct": lambda row: "both_correct" if row["both_correct"] else "not_both_correct",
        "length_group": lambda row: row["length_group"],
    }, ("spearman_abs_margin", "top_k_overlap_rate", "top_k_jaccard", "sign_agreement", "prediction_agreement"))
    for row in subgroup_rows:
        if "mean_correct" in row:
            row["accuracy"] = row.pop("mean_correct")
    write_csv(METRICS_DIR / "subgroup_results.csv", subgroup_rows)

    # ---- runtime
    runtime_rows = []
    for model_key in MODEL_KEYS:
        selected = by_model[model_key]
        runtimes = [row["shap_runtime_seconds"] for row in selected]
        evaluations = [row["shap_model_evaluations"] for row in selected]
        stats = em.describe(runtimes)
        runtime_rows.append(
            {
                "model": model_key, "device": selected[0]["device"], "sentences": len(selected),
                "total_shap_runtime_seconds": float(np.sum(runtimes)), "mean_shap_runtime_seconds": stats["mean"],
                "median_shap_runtime_seconds": stats["median"], "p95_shap_runtime_seconds": stats["p95"], "max_shap_runtime_seconds": stats["max"],
                "total_model_evaluations": int(np.sum(evaluations)), "mean_model_evaluations": float(np.mean(evaluations)),
                "median_model_evaluations": float(np.median(evaluations)), "max_model_evaluations": int(np.max(evaluations)),
                "sentences_reaching_budget": int(np.sum(np.asarray(evaluations) >= int(config["shap"]["max_evals"]))),
                "mean_seconds_per_1000_evaluations": float(np.sum(runtimes) / np.sum(evaluations) * 1000),
                "total_faithfulness_runtime_seconds": float(np.sum([row["faithfulness_runtime_seconds"] for row in selected])),
                "total_faithfulness_model_evaluations": int(np.sum([row["faithfulness_model_evaluations"] for row in selected])),
                "max_evals_budget": int(config["shap"]["max_evals"]),
                "shap_batch_size": int(config["batch_sizes"]["shap"][model_key]),
            }
        )
    write_csv(METRICS_DIR / "runtime_summary.csv", runtime_rows)

    # ---- acceptance checks for this stage
    words_expected = {model_key: sum(len(record["words"]) for record in records[model_key]) for model_key in MODEL_KEYS}
    checks = {
        "bert_500_records": len(records["bert"]) == 500,
        "clip_500_records": len(records["clip"]) == 500,
        "sentence_rows_1000": len(sentence_rows) == 1000,
        "word_rows_complete": len(word_rows) == sum(words_expected.values()),
        "pairwise_rows_500": len(pair_rows) == 500,
        "same_sentence_order": [r["sentence_id"] for r in records["bert"]] == [r["sentence_id"] for r in records["clip"]] == [row["sentence_id"] for row in rows],
        "additivity_all_pass": all(row["all_pass"] for row in additivity_rows if row["check"] == "additivity_residual"),
        "margin_linearity_all_pass": all(row["all_pass"] for row in additivity_rows if row["check"] == "margin_linearity"),
        "bootstrap_1000_iterations": samples.shape[0] == 1000 and all(row["iterations"] == 1000 for row in ci_rows),
        "faithfulness_rows_complete": len(faith_rows) == 1000 * len(config["faithfulness"]["fractions"]) * (2 + int(config["faithfulness"]["random_repeats"])),
    }
    save_json(METRIC_CHECKS, {
        "checks": checks, "gate_passed": all(checks.values()), "dataset_sha256": dataset_hash(config),
        "config_sha256": config_sha256(), "word_rows": words_expected, "completed_at_utc": utc_now(),
    })
    print("[metrics] checks:", checks)
    if not all(checks.values()):
        raise RuntimeError("Metric-stage acceptance checks failed.")


# ------------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=("prepare", "score", "smoke", "shap", "metrics", "plots", "wordbars", "report", "all"))
    parser.add_argument("--model", choices=("bert", "clip", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None, help="process at most N new sentences in this invocation (resume test)")
    args = parser.parse_args()
    config = load_config()
    stages = ["prepare", "score", "smoke", "shap", "metrics", "plots", "wordbars", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        started = time.perf_counter()
        if stage == "prepare":
            manifest = prepare_dataset(config)
            print(f"[prepare] dataset verified: sha256={manifest['sha256_copied']}, checks={manifest['all_checks_passed']}")
        elif stage == "score":
            stage_score(config)
        elif stage == "smoke":
            stage_smoke(config)
        elif stage == "shap":
            for model_key in (MODEL_KEYS if args.model == "both" else (args.model,)):
                stage_shap(config, model_key, args.limit)
        elif stage == "metrics":
            stage_metrics(config)
        elif stage == "plots":
            from plotting import make_all_plots

            make_all_plots(config)
        elif stage == "wordbars":
            from word_bar_plots import make_word_bar_plots

            make_word_bar_plots(config)
        elif stage == "report":
            from report_builder import build_reports

            build_reports(config)
        print(f"[{stage}] completed in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
