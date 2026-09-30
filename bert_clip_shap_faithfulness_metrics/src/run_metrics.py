"""Four explanation metrics for the existing BERT / CLIP-text whole-word POS-NEG margin SHAP.

Steps (every gate must pass before any metric file is written):
  1. load the 500 validated SHAP records per model with the source experiment's own
     record validation (text hashes, settings signature, stored scores, completeness),
     and build the model-independent SST word-sentiment reference (rationale proxy);
  2. rebuild the |phi_margin| ranking and the deletion texts and require that they equal
     the stored ``shap_abs`` deletion rows exactly (same ranking, same masker);
  3. load each frozen scorer, re-score the 500 originals and the 2,000 stored deletion
     texts and require agreement within the source batch-parity tolerance;
  4. score the 2,000 new sufficiency texts (top-k content words + punctuation kept);
  5. compute per-sentence metrics, reproduce the stored deletion AOPC, write outputs.

Stored original scores and stored deletion scores are used for the metrics; re-scoring
is only a check. No SHAP value is recomputed, no model is trained, and nothing inside
the source experiment folder is written (verified by hashing it before and after).
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

# In the rfem environment shap (pandas/pyarrow) must be imported before CUDA initialises.
import shap  # noqa: F401,E402

from source_link import OUTPUT_DIR, SOURCE_ROOT, load_own_config, source_file_hashes  # noqa: E402
from metrics_core import (  # noqa: E402
    comprehensiveness_keep,
    fraction_label,
    rank_content_words,
    rationale_columns,
    sentence_metrics,
    sufficiency_keep,
    top_k_by_fraction,
)
from rationale_reference import SstLexicon, precision_recall_f1, sentence_reference  # noqa: E402
from io_utils import VALUES_DIR, environment_versions, load_config, load_json, read_csv, save_json, sha256_file, utc_now, write_csv  # noqa: E402
from model_outputs import MARGIN, predict_label, score_in_batches  # noqa: E402
from run_experiment import _release_memory, build_scorer, load_complete_records, records_path, setup  # noqa: E402
from whole_word_masker import create_word_masker, render_coalition  # noqa: E402


LOG_PATH = OUTPUT_DIR / "run.log"
MODEL_LABELS = {"bert": "BERT", "clip": "CLIP"}
RATIONALE_METRICS = ("precision", "recall", "f1")
TABLE_ROWS = {
    "Comprehensiveness (higher = better)": "comprehensiveness",
    "Sufficiency (lower = better)": "sufficiency",
    "Deletion AOPC (higher = better)": "deletion_aopc",
    "Rationale Precision, SST lexical proxy (higher = better)": "rationale_precision",
    "Rationale Recall, SST lexical proxy (higher = better)": "rationale_recall",
    "Rationale F1, SST lexical proxy (higher = better)": "rationale_f1",
}


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


# ------------------------------------------------------------------ planning

def build_plans(rows: list[dict[str, str]], records: list[dict[str, Any]], masker: Any, fractions: list[float]) -> list[dict[str, Any]]:
    """Ranking, top-k sets and perturbed texts per sentence, checked against the stored deletion rows."""
    plans = []
    for row, record in zip(rows, records, strict=True):
        values = np.asarray(record["word_values"], dtype=np.float64)[:, MARGIN]
        order = rank_content_words(record["is_content"], values)
        stored = {float(item["fraction"]): item for item in record["faithfulness_rows"] if item["ranking"] == "shap_abs"}
        cuts = []
        for fraction, k, top in top_k_by_fraction(order, fractions):
            comp_text = render_coalition(masker, row["text"], comprehensiveness_keep(record["is_content"], top))
            suff_text = render_coalition(masker, row["text"], sufficiency_keep(record["is_content"], top))
            reference = stored[fraction]
            cuts.append(
                {
                    "fraction": fraction,
                    "k": k,
                    "top": top,
                    "top_words": [record["words"][index] for index in top],
                    "comp_text": comp_text,
                    "suff_text": suff_text,
                    "stored_comp_margin": float(reference["margin_after"]),
                    "stored_normalized_drop": float(reference["normalized_drop"]),
                    "ranking_matches_stored": [int(v) for v in str(reference["deleted_word_indices"]).split()] == top
                    and int(reference["deleted_count"]) == k,
                    "comp_text_matches_stored": comp_text == reference["perturbed_text"],
                }
            )
        plans.append(
            {
                "row_index": int(record["row_index"]),
                "sentence_id": record["sentence_id"],
                "text": row["text"],
                "gold_label": record["gold_label"],
                "prediction": record["prediction"],
                "correct": int(record["correct"]),
                "content_words": int(sum(record["is_content"])),
                "stored_scores": np.asarray(record["scores"], dtype=np.float64),
                "stored_aopc": float(record["faithfulness_summary"]["aopc_shap_abs"]),
                "cuts": cuts,
            }
        )
    return plans


# ------------------------------------------------------------------ scoring

def score_model(model_key: str, config: dict[str, Any], plans: list[dict[str, Any]], device: Any) -> dict[str, Any]:
    started = time.perf_counter()
    scorer, info = build_scorer(model_key, config, device)
    batch = int(config["batch_sizes"]["score"][model_key])
    originals = score_in_batches(scorer, [plan["text"] for plan in plans], batch)
    comp = score_in_batches(scorer, [cut["comp_text"] for plan in plans for cut in plan["cuts"]], batch)
    suff = score_in_batches(scorer, [cut["suff_text"] for plan in plans for cut in plan["cuts"]], batch)
    evaluations = int(scorer.text_evaluations)
    device_name = str(scorer.device)
    del scorer
    _release_memory()
    return {
        "originals": originals,
        "comp": comp,
        "suff": suff,
        "semantic_check_passed": bool(info["semantic_check_passed"]),
        "frozen": bool(info["frozen"]["frozen"]),
        "model_name": info["model_name"],
        "model_evaluations": evaluations,
        "device": device_name,
        "runtime_seconds": time.perf_counter() - started,
    }


# ------------------------------------------------------------------ outputs

def per_sentence_rows(
    model_key: str,
    plans: list[dict[str, Any]],
    suff_margins: np.ndarray,
    epsilon: float,
    scale: float,
    references: dict[str, list[int]],
) -> list[dict[str, Any]]:
    rows = []
    n_fractions = len(plans[0]["cuts"])
    for index, plan in enumerate(plans):
        cuts = plan["cuts"]
        suff = suff_margins[index * n_fractions : (index + 1) * n_fractions]
        metrics = sentence_metrics(plan["stored_scores"][MARGIN], [cut["stored_comp_margin"] for cut in cuts], suff, epsilon)
        row: dict[str, Any] = {
            "model": model_key,
            "row_index": plan["row_index"],
            "sentence_id": plan["sentence_id"],
            "gold_label": plan["gold_label"],
            "prediction": plan["prediction"],
            "correct": plan["correct"],
            "content_word_count": plan["content_words"],
            "original_margin": float(plan["stored_scores"][MARGIN]),
            "direction": metrics["direction"],
            "original_decision_score": metrics["decision_score"],
            "model_scale_mean_abs_decision_score": scale,
            "comprehensiveness": metrics["raw_comprehensiveness"] / scale,
            "sufficiency": metrics["raw_sufficiency"] / scale,
            "deletion_aopc": metrics["deletion_aopc"],
            "comprehensiveness_raw": metrics["raw_comprehensiveness"],
            "sufficiency_raw": metrics["raw_sufficiency"],
        }
        for position, cut in enumerate(cuts):
            label = fraction_label(cut["fraction"])
            row[f"top_k_{label}"] = cut["k"]
            row[f"comprehensiveness_{label}"] = metrics["comp_by_fraction"][position] / scale
            row[f"sufficiency_{label}"] = metrics["suff_by_fraction"][position] / scale
            row[f"deletion_normalized_drop_{label}"] = metrics["normalized_drop_by_fraction"][position]
            row[f"comprehensiveness_raw_{label}"] = metrics["comp_by_fraction"][position]
            row[f"sufficiency_raw_{label}"] = metrics["suff_by_fraction"][position]
        reference = references[plan["sentence_id"]]
        row["rationale_reference_word_count"] = len(reference)
        row["rationale_evaluated"] = int(bool(reference))
        # rows: fractions; columns: precision, recall, F1
        scores = np.asarray([precision_recall_f1(cut["top"], reference) for cut in cuts]) if reference else None
        for position, name in enumerate(RATIONALE_METRICS):
            row[f"rationale_{name}"] = float(np.mean(scores[:, position])) if reference else "NA"
            for fraction_position, cut in enumerate(cuts):
                value = float(scores[fraction_position, position]) if reference else "NA"
                row[f"rationale_{name}_{fraction_label(cut['fraction'])}"] = value
        rows.append(row)
    return rows


def audit_rows(model_key: str, plans: list[dict[str, Any]], scored: dict[str, Any]) -> list[dict[str, Any]]:
    rows, position = [], 0
    for plan in plans:
        for cut in plan["cuts"]:
            comp_rescored = float(scored["comp"][position, MARGIN])
            suff_margin = float(scored["suff"][position, MARGIN])
            rows.append(
                {
                    "model": model_key,
                    "sentence_id": plan["sentence_id"],
                    "fraction": cut["fraction"],
                    "top_k": cut["k"],
                    "content_word_count": plan["content_words"],
                    "top_word_indices": " ".join(str(index) for index in cut["top"]),
                    "top_words": " | ".join(cut["top_words"]),
                    "original_margin": float(plan["stored_scores"][MARGIN]),
                    "original_prediction": plan["prediction"],
                    "comprehensiveness_text": cut["comp_text"],
                    "comprehensiveness_margin_stored": cut["stored_comp_margin"],
                    "comprehensiveness_margin_rescored": comp_rescored,
                    "comprehensiveness_prediction": predict_label(cut["stored_comp_margin"]),
                    "sufficiency_text": cut["suff_text"],
                    "sufficiency_margin": suff_margin,
                    "sufficiency_prediction": predict_label(suff_margin),
                }
            )
            position += 1
    return rows


def aggregate_rows(sentence_rows: list[dict[str, Any]], fractions: list[float]) -> list[dict[str, Any]]:
    rows = []
    for model_key in MODEL_LABELS:
        subset = [row for row in sentence_rows if row["model"] == model_key]
        evaluated = [row for row in subset if row["rationale_evaluated"]]
        for fraction in fractions:
            label = fraction_label(fraction)
            rows.append(
                {
                    "model": model_key,
                    "fraction": fraction,
                    "sentences": len(subset),
                    "mean_top_k": float(np.mean([row[f"top_k_{label}"] for row in subset])),
                    "comprehensiveness": float(np.mean([row[f"comprehensiveness_{label}"] for row in subset])),
                    "sufficiency": float(np.mean([row[f"sufficiency_{label}"] for row in subset])),
                    "deletion_normalized_drop": float(np.mean([row[f"deletion_normalized_drop_{label}"] for row in subset])),
                    "comprehensiveness_raw": float(np.mean([row[f"comprehensiveness_raw_{label}"] for row in subset])),
                    "sufficiency_raw": float(np.mean([row[f"sufficiency_raw_{label}"] for row in subset])),
                    "rationale_sentences": len(evaluated),
                    **{
                        f"rationale_{name}": float(np.mean([row[f"rationale_{name}_{label}"] for row in evaluated]))
                        for name in RATIONALE_METRICS
                    },
                }
            )
    return rows


def table_values(sentence_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Means over all sentences; rationale metrics over the sentences with a non-empty reference."""
    values: dict[str, dict[str, float]] = {}
    for model_key in MODEL_LABELS:
        subset = [row for row in sentence_rows if row["model"] == model_key]
        evaluated = [row for row in subset if row["rationale_evaluated"]]
        values[model_key] = {
            name: float(np.mean([row[column] for row in (evaluated if column.startswith("rationale_") else subset)]))
            for name, column in TABLE_ROWS.items()
        }
    return values


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def write_tables(
    values: dict[str, dict[str, Any]],
    scales: dict[str, float],
    near_zero: dict[str, int],
    fractions: list[float],
    rationale: dict[str, Any],
) -> None:
    metrics = list(values["bert"])
    write_csv(OUTPUT_DIR / "comparison_table.csv", [{"Metric": name, "BERT": values["bert"][name], "CLIP": values["clip"][name]} for name in metrics])
    percent = ", ".join(f"{int(round(f * 100))}%" for f in fractions)
    markdown = [
        "# BERT vs CLIP-text: explanation metrics for whole-word POS−NEG margin Partition SHAP",
        "",
        "500 SST-2 sentences (250 NEG / 250 POS). Both models use the same sentences, the same stored",
        "SHAP values (POS−NEG margin), the same word ranking, the same whole-word deletion masker and",
        f"the same fractions ({percent}). Values are means over the 500 sentences; rationale metrics are means",
        f"over the {rationale['evaluated_sentences']} sentences that contain at least one reference sentiment word (identical for both models).",
        "",
        "| Metric | BERT | CLIP |",
        "|---|---|---|",
        *[f"| {name} | {_fmt(values['bert'][name])} | {_fmt(values['clip'][name])} |" for name in metrics],
        "",
        "**Definitions.** d = +1 for a POS prediction, −1 for NEG; decision score s(x) = d·(POS−NEG margin).",
        "Content words are ranked by |φ_margin| (ties by position); k = max(1, ⌈f·n_content⌉); punctuation is never removed.",
        "",
        "- **Comprehensiveness** = mean over f of [s(x) − s(x without its top-k words)] / S_model.",
        "- **Sufficiency** = mean over f of [s(x) − s(only the top-k words, punctuation kept)] / S_model.",
        f"- S_model = mean |s(x)| over the 500 sentences: BERT {scales['bert']:.6g} (logit units), CLIP {scales['clip']:.6g} (cosine units).",
        "- **Deletion AOPC** = mean over f of [s(x) − s(x without its top-k words)] / (|s(x)| + 1e-6); the source experiment's",
        "  definition, reproduced from its stored deletion scores. Per-sentence normalization divides by each sentence's own",
        f"  margin; {near_zero['clip']} CLIP and {near_zero['bert']} BERT sentences have |margin| < 0.001.",
        "- **Rationale Precision / Recall / F1 (SST lexical proxy)**: the evaluation set has no human rationale annotations, so the",
        "  reference is built from the Stanford Sentiment Treebank's human word-level sentiment ratings (0–1 positivity, each word",
        "  rated out of context). Reference words = content words rated outside SST's neutral interval (0.4, 0.6]",
        f"  ({rationale['reference_words']} of {rationale['content_words']} content words). SHAP-selected words = the same top-k words as above.",
        "  Per sentence: P = |top-k ∩ ref| / k, R = |top-k ∩ ref| / |ref|, F1 = 2PR / (P + R), averaged over the four fractions.",
        f"  {rationale['excluded_sentences']} sentences without any reference word are excluded (recall undefined).",
        "  These ratings are context-free lexical sentiment, not rationales for each sentence's label, so this is a proxy for",
        "  token-level rationale F1.",
        "",
    ]
    (OUTPUT_DIR / "comparison_table.md").write_text("\n".join(markdown), encoding="utf-8")
    arrows = {"(higher = better)": "$\\uparrow$", "(lower = better)": "$\\downarrow$"}
    latex_names = {name: name for name in metrics}
    for name in metrics:
        for phrase, arrow in arrows.items():
            latex_names[name] = latex_names[name].replace(phrase, arrow)
    latex = [
        "% Generated by bert_clip_shap_faithfulness_metrics/src/run_metrics.py",
        "\\begin{tabular}{lrr}",
        "\\hline",
        "Metric & BERT & CLIP \\\\",
        "\\hline",
        *[f"{latex_names[name]} & {_fmt(values['bert'][name])} & {_fmt(values['clip'][name])} \\\\" for name in metrics],
        "\\hline",
        "\\end{tabular}",
        "",
    ]
    (OUTPUT_DIR / "comparison_table.tex").write_text("\n".join(latex), encoding="utf-8")


# ------------------------------------------------------------------ main

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    started = time.perf_counter()
    own = load_own_config()
    config = load_config()  # the source experiment's config.json
    fractions = [float(value) for value in own["fractions"]]
    epsilon = float(own["normalization_epsilon"])
    hashes_before = source_file_hashes()
    log(f"source experiment: {SOURCE_ROOT} ({len(hashes_before)} files hashed)")

    checks: dict[str, Any] = {
        "config_fractions_equal_source": fractions == [float(v) for v in config["faithfulness"]["fractions"]],
        "config_epsilon_equals_source": epsilon == float(config["faithfulness"]["normalization_epsilon"]),
        "config_models_equal_source": list(own["models"]) == list(MODEL_LABELS),
    }

    rows, records = load_complete_records(config)  # source validation of all 1,000 records
    checks["records_500_per_model"] = all(len(records[m]) == len(rows) == 500 for m in MODEL_LABELS)
    checks["same_sentence_order_both_models"] = [r["sentence_id"] for r in records["bert"]] == [r["sentence_id"] for r in records["clip"]]
    log(f"loaded {len(rows)} sentences x {len(MODEL_LABELS)} models of validated SHAP records")

    settings = own["rationale"]
    found = rationale_columns(list(rows[0].keys()), settings["reference_columns_searched"])
    if found:
        raise RuntimeError(f"Possible human rationale columns {found}: use them instead of the SST lexical proxy.")
    lexicon = SstLexicon(Path(settings["sst_dictionary_path"]), Path(settings["sst_sentiment_labels_path"]))
    reference_rows, references = [], {}
    for record in records["bert"]:  # word units and content flags are model-independent (checked below)
        words = sentence_reference(lexicon, record["words"], record["is_content"])
        references[record["sentence_id"]] = [word["word_index"] for word in words if word["is_reference"]]
        reference_rows.extend({"sentence_id": record["sentence_id"], **word} for word in words)
    checks["word_units_identical_for_both_models"] = all(
        bert["words"] == clip["words"] and bert["is_content"] == clip["is_content"]
        for bert, clip in zip(records["bert"], records["clip"], strict=True)
    )
    manifest = load_json(SOURCE_ROOT / "data" / "dataset_manifest.json")
    checks["sst2_dev_matches_dataset_manifest"] = sha256_file(Path(settings["sst2_dev_path"])) == manifest["original_sst2_source"]["sha256"]
    sources = Counter(word["lookup_source"] for word in reference_rows)
    checks["sst_reference_rates_every_content_word"] = sources.get("not_found", 0) == 0
    rationale = {
        "dataset_has_human_rationale_annotations": False,
        "dataset_columns": list(rows[0].keys()),
        "reference": settings["reference"],
        "reference_rule": settings["reference_rule"],
        "selection_rule": settings["selection_rule"],
        "empty_reference_rule": settings["empty_reference_rule"],
        "sst_files_sha256": {
            "dictionary.txt": sha256_file(Path(settings["sst_dictionary_path"])),
            "sentiment_labels.txt": sha256_file(Path(settings["sst_sentiment_labels_path"])),
            "dev.tsv": sha256_file(Path(settings["sst2_dev_path"])),
        },
        "content_words": len(reference_rows),
        "reference_words": int(sum(word["is_reference"] for word in reference_rows)),
        "lookup_sources": dict(sources),
        "evaluated_sentences": sum(bool(indices) for indices in references.values()),
        "excluded_sentences": sum(not indices for indices in references.values()),
        "excluded_sentence_ids": [sentence_id for sentence_id, indices in references.items() if not indices],
    }
    log(f"SST lexical reference: {rationale['reference_words']}/{rationale['content_words']} content words are sentiment words "
        f"({dict(sources)}); {rationale['evaluated_sentences']} sentences evaluated, {rationale['excluded_sentences']} without reference words")

    masker = create_word_masker()
    plans = {model_key: build_plans(rows, records[model_key], masker, fractions) for model_key in MODEL_LABELS}
    for model_key in MODEL_LABELS:
        cuts = [cut for plan in plans[model_key] for cut in plan["cuts"]]
        checks[f"{model_key}_ranking_matches_stored_deletion"] = all(cut["ranking_matches_stored"] for cut in cuts)
        checks[f"{model_key}_comprehensiveness_texts_match_stored"] = all(cut["comp_text_matches_stored"] for cut in cuts)
        checks[f"{model_key}_sufficiency_texts_nonempty"] = all(cut["suff_text"].strip() for cut in cuts)
    log("ranking and deletion texts rebuilt: " + ", ".join(f"{k}={v}" for k, v in checks.items() if "ranking" in k or "texts" in k))
    if not all(checks.values()):
        save_json(OUTPUT_DIR / "checks.json", {"gate_passed": False, "checks": checks})
        raise RuntimeError("Pre-scoring gate failed; see outputs/checks.json.")

    device = setup(config)
    scored, scoring_info = {}, {}
    for model_key in MODEL_LABELS:
        result = score_model(model_key, config, plans[model_key], device)
        tolerance = float(config["tolerances"]["batch_parity"][model_key])
        stored_originals = np.stack([plan["stored_scores"] for plan in plans[model_key]])
        stored_comp = np.asarray([cut["stored_comp_margin"] for plan in plans[model_key] for cut in plan["cuts"]])
        original_error = float(np.max(np.abs(result["originals"] - stored_originals)))
        comp_error = float(np.max(np.abs(result["comp"][:, MARGIN] - stored_comp)))
        predictions_same = [predict_label(m) for m in result["originals"][:, MARGIN]] == [plan["prediction"] for plan in plans[model_key]]
        checks[f"{model_key}_model_semantic_check"] = result["semantic_check_passed"]
        checks[f"{model_key}_model_frozen"] = result["frozen"]
        checks[f"{model_key}_rescored_originals_match_stored"] = original_error <= tolerance
        checks[f"{model_key}_rescored_deletions_match_stored"] = comp_error <= tolerance
        checks[f"{model_key}_rescored_predictions_identical"] = predictions_same
        checks[f"{model_key}_sufficiency_scores_finite"] = bool(np.all(np.isfinite(result["suff"])))
        scoring_info[model_key] = {
            "model_name": result["model_name"],
            "device": result["device"],
            "model_evaluations": result["model_evaluations"],
            "runtime_seconds": result["runtime_seconds"],
            "tolerance": tolerance,
            "max_abs_rescored_original_minus_stored": original_error,
            "max_abs_rescored_deletion_margin_minus_stored": comp_error,
            "sufficiency_texts_scored": int(result["suff"].shape[0]),
        }
        scored[model_key] = result
        log(f"{model_key}: {result['model_evaluations']} evaluations in {result['runtime_seconds']:.1f}s on {result['device']} | "
            f"max |rescored-stored| originals {original_error:.2e}, deletions {comp_error:.2e} (tol {tolerance:.0e})")

    sentence_rows, audit, scales, near_zero = [], [], {}, {}
    source_aopc = {(row["model"], row["sentence_id"]): float(row["aopc_shap_abs"]) for row in read_csv(VALUES_DIR / "sentence_results.csv")}
    for model_key in MODEL_LABELS:
        margins = np.asarray([plan["stored_scores"][MARGIN] for plan in plans[model_key]])
        scales[model_key] = float(np.mean(np.abs(margins)))
        near_zero[model_key] = int(np.sum(np.abs(margins) < 1e-3))
        model_rows = per_sentence_rows(
            model_key, plans[model_key], scored[model_key]["suff"][:, MARGIN], epsilon, scales[model_key], references
        )
        drop_error = max(
            abs(row[f"deletion_normalized_drop_{fraction_label(cut['fraction'])}"] - cut["stored_normalized_drop"])
            for row, plan in zip(model_rows, plans[model_key], strict=True)
            for cut in plan["cuts"]
        )
        aopc_error = max(abs(row["deletion_aopc"] - plan["stored_aopc"]) for row, plan in zip(model_rows, plans[model_key], strict=True))
        csv_error = max(abs(row["deletion_aopc"] - source_aopc[(model_key, row["sentence_id"])]) for row in model_rows)
        tolerance = float(own["aopc_reproduction_tolerance"])
        checks[f"{model_key}_normalized_drops_reproduce_stored"] = drop_error <= tolerance
        checks[f"{model_key}_deletion_aopc_reproduces_stored_records"] = aopc_error <= tolerance
        checks[f"{model_key}_deletion_aopc_reproduces_sentence_results_csv"] = csv_error <= tolerance
        scoring_info[model_key].update(
            {
                "max_abs_aopc_minus_stored_record": aopc_error,
                "max_abs_aopc_minus_sentence_results_csv": csv_error,
                "max_abs_normalized_drop_minus_stored": drop_error,
            }
        )
        sentence_rows.extend(model_rows)
        audit.extend(audit_rows(model_key, plans[model_key], scored[model_key]))

    hashes_after = source_file_hashes()
    checks["source_experiment_unchanged"] = hashes_after == hashes_before
    gate = all(checks.values())
    summary = {
        "gate_passed": gate,
        "checks": checks,
        "completed_at_utc": utc_now(),
        "runtime_seconds": time.perf_counter() - started,
        "source_experiment": str(SOURCE_ROOT),
        "source_inputs_sha256": {
            "bert_shap_records.jsonl": sha256_file(records_path("bert")),
            "clip_shap_records.jsonl": sha256_file(records_path("clip")),
            "sentence_results.csv": sha256_file(VALUES_DIR / "sentence_results.csv"),
            "model_scores.csv": sha256_file(VALUES_DIR / "model_scores.csv"),
            "config.json": sha256_file(SOURCE_ROOT / "config.json"),
            "evaluation_500.csv": sha256_file(SOURCE_ROOT / "data" / "evaluation_500.csv"),
        },
        "source_files_hashed": len(hashes_before),
        "fractions": fractions,
        "normalization_epsilon": epsilon,
        "comprehensiveness_sufficiency_scale": scales,
        "sentences_with_abs_margin_below_1e-3": near_zero,
        "scoring": scoring_info,
        "rationale": rationale,
        "environment": environment_versions(),
    }
    save_json(OUTPUT_DIR / "checks.json", summary)
    if not gate:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Gate failed ({failed}); metric files were not written. See outputs/checks.json.")

    values = table_values(sentence_rows)
    write_csv(OUTPUT_DIR / "per_sentence_metrics.csv", sentence_rows)
    write_csv(OUTPUT_DIR / "perturbation_audit.csv", audit)
    write_csv(OUTPUT_DIR / "aggregate_by_fraction.csv", aggregate_rows(sentence_rows, fractions))
    write_csv(OUTPUT_DIR / "rationale_reference_words.csv", reference_rows)
    write_tables(values, scales, near_zero, fractions, rationale)
    for name in values["bert"]:
        log(f"{name:56s} BERT {_fmt(values['bert'][name]):>12s}   CLIP {_fmt(values['clip'][name]):>12s}")
    log(f"all {len(checks)} checks passed; outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
