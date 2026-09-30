"""Part 6: evaluate both frozen CLIP conditions on the held-out 500 sentences."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch

from classifier_head import FrozenClipLinearScorer, classification_metrics
from clip_backend import ClipSentimentScorer, choose_device, load_clip
from evaluation_utils import (
    bootstrap_metrics,
    compute_faithfulness,
    explanation_concentration,
    labels_from_rows,
    length_group,
    score_in_batches,
    select_representative_sentences,
    subgroup_rows,
)
from io_utils import load_json, project_path, read_csv, save_json, sha256_file, write_csv
from plotting import (
    save_part6_classification,
    save_part6_efficiency,
    save_part6_example_comparison,
    save_part6_faithfulness,
    save_part6_global_importance,
    save_part6_subgroups,
)
from prompting import OUTPUT_NAMES, build_prototypes
from shap_utils import additivity, unpack_explanation
from word_masking import create_word_masker, split_word_units


DEFAULT_CONFIG = project_path("configs/part6_full_evaluation.json")
CONDITIONS = ("zero_shot_prompt", "frozen_linear_head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-sentences", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    part5 = load_json(project_path(config["part5_checks_path"]))
    budget_manifest = load_json(project_path(config["frozen_budget_manifest_path"]))
    if not part5.get("gate_passed"):
        raise RuntimeError("Part 5 gate must pass before final evaluation.")
    budget = int(budget_manifest["selected_budget"])
    if budget != int(part5["selected_budget"]):
        raise RuntimeError("Frozen SHAP budget and Part 5 checks disagree.")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    evaluation_path = project_path(config["evaluation_path"])
    evaluation_manifest = load_json(project_path(config["evaluation_manifest_path"]))
    expected_hash = evaluation_manifest["outputs"]["final_evaluation"]["sha256"]
    actual_hash = sha256_file(evaluation_path)
    if actual_hash != expected_hash:
        raise RuntimeError("The frozen final-evaluation dataset hash has changed.")
    rows = read_csv(evaluation_path)
    complete_run = args.max_sentences is None
    if args.max_sentences is not None:
        rows = rows[: args.max_sentences]
    if complete_run and len(rows) != 500:
        raise RuntimeError(f"Expected 500 final rows, found {len(rows)}.")

    output_root = project_path(config["output_dir"])
    values_dir, plots_dir, reports_dir = (
        output_root / "values",
        output_root / "plots",
        output_root / "reports",
    )
    for directory in (values_dir, plots_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.time()
    device = choose_device(str(config["device"]))
    print(f"Part 6: loading {config['model_name']} on {device}")
    model, tokenizer = load_clip(
        str(config["model_name"]), device, bool(config["local_files_only"])
    )
    prompt_manifest = load_json(project_path(config["frozen_prompt_manifest_path"]))
    prompt_family = {
        "NEG": prompt_manifest["selected_prompts"]["NEG"],
        "POS": prompt_manifest["selected_prompts"]["POS"],
    }
    prototypes, _ = build_prototypes(
        model, tokenizer, prompt_family, device, int(config["max_length"])
    )
    saved_head = np.load(project_path(config["trained_head_path"]))
    scorers = {
        "zero_shot_prompt": ClipSentimentScorer(
            model, tokenizer, prototypes, device, int(config["max_length"])
        ),
        "frozen_linear_head": FrozenClipLinearScorer(
            model,
            tokenizer,
            saved_head["weights"],
            saved_head["bias"],
            device,
            int(config["max_length"]),
        ),
    }
    texts = [row["text"] for row in rows]
    print("Scoring the frozen final dataset before any SHAP-based example inspection")
    all_scores = {
        condition: score_in_batches(scorer, texts, int(config["score_batch_size"]))
        for condition, scorer in scorers.items()
    }
    representatives = select_representative_sentences(
        rows, all_scores["zero_shot_prompt"], all_scores["frozen_linear_head"]
    )
    write_csv(values_dir / "presentation_selection.csv", representatives)
    representative_ids = {row["sentence_id"] for row in representatives}

    masker = create_word_masker()
    explainers = {
        condition: shap.Explainer(
            scorer,
            masker,
            algorithm="partition",
            output_names=list(OUTPUT_NAMES),
        )
        for condition, scorer in scorers.items()
    }
    sentence_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []
    faithfulness_rows: list[dict[str, Any]] = []
    example_store: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    total = len(CONDITIONS) * len(rows)
    progress = 0

    for condition_index, condition in enumerate(CONDITIONS):
        scorer = scorers[condition]
        explainer = explainers[condition]
        for row_index, row in enumerate(rows):
            progress += 1
            sentence_id, text, gold = row["sentence_id"], row["text"], row["gold_label"]
            if progress == 1 or progress % 10 == 0:
                print(f"Part 6 SHAP [{progress}/{total}] {condition} | {sentence_id}")
            units = split_word_units(text)
            content_mask = np.asarray([unit.is_content for unit in units], dtype=bool)
            scores = np.asarray(all_scores[condition][row_index], dtype=float)
            before = int(scorer.text_evaluations)
            shap_started = time.perf_counter()
            explanation = explainer(
                [text],
                max_evals=budget,
                batch_size=int(config["shap_batch_size"]),
                silent=True,
            )
            shap_runtime = time.perf_counter() - shap_started
            shap_evaluations = int(scorer.text_evaluations) - before
            _, values, bases = unpack_explanation(explanation, len(OUTPUT_NAMES))
            word_values = values[1:-1]
            if len(word_values) != len(units):
                raise ValueError(f"{condition}/{sentence_id}: whole-word alignment failed.")
            calculation = additivity(scores, bases, values)
            maximum_residual = float(np.max(np.abs(calculation["residual"])))
            margin_linearity_error = float(
                np.max(np.abs(values[:, 2] - (values[:, 1] - values[:, 0])))
            )
            prediction = "POS" if scores[2] > 0 else "NEG"
            correct = int(prediction == gold)
            faith_rows, faith_summary = compute_faithfulness(
                scorer,
                masker,
                text,
                word_values[:, 2],
                float(scores[2]),
                config["faithfulness_fractions"],
                int(config["random_baseline_repeats"]),
                seed + condition_index * 100_000 + row_index,
            )
            concentration = explanation_concentration(word_values[content_mask, 2])
            summary = {
                "condition": condition,
                "sentence_id": sentence_id,
                "text": text,
                "gold_label": gold,
                "prediction": prediction,
                "correct": correct,
                "source_row_id": row["source_row_id"],
                "word_count": len(units),
                "content_word_count": int(content_mask.sum()),
                "length_group": length_group(
                    int(content_mask.sum()),
                    int(config["short_max_words"]),
                    int(config["medium_max_words"]),
                ),
                "negative_score": float(scores[0]),
                "positive_score": float(scores[1]),
                "margin_score": float(scores[2]),
                "negative_base": float(bases[0]),
                "positive_base": float(bases[1]),
                "margin_base": float(bases[2]),
                "maximum_absolute_additivity_residual": maximum_residual,
                "maximum_margin_linearity_error": margin_linearity_error,
                "shap_runtime_seconds": shap_runtime,
                "shap_model_evaluations": shap_evaluations,
                **concentration,
                **faith_summary,
            }
            sentence_rows.append(summary)
            for word_index, unit in enumerate(units):
                word_rows.append(
                    {
                        "condition": condition,
                        "sentence_id": sentence_id,
                        "word_index": word_index,
                        "word": unit.text,
                        "is_content": int(unit.is_content),
                        "negative_shap": float(word_values[word_index, 0]),
                        "positive_shap": float(word_values[word_index, 1]),
                        "margin_shap": float(word_values[word_index, 2]),
                    }
                )
            for faith_row in faith_rows:
                faithfulness_rows.append(
                    {
                        "condition": condition,
                        "sentence_id": sentence_id,
                        "gold_label": gold,
                        "prediction": prediction,
                        "correct": correct,
                        **faith_row,
                    }
                )
            if sentence_id in representative_ids:
                example_store[sentence_id][condition] = {
                    "word_values": word_values,
                    "scores": scores,
                    "bases": bases,
                    "prediction": prediction,
                    "maximum_residual": maximum_residual,
                    "words": [unit.text for unit in units],
                }
        # Preserve a complete condition immediately in case a later model is interrupted.
        write_csv(values_dir / "sentence_results.csv", sentence_rows)
        write_csv(values_dir / "word_shap_values.csv", word_rows)
        write_csv(values_dir / "faithfulness_by_fraction.csv", faithfulness_rows)

    labels = labels_from_rows(rows)
    metric_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    subgroup_result_rows: list[dict[str, Any]] = []
    bootstrap_fields = [
        "top_deletion_aopc",
        "random_deletion_aopc",
        "mean_sufficiency_gap",
        "top_5_concentration",
        "shap_runtime_seconds",
    ]
    for condition_index, condition in enumerate(CONDITIONS):
        selected_sentences = [row for row in sentence_rows if row["condition"] == condition]
        metrics = classification_metrics(labels, all_scores[condition][:, :2])
        metric_rows.append({"condition": condition, **metrics})
        for confidence in bootstrap_metrics(
            labels,
            all_scores[condition],
            selected_sentences,
            bootstrap_fields,
            int(config["bootstrap_iterations"]),
            seed + condition_index,
        ):
            confidence_rows.append({"condition": condition, **confidence})
        subgroup_result_rows.extend(subgroup_rows(condition, selected_sentences))

    grouped_words: dict[tuple[str, str], list[float]] = defaultdict(list)
    signed_words: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in word_rows:
        if int(row["is_content"]):
            key = (str(row["condition"]), str(row["word"]).casefold())
            value = float(row["margin_shap"])
            grouped_words[key].append(abs(value))
            signed_words[key].append(value)
    global_word_rows = [
        {
            "condition": condition,
            "word": word,
            "occurrences": len(values),
            "mean_absolute_margin_shap": float(np.mean(values)),
            "mean_signed_margin_shap": float(np.mean(signed_words[(condition, word)])),
        }
        for (condition, word), values in grouped_words.items()
        if len(values) >= 2
    ]

    write_csv(values_dir / "classification_metrics.csv", metric_rows)
    write_csv(values_dir / "bootstrap_confidence_intervals.csv", confidence_rows)
    write_csv(values_dir / "subgroup_metrics.csv", subgroup_result_rows)
    write_csv(values_dir / "global_word_importance.csv", global_word_rows)

    save_part6_classification(plots_dir / "classification_metrics_and_confusion", metric_rows)
    save_part6_global_importance(plots_dir / "global_word_importance", global_word_rows)
    save_part6_faithfulness(plots_dir / "faithfulness_deletion_curves", faithfulness_rows)
    save_part6_subgroups(plots_dir / "subgroup_results", subgroup_result_rows)
    save_part6_efficiency(plots_dir / "efficiency_and_additivity", sentence_rows)
    for selection in representatives:
        sentence_id = selection["sentence_id"]
        result = example_store[sentence_id]
        if set(result) != set(CONDITIONS):
            raise RuntimeError(f"Missing presentation explanation for {sentence_id}.")
        save_part6_example_comparison(
            plots_dir / "presentation_examples" / f"{sentence_id}_clip_conditions",
            selection,
            result[CONDITIONS[0]]["words"],
            result,
        )

    maximum_residual = max(float(row["maximum_absolute_additivity_residual"]) for row in sentence_rows)
    maximum_linearity = max(float(row["maximum_margin_linearity_error"]) for row in sentence_rows)
    class_counts = {label: sum(row["gold_label"] == label for row in rows) for label in ("NEG", "POS")}
    png_stems = {path.relative_to(plots_dir).with_suffix("") for path in plots_dir.rglob("*.png")}
    pdf_stems = {path.relative_to(plots_dir).with_suffix("") for path in plots_dir.rglob("*.pdf")}
    checks = {
        "complete_run": complete_run,
        "evaluation_rows": len(rows),
        "class_counts": class_counts,
        "dataset_sha256": actual_hash,
        "frozen_shap_budget": budget,
        "conditions": list(CONDITIONS),
        "sentence_result_rows": len(sentence_rows),
        "word_result_rows": len(word_rows),
        "faithfulness_rows": len(faithfulness_rows),
        "presentation_examples": len(representatives),
        "aggregate_plot_pairs": 5,
        "maximum_additivity_residual": maximum_residual,
        "maximum_margin_linearity_error": maximum_linearity,
        "margin_linearity_tolerance": float(config["margin_linearity_tolerance"]),
        "margin_linearity_tolerance_rationale": "1e-6 covers float32 scorer arithmetic while remaining below the 1e-5 additivity tolerance.",
        "tests": {
            "part5_gate_passed": bool(part5["gate_passed"]),
            "dataset_hash_matches_manifest": actual_hash == expected_hash,
            "balanced_500_rows": len(rows) == 500 and class_counts == {"NEG": 250, "POS": 250},
            "both_conditions_complete": len(sentence_rows) == 2 * len(rows),
            "additivity": maximum_residual <= float(config["additivity_tolerance"]),
            "margin_linearity": maximum_linearity <= float(config["margin_linearity_tolerance"]),
            "five_pre_shap_examples": len(representatives) == 5
            and all(str(row["selection_uses_shap_values"]) in {"False", "false", "0"} for row in representatives),
            "plot_png_pdf_pairing": png_stems == pdf_stems and len(png_stems) == 10,
        },
        "runtime_seconds": time.time() - started,
    }
    checks["gate_passed"] = complete_run and all(checks["tests"].values())
    save_json(values_dir / "part6_checks.json", checks)
    save_json(values_dir / "run_config.json", {**config, "device_resolved": str(device), "frozen_shap_budget": budget})
    result_manifest = {
        "evaluation_dataset_sha256": actual_hash,
        "sentence_results_sha256": sha256_file(values_dir / "sentence_results.csv"),
        "word_shap_values_sha256": sha256_file(values_dir / "word_shap_values.csv"),
        "faithfulness_sha256": sha256_file(values_dir / "faithfulness_by_fraction.csv"),
        "classification_metrics_sha256": sha256_file(values_dir / "classification_metrics.csv"),
        "frozen_shap_budget": budget,
        "conditions": list(CONDITIONS),
        "gate_passed": checks["gate_passed"],
    }
    save_json(project_path("configs/frozen_part6_results_manifest.json"), result_manifest)

    zero = next(row for row in metric_rows if row["condition"] == "zero_shot_prompt")
    trained = next(row for row in metric_rows if row["condition"] == "frozen_linear_head")
    report = f"""# Part 6 Results — Final 500-Sentence CLIP Evaluation

Both frozen CLIP conditions were evaluated exactly once on the previously untouched,
balanced 500-sentence SST-2 development subset. Whole-word Partition SHAP used the
Part 5 maximum budget of {budget} evaluations.

| Condition | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Zero-shot prompt ensemble | {zero['accuracy']:.4f} | {zero['macro_f1']:.4f} | {zero['balanced_accuracy']:.4f} |
| Frozen CLIP + linear head | {trained['accuracy']:.4f} | {trained['macro_f1']:.4f} | {trained['balanced_accuracy']:.4f} |

## Explanation verification

- Maximum additivity residual: `{maximum_residual:.3e}`
- Maximum direct-versus-derived margin SHAP error: `{maximum_linearity:.3e}`
- Complete per-sentence rows: `{len(sentence_rows)}`
- Faithfulness fraction rows: `{len(faithfulness_rows)}`
- Five presentation examples were selected without inspecting SHAP values.
- Part 6 gate passed: **{checks['gate_passed']}**

Bootstrap confidence intervals, subgroup results, word values and faithfulness
measurements are stored as complete CSV tables. Raw score magnitudes are interpreted
within condition because cosine similarities and trained-head logits have different
scales.
"""
    (reports_dir / "PART6_RESULTS.md").write_text(report, encoding="utf-8")
    (output_root.parent / "PART6_SUMMARY.md").write_text(report, encoding="utf-8")
    print(f"Part 6 gate passed: {checks['gate_passed']}")
    print(f"Accuracy: zero-shot={zero['accuracy']:.4f}, trained={trained['accuracy']:.4f}")
    if complete_run and not checks["gate_passed"]:
        raise RuntimeError("Part 6 gate failed; Part 7 must not proceed.")


if __name__ == "__main__":
    main()
