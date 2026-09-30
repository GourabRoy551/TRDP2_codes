"""Part 5: select a reliable whole-word Partition-SHAP evaluation budget."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch
from scipy.stats import spearmanr

from classifier_head import FrozenClipLinearScorer
from clip_backend import ClipSentimentScorer, choose_device, load_clip
from io_utils import load_json, project_path, read_csv, save_json, sha256_file, write_csv
from plotting import (
    save_budget_additivity,
    save_budget_efficiency,
    save_budget_selection_summary,
    save_budget_stability,
)
from prompting import OUTPUT_NAMES, build_prototypes
from shap_utils import additivity, unpack_explanation
from word_masking import create_word_masker, split_word_units


DEFAULT_CONFIG = project_path("configs/part5_shap_budget.json")
CONDITIONS = ("zero_shot_prompt", "frozen_linear_head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Development-only smoke-test option; omit for the approved 40 sentences.",
    )
    return parser.parse_args()


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Return a stable importance-rank correlation for short or tied vectors."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size <= 1 or np.allclose(left, right, atol=1e-15, rtol=0.0):
        return 1.0
    result = float(spearmanr(left, right).statistic)
    return result if np.isfinite(result) else 0.0


def important_indices(values: np.ndarray, k: int) -> set[int]:
    count = min(int(k), len(values))
    return set(np.argsort(-np.abs(values), kind="stable")[:count].tolist())


def sign_agreement(left: np.ndarray, right: np.ndarray, tolerance: float = 1e-12) -> float:
    def tolerant_sign(values: np.ndarray) -> np.ndarray:
        return np.where(values > tolerance, 1, np.where(values < -tolerance, -1, 0))

    if not len(left):
        return 1.0
    return float(np.mean(tolerant_sign(left) == tolerant_sign(right)))


def explain(
    explainer: Any,
    scorer: Any,
    text: str,
    budget: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Run one explanation and return full features, bases and cost measures."""
    before = int(scorer.text_evaluations)
    started = time.perf_counter()
    explanation = explainer(
        [text], max_evals=budget, batch_size=batch_size, silent=True
    )
    runtime = time.perf_counter() - started
    evaluations = int(scorer.text_evaluations) - before
    _, values, bases = unpack_explanation(explanation, len(OUTPUT_NAMES))
    scores = np.asarray(scorer([text])[0], dtype=float)
    return scores, values, bases, runtime, evaluations


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    part3 = load_json(project_path(config["part3_checks_path"]))
    part4 = load_json(project_path(config["part4_checks_path"]))
    if not part3.get("gate_passed") or not part4.get("gate_passed"):
        raise RuntimeError("Parts 3 and 4 must pass before Part 5 can start.")

    budgets = sorted({int(value) for value in config["budgets"]})
    reference_budget = int(config["reference_budget"])
    if reference_budget not in budgets:
        raise ValueError("The reference budget must be included in budgets.")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    stability_path = project_path(config["stability_path"])
    stability_rows = read_csv(stability_path)
    complete_run = args.max_sentences is None
    if args.max_sentences is not None:
        stability_rows = stability_rows[: args.max_sentences]
    if not stability_rows:
        raise ValueError("The Part 5 stability set is empty.")

    output_root = project_path(config["output_dir"])
    values_dir = output_root / "values"
    plots_dir = output_root / "plots"
    reports_dir = output_root / "reports"
    for directory in (values_dir, plots_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.time()
    device = choose_device(str(config["device"]))
    print(f"Part 5: loading {config['model_name']} on {device}")
    model, tokenizer = load_clip(
        str(config["model_name"]), device, bool(config["local_files_only"])
    )

    frozen_prompts = load_json(project_path(config["frozen_prompt_manifest_path"]))
    prompt_family = {
        "NEG": frozen_prompts["selected_prompts"]["NEG"],
        "POS": frozen_prompts["selected_prompts"]["POS"],
    }
    prototypes, _ = build_prototypes(
        model, tokenizer, prompt_family, device, int(config["max_length"])
    )
    head = np.load(project_path(config["trained_head_path"]))
    if list(head["class_names"].astype(str)) != ["NEG", "POS"]:
        raise ValueError("The saved trained head does not use NEG/POS class order.")

    scorers = {
        "zero_shot_prompt": ClipSentimentScorer(
            model, tokenizer, prototypes, device, int(config["max_length"])
        ),
        "frozen_linear_head": FrozenClipLinearScorer(
            model,
            tokenizer,
            head["weights"],
            head["bias"],
            device,
            int(config["max_length"]),
        ),
    }
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

    run_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []
    result_store: dict[tuple[str, str, int], dict[str, Any]] = {}
    total = len(CONDITIONS) * len(stability_rows) * len(budgets)
    progress = 0

    for condition_index, condition in enumerate(CONDITIONS):
        scorer = scorers[condition]
        explainer = explainers[condition]
        for sentence_index, row in enumerate(stability_rows):
            sentence_id = row["sentence_id"]
            text = row["text"]
            gold = row["gold_label"]
            units = split_word_units(text)
            content_mask = np.asarray([unit.is_content for unit in units], dtype=bool)
            for budget in budgets:
                progress += 1
                print(
                    f"Part 5 [{progress}/{total}] {condition} | "
                    f"{sentence_id} | max_evals={budget}"
                )
                # Resetting the seed makes any stochastic implementation detail
                # reproducible without sharing randomness between sentences.
                np.random.seed(seed + condition_index * 10_000 + sentence_index)
                scores, values, bases, runtime, evaluations = explain(
                    explainer,
                    scorer,
                    text,
                    budget,
                    int(config["batch_size"]),
                )
                direct_values = values[1:-1]
                if len(direct_values) != len(units):
                    raise ValueError(
                        f"{condition}/{sentence_id}: whole-word SHAP alignment failed."
                    )
                calculation = additivity(scores, bases, values)
                prediction = "POS" if scores[2] > 0 else "NEG"
                maximum_residual = float(np.max(np.abs(calculation["residual"])))
                run_rows.append(
                    {
                        "condition": condition,
                        "sentence_id": sentence_id,
                        "text": text,
                        "gold_label": gold,
                        "prediction": prediction,
                        "correct": int(prediction == gold),
                        "budget": budget,
                        "word_count": len(units),
                        "content_word_count": int(content_mask.sum()),
                        "negative_score": float(scores[0]),
                        "positive_score": float(scores[1]),
                        "margin_score": float(scores[2]),
                        "negative_base": float(bases[0]),
                        "positive_base": float(bases[1]),
                        "margin_base": float(bases[2]),
                        "negative_residual": float(calculation["residual"][0]),
                        "positive_residual": float(calculation["residual"][1]),
                        "margin_residual": float(calculation["residual"][2]),
                        "maximum_absolute_additivity_residual": maximum_residual,
                        "runtime_seconds": runtime,
                        "actual_model_evaluations": evaluations,
                    }
                )
                for word_index, unit in enumerate(units):
                    word_rows.append(
                        {
                            "condition": condition,
                            "sentence_id": sentence_id,
                            "budget": budget,
                            "word_index": word_index,
                            "word": unit.text,
                            "is_content": int(unit.is_content),
                            "negative_shap": float(direct_values[word_index, 0]),
                            "positive_shap": float(direct_values[word_index, 1]),
                            "margin_shap": float(direct_values[word_index, 2]),
                        }
                    )
                result_store[(condition, sentence_id, budget)] = {
                    "scores": scores,
                    "values": values,
                    "word_values": direct_values,
                    "bases": bases,
                    "content_mask": content_mask,
                    "words": [unit.text for unit in units],
                }

    comparison_rows: list[dict[str, Any]] = []
    top_k = int(config["top_k"])
    for condition in CONDITIONS:
        for row in stability_rows:
            sentence_id = row["sentence_id"]
            reference = result_store[(condition, sentence_id, reference_budget)]
            reference_margin = reference["word_values"][:, 2][reference["content_mask"]]
            reference_top = important_indices(reference_margin, top_k)
            reference_dominant = int(np.argmax(np.abs(reference_margin))) if len(reference_margin) else -1
            for budget in budgets:
                current = result_store[(condition, sentence_id, budget)]
                current_margin = current["word_values"][:, 2][current["content_mask"]]
                current_top = important_indices(current_margin, top_k)
                union = current_top | reference_top
                overlap_denominator = max(1, min(top_k, len(reference_margin)))
                current_dominant = int(np.argmax(np.abs(current_margin))) if len(current_margin) else -1
                comparison_rows.append(
                    {
                        "condition": condition,
                        "sentence_id": sentence_id,
                        "budget": budget,
                        "reference_budget": reference_budget,
                        "abs_rank_spearman": safe_spearman(
                            np.abs(current_margin), np.abs(reference_margin)
                        ),
                        "top_k": top_k,
                        "top_k_overlap_count": len(current_top & reference_top),
                        "top_k_overlap_rate": len(current_top & reference_top) / overlap_denominator,
                        "top_k_jaccard": len(current_top & reference_top) / len(union) if union else 1.0,
                        "sign_agreement": sign_agreement(current_margin, reference_margin),
                        "maximum_absolute_margin_shap_difference": float(
                            np.max(np.abs(current_margin - reference_margin))
                            if len(current_margin)
                            else 0.0
                        ),
                        "mean_absolute_margin_shap_difference": float(
                            np.mean(np.abs(current_margin - reference_margin))
                            if len(current_margin)
                            else 0.0
                        ),
                        "dominant_word_same": int(current_dominant == reference_dominant),
                        "dominant_direction_same": int(
                            current_dominant == reference_dominant
                            and (
                                current_dominant < 0
                                or np.sign(current_margin[current_dominant])
                                == np.sign(reference_margin[reference_dominant])
                            )
                        ),
                        "prediction_unchanged": int(
                            np.sign(current["scores"][2]) == np.sign(reference["scores"][2])
                        ),
                    }
                )

    threshold_values = {
        "spearman": float(config["minimum_median_spearman"]),
        "top_k_overlap": float(config["minimum_median_top_k_overlap"]),
        "sign_agreement": float(config["minimum_median_sign_agreement"]),
    }
    summary_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for budget in budgets:
            comparisons = [
                row
                for row in comparison_rows
                if row["condition"] == condition and int(row["budget"]) == budget
            ]
            runs = [
                row
                for row in run_rows
                if row["condition"] == condition and int(row["budget"]) == budget
            ]
            median_spearman = float(np.median([row["abs_rank_spearman"] for row in comparisons]))
            median_overlap = float(np.median([row["top_k_overlap_rate"] for row in comparisons]))
            median_sign = float(np.median([row["sign_agreement"] for row in comparisons]))
            maximum_residual = max(float(row["maximum_absolute_additivity_residual"]) for row in runs)
            thresholds_passed = {
                "spearman_passed": median_spearman >= threshold_values["spearman"],
                "top_k_overlap_passed": median_overlap >= threshold_values["top_k_overlap"],
                "sign_agreement_passed": median_sign >= threshold_values["sign_agreement"],
                "additivity_passed": maximum_residual <= float(config["additivity_tolerance"]),
                "prediction_conclusion_unchanged": all(int(row["prediction_unchanged"]) == 1 for row in comparisons),
            }
            summary_rows.append(
                {
                    "condition": condition,
                    "budget": budget,
                    "sentences": len(runs),
                    "correct_predictions": sum(int(row["correct"]) for row in runs),
                    "incorrect_predictions": sum(1 - int(row["correct"]) for row in runs),
                    "median_abs_rank_spearman": median_spearman,
                    "mean_abs_rank_spearman": float(np.mean([row["abs_rank_spearman"] for row in comparisons])),
                    "median_top_k_overlap_rate": median_overlap,
                    "mean_top_k_overlap_rate": float(np.mean([row["top_k_overlap_rate"] for row in comparisons])),
                    "median_top_k_jaccard": float(np.median([row["top_k_jaccard"] for row in comparisons])),
                    "median_sign_agreement": median_sign,
                    "mean_sign_agreement": float(np.mean([row["sign_agreement"] for row in comparisons])),
                    "dominant_word_agreement_rate": float(np.mean([row["dominant_word_same"] for row in comparisons])),
                    "dominant_direction_agreement_rate": float(np.mean([row["dominant_direction_same"] for row in comparisons])),
                    "maximum_margin_shap_difference": max(float(row["maximum_absolute_margin_shap_difference"]) for row in comparisons),
                    "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in runs])),
                    "median_runtime_seconds": float(np.median([row["runtime_seconds"] for row in runs])),
                    "mean_actual_model_evaluations": float(np.mean([row["actual_model_evaluations"] for row in runs])),
                    "maximum_additivity_residual": maximum_residual,
                    **thresholds_passed,
                    "all_thresholds_passed": all(thresholds_passed.values()),
                }
            )

    eligible = []
    for budget in budgets:
        rows_at_budget = [row for row in summary_rows if int(row["budget"]) == budget]
        if len(rows_at_budget) == len(CONDITIONS) and all(
            bool(row["all_thresholds_passed"]) for row in rows_at_budget
        ):
            eligible.append(budget)
    if not eligible:
        raise RuntimeError("No tested SHAP budget satisfied the stability gate.")
    selected_budget = min(eligible)

    determinism_rows: list[dict[str, Any]] = []
    requested_repeat_ids = set(config["determinism_sentence_ids"])
    repeat_rows = [row for row in stability_rows if row["sentence_id"] in requested_repeat_ids]
    if not repeat_rows:
        repeat_rows = stability_rows[: min(5, len(stability_rows))]
    for condition_index, condition in enumerate(CONDITIONS):
        scorer = scorers[condition]
        explainer = explainers[condition]
        for sentence_index, row in enumerate(repeat_rows):
            sentence_id = row["sentence_id"]
            print(f"Determinism repeat: {condition} | {sentence_id} | {selected_budget}")
            np.random.seed(seed + condition_index * 10_000 + stability_rows.index(row))
            scores, values, bases, runtime, evaluations = explain(
                explainer,
                scorer,
                row["text"],
                selected_budget,
                int(config["batch_size"]),
            )
            original = result_store[(condition, sentence_id, selected_budget)]
            determinism_rows.append(
                {
                    "condition": condition,
                    "sentence_id": sentence_id,
                    "budget": selected_budget,
                    "maximum_absolute_shap_difference": float(np.max(np.abs(values - original["values"]))),
                    "maximum_absolute_base_difference": float(np.max(np.abs(bases - original["bases"]))),
                    "maximum_absolute_score_difference": float(np.max(np.abs(scores - original["scores"]))),
                    "repeat_runtime_seconds": runtime,
                    "repeat_actual_model_evaluations": evaluations,
                }
            )

    maximum_repeat_difference = max(
        max(
            float(row["maximum_absolute_shap_difference"]),
            float(row["maximum_absolute_base_difference"]),
            float(row["maximum_absolute_score_difference"]),
        )
        for row in determinism_rows
    )

    write_csv(values_dir / "budget_run_summary.csv", run_rows)
    write_csv(values_dir / "budget_word_shap_values.csv", word_rows)
    write_csv(values_dir / "budget_comparisons.csv", comparison_rows)
    write_csv(values_dir / "budget_stability_summary.csv", summary_rows)
    write_csv(values_dir / "determinism_checks.csv", determinism_rows)
    save_budget_stability(
        plots_dir / "stability_metrics",
        summary_rows,
        reference_budget,
        threshold_values,
    )
    save_budget_efficiency(plots_dir / "runtime_and_evaluations", summary_rows)
    save_budget_additivity(plots_dir / "additivity_matrix", summary_rows)
    save_budget_selection_summary(
        plots_dir / "budget_selection_summary",
        summary_rows,
        selected_budget,
        determinism_rows,
    )

    selected_summaries = [
        row for row in summary_rows if int(row["budget"]) == selected_budget
    ]
    expected_runs = len(CONDITIONS) * len(stability_rows) * len(budgets)
    checks = {
        "complete_run": complete_run,
        "stability_sentences": len(stability_rows),
        "conditions": list(CONDITIONS),
        "budgets": budgets,
        "reference_budget": reference_budget,
        "selected_budget": selected_budget,
        "run_rows": len(run_rows),
        "expected_run_rows": expected_runs,
        "comparison_rows": len(comparison_rows),
        "determinism_rows": len(determinism_rows),
        "maximum_determinism_difference": maximum_repeat_difference,
        "thresholds": threshold_values,
        "additivity_tolerance": float(config["additivity_tolerance"]),
        "final_evaluation_used": False,
        "tests": {
            "part3_gate_passed": bool(part3["gate_passed"]),
            "part4_gate_passed": bool(part4["gate_passed"]),
            "all_requested_runs_present": len(run_rows) == expected_runs,
            "both_conditions_pass_selected_budget": len(selected_summaries) == len(CONDITIONS)
            and all(bool(row["all_thresholds_passed"]) for row in selected_summaries),
            "smallest_passing_budget_selected": selected_budget == min(eligible),
            "deterministic_repeats": maximum_repeat_difference
            <= float(config["determinism_tolerance"]),
            "final_evaluation_not_used": True,
        },
        "runtime_seconds": time.time() - started,
    }
    checks["gate_passed"] = complete_run and all(checks["tests"].values())
    save_json(values_dir / "part5_checks.json", checks)
    save_json(values_dir / "run_config.json", {**config, "device_resolved": str(device)})

    manifest = {
        "selected_budget": selected_budget,
        "reference_budget": reference_budget,
        "batch_size": int(config["batch_size"]),
        "masking_level": "whole_word",
        "algorithm": "partition",
        "explained_outputs": list(OUTPUT_NAMES),
        "conditions_validated": list(CONDITIONS),
        "thresholds": threshold_values,
        "additivity_tolerance": float(config["additivity_tolerance"]),
        "selection_dataset": str(config["stability_path"]),
        "selection_dataset_sha256": sha256_file(stability_path),
        "selection_uses_final_evaluation": False,
        "eligible_budgets": eligible,
    }
    save_json(project_path("configs/frozen_shap_budget_manifest.json"), manifest)

    summary_lines = []
    for row in selected_summaries:
        summary_lines.append(
            f"- `{row['condition']}`: median Spearman `{row['median_abs_rank_spearman']:.3f}`, "
            f"top-{top_k} overlap `{row['median_top_k_overlap_rate']:.3f}`, "
            f"sign agreement `{row['median_sign_agreement']:.3f}`, "
            f"mean Spearman `{row['mean_abs_rank_spearman']:.3f}`, "
            f"mean top-{top_k} overlap `{row['mean_top_k_overlap_rate']:.3f}`, "
            f"dominant-word agreement `{row['dominant_word_agreement_rate']:.3f}`, "
            f"mean runtime `{row['mean_runtime_seconds']:.3f}` s, "
            f"mean actual evaluations `{row['mean_actual_model_evaluations']:.1f}`."
        )
    report = f"""# Part 5 Results — Whole-Word SHAP Budget and Stability

Partition SHAP was measured on all {len(stability_rows)} development-only stability
sentences for the zero-shot prompt model and the frozen-CLIP linear head. Requested
budgets were {', '.join(map(str, budgets))}; {reference_budget} evaluations was the
predeclared reference. The final 500-sentence evaluation set was not used.

## Selected budget

The smallest budget satisfying every predeclared criterion for both conditions was
**{selected_budget} maximum evaluations per sentence**.

{chr(10).join(summary_lines)}

## Verification

- Maximum deterministic-repeat difference: `{maximum_repeat_difference:.3e}`
- Part 3 gate inherited: `{part3['gate_passed']}`
- Part 4 gate inherited: `{part4['gate_passed']}`
- Part 5 gate passed: **{checks['gate_passed']}**

The selected value is a maximum budget. Partition SHAP may perform fewer actual model
evaluations for short sentences when the hierarchical coalition tree is exhausted.
The zero-shot cosine similarities and trained-head logits use different numerical
scales, so stability is assessed within each condition rather than by comparing raw
SHAP magnitudes between the two models.
"""
    (reports_dir / "PART5_RESULTS.md").write_text(report, encoding="utf-8")
    (output_root.parent / "PART5_SUMMARY.md").write_text(report, encoding="utf-8")

    print(f"Part 5 selected budget: {selected_budget}")
    print(f"Part 5 gate passed: {checks['gate_passed']}")
    if complete_run and not checks["gate_passed"]:
        raise RuntimeError("Part 5 gate failed; Part 6 must not proceed.")


if __name__ == "__main__":
    main()
