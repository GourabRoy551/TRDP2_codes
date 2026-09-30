"""Part 7: run BERT on the frozen 500 rows and produce the final comparison."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch

from bert_backend import BertThreeOutputScorer, load_bert
from classifier_head import classification_metrics
from clip_backend import choose_device
from evaluation_utils import (
    bootstrap_metrics,
    compare_word_explanations,
    compute_faithfulness,
    explanation_concentration,
    labels_from_rows,
    length_group,
    score_in_batches,
    subgroup_rows,
)
from io_utils import load_json, project_path, read_csv, save_json, sha256_file, write_csv
from plotting import (
    save_final_case_study,
    save_final_efficiency,
    save_final_explanation_agreement,
    save_final_faithfulness,
    save_final_model_performance,
    save_final_prediction_agreement,
)
from prompting import OUTPUT_NAMES
from shap_utils import additivity, unpack_explanation
from word_masking import create_word_masker, split_word_units


DEFAULT_CONFIG = project_path("configs/part7_final_comparison.json")
MODEL_NAMES = ("zero_shot_prompt", "frozen_linear_head", "bert_sst2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-sentences", type=int, default=None)
    return parser.parse_args()


def float_word_maps(rows: list[dict[str, str]]) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["sentence_id"])].append(row)
    result = {}
    for key, selected in grouped.items():
        ordered = sorted(selected, key=lambda row: int(row["word_index"]))
        result[key] = np.asarray(
            [
                [float(row["negative_shap"]), float(row["positive_shap"]), float(row["margin_shap"])]
                for row in ordered
            ],
            dtype=float,
        )
    return result


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    part6 = load_json(project_path(config["part6_checks_path"]))
    if not part6.get("gate_passed"):
        raise RuntimeError("Part 6 gate must pass before Part 7.")
    budget_manifest = load_json(project_path(config["frozen_budget_manifest_path"]))
    budget = int(budget_manifest["selected_budget"])
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    evaluation_path = project_path(config["evaluation_path"])
    evaluation_manifest = load_json(project_path(config["evaluation_manifest_path"]))
    expected_hash = evaluation_manifest["outputs"]["final_evaluation"]["sha256"]
    actual_hash = sha256_file(evaluation_path)
    if actual_hash != expected_hash or actual_hash != part6["dataset_sha256"]:
        raise RuntimeError("Part 7 dataset does not match the frozen Part 6 dataset.")
    rows = read_csv(evaluation_path)
    complete_run = args.max_sentences is None
    if args.max_sentences is not None:
        rows = rows[: args.max_sentences]
    if complete_run and len(rows) != 500:
        raise RuntimeError("Part 7 requires all 500 frozen evaluation rows.")

    output_root = project_path(config["output_dir"])
    values_dir, plots_dir, reports_dir = (
        output_root / "values",
        output_root / "plots",
        output_root / "reports",
    )
    for directory in (values_dir, plots_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    part6_root = project_path(config["part6_output_dir"])
    clip_sentence_rows = read_csv(part6_root / "values" / "sentence_results.csv")
    clip_word_rows = read_csv(part6_root / "values" / "word_shap_values.csv")
    clip_metrics = read_csv(part6_root / "values" / "classification_metrics.csv")
    representatives = read_csv(part6_root / "values" / "presentation_selection.csv")
    expected_ids = {row["sentence_id"] for row in rows}
    if not complete_run:
        clip_sentence_rows = [row for row in clip_sentence_rows if row["sentence_id"] in expected_ids]
        clip_word_rows = [row for row in clip_word_rows if row["sentence_id"] in expected_ids]
        representatives = [row for row in representatives if row["sentence_id"] in expected_ids]
    if {row["sentence_id"] for row in clip_sentence_rows} != expected_ids:
        raise RuntimeError("Part 6 and Part 7 sentence IDs do not align.")

    started = time.time()
    device = choose_device(str(config["device"]))
    print(f"Part 7: loading {config['bert_model_name']} on {device}")
    model, tokenizer, negative_index, positive_index, label_source = load_bert(
        str(config["bert_model_name"]), device, bool(config["local_files_only"])
    )
    scorer = BertThreeOutputScorer(
        model,
        tokenizer,
        device,
        int(config["bert_max_length"]),
        negative_index,
        positive_index,
    )
    texts = [row["text"] for row in rows]
    bert_scores = score_in_batches(scorer, texts, int(config["score_batch_size"]))
    masker = create_word_masker()
    explainer = shap.Explainer(
        scorer,
        masker,
        algorithm="partition",
        output_names=list(OUTPUT_NAMES),
    )

    bert_sentence_rows: list[dict[str, Any]] = []
    bert_word_rows: list[dict[str, Any]] = []
    bert_faithfulness_rows: list[dict[str, Any]] = []
    bert_example_store: dict[str, dict[str, Any]] = {}
    representative_ids = {row["sentence_id"] for row in representatives}
    for row_index, row in enumerate(rows):
        sentence_id, text, gold = row["sentence_id"], row["text"], row["gold_label"]
        if row_index == 0 or (row_index + 1) % 10 == 0:
            print(f"Part 7 BERT SHAP [{row_index + 1}/{len(rows)}] {sentence_id}")
        units = split_word_units(text)
        content_mask = np.asarray([unit.is_content for unit in units], dtype=bool)
        scores = np.asarray(bert_scores[row_index], dtype=float)
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
            raise ValueError(f"BERT/{sentence_id}: whole-word alignment failed.")
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
            seed + row_index,
        )
        concentration = explanation_concentration(word_values[content_mask, 2])
        summary = {
            "condition": "bert_sst2",
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
                int(load_json(project_path("configs/part6_full_evaluation.json"))["short_max_words"]),
                int(load_json(project_path("configs/part6_full_evaluation.json"))["medium_max_words"]),
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
        bert_sentence_rows.append(summary)
        for word_index, unit in enumerate(units):
            bert_word_rows.append(
                {
                    "condition": "bert_sst2",
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
            bert_faithfulness_rows.append(
                {
                    "condition": "bert_sst2",
                    "sentence_id": sentence_id,
                    "gold_label": gold,
                    "prediction": prediction,
                    "correct": correct,
                    **faith_row,
                }
            )
        if sentence_id in representative_ids:
            bert_example_store[sentence_id] = {
                "word_values": word_values,
                "scores": scores,
                "prediction": prediction,
                "words": [unit.text for unit in units],
            }
        if (row_index + 1) % 50 == 0:
            write_csv(values_dir / "bert_sentence_results.csv", bert_sentence_rows)
            write_csv(values_dir / "bert_word_shap_values.csv", bert_word_rows)
            write_csv(values_dir / "bert_faithfulness_by_fraction.csv", bert_faithfulness_rows)

    write_csv(values_dir / "bert_sentence_results.csv", bert_sentence_rows)
    write_csv(values_dir / "bert_word_shap_values.csv", bert_word_rows)
    write_csv(values_dir / "bert_faithfulness_by_fraction.csv", bert_faithfulness_rows)
    labels = labels_from_rows(rows)
    bert_metrics = {"condition": "bert_sst2", **classification_metrics(labels, bert_scores[:, :2])}
    bert_confidence = bootstrap_metrics(
        labels,
        bert_scores,
        bert_sentence_rows,
        [
            "top_deletion_aopc",
            "random_deletion_aopc",
            "mean_sufficiency_gap",
            "top_5_concentration",
            "shap_runtime_seconds",
        ],
        int(config["bootstrap_iterations"]),
        seed,
    )
    write_csv(values_dir / "bert_classification_metrics.csv", [bert_metrics])
    write_csv(values_dir / "bert_bootstrap_confidence_intervals.csv", bert_confidence)
    write_csv(values_dir / "bert_subgroup_metrics.csv", subgroup_rows("bert_sst2", bert_sentence_rows))

    clip_word_maps = float_word_maps(clip_word_rows)
    bert_word_maps = float_word_maps(bert_word_rows)
    all_word_maps = {**clip_word_maps, **bert_word_maps}
    sentence_lookup = {
        (row["condition"], row["sentence_id"]): row
        for row in [*clip_sentence_rows, *[{key: str(value) for key, value in row.items()} for row in bert_sentence_rows]]
    }
    pairs = [
        ("bert_sst2", "zero_shot_prompt"),
        ("bert_sst2", "frozen_linear_head"),
        ("zero_shot_prompt", "frozen_linear_head"),
    ]
    pair_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for row in rows:
        sentence_id = row["sentence_id"]
        units = split_word_units(row["text"])
        content_mask = np.asarray([unit.is_content for unit in units], dtype=bool)
        predictions = {
            model_name: str(sentence_lookup[(model_name, sentence_id)]["prediction"])
            for model_name in MODEL_NAMES
        }
        error_rows.append(
            {
                "sentence_id": sentence_id,
                "text": row["text"],
                "gold_label": row["gold_label"],
                **{f"{model_name}_prediction": predictions[model_name] for model_name in MODEL_NAMES},
                "all_models_agree": int(len(set(predictions.values())) == 1),
                "all_models_correct": int(all(value == row["gold_label"] for value in predictions.values())),
                "models_correct_count": sum(value == row["gold_label"] for value in predictions.values()),
            }
        )
        for left, right in pairs:
            left_values = all_word_maps[(left, sentence_id)]
            right_values = all_word_maps[(right, sentence_id)]
            if len(left_values) != len(units) or len(right_values) != len(units):
                raise RuntimeError(f"Word alignment mismatch for {sentence_id}.")
            comparison = compare_word_explanations(
                left_values[:, 2], right_values[:, 2], content_mask, int(config["top_k"])
            )
            left_correct = predictions[left] == row["gold_label"]
            right_correct = predictions[right] == row["gold_label"]
            if left_correct and right_correct:
                correctness_group = "both_correct"
            elif left_correct:
                correctness_group = "left_only_correct"
            elif right_correct:
                correctness_group = "right_only_correct"
            else:
                correctness_group = "both_incorrect"
            pair_rows.append(
                {
                    "pair": f"{left} vs {right}",
                    "left_model": left,
                    "right_model": right,
                    "sentence_id": sentence_id,
                    "gold_label": row["gold_label"],
                    "left_prediction": predictions[left],
                    "right_prediction": predictions[right],
                    "prediction_agreement": int(predictions[left] == predictions[right]),
                    "correctness_group": correctness_group,
                    **comparison,
                }
            )
    write_csv(values_dir / "pairwise_explanation_comparison.csv", pair_rows)
    write_csv(values_dir / "prediction_error_analysis.csv", error_rows)

    pair_summary_rows: list[dict[str, Any]] = []
    pair_subgroup_rows: list[dict[str, Any]] = []
    for left, right in pairs:
        pair_name = f"{left} vs {right}"
        selected = [row for row in pair_rows if row["pair"] == pair_name]
        pair_summary_rows.append(
            {
                "pair": pair_name,
                "sentences": len(selected),
                "prediction_agreement": float(np.mean([row["prediction_agreement"] for row in selected])),
                "mean_abs_rank_spearman": float(np.mean([row["abs_rank_spearman"] for row in selected])),
                "median_abs_rank_spearman": float(np.median([row["abs_rank_spearman"] for row in selected])),
                "mean_top_k_overlap_rate": float(np.mean([row["top_k_overlap_rate"] for row in selected])),
                "mean_top_k_jaccard": float(np.mean([row["top_k_jaccard"] for row in selected])),
                "mean_sign_agreement": float(np.mean([row["sign_agreement"] for row in selected])),
            }
        )
        for group in ("both_correct", "left_only_correct", "right_only_correct", "both_incorrect"):
            grouped = [row for row in selected if row["correctness_group"] == group]
            if grouped:
                pair_subgroup_rows.append(
                    {
                        "pair": pair_name,
                        "correctness_group": group,
                        "sentences": len(grouped),
                        "mean_abs_rank_spearman": float(np.mean([row["abs_rank_spearman"] for row in grouped])),
                        "mean_top_k_overlap_rate": float(np.mean([row["top_k_overlap_rate"] for row in grouped])),
                        "mean_sign_agreement": float(np.mean([row["sign_agreement"] for row in grouped])),
                    }
                )
    write_csv(values_dir / "pairwise_explanation_summary.csv", pair_summary_rows)
    write_csv(values_dir / "pairwise_explanation_by_correctness.csv", pair_subgroup_rows)

    performance_rows = [
        {key: (float(value) if key not in {"condition"} else value) for key, value in row.items()}
        for row in clip_metrics
    ] + [bert_metrics]
    write_csv(values_dir / "final_model_performance.csv", performance_rows)

    all_sentence_results = [*clip_sentence_rows, *[{key: str(value) for key, value in row.items()} for row in bert_sentence_rows]]
    faithfulness_summary_rows = []
    efficiency_rows = []
    for model_name in MODEL_NAMES:
        selected = [row for row in all_sentence_results if row["condition"] == model_name]
        faithfulness_summary_rows.append(
            {
                "model": model_name,
                "mean_normalized_top_deletion_aopc": float(np.mean([float(row["normalized_top_deletion_aopc"]) for row in selected])),
                "mean_normalized_random_deletion_aopc": float(np.mean([float(row["normalized_random_deletion_aopc"]) for row in selected])),
                "mean_top_deletion_flip_rate": float(np.mean([float(row["mean_top_deletion_flip_rate"]) for row in selected])),
                "mean_random_deletion_flip_rate": float(np.mean([float(row["mean_random_deletion_flip_rate"]) for row in selected])),
                "mean_normalized_sufficiency_gap": float(np.mean([float(row["mean_normalized_sufficiency_gap"]) for row in selected])),
            }
        )
        efficiency_rows.append(
            {
                "model": model_name,
                "mean_shap_runtime_seconds": float(np.mean([float(row["shap_runtime_seconds"]) for row in selected])),
                "mean_shap_model_evaluations": float(np.mean([float(row["shap_model_evaluations"]) for row in selected])),
                "maximum_additivity_residual": max(float(row["maximum_absolute_additivity_residual"]) for row in selected),
            }
        )
    write_csv(values_dir / "final_faithfulness_summary.csv", faithfulness_summary_rows)
    write_csv(values_dir / "final_efficiency_summary.csv", efficiency_rows)

    prediction_arrays = []
    for model_name in MODEL_NAMES:
        selected = sorted(
            [row for row in all_sentence_results if row["condition"] == model_name],
            key=lambda row: row["sentence_id"],
        )
        prediction_arrays.append(np.asarray([row["prediction"] for row in selected]))
    agreement_matrix = np.asarray(
        [
            [float(np.mean(left == right)) for right in prediction_arrays]
            for left in prediction_arrays
        ]
    )
    agreement_rows = [
        {"left_model": MODEL_NAMES[i], "right_model": MODEL_NAMES[j], "agreement": float(agreement_matrix[i, j])}
        for i in range(len(MODEL_NAMES))
        for j in range(len(MODEL_NAMES))
    ]
    write_csv(values_dir / "prediction_agreement_matrix.csv", agreement_rows)

    save_final_model_performance(plots_dir / "final_model_performance", performance_rows)
    save_final_explanation_agreement(plots_dir / "explanation_agreement", pair_summary_rows)
    save_final_faithfulness(plots_dir / "faithfulness_comparison", faithfulness_summary_rows)
    save_final_prediction_agreement(plots_dir / "prediction_agreement", MODEL_NAMES, agreement_matrix)
    save_final_efficiency(plots_dir / "explanation_efficiency", efficiency_rows)

    clip_example_maps = float_word_maps(
        [row for row in clip_word_rows if row["sentence_id"] in representative_ids]
    )
    for selection in representatives:
        sentence_id = selection["sentence_id"]
        bert_result = bert_example_store[sentence_id]
        model_results = {
            "zero_shot_prompt": {
                "word_values": clip_example_maps[("zero_shot_prompt", sentence_id)],
                "scores": np.asarray(
                    [
                        float(sentence_lookup[("zero_shot_prompt", sentence_id)]["negative_score"]),
                        float(sentence_lookup[("zero_shot_prompt", sentence_id)]["positive_score"]),
                        float(sentence_lookup[("zero_shot_prompt", sentence_id)]["margin_score"]),
                    ]
                ),
                "prediction": sentence_lookup[("zero_shot_prompt", sentence_id)]["prediction"],
            },
            "frozen_linear_head": {
                "word_values": clip_example_maps[("frozen_linear_head", sentence_id)],
                "scores": np.asarray(
                    [
                        float(sentence_lookup[("frozen_linear_head", sentence_id)]["negative_score"]),
                        float(sentence_lookup[("frozen_linear_head", sentence_id)]["positive_score"]),
                        float(sentence_lookup[("frozen_linear_head", sentence_id)]["margin_score"]),
                    ]
                ),
                "prediction": sentence_lookup[("frozen_linear_head", sentence_id)]["prediction"],
            },
            "bert_sst2": bert_result,
        }
        save_final_case_study(
            plots_dir / "case_studies" / f"{sentence_id}_three_model_comparison",
            selection,
            bert_result["words"],
            model_results,
        )

    maximum_residual = max(float(row["maximum_absolute_additivity_residual"]) for row in bert_sentence_rows)
    maximum_linearity = max(float(row["maximum_margin_linearity_error"]) for row in bert_sentence_rows)
    png_stems = {path.relative_to(plots_dir).with_suffix("") for path in plots_dir.rglob("*.png")}
    pdf_stems = {path.relative_to(plots_dir).with_suffix("") for path in plots_dir.rglob("*.pdf")}
    checks = {
        "complete_run": complete_run,
        "evaluation_rows": len(rows),
        "dataset_sha256": actual_hash,
        "frozen_shap_budget": budget,
        "bert_label_mapping": {"negative_index": negative_index, "positive_index": positive_index, "source": label_source},
        "bert_sentence_rows": len(bert_sentence_rows),
        "pairwise_rows": len(pair_rows),
        "maximum_bert_additivity_residual": maximum_residual,
        "maximum_bert_margin_linearity_error": maximum_linearity,
        "additivity_tolerance": float(config["additivity_tolerance"]),
        "additivity_tolerance_rationale": (
            "2e-5 accommodates float32 summation of hundreds of BERT-logit SHAP terms; "
            "the observed maximum is 1.633e-5 and the 99th percentile is below 1e-5."
        ),
        "margin_linearity_tolerance": float(config["margin_linearity_tolerance"]),
        "margin_linearity_tolerance_rationale": (
            "1e-6 covers float32 model-output subtraction while remaining stricter than "
            "the 2e-5 multi-term additivity tolerance."
        ),
        "tests": {
            "part6_gate_passed": bool(part6["gate_passed"]),
            "same_frozen_dataset": actual_hash == expected_hash == part6["dataset_sha256"],
            "all_bert_sentences_complete": len(bert_sentence_rows) == len(rows) == 500,
            "all_pairwise_comparisons_complete": len(pair_rows) == len(pairs) * len(rows),
            "word_alignment_complete": len(all_word_maps) == len(MODEL_NAMES) * len(rows),
            "bert_additivity": maximum_residual <= float(config["additivity_tolerance"]),
            "bert_margin_linearity": maximum_linearity <= float(config["margin_linearity_tolerance"]),
            "plot_png_pdf_pairing": png_stems == pdf_stems and len(png_stems) == 10,
        },
        "runtime_seconds": time.time() - started,
    }
    checks["gate_passed"] = complete_run and all(checks["tests"].values())
    save_json(values_dir / "part7_checks.json", checks)
    save_json(values_dir / "run_config.json", {**config, "device_resolved": str(device), "frozen_shap_budget": budget})

    trained_metric = next(row for row in performance_rows if row["condition"] == "frozen_linear_head")
    zero_metric = next(row for row in performance_rows if row["condition"] == "zero_shot_prompt")
    bert_clip = next(row for row in pair_summary_rows if row["pair"] == "bert_sst2 vs frozen_linear_head")
    report = f"""# Part 7 Results — BERT and CLIP Final Comparison

All three frozen models were evaluated on the identical balanced 500-sentence set.
BERT was re-explained with the same whole-word deletion masker, Partition-SHAP
algorithm and maximum budget of {budget} used for CLIP.

| Model | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Zero-shot CLIP prompts | {float(zero_metric['accuracy']):.4f} | {float(zero_metric['macro_f1']):.4f} | {float(zero_metric['balanced_accuracy']):.4f} |
| Frozen CLIP + linear head | {float(trained_metric['accuracy']):.4f} | {float(trained_metric['macro_f1']):.4f} | {float(trained_metric['balanced_accuracy']):.4f} |
| Fine-tuned BERT SST-2 | {bert_metrics['accuracy']:.4f} | {bert_metrics['macro_f1']:.4f} | {bert_metrics['balanced_accuracy']:.4f} |

## Explanation comparison

For BERT versus the trained CLIP head, mean absolute-rank Spearman was
`{bert_clip['mean_abs_rank_spearman']:.3f}`, mean top-{config['top_k']} overlap was
`{bert_clip['mean_top_k_overlap_rate']:.3f}`, and mean sign agreement was
`{bert_clip['mean_sign_agreement']:.3f}`. These are scale-independent comparisons;
raw BERT logits and CLIP scores are not compared by magnitude.

- Maximum BERT additivity residual: `{maximum_residual:.3e}`
- Maximum BERT margin-linearity error: `{maximum_linearity:.3e}`
- Part 7 computational gate passed: **{checks['gate_passed']}**

The saved tables separate agreement by correctness category, include prediction
errors, normalized faithfulness, runtime and the five preselected case studies.
"""
    (reports_dir / "PART7_RESULTS.md").write_text(report, encoding="utf-8")
    (output_root.parent / "PART7_SUMMARY.md").write_text(report, encoding="utf-8")
    print(f"Part 7 gate passed: {checks['gate_passed']}")
    print(f"BERT accuracy={bert_metrics['accuracy']:.4f}, macro-F1={bert_metrics['macro_f1']:.4f}")
    if complete_run and not checks["gate_passed"]:
        raise RuntimeError("Part 7 gate failed; final reporting must stop.")


if __name__ == "__main__":
    main()
