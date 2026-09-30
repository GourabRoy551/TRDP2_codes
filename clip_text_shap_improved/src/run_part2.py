"""Part 2: validate balanced prompt families and freeze the best ensemble."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from clip_backend import (
    ClipSentimentScorer,
    MarginOnlyScorer,
    choose_device,
    encode_in_batches,
    load_clip,
    prototypes_for_families,
)
from io_utils import (
    load_json,
    project_path,
    read_csv,
    save_json,
    sha256_file,
    write_csv,
)
from plotting import (
    save_agreement_matrix,
    save_confusion_grid,
    save_margin_distribution,
    save_prompt_metrics,
    save_shap_sensitivity,
)
from prompting import load_prompt_registry
from shap_utils import (
    additivity,
    aggregate_bpe,
    create_text_masker,
    merge_word_values,
    model_tokens,
    unpack_explanation,
)


DEFAULT_CONFIG = project_path("configs/part2_prompt_validation.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-shap-sensitivity",
        action="store_true",
        help="Run classification validation only (not used for the approved full run).",
    )
    return parser.parse_args()


def validation_metrics(
    family: str, labels: np.ndarray, scores: np.ndarray, prototype_cosine: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    margins = scores[:, 1] - scores[:, 0]
    predictions = (margins > 0).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    negative_margins = margins[labels == 0]
    positive_margins = margins[labels == 1]
    summary = {
        "family": family,
        "rows": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "negative_precision": float(precision[0]),
        "negative_recall": float(recall[0]),
        "negative_f1": float(f1[0]),
        "negative_support": int(support[0]),
        "positive_precision": float(precision[1]),
        "positive_recall": float(recall[1]),
        "positive_f1": float(f1[1]),
        "positive_support": int(support[1]),
        "true_neg": int(matrix[0, 0]),
        "false_pos": int(matrix[0, 1]),
        "false_neg": int(matrix[1, 0]),
        "true_pos": int(matrix[1, 1]),
        "mean_negative_gold_margin": float(np.mean(negative_margins)),
        "mean_positive_gold_margin": float(np.mean(positive_margins)),
        "margin_separation": float(np.mean(positive_margins) - np.mean(negative_margins)),
        "mean_absolute_margin": float(np.mean(np.abs(margins))),
        "near_zero_margin_fraction": float(np.mean(np.abs(margins) < 0.005)),
        "negative_positive_prototype_cosine": prototype_cosine,
    }
    rows = [
        {
            "family": family,
            "row_index": index,
            "negative_score": float(scores[index, 0]),
            "positive_score": float(scores[index, 1]),
            "margin": float(margins[index]),
            "prediction": "POS" if predictions[index] else "NEG",
        }
        for index in range(len(labels))
    ]
    return summary, rows


def choose_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Predeclared lexicographic rule: macro-F1, accuracy, separation, name."""
    return sorted(
        rows,
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["accuracy"]),
            -float(row["margin_separation"]),
            str(row["family"]),
        ),
    )[0]


def correlation_or_identity(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, right, atol=1e-15, rtol=0.0):
        return 1.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def prompt_shap_sensitivity(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prototypes: dict[str, torch.Tensor],
    stability: list[dict[str, str]],
    selected: str,
    max_length: int,
    max_evals: int,
    batch_size: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    masker = create_text_masker(tokenizer)
    maps: dict[tuple[str, str], dict[str, float]] = {}
    word_rows: list[dict[str, Any]] = []
    additivity_rows: list[dict[str, Any]] = []
    total = len(prototypes) * len(stability)
    progress = 0
    for family_name, family_prototypes in prototypes.items():
        scorer = ClipSentimentScorer(
            model, tokenizer, family_prototypes.to(device), device, max_length
        )
        explainer = shap.Explainer(
            MarginOnlyScorer(scorer),
            masker,
            algorithm="partition",
            output_names=["MARGIN"],
        )
        for row in stability:
            progress += 1
            sentence_id, text = row["sentence_id"], row["text"]
            if progress == 1 or progress % 10 == 0:
                print(f"Prompt SHAP sensitivity [{progress}/{total}]")
            tokens = model_tokens(tokenizer, text, max_length)
            margin_score = float(scorer([text])[0, 2])
            explanation = explainer(
                [text], max_evals=max_evals, batch_size=batch_size, silent=True
            )
            features, values, bases = unpack_explanation(explanation, 1)
            if len(tokens) != len(features):
                raise ValueError(f"{family_name}/{sentence_id}: token alignment failed.")
            calculation = additivity([margin_score], bases, values)
            groups = aggregate_bpe(features, tokens, values)
            word_map = merge_word_values(groups, 0)
            maps[(family_name, sentence_id)] = word_map
            additivity_rows.append(
                {
                    "family": family_name,
                    "sentence_id": sentence_id,
                    "margin_score": margin_score,
                    "base_value": float(bases[0]),
                    "shap_sum": float(calculation["shap_sum"][0]),
                    "reconstructed": float(calculation["reconstructed"][0]),
                    "residual": float(calculation["residual"][0]),
                    "model_text_evaluations": scorer.text_evaluations,
                }
            )
            for word_index, group in enumerate(
                group for group in groups if not group["is_special"]
            ):
                word_rows.append(
                    {
                        "family": family_name,
                        "sentence_id": sentence_id,
                        "gold_label": row["gold_label"],
                        "word_index": word_index,
                        "word": group["word"],
                        "is_content": int(group["is_content"]),
                        "margin_shap": float(group["values"][0]),
                    }
                )

    comparison_rows: list[dict[str, Any]] = []
    for family_name in prototypes:
        for row in stability:
            sentence_id = row["sentence_id"]
            reference = maps[(selected, sentence_id)]
            candidate = maps[(family_name, sentence_id)]
            words = sorted(set(reference) | set(candidate))
            ref = np.asarray([reference.get(word, 0.0) for word in words])
            other = np.asarray([candidate.get(word, 0.0) for word in words])
            ref_top = set(
                word
                for word, _ in sorted(
                    reference.items(), key=lambda item: abs(item[1]), reverse=True
                )[:top_k]
            )
            other_top = set(
                word
                for word, _ in sorted(
                    candidate.items(), key=lambda item: abs(item[1]), reverse=True
                )[:top_k]
            )
            union = ref_top | other_top
            comparison_rows.append(
                {
                    "family": family_name,
                    "reference_family": selected,
                    "sentence_id": sentence_id,
                    "common_union_word_count": len(words),
                    "spearman_abs_rank": correlation_or_identity(np.abs(ref), np.abs(other)),
                    "top_k": top_k,
                    "top_k_overlap": len(ref_top & other_top),
                    "top_k_jaccard": len(ref_top & other_top) / len(union) if union else 1.0,
                    "sign_agreement": float(np.mean(np.sign(ref) == np.sign(other))) if len(words) else 1.0,
                }
            )
    summary_rows: list[dict[str, Any]] = []
    for family_name in prototypes:
        rows = [row for row in comparison_rows if row["family"] == family_name]
        summary_rows.append(
            {
                "family": family_name,
                "reference_family": selected,
                "sentences": len(rows),
                "mean_spearman_abs_rank": float(np.mean([row["spearman_abs_rank"] for row in rows])),
                "mean_top_k_jaccard": float(np.mean([row["top_k_jaccard"] for row in rows])),
                "mean_sign_agreement": float(np.mean([row["sign_agreement"] for row in rows])),
            }
        )
    return word_rows, additivity_rows, comparison_rows, summary_rows


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    validation_path = project_path(config["validation_path"])
    stability_path = project_path(config["stability_path"])
    registry_path = project_path(config["prompt_registry_path"])
    validation = read_csv(validation_path)
    stability = read_csv(stability_path)
    labels = np.asarray([0 if row["gold_label"] == "NEG" else 1 for row in validation])
    texts = [row["text"] for row in validation]

    output_root = project_path(config["output_dir"])
    values_dir = output_root / "values"
    plots_dir = output_root / "plots"
    reports_dir = output_root / "reports"
    for directory in (values_dir, plots_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.time()
    device = choose_device(str(config["device"]))
    print(f"Part 2: loading {config['model_name']} on {device}")
    model, tokenizer = load_clip(
        str(config["model_name"]), device, bool(config["local_files_only"])
    )
    families = load_prompt_registry(registry_path)
    prototypes, prompt_diagnostics = prototypes_for_families(
        model, tokenizer, families, device, int(config["max_length"])
    )
    print(f"Encoding {len(texts)} validation sentences once for all prompt families")
    embeddings = encode_in_batches(
        model,
        tokenizer,
        texts,
        device,
        int(config["max_length"]),
        int(config["embedding_batch_size"]),
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    predictions_by_family: dict[str, np.ndarray] = {}
    margins_by_family: dict[str, np.ndarray] = {}
    for family_name, family_prototypes in prototypes.items():
        class_scores = (embeddings @ family_prototypes.T).numpy()
        prototype_cosine = float(torch.dot(family_prototypes[0], family_prototypes[1]))
        metrics, rows = validation_metrics(
            family_name, labels, class_scores, prototype_cosine
        )
        metric_rows.append(metrics)
        for dataset_row, prediction_row in zip(validation, rows, strict=True):
            prediction_rows.append(
                {
                    **prediction_row,
                    "sentence_id": dataset_row["sentence_id"],
                    "source_row_id": dataset_row["source_row_id"],
                    "gold_label": dataset_row["gold_label"],
                    "correct": int(prediction_row["prediction"] == dataset_row["gold_label"]),
                }
            )
        margins = class_scores[:, 1] - class_scores[:, 0]
        margins_by_family[family_name] = margins
        predictions_by_family[family_name] = margins > 0
        print(
            f"{family_name}: macro-F1={metrics['macro_f1']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, separation={metrics['margin_separation']:+.4f}"
        )

    selected_row = choose_family(metric_rows)
    selected = str(selected_row["family"])
    ordered_families = list(families)
    agreement = np.asarray(
        [
            [
                np.mean(predictions_by_family[left] == predictions_by_family[right])
                for right in ordered_families
            ]
            for left in ordered_families
        ]
    )
    agreement_rows = [
        {
            "family_a": left,
            "family_b": right,
            "prediction_agreement": float(agreement[i, j]),
            "margin_spearman": correlation_or_identity(
                margins_by_family[left], margins_by_family[right]
            ),
        }
        for i, left in enumerate(ordered_families)
        for j, right in enumerate(ordered_families)
    ]

    frozen_manifest = {
        "selected_family": selected,
        "selection_rule": config["selection_rule"],
        "selection_uses_final_evaluation": False,
        "validation_path": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "validation_rows": len(validation),
        "validation_label_counts": {
            "NEG": int(np.sum(labels == 0)),
            "POS": int(np.sum(labels == 1)),
        },
        "prompt_registry_path": str(registry_path),
        "prompt_registry_sha256": sha256_file(registry_path),
        "selected_prompts": {
            "NEG": families[selected]["NEG"],
            "POS": families[selected]["POS"],
        },
        "selected_validation_metrics": selected_row,
        "all_family_metrics": metric_rows,
        "model_name": config["model_name"],
        "class_order": ["NEG", "POS"],
        "seed": seed,
    }
    save_json(project_path(config["frozen_manifest_path"]), frozen_manifest)

    write_csv(values_dir / "prompt_performance.csv", metric_rows)
    write_csv(values_dir / "validation_predictions.csv", prediction_rows)
    write_csv(values_dir / "prompt_pairwise_agreement.csv", agreement_rows)
    write_csv(values_dir / "prompt_diagnostics.csv", prompt_diagnostics)
    save_prompt_metrics(plots_dir / "prompt_validation_metrics", metric_rows, selected)
    save_confusion_grid(plots_dir / "prompt_confusion_matrices", metric_rows, selected)
    save_agreement_matrix(plots_dir / "prompt_prediction_agreement", ordered_families, agreement)
    save_margin_distribution(plots_dir / "prompt_margin_distributions", prediction_rows, selected)

    maximum_shap_residual = None
    sensitivity_summary: list[dict[str, Any]] = []
    if not args.skip_shap_sensitivity:
        word_rows, additivity_rows, comparison_rows, sensitivity_summary = prompt_shap_sensitivity(
            model,
            tokenizer,
            device,
            prototypes,
            stability,
            selected,
            int(config["max_length"]),
            int(config["shap_max_evals"]),
            int(config["shap_batch_size"]),
            int(config["top_k"]),
        )
        maximum_shap_residual = max(abs(float(row["residual"])) for row in additivity_rows)
        write_csv(values_dir / "stability_margin_shap_words.csv", word_rows)
        write_csv(values_dir / "stability_margin_shap_additivity.csv", additivity_rows)
        write_csv(values_dir / "prompt_shap_sentence_comparison.csv", comparison_rows)
        write_csv(values_dir / "prompt_shap_sensitivity_summary.csv", sensitivity_summary)
        save_shap_sensitivity(
            plots_dir / "prompt_shap_sensitivity", sensitivity_summary, selected
        )

    run_summary = {
        "selected_family": selected,
        "validation_rows": len(validation),
        "stability_rows": len(stability),
        "prompt_family_count": len(families),
        "selected_metrics": selected_row,
        "maximum_stability_shap_additivity_residual": maximum_shap_residual,
        "shap_sensitivity_completed": not args.skip_shap_sensitivity,
        "final_evaluation_used": False,
        "runtime_seconds": time.time() - started,
    }
    save_json(values_dir / "part2_summary.json", run_summary)
    save_json(
        values_dir / "run_config.json",
        {**config, "config_path": str(config_path), "device_resolved": str(device)},
    )

    sensitivity_lines = ""
    if sensitivity_summary:
        sensitivity_lines = "\n".join(
            f"- `{row['family']}`: Spearman={row['mean_spearman_abs_rank']:.3f}, "
            f"top-{config['top_k']} Jaccard={row['mean_top_k_jaccard']:.3f}"
            for row in sensitivity_summary
        )
    report = f"""# Part 2 Results — Prompt Validation and Robustness

Five balanced prompt families were evaluated on the internal validation split only
({len(validation):,} rows). The final 500-sentence evaluation set was not read.

## Frozen prompt family

`{selected}` was selected using the predeclared lexicographic rule: macro-F1,
accuracy, margin separation, then family name. Its validation macro-F1 was
`{float(selected_row['macro_f1']):.4f}` and accuracy was
`{float(selected_row['accuracy']):.4f}`.

## Development-only SHAP prompt sensitivity

Partition SHAP margin explanations were compared on the fixed 40-sentence stability
subset with a provisional budget of {config['shap_max_evals']} evaluations. This is
a prompt-robustness check, not the final Part 5 budget decision.

{sensitivity_lines}

Maximum margin-SHAP additivity residual: `{maximum_shap_residual}`.

The selected prompts are frozen in `configs/frozen_prompt_manifest.json`. Part 3 has
not been started.
"""
    (reports_dir / "PART2_RESULTS.md").write_text(report, encoding="utf-8")
    print(f"Selected prompt family: {selected}")
    print(f"Final evaluation rows used: 0")
    print(f"Part 2 outputs: {output_root}")


if __name__ == "__main__":
    main()
