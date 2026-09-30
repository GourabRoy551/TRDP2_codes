"""Part 3: compare raw CLIP-BPE SHAP with genuine whole-word Partition SHAP."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch
from scipy.stats import spearmanr

from clip_backend import ClipSentimentScorer, choose_device, load_clip
from io_utils import load_json, project_path, read_csv, save_json, write_csv
from plotting import save_part3_aggregate, save_part3_word_comparison
from prompting import OUTPUT_NAMES, build_prototypes
from shap_utils import (
    additivity,
    create_text_masker,
    model_tokens,
    unpack_explanation,
)
from word_masking import (
    audit_mask_coalitions,
    create_word_masker,
    map_clip_bpe_to_words,
    split_word_units,
)


DEFAULT_CONFIG = project_path("configs/part3_word_partition.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-sentences", type=int, default=None)
    return parser.parse_args()


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, right, atol=1e-15, rtol=0.0):
        return 1.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def top_indices(values: np.ndarray, k: int) -> set[int]:
    count = min(k, len(values))
    return set(np.argsort(-np.abs(values))[:count].tolist())


def mask_words(masker: Any, text: str, removed: set[int]) -> str:
    feature_count = int(masker.shape(text)[1])
    mask = np.ones(feature_count, dtype=bool)
    for word_index in removed:
        mask[word_index + 1] = False
    return str(np.asarray(masker(mask, text)[0], dtype=object).reshape(-1)[0])


def supportive_indices(values: np.ndarray, prediction: str, k: int) -> set[int]:
    signed = values if prediction == "POS" else -values
    positive = [index for index in np.argsort(-signed) if signed[index] > 0]
    return set(positive[: min(k, len(positive))])


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    stability = read_csv(project_path(config["stability_path"]))
    if args.max_sentences is not None:
        stability = stability[: args.max_sentences]

    frozen = load_json(project_path(config["frozen_prompt_manifest_path"]))
    selected_family = str(frozen["selected_family"])
    family = {
        "NEG": frozen["selected_prompts"]["NEG"],
        "POS": frozen["selected_prompts"]["POS"],
    }
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
    print(f"Part 3: loading {config['model_name']} on {device}")
    model, tokenizer = load_clip(
        str(config["model_name"]), device, bool(config["local_files_only"])
    )
    prototypes, prompt_rows = build_prototypes(
        model, tokenizer, family, device, int(config["max_length"])
    )
    scorer = ClipSentimentScorer(
        model, tokenizer, prototypes, device, int(config["max_length"])
    )
    bpe_masker = create_text_masker(tokenizer)
    word_masker = create_word_masker()
    bpe_explainer = shap.Explainer(
        scorer, bpe_masker, algorithm="partition", output_names=list(OUTPUT_NAMES)
    )
    word_explainer = shap.Explainer(
        scorer, word_masker, algorithm="partition", output_names=list(OUTPUT_NAMES)
    )

    comparison_rows: list[dict[str, Any]] = []
    bpe_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    hierarchy_rows: list[dict[str, Any]] = []
    presentation_ids = set(config["presentation_sentence_ids"])

    for item_index, row in enumerate(stability, start=1):
        sentence_id, text, gold = row["sentence_id"], row["text"], row["gold_label"]
        print(f"Part 3 [{item_index}/{len(stability)}] {sentence_id}")
        scores = scorer([text])[0]
        prediction = "POS" if scores[2] > 0 else "NEG"
        clip_tokens = model_tokens(tokenizer, text, int(config["max_length"]))
        units = split_word_units(text)
        mapping = map_clip_bpe_to_words(tokenizer, text)
        for mapping_row in mapping:
            mapping_rows.append({"sentence_id": sentence_id, **mapping_row})

        before = scorer.text_evaluations
        bpe_started = time.perf_counter()
        bpe_explanation = bpe_explainer(
            [text],
            max_evals=int(config["max_evals"]),
            batch_size=int(config["batch_size"]),
            silent=True,
        )
        bpe_runtime = time.perf_counter() - bpe_started
        bpe_evaluations = scorer.text_evaluations - before
        bpe_features, bpe_values, bpe_bases = unpack_explanation(bpe_explanation, 3)
        if len(clip_tokens) != len(bpe_features):
            raise ValueError(f"{sentence_id}: CLIP-BPE feature alignment failed.")
        bpe_add = additivity(scores, bpe_bases, bpe_values)

        bpe_word_values = np.zeros((len(units), 3), dtype=float)
        special_values = np.zeros(3, dtype=float)
        for mapping_row in mapping:
            index = int(mapping_row["bpe_index"])
            if mapping_row["word_index"] == "":
                special_values += bpe_values[index]
            else:
                bpe_word_values[int(mapping_row["word_index"])] += bpe_values[index]
        conservation_error = float(
            np.max(np.abs(bpe_values.sum(axis=0) - bpe_word_values.sum(axis=0) - special_values))
        )

        before = scorer.text_evaluations
        word_started = time.perf_counter()
        word_explanation = word_explainer(
            [text],
            max_evals=int(config["max_evals"]),
            batch_size=int(config["batch_size"]),
            silent=True,
        )
        word_runtime = time.perf_counter() - word_started
        word_evaluations = scorer.text_evaluations - before
        word_features, word_values_all, word_bases = unpack_explanation(word_explanation, 3)
        if len(word_features) != len(units) + 2:
            raise ValueError(
                f"{sentence_id}: expected {len(units) + 2} word features, got {len(word_features)}."
            )
        direct_word_values = word_values_all[1:-1]
        word_add = additivity(scores, word_bases, word_values_all)

        audit = audit_mask_coalitions(word_masker, tokenizer, text, seed + item_index)
        audit_rows.append({"sentence_id": sentence_id, **audit})
        hierarchy = np.asarray(word_masker.clustering(text), dtype=float)
        for merge_index, link in enumerate(hierarchy):
            hierarchy_rows.append(
                {
                    "sentence_id": sentence_id,
                    "merge_index": merge_index,
                    "left_child": int(link[0]),
                    "right_child": int(link[1]),
                    "normalized_height": float(link[2]),
                    "cluster_size": int(link[3]),
                }
            )

        bpe_top = top_indices(bpe_word_values[:, 2], int(config["top_k"]))
        word_top = top_indices(direct_word_values[:, 2], int(config["top_k"]))
        union = bpe_top | word_top
        rank_correlation = safe_spearman(
            np.abs(bpe_word_values[:, 2]), np.abs(direct_word_values[:, 2])
        )
        sign_agreement = float(
            np.mean(np.sign(bpe_word_values[:, 2]) == np.sign(direct_word_values[:, 2]))
        )
        bpe_support = supportive_indices(
            bpe_word_values[:, 2], prediction, int(config["top_k"])
        )
        word_support = supportive_indices(
            direct_word_values[:, 2], prediction, int(config["top_k"])
        )
        prediction_sign = 1.0 if prediction == "POS" else -1.0
        bpe_masked_margin = float(scorer([mask_words(word_masker, text, bpe_support)])[0, 2])
        word_masked_margin = float(scorer([mask_words(word_masker, text, word_support)])[0, 2])

        comparison = {
            "sentence_id": sentence_id,
            "text": text,
            "gold_label": gold,
            "prediction": prediction,
            "margin_score": float(scores[2]),
            "word_count": len(units),
            "bpe_feature_count": len(clip_tokens),
            "bpe_max_abs_additivity_residual": float(np.max(np.abs(bpe_add["residual"]))),
            "word_max_abs_additivity_residual": float(np.max(np.abs(word_add["residual"]))),
            "bpe_to_word_conservation_error": conservation_error,
            "partial_word_mask_violations": audit["partial_word_mask_violations"],
            "unassigned_non_special_bpe": audit["unassigned_non_special_bpe"],
            "margin_abs_rank_spearman": rank_correlation,
            "top_k": int(config["top_k"]),
            "top_k_overlap": len(bpe_top & word_top),
            "top_k_jaccard": len(bpe_top & word_top) / len(union) if union else 1.0,
            "margin_sign_agreement": sign_agreement,
            "bpe_supportive_deletion_drop": prediction_sign * (float(scores[2]) - bpe_masked_margin),
            "word_supportive_deletion_drop": prediction_sign * (float(scores[2]) - word_masked_margin),
            "bpe_runtime_seconds": bpe_runtime,
            "word_runtime_seconds": word_runtime,
            "bpe_model_evaluations": bpe_evaluations,
            "word_model_evaluations": word_evaluations,
        }
        comparison_rows.append(comparison)

        for feature_index, (feature, token) in enumerate(
            zip(bpe_features, clip_tokens, strict=True)
        ):
            bpe_rows.append(
                {
                    "sentence_id": sentence_id,
                    "feature_index": feature_index,
                    "feature_text": feature,
                    "clip_bpe_token": token,
                    "negative_shap": float(bpe_values[feature_index, 0]),
                    "positive_shap": float(bpe_values[feature_index, 1]),
                    "margin_shap": float(bpe_values[feature_index, 2]),
                }
            )
        for word_index, unit in enumerate(units):
            word_rows.append(
                {
                    "sentence_id": sentence_id,
                    "word_index": word_index,
                    "word": unit.text,
                    "offset_start": unit.start,
                    "offset_end": unit.end,
                    "is_content": int(unit.is_content),
                    "bpe_aggregated_negative_shap": float(bpe_word_values[word_index, 0]),
                    "bpe_aggregated_positive_shap": float(bpe_word_values[word_index, 1]),
                    "bpe_aggregated_margin_shap": float(bpe_word_values[word_index, 2]),
                    "direct_word_negative_shap": float(direct_word_values[word_index, 0]),
                    "direct_word_positive_shap": float(direct_word_values[word_index, 1]),
                    "direct_word_margin_shap": float(direct_word_values[word_index, 2]),
                }
            )
        if sentence_id in presentation_ids:
            save_part3_word_comparison(
                plots_dir / "presentation_examples" / f"{sentence_id}_bpe_vs_word",
                sentence_id,
                text,
                gold,
                [unit.text for unit in units],
                bpe_word_values,
                direct_word_values,
                comparison["bpe_max_abs_additivity_residual"],
                comparison["word_max_abs_additivity_residual"],
                rank_correlation,
            )

    write_csv(values_dir / "sentence_comparison.csv", comparison_rows)
    write_csv(values_dir / "bpe_shap_values.csv", bpe_rows)
    write_csv(values_dir / "word_shap_values.csv", word_rows)
    write_csv(values_dir / "bpe_to_word_mapping.csv", mapping_rows)
    write_csv(values_dir / "masking_audit.csv", audit_rows)
    write_csv(values_dir / "word_partition_hierarchy.csv", hierarchy_rows)
    write_csv(values_dir / "prompt_diagnostics.csv", prompt_rows)
    save_part3_aggregate(plots_dir / "part3_agreement_and_runtime", comparison_rows)

    max_bpe_residual = max(float(row["bpe_max_abs_additivity_residual"]) for row in comparison_rows)
    max_word_residual = max(float(row["word_max_abs_additivity_residual"]) for row in comparison_rows)
    max_conservation = max(float(row["bpe_to_word_conservation_error"]) for row in comparison_rows)
    partial_violations = sum(int(row["partial_word_mask_violations"]) for row in comparison_rows)
    unassigned = sum(int(row["unassigned_non_special_bpe"]) for row in comparison_rows)
    checks = {
        "sentences": len(comparison_rows),
        "selected_prompt_family": selected_family,
        "maximum_bpe_additivity_residual": max_bpe_residual,
        "maximum_word_additivity_residual": max_word_residual,
        "maximum_bpe_to_word_conservation_error": max_conservation,
        "partial_word_mask_violations": partial_violations,
        "unassigned_non_special_bpe": unassigned,
        "mean_margin_abs_rank_spearman": float(np.mean([row["margin_abs_rank_spearman"] for row in comparison_rows])),
        "mean_top_k_jaccard": float(np.mean([row["top_k_jaccard"] for row in comparison_rows])),
        "mean_bpe_runtime_seconds": float(np.mean([row["bpe_runtime_seconds"] for row in comparison_rows])),
        "mean_word_runtime_seconds": float(np.mean([row["word_runtime_seconds"] for row in comparison_rows])),
        "tests": {
            "bpe_additivity": max_bpe_residual <= float(config["additivity_tolerance"]),
            "word_additivity": max_word_residual <= float(config["additivity_tolerance"]),
            "bpe_word_conservation": max_conservation <= float(config["conservation_tolerance"]),
            "no_partial_word_masking": partial_violations == 0,
            "all_non_special_bpe_assigned": unassigned == 0,
        },
        "runtime_seconds": time.time() - started,
    }
    checks["gate_passed"] = all(checks["tests"].values())
    save_json(values_dir / "part3_checks.json", checks)
    save_json(
        values_dir / "run_config.json",
        {**config, "device_resolved": str(device), "selected_prompt_family": selected_family},
    )
    report = f"""# Part 3 Results — Whole-Word Hierarchical Partition SHAP

Raw CLIP-BPE and genuine whole-word masking were compared on {len(comparison_rows)}
development-only stability sentences using the frozen `{selected_family}` prompts.

## Gate checks

- Maximum BPE additivity residual: `{max_bpe_residual:.3e}`
- Maximum whole-word additivity residual: `{max_word_residual:.3e}`
- Maximum BPE-to-word conservation error: `{max_conservation:.3e}`
- Partial-word masking violations: `{partial_violations}`
- Unassigned non-special BPE tokens: `{unassigned}`
- Gate passed: **{checks['gate_passed']}**

## Comparison

- Mean absolute-margin word-rank Spearman: `{checks['mean_margin_abs_rank_spearman']:.3f}`
- Mean top-{config['top_k']} Jaccard: `{checks['mean_top_k_jaccard']:.3f}`
- Mean BPE runtime: `{checks['mean_bpe_runtime_seconds']:.3f}` seconds
- Mean whole-word runtime: `{checks['mean_word_runtime_seconds']:.3f}` seconds

The direct word masker treats contractions and hyphenated compounds as indivisible
features. Punctuation remains explicit. Part 4 may proceed only because every gate
check passed.
"""
    (reports_dir / "PART3_RESULTS.md").write_text(report, encoding="utf-8")
    print(f"Part 3 gate passed: {checks['gate_passed']}")
    print(f"Mean BPE/word rank agreement: {checks['mean_margin_abs_rank_spearman']:.3f}")
    if not checks["gate_passed"]:
        raise RuntimeError("Part 3 gate failed; Part 4 must not proceed.")


if __name__ == "__main__":
    main()
