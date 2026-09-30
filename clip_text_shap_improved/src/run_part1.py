"""Part 1: reproduce the baseline and explain NEG, POS, and POS−NEG directly."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch

from clip_backend import ClipSentimentScorer, choose_device, load_clip
from io_utils import (
    load_json,
    project_path,
    read_csv,
    save_json,
    sha256_file,
    write_csv,
)
from plotting import save_part1_aggregate, save_part1_overview
from prompting import OUTPUT_NAMES, build_prototypes, load_prompt_registry
from shap_utils import (
    additivity,
    aggregate_bpe,
    create_text_masker,
    model_tokens,
    unpack_explanation,
)


DEFAULT_CONFIG = project_path("configs/part1_margin.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-sentences", type=int, default=None)
    return parser.parse_args()


def check_pilot_parity(
    prepared: list[dict[str, str]], legacy: list[dict[str, str]]
) -> None:
    prepared_rows = {
        row["sentence_id"]: (row["text"], row["gold_label"].upper())
        for row in prepared
    }
    legacy_rows = {
        row["sentence_id"]: (row["text"], row["gold_label"].upper())
        for row in legacy
    }
    if prepared_rows != legacy_rows:
        raise ValueError("Prepared pilot texts or labels differ from the completed baseline.")


def score_parity_rows(
    summaries: list[dict[str, Any]], legacy_summary: list[dict[str, str]]
) -> list[dict[str, Any]]:
    old = {row["sentence_id"]: row for row in legacy_summary}
    rows = []
    for current in summaries:
        legacy = old[current["sentence_id"]]
        negative_error = float(current["negative_score"]) - float(legacy["negative_score"])
        positive_error = float(current["positive_score"]) - float(legacy["positive_score"])
        rows.append(
            {
                "sentence_id": current["sentence_id"],
                "new_negative_score": current["negative_score"],
                "legacy_negative_score": legacy["negative_score"],
                "negative_difference": negative_error,
                "new_positive_score": current["positive_score"],
                "legacy_positive_score": legacy["positive_score"],
                "positive_difference": positive_error,
                "new_prediction": current["prediction"],
                "legacy_prediction": legacy["prediction"],
                "prediction_matches": int(current["prediction"] == legacy["prediction"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    pilot_path = project_path(config["pilot_path"])
    pilot = read_csv(pilot_path)
    legacy_pilot = read_csv(project_path(config["legacy_pilot_path"]))
    check_pilot_parity(pilot, legacy_pilot)
    if args.max_sentences is not None:
        pilot = pilot[: args.max_sentences]
    if not pilot:
        raise RuntimeError("The pilot dataset is empty.")

    output_root = project_path(config["output_dir"])
    values_dir = output_root / "values"
    plots_dir = output_root / "plots"
    reports_dir = output_root / "reports"
    for directory in (values_dir, plots_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    start = time.time()
    device = choose_device(str(config["device"]))
    print(f"Part 1: loading {config['model_name']} on {device}")
    model, tokenizer = load_clip(
        str(config["model_name"]), device, bool(config["local_files_only"])
    )
    families = load_prompt_registry(project_path(config["prompt_registry_path"]))
    family_name = str(config["prompt_family"])
    prototypes, prompt_rows = build_prototypes(
        model,
        tokenizer,
        families[family_name],
        device,
        int(config["max_length"]),
    )
    scorer = ClipSentimentScorer(
        model, tokenizer, prototypes, device, int(config["max_length"])
    )
    masker = create_text_masker(tokenizer)
    explainer = shap.Explainer(
        scorer,
        masker,
        algorithm="partition",
        output_names=list(OUTPUT_NAMES),
    )

    summaries: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []
    presentation_ids = set(config["presentation_sentence_ids"])

    for item_index, row in enumerate(pilot, start=1):
        sentence_id = row["sentence_id"]
        text = row["text"]
        gold = row["gold_label"].upper()
        print(f"[{item_index}/{len(pilot)}] {sentence_id}")
        tokens = model_tokens(tokenizer, text, int(config["max_length"]))
        scores = scorer([text])[0]
        explanation = explainer(
            [text],
            max_evals=int(config["max_evals"]),
            batch_size=int(config["batch_size"]),
            silent=True,
        )
        features, values, bases = unpack_explanation(explanation, 3)
        if len(tokens) != len(features):
            raise ValueError(
                f"{sentence_id}: {len(tokens)} CLIP tokens but {len(features)} SHAP features."
            )
        calculation = additivity(scores, bases, values)
        groups = aggregate_bpe(features, tokens, values)
        display_groups = [group for group in groups if not group["is_special"]]
        prediction = "POS" if scores[1] > scores[0] else "NEG"

        token_errors = values[:, 2] - (values[:, 1] - values[:, 0])
        base_error = float(bases[2] - (bases[1] - bases[0]))
        score_error = float(scores[2] - (scores[1] - scores[0]))
        word_errors = np.asarray(
            [
                float(group["values"][2] - (group["values"][1] - group["values"][0]))
                for group in display_groups
            ]
        )
        max_linearity = max(
            abs(base_error),
            abs(score_error),
            float(np.max(np.abs(token_errors))) if token_errors.size else 0.0,
            float(np.max(np.abs(word_errors))) if word_errors.size else 0.0,
        )
        max_additivity = float(np.max(np.abs(calculation["residual"])))

        summary = {
            "sentence_id": sentence_id,
            "text": text,
            "gold_label": gold,
            "prediction": prediction,
            "correct": int(gold == prediction),
            "negative_score": float(scores[0]),
            "positive_score": float(scores[1]),
            "margin_score": float(scores[2]),
            "negative_base_value": float(bases[0]),
            "positive_base_value": float(bases[1]),
            "margin_base_value": float(bases[2]),
            "negative_shap_sum": float(calculation["shap_sum"][0]),
            "positive_shap_sum": float(calculation["shap_sum"][1]),
            "margin_shap_sum": float(calculation["shap_sum"][2]),
            "negative_reconstructed": float(calculation["reconstructed"][0]),
            "positive_reconstructed": float(calculation["reconstructed"][1]),
            "margin_reconstructed": float(calculation["reconstructed"][2]),
            "negative_additivity_residual": float(calculation["residual"][0]),
            "positive_additivity_residual": float(calculation["residual"][1]),
            "margin_additivity_residual": float(calculation["residual"][2]),
            "max_abs_additivity_residual": max_additivity,
            "score_margin_linearity_error": score_error,
            "base_margin_linearity_error": base_error,
            "max_abs_token_margin_linearity_error": float(np.max(np.abs(token_errors))),
            "max_abs_word_margin_linearity_error": float(np.max(np.abs(word_errors))),
            "max_abs_margin_linearity_error": max_linearity,
            "clip_bpe_feature_count": len(tokens),
            "display_word_count": len(display_groups),
        }
        summaries.append(summary)

        for feature_index, (feature, token) in enumerate(
            zip(features, tokens, strict=True)
        ):
            token_rows.append(
                {
                    "sentence_id": sentence_id,
                    "feature_index": feature_index,
                    "feature_text": feature,
                    "clip_bpe_token": token,
                    "negative_shap": float(values[feature_index, 0]),
                    "positive_shap": float(values[feature_index, 1]),
                    "direct_margin_shap": float(values[feature_index, 2]),
                    "derived_margin_shap": float(values[feature_index, 1] - values[feature_index, 0]),
                    "direct_minus_derived": float(token_errors[feature_index]),
                }
            )
        for word_index, group in enumerate(display_groups):
            group_values = np.asarray(group["values"], dtype=float)
            word_rows.append(
                {
                    "sentence_id": sentence_id,
                    "word_index": word_index,
                    "word": group["word"],
                    "feature_indices": " | ".join(map(str, group["feature_indices"])),
                    "is_content": int(group["is_content"]),
                    "negative_shap": float(group_values[0]),
                    "positive_shap": float(group_values[1]),
                    "direct_margin_shap": float(group_values[2]),
                    "derived_margin_shap": float(group_values[1] - group_values[0]),
                    "direct_minus_derived": float(word_errors[word_index]),
                }
            )

        if sentence_id in presentation_ids:
            word_matrix = np.asarray([group["values"] for group in display_groups])
            save_part1_overview(
                plots_dir / "presentation_examples" / f"{sentence_id}_margin_overview",
                sentence_id,
                text,
                gold,
                prediction,
                [str(group["word"]) for group in display_groups],
                word_matrix,
                scores,
                bases,
                calculation["shap_sum"],
                calculation["reconstructed"],
                calculation["residual"],
            )

    legacy_summary = read_csv(project_path(config["legacy_summary_path"]))
    parity = score_parity_rows(summaries, legacy_summary)
    max_parity_error = max(
        max(abs(float(row["negative_difference"])), abs(float(row["positive_difference"])))
        for row in parity
    )
    prediction_mismatches = sum(not int(row["prediction_matches"]) for row in parity)
    maximum_additivity = max(float(row["max_abs_additivity_residual"]) for row in summaries)
    maximum_linearity = max(float(row["max_abs_margin_linearity_error"]) for row in summaries)

    write_csv(values_dir / "sentence_summary.csv", summaries)
    write_csv(values_dir / "token_shap_values.csv", token_rows)
    write_csv(values_dir / "word_shap_values.csv", word_rows)
    write_csv(values_dir / "baseline_parity.csv", parity)
    write_csv(values_dir / "prompt_diagnostics.csv", prompt_rows)
    save_part1_aggregate(plots_dir / "part1_all_sentence_checks", summaries)

    checks = {
        "pilot_rows": len(pilot),
        "pilot_sha256": sha256_file(pilot_path),
        "prompt_family": family_name,
        "maximum_absolute_baseline_score_difference": max_parity_error,
        "baseline_prediction_mismatches": prediction_mismatches,
        "maximum_absolute_additivity_residual": maximum_additivity,
        "maximum_absolute_margin_linearity_error": maximum_linearity,
        "presentation_sentence_ids": sorted(presentation_ids),
        "model_text_evaluations": scorer.text_evaluations,
        "runtime_seconds": time.time() - start,
        "tests": {
            "baseline_parity": max_parity_error <= float(config["baseline_score_tolerance"]),
            "prediction_parity": prediction_mismatches == 0,
            "three_output_additivity": maximum_additivity <= float(config["additivity_tolerance"]),
            "direct_derived_margin_linearity": maximum_linearity <= float(config["margin_linearity_tolerance"]),
        },
    }
    checks["gate_passed"] = all(checks["tests"].values())
    save_json(values_dir / "part1_checks.json", checks)
    save_json(
        values_dir / "run_config.json",
        {
            **config,
            "config_path": str(config_path),
            "device_resolved": str(device),
            "output_names": list(OUTPUT_NAMES),
            "score_definition": "NEG and POS prototype cosine similarities plus POS-minus-NEG margin",
            "shap_algorithm": "partition",
            "masking": "empty-string deletion",
        },
    )

    report = f"""# Part 1 Results — Baseline Parity and Discriminative Margin

The independent runner processed {len(pilot)} pilot sentences with Partition SHAP and
three explained outputs: NEG similarity, POS similarity and POS−NEG margin.

## Gate checks

- Maximum baseline score difference: `{max_parity_error:.3e}`
- Baseline prediction mismatches: `{prediction_mismatches}`
- Maximum three-output additivity residual: `{maximum_additivity:.3e}`
- Maximum direct-versus-derived margin error: `{maximum_linearity:.3e}`
- Gate passed: **{checks['gate_passed']}**

The direct margin explanation was compared token by token and word by word with
`POS SHAP − NEG SHAP`. Five predefined presentation examples were plotted in both
PNG and PDF; full numerical tables were retained for all pilot sentences.
"""
    (reports_dir / "PART1_RESULTS.md").write_text(report, encoding="utf-8")
    print(f"Part 1 gate passed: {checks['gate_passed']}")
    print(f"Maximum baseline difference: {max_parity_error:.3e}")
    print(f"Maximum additivity residual: {maximum_additivity:.3e}")
    print(f"Maximum margin linearity error: {maximum_linearity:.3e}")
    if not checks["gate_passed"]:
        raise RuntimeError("Part 1 gate failed; Part 2 must not proceed.")


if __name__ == "__main__":
    main()
