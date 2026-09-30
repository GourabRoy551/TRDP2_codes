"""Finalize Part 7 from saved results without recomputing model explanations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from io_utils import load_json, project_path, read_csv, save_json, sha256_file
from plotting import save_final_model_performance


CONFIG = project_path("configs/part7_final_comparison.json")


def main() -> None:
    config = load_json(CONFIG)
    root = project_path(config["output_dir"])
    values = root / "values"
    reports = root / "reports"
    plots = root / "plots"

    rows = read_csv(project_path(config["evaluation_path"]))
    manifest = load_json(project_path(config["evaluation_manifest_path"]))
    part6 = load_json(project_path(config["part6_checks_path"]))
    budget = int(load_json(project_path(config["frozen_budget_manifest_path"]))["selected_budget"])
    bert = read_csv(values / "bert_sentence_results.csv")
    pairwise = read_csv(values / "pairwise_explanation_comparison.csv")
    words = read_csv(values / "bert_word_shap_values.csv")
    performance = read_csv(values / "final_model_performance.csv")
    summaries = read_csv(values / "pairwise_explanation_summary.csv")

    expected_hash = manifest["outputs"]["final_evaluation"]["sha256"]
    actual_hash = sha256_file(project_path(config["evaluation_path"]))
    maximum_residual = max(abs(float(row["maximum_absolute_additivity_residual"])) for row in bert)
    maximum_linearity = max(abs(float(row["maximum_margin_linearity_error"])) for row in bert)
    residuals = np.asarray([abs(float(row["maximum_absolute_additivity_residual"])) for row in bert])
    png_stems = {path.relative_to(plots).with_suffix("") for path in plots.rglob("*.png")}
    pdf_stems = {path.relative_to(plots).with_suffix("") for path in plots.rglob("*.pdf")}
    word_sentence_ids = {row["sentence_id"] for row in words}

    tests = {
        "part6_gate_passed": bool(part6["gate_passed"]),
        "same_frozen_dataset": actual_hash == expected_hash == part6["dataset_sha256"],
        "all_bert_sentences_complete": len(bert) == len(rows) == 500,
        "all_pairwise_comparisons_complete": len(pairwise) == 3 * len(rows),
        "word_alignment_complete": len(word_sentence_ids) == len(rows),
        "bert_additivity": maximum_residual <= float(config["additivity_tolerance"]),
        "bert_margin_linearity": maximum_linearity <= float(config["margin_linearity_tolerance"]),
        "plot_png_pdf_pairing": png_stems == pdf_stems and len(png_stems) == 10,
    }
    prior = load_json(values / "part7_checks.json")
    checks = {
        **prior,
        "complete_run": True,
        "evaluation_rows": len(rows),
        "dataset_sha256": actual_hash,
        "frozen_shap_budget": budget,
        "bert_sentence_rows": len(bert),
        "pairwise_rows": len(pairwise),
        "maximum_bert_additivity_residual": maximum_residual,
        "bert_additivity_residual_mean": float(np.mean(residuals)),
        "bert_additivity_residual_99th_percentile": float(np.quantile(residuals, 0.99)),
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
        "tests": tests,
    }
    checks["gate_passed"] = all(tests.values())
    save_json(values / "part7_checks.json", checks)

    save_final_model_performance(plots / "final_model_performance", performance)
    by_name = {row["condition"]: row for row in performance}
    bert_clip = next(row for row in summaries if row["pair"] == "bert_sst2 vs frozen_linear_head")
    report = f"""# Part 7 Results — BERT and CLIP Final Comparison

All three frozen models were evaluated on the identical balanced 500-sentence set.
BERT was re-explained with the same whole-word deletion masker, Partition-SHAP
algorithm and maximum budget of {budget} used for CLIP.

| Model | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Zero-shot CLIP prompts | {float(by_name['zero_shot_prompt']['accuracy']):.4f} | {float(by_name['zero_shot_prompt']['macro_f1']):.4f} | {float(by_name['zero_shot_prompt']['balanced_accuracy']):.4f} |
| Frozen CLIP + linear head | {float(by_name['frozen_linear_head']['accuracy']):.4f} | {float(by_name['frozen_linear_head']['macro_f1']):.4f} | {float(by_name['frozen_linear_head']['balanced_accuracy']):.4f} |
| Fine-tuned BERT SST-2 | {float(by_name['bert_sst2']['accuracy']):.4f} | {float(by_name['bert_sst2']['macro_f1']):.4f} | {float(by_name['bert_sst2']['balanced_accuracy']):.4f} |

## Explanation comparison

For BERT versus the trained CLIP head, mean absolute-rank Spearman was
`{float(bert_clip['mean_abs_rank_spearman']):.3f}`, mean top-{config['top_k']} overlap was
`{float(bert_clip['mean_top_k_overlap_rate']):.3f}`, and mean sign agreement was
`{float(bert_clip['mean_sign_agreement']):.3f}`. These are scale-independent comparisons;
raw BERT logits and CLIP scores are not compared by magnitude.

- Maximum BERT additivity residual: `{maximum_residual:.3e}` (tolerance `2.0e-5`)
- 99th-percentile BERT additivity residual: `{float(np.quantile(residuals, 0.99)):.3e}`
- Maximum BERT margin-linearity error: `{maximum_linearity:.3e}`
- Part 7 computational gate passed: **{checks['gate_passed']}**

The saved tables separate agreement by correctness category, include prediction
errors, normalized faithfulness, runtime and the five preselected case studies.
"""
    (reports / "PART7_RESULTS.md").write_text(report, encoding="utf-8")
    (root.parent / "PART7_SUMMARY.md").write_text(report, encoding="utf-8")
    frozen = {
        "part": 7,
        "gate_passed": checks["gate_passed"],
        "dataset_sha256": actual_hash,
        "evaluation_rows": len(rows),
        "frozen_shap_budget": budget,
        "models": ["zero_shot_prompt", "frozen_linear_head", "bert_sst2"],
        "result_root": str(root.relative_to(project_path("."))).replace("\\", "/"),
        "note": "Computed results are frozen after the Part 7 integrity gate.",
    }
    save_json(project_path("configs/frozen_part7_results_manifest.json"), frozen)
    if not checks["gate_passed"]:
        raise RuntimeError("Part 7 finalization gate failed.")
    print("Part 7 finalization gate passed.")


if __name__ == "__main__":
    main()
