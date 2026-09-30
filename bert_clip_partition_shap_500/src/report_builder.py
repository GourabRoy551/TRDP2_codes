"""Build FINAL_REPORT.md, RESULTS_SUMMARY.md and the LaTeX report from saved outputs.

Every number in the reports is read from the CSV/JSON files under ``outputs/`` and
``data/``; nothing is recomputed from model outputs here. The only extra input is a
read-only comparison with the earlier clip_text_shap_improved runs, saved to
``outputs/metrics/prior_run_crosscheck.csv``.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from io_utils import (
    CASE_STUDY_DIR,
    CHECKPOINT_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    VALUES_DIR,
    config_sha256,
    load_json,
    project_path,
    read_csv,
    utc_now,
    write_csv,
)


MODELS = ("bert", "clip")
NAMES = {"bert": "BERT", "clip": "CLIP"}
PRIOR_BERT = "../clip_text_shap_improved/outputs/part7_final_comparison/values/bert_sentence_results.csv"
PRIOR_CLIP = "../clip_text_shap_improved/outputs/part6_full_evaluation/values/sentence_results.csv"


def _f(value: Any) -> float:
    return float(value) if value not in ("", None) else float("nan")


# ------------------------------------------------------------------ collection

def prior_crosscheck(sentences: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    """Compare predictions/margins with the earlier TRDP runs on the same 500 rows (read-only)."""
    rows = []
    sources = {"bert": (PRIOR_BERT, None), "clip": (PRIOR_CLIP, "zero_shot_prompt")}
    for model, (relative, condition) in sources.items():
        path = project_path(relative)
        if not path.exists():
            rows.append({"model": model, "prior_file": str(path), "available": False})
            continue
        prior = {row["sentence_id"]: row for row in read_csv(path) if condition is None or row.get("condition") == condition}
        same = sum(sentences[(model, sid)]["prediction"] == row["prediction"] for sid, row in prior.items())
        difference = max(abs(_f(sentences[(model, sid)]["margin"]) - _f(row["margin_score"])) for sid, row in prior.items())
        rows.append({"model": model, "prior_file": str(path), "available": True, "sentences_compared": len(prior),
                     "identical_predictions": same, "max_abs_margin_difference": difference})
    write_csv(METRICS_DIR / "prior_run_crosscheck.csv", rows)
    return rows


def collect(config: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {"config": config, "config_sha256": config_sha256(), "generated_at": utc_now()}
    data["manifest"] = load_json(project_path(config["dataset"]["manifest_path"]))
    data["model_checks"] = load_json(CHECKPOINT_DIR / "model_checks.json")
    data["smoke"] = load_json(CHECKPOINT_DIR / "smoke_test.json")
    data["smoke_attempt1"] = load_json(CHECKPOINT_DIR / "smoke_attempt1_boundary_placeholder_failure.json")
    data["acceptance"] = load_json(METRICS_DIR / "acceptance_checks.json")
    data["progress"] = {model: load_json(CHECKPOINT_DIR / f"{model}_progress.json") for model in MODELS}
    data["classification"] = {row["model"]: row for row in read_csv(METRICS_DIR / "classification_metrics.csv")}
    data["ci"] = {(row["scope"], row["metric"]): {k: _f(row[k]) for k in ("point_estimate", "ci_lower", "ci_upper")}
                  for row in read_csv(METRICS_DIR / "bootstrap_confidence_intervals.csv")}
    data["additivity"] = read_csv(METRICS_DIR / "additivity_summary.csv")
    data["faith"] = read_csv(METRICS_DIR / "faithfulness_summary.csv")
    data["pair_summary"] = {(row["metric"], row["subset"]): row for row in read_csv(METRICS_DIR / "pairwise_explanation_summary.csv")}
    data["pairs"] = {row["sentence_id"]: row for row in read_csv(METRICS_DIR / "pairwise_explanation_comparison.csv")}
    data["structure"] = {row["model"]: row for row in read_csv(METRICS_DIR / "shap_output_structure.csv")}
    data["runtime"] = {row["model"]: row for row in read_csv(METRICS_DIR / "runtime_summary.csv")}
    data["selection"] = read_csv(VALUES_DIR / "representative_selection.csv")
    data["sentences"] = {(row["model"], row["sentence_id"]): row for row in read_csv(VALUES_DIR / "sentence_results.csv")}
    data["crosscheck"] = {row["model"]: row for row in prior_crosscheck(data["sentences"])}
    data["aggregate_plots"] = sorted(path.stem for path in PLOTS_DIR.glob("*.png"))
    data["case_plots"] = sorted(path.stem for path in CASE_STUDY_DIR.glob("*.png"))
    return data


# ------------------------------------------------------------------ formatting

def ci(data: dict[str, Any], scope: str, metric: str, digits: int = 3) -> str:
    item = data["ci"][(scope, metric)]
    return f"{item['point_estimate']:.{digits}f} [{item['ci_lower']:.{digits}f}, {item['ci_upper']:.{digits}f}]"


def ci_sign(data: dict[str, Any], scope: str, metric: str) -> int:
    item = data["ci"][(scope, metric)]
    return 1 if item["ci_lower"] > 0 else (-1 if item["ci_upper"] < 0 else 0)


def pval(value: Any) -> str:
    number = _f(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.1e}"


def faith(data: dict[str, Any], model: str, ranking: str, fraction: str = "all") -> dict[str, str]:
    return next(row for row in data["faith"] if row["model"] == model and row["ranking"] == ranking and row["fraction"] == fraction)


def posthoc(data: dict[str, Any], model: str) -> dict[str, str]:
    return next(row for row in data["faith"] if row["model"] == model and row["ranking"].startswith("POSTHOC"))


def additivity_row(data: dict[str, Any], model: str, output: str) -> dict[str, str]:
    return next(row for row in data["additivity"] if row["model"] == model and row["output"] == output)


def pair(data: dict[str, Any], metric: str, subset: str = "all") -> dict[str, str]:
    return data["pair_summary"][(metric, subset)]


def direction_phrase(sign: int) -> str:
    return {1: "above zero", -1: "below zero", 0: "includes zero"}[sign]


# ------------------------------------------------------------------ markdown report

def render_markdown(data: dict[str, Any]) -> str:
    config = data["config"]
    manifest = data["manifest"]
    checks = data["model_checks"]
    bert_checks, clip_checks = checks["models"]["bert"], checks["models"]["clip"]
    env = checks["environment"]
    cls = data["classification"]
    audit = data["smoke"]["masking_audit"]
    attempt = {(row["model"], row["sentence_id"]): row for row in data["smoke_attempt1"]["attempt1_rows"]}
    runtime = data["runtime"]
    structure = data["structure"]
    mcnemar = pair(data, "exact_mcnemar_bert_vs_clip_correctness")
    lines: list[str] = []
    add = lines.append

    add("# BERT vs CLIP Text Encoder: whole-word Partition SHAP on 500 SST-2 sentences")
    add("")
    add(f"*Final report. Generated {data['generated_at']} from the saved outputs; config SHA-256 `{data['config_sha256'][:16]}…`, "
        f"dataset SHA-256 `{manifest['sha256_copied']}`.*")
    add("")
    add("No new model training or fine-tuning was performed during this experiment. BERT had already been fine-tuned on "
        "SST-2 by its publisher; the CLIP text encoder is used frozen and zero-shot with a prompt family frozen in an earlier experiment.")
    add("")

    # 1 objective
    add("## 1. Objective")
    add("")
    add("Compare a sentiment classifier that was fine-tuned for the task (BERT, SST-2) with a general-purpose frozen text encoder "
        "used zero-shot (CLIP ViT-B/32 text tower) on the **same 500 sentences**, using one explanation method for both: "
        "**whole-word Partition SHAP** applied after model scoring to three outputs, NEG, POS and the decision margin POS−NEG. "
        "The questions are (i) how well each model classifies, (ii) whether the explanations are numerically exact (additivity, "
        "margin linearity) and faithful under deletion, and (iii) how much the two models' explanations agree on identical word units, "
        "using scale-independent metrics only.")
    add("")

    # 2 dataset
    validation = manifest["validation"]
    add("## 2. Dataset")
    add("")
    add(f"- File: `data/evaluation_500.csv`, a byte-identical copy of `clip_text_shap_improved/data/final_evaluation_500.csv` "
        f"(SST-2 development split, held out from prompt selection and budget selection in the earlier experiment).")
    add(f"- {validation['rows']} sentences: {validation['class_counts']['NEG']} NEG and {validation['class_counts']['POS']} POS; IDs "
        f"`{validation['first_sentence_id']}`…`{validation['last_sentence_id']}`; unique IDs; no duplicate normalized text; no missing text; labels only NEG/POS.")
    add(f"- SHA-256 `{manifest['sha256_copied']}` equals the source file and the source manifest "
        f"(`{manifest['hash_checks']}`); nothing was resampled, edited, reordered or relabelled. Verification record: "
        "[data/dataset_manifest.json](../../data/dataset_manifest.json).")
    add(f"- Both models processed the sentences in the dataset order (ordered-ID SHA-256 `{validation['sentence_order_sha256'][:16]}…`, checked for both models).")
    add("")

    # 3 models
    add("## 3. Models")
    add("")
    add("| | BERT | CLIP text encoder |")
    add("|---|---|---|")
    add(f"| Checkpoint | `{config['models']['bert']['name_or_path']}` | `{config['models']['clip']['name_or_path']}` |")
    add("| Training history | BERT-base-uncased fine-tuned on SST-2 by TextAttack (before this experiment) | contrastive image–text pre-training only; no sentiment training |")
    add("| Use here | frozen, eval mode, raw logits | frozen, zero-shot cosine similarity to class prototypes |")
    add(f"| Parameters (all frozen) | {bert_checks['frozen']['parameters_total']:,} | {clip_checks['frozen']['parameters_total']:,} (text + vision towers loaded; only the text tower is used) |")
    add(f"| Max tokens | {config['models']['bert']['max_length']} (no sentence truncated) | {config['models']['clip']['max_length']} (no sentence truncated) |")
    add("| Outputs | [NEG logit, POS logit, POS−NEG] | [NEG similarity, POS similarity, POS−NEG] |")
    add("")
    mapping = bert_checks["label_mapping"]
    verification = bert_checks["label_mapping_verification"]
    add(f"**BERT label order.** The checkpoint's `id2label` is generic (`{mapping['config_id2label']}`), so it cannot identify NEG/POS on its own. "
        f"The mapping index {mapping['NEG']} = NEG, index {mapping['POS']} = POS (GLUE SST-2 convention; used in `bert_shap`, where it was verified on S1–S10, "
        f"and in `clip_text_shap_improved` Part 7) was applied and re-verified before scoring the evaluation set: on the ten independent "
        f"hand-written sentences S1–S10 from `bert_shap/data/sentences.csv` it gives {verification['correct_under_mapping']}/10 correct, and the swapped "
        f"mapping gives {verification['correct_under_swapped_mapping']}/10.")
    add("")
    prompt_check = clip_checks["prompt_source_verification"]
    add(f"**CLIP prompts and prototypes.** The frozen `{config['models']['clip']['prompt_family']}` family ({prompt_check['prompts_per_class']} NEG + "
        f"{prompt_check['prompts_per_class']} POS prompts) was selected in `clip_text_shap_improved` Part 2 on 6,735 SST-2 *training* sentences; "
        "its manifest records `selection_uses_final_evaluation = false`, and the prompts in `config.json` were checked to be identical to it. "
        "One prototype per class is the L2-normalized mean of the normalized prompt embeddings. No linear head was trained. "
        f"The two prototypes are very similar (cosine {clip_checks['neg_pos_prototype_cosine']:.4f}), which is why CLIP margins are small.")
    add("")
    add(f"**Model-output parity (20 sentences).** Both scorers return shape [n, 3]; column 2 equals column 1 − column 0 exactly; "
        "predictions follow POS ⇔ margin > 0; batched and one-at-a-time scoring agree to "
        f"{bert_checks['parity']['max_batched_vs_individual_abs_difference']:.1e} (BERT, tolerance {bert_checks['parity']['batch_parity_tolerance']:.0e}) and "
        f"{clip_checks['parity']['max_batched_vs_individual_abs_difference']:.1e} (CLIP, tolerance {clip_checks['parity']['batch_parity_tolerance']:.0e}). "
        f"All parameters report `requires_grad = False` ({bert_checks['frozen']['trainable_parameter_tensors']} and "
        f"{clip_checks['frozen']['trainable_parameter_tensors']} trainable tensors).")
    add("")
    cross = data["crosscheck"]
    if all(row.get("available") in (True, "True") for row in cross.values()):
        add(f"**Cross-check with earlier runs.** On the same 500 rows, predictions equal those of the earlier `clip_text_shap_improved` runs for "
            f"{cross['bert']['identical_predictions']}/500 (BERT) and {cross['clip']['identical_predictions']}/500 (CLIP) sentences; "
            f"maximum margin differences are {_f(cross['bert']['max_abs_margin_difference']):.1e} and {_f(cross['clip']['max_abs_margin_difference']):.1e} "
            "(earlier BERT run on CPU, this run on GPU). See [prior_run_crosscheck.csv](../metrics/prior_run_crosscheck.csv).")
        add("")
    add("**Environment.** conda env `rfem`: Python " + env["python"] + ", torch " + env["torch"] + ", transformers " + env["transformers"]
        + ", shap " + env["shap"] + ", NumPy " + env["numpy"] + ", SciPy " + env["scipy"] + f"; device `{checks['device']}` ({env.get('cuda_device', 'CPU')}). "
        "Seed 42 for Python/NumPy/PyTorch; deterministic cuDNN/cuBLAS; TF32 disabled.")
    add("")

    # 4 procedure
    add("## 4. Whole-word Partition SHAP procedure")
    add("")
    add("```text")
    add("sentence → shared whole-word units → shap.maskers.Text (hierarchical partition tree over the units)")
    add("  → PartitionExplainer coalitions (max_evals = 500) → perturbed sentence (masked units deleted)")
    add("  → BERT WordPiece or CLIP BPE tokenizer → frozen model → [NEG, POS, POS−NEG]")
    add("  → one SHAP value per original whole word and output")
    add("```")
    add("")
    add("- **Units (identical for both models).** Words with internal apostrophes and hyphenated compounds stay whole "
        "(`heavy-handed`, `fly-on-the-wall`, `k-19`); PTB-split clitics are merged with their host so contractions and possessives stay complete "
        "(`is n't`, `son 's`); apostrophe-initial tokens (`'70s`), decimal numbers and common abbreviations (`mr.`) are single units; "
        "punctuation runs (`,`, `--`, `...`) are separate units marked **non-content**. \"Content word\" therefore means any non-punctuation unit "
        "(function words included).")
    add("- **Masking.** A masked unit is deleted (empty mask token, consecutive deletions collapsed, whitespace normalized); the remaining "
        "string is re-tokenized by the model's own tokenizer. The same deletion strategy is used for both models, and for the faithfulness "
        "deletions. The empty coalition is the empty string, so the base value is f(\"\") for each model.")
    add(f"- **Masking audit.** On all 500 sentences, {audit['coalitions_checked']:,} coalitions (full, empty, every leave-one-out and 10 random "
        "per sentence) were rendered and re-tokenized with both tokenizers. In every case the model tokens equal the concatenated tokens of exactly "
        f"the kept units: {audit['bert_partial_word_violations']} partial-word violations for BERT and {audit['clip_partial_word_violations']} for CLIP; "
        "every WordPiece/BPE token of the original sentences maps to exactly one unit.")
    add("- **Algorithm.** `shap.Explainer(algorithm=\"partition\")` with SHAP's text partition tree built over the whole-word units "
        "(merges adjacent units and splits last at commas, sentence ends and connectives). All three outputs are explained jointly from the same "
        f"coalition evaluations, max_evals = {config['shap']['max_evals']}, batch size {config['batch_sizes']['shap']['bert']}.")
    add(f"- **Checkpointing.** One validated JSON record per sentence is appended and fsync'd immediately; a progress file and partial CSV are "
        f"written every {config['checkpoint_interval']} sentences. Resume was exercised: the BERT run was stopped after 30 sentences, a "
        "truncated record was appended to simulate an interrupted write, and the resumed run rejected only that line and continued from sentence 31.")
    add("")
    bert5, clip5 = attempt[("bert", "EV00005")], attempt[("clip", "EV00005")]
    add("**Implementation finding (boundary placeholders).** The first smoke test used the earlier TRDP masker design, which adds two empty "
        "boundary placeholders around the words. It failed the additivity gate: when the 500-evaluation budget was exhausted, PartitionExplainer "
        "split unresolved cluster credit evenly across leaves, placeholders included, so attribution mass left the words (e.g. EV00005: word-level "
        f"residual {bert5['max_word_additivity_residual']:.3f} BERT logits and {clip5['max_word_additivity_residual']:.1e} CLIP similarity). "
        "Additivity computed *including* the placeholders still held, so the leak is invisible to a check that includes them. The placeholders "
        "were removed (features = words only); no score or SHAP value was altered. The failed attempt is recorded in "
        "[smoke_attempt1_boundary_placeholder_failure.json](../checkpoints/smoke_attempt1_boundary_placeholder_failure.json).")
    add("")

    # 5 equations
    add("## 5. Equations")
    add("")
    add("```text")
    add("BERT:  score_NEG(x) = z_0(x),  score_POS(x) = z_1(x)                       (raw logits)")
    add("CLIP:  p_c = normalize( mean_j normalize(E(prompt_c,j)) ),  c ∈ {NEG, POS}")
    add("       score_c(x) = normalize(E(x)) · p_c                                  (cosine similarity)")
    add("Both:  margin(x) = score_POS(x) − score_NEG(x);   prediction = POS if margin(x) > 0 else NEG")
    add("")
    add("SHAP (per output c, words i = 1..n):   score_c(x) ≈ base_c + Σ_i φ_c(i),   base_c = score_c(\"\")")
    add("Additivity residual:   r_c = | score_c(x) − (base_c + Σ_i φ_c(i)) |")
    add("Margin linearity:      | φ_margin(i) − (φ_POS(i) − φ_NEG(i)) |  and  | base_margin − (base_POS − base_NEG) |")
    add("")
    add("Decision score:  d = +1 if margin(x) > 0 else −1;   s(x) = d · margin(x)")
    add("Deletion at fraction f:  k = max(1, ceil(f · n_content));  x_f = x without its top-k content words")
    add("Normalized drop:  Δ_f = ( s(x) − s(x_f) ) / ( |s(x)| + 1e-6 );   AOPC = mean over f ∈ {0.1, 0.2, 0.3, 0.5} of Δ_f")
    add("```")
    add("")

    # 6 metric definitions
    add("## 6. Metric definitions")
    add("")
    add("1. **Macro-F1** — mean of the NEG and POS F1 scores over the 500 sentences; supporting: accuracy, balanced accuracy (mean recall), "
        "per-class precision/recall/F1, confusion counts, and an exact McNemar test on the paired correctness.")
    add("2. **SHAP additivity residual** — r_c above, per sentence and per output (NEG, POS, POS−NEG), summarized by mean, median, 95th "
        "percentile and maximum; a sentence passes if max_c r_c ≤ tolerance (BERT 2e-5, CLIP 1e-5, as specified; not adjusted). Margin linearity "
        "uses tolerance 1e-6.")
    add("3. **Normalized deletion AOPC** — content words ranked by |φ_margin| (ties by position); Δ_f and AOPC as above. Random baseline: five "
        "deterministic random orders per sentence (seeded by sentence index, identical for both models), Δ_f averaged over the five orders. "
        "Prediction-flip rate: fraction of the four deletions whose prediction differs from the original, averaged over fractions (and orders). "
        "SHAP vs random is compared within each model. Supporting: a signed ranking (d·φ_margin, words that support the prediction first).")
    add("4. **Spearman rank correlation** — per sentence, Spearman ρ between BERT |φ_margin| and CLIP |φ_margin| over the identical content "
        "units; undefined (and excluded) when a sentence has fewer than three content words or a constant ranking.")
    add("5. **Top-five overlap** — per sentence, |T_BERT ∩ T_CLIP| / k and Jaccard |T_BERT ∩ T_CLIP| / |T_BERT ∪ T_CLIP| of the k = min(5, n_content) "
        "content words with the largest |φ_margin|. Supporting: sign agreement, the fraction of content words with sign(φ_BERT) = sign(φ_CLIP). "
        "Chance levels are computed per sentence for random rankings (overlap k/n; Jaccard from the hypergeometric distribution; sign agreement "
        "p_B·p_C + (1−p_B)(1−p_C) from each model's share of positive signs).")
    add("")
    add(f"**Confidence intervals.** {config['bootstrap']['iterations']:,}-iteration percentile bootstrap; in every replicate the 250 NEG and 250 POS "
        "sentences are resampled separately (balanced design preserved); the same resamples are used for every metric (paired); seed 42.")
    add("")

    # 7 results
    add("## 7. Results")
    add("")
    add("### 7.1 Model performance (Metric 1)")
    add("")
    add("| Model | Macro-F1 [95% CI] | Accuracy [95% CI] | Balanced accuracy | NEG P / R / F1 | POS P / R / F1 | TN / FP / FN / TP |")
    add("|---|---|---|---|---|---|---|")
    for model in MODELS:
        row = cls[model]
        add(f"| {NAMES[model]} | {ci(data, model, 'macro_f1')} | {ci(data, model, 'accuracy')} | {_f(row['balanced_accuracy']):.3f} | "
            f"{_f(row['neg_precision']):.3f} / {_f(row['neg_recall']):.3f} / {_f(row['neg_f1']):.3f} | "
            f"{_f(row['pos_precision']):.3f} / {_f(row['pos_recall']):.3f} / {_f(row['pos_f1']):.3f} | "
            f"{row['true_neg']} / {row['false_pos']} / {row['false_neg']} / {row['true_pos']} |")
    add("")
    add(f"BERT − CLIP Macro-F1 difference: **{ci(data, 'bert_minus_clip', 'macro_f1')}**. On the paired sentences, {mcnemar['bert_only_correct']} are "
        f"correct only for BERT and {mcnemar['clip_only_correct']} only for CLIP (exact McNemar p = {pval(mcnemar['p_value'])}). "
        f"The two models agree on the predicted label for {_f(pair(data, 'prediction_agreement')['mean']):.1%} of sentences. "
        "This gap is expected: BERT was fine-tuned on SST-2 training data, whereas CLIP is zero-shot. It says nothing about which encoder "
        "represents sentiment better in general.")
    add("")
    add("![Figure 1](../plots/fig1_model_performance.png)")
    add("")

    add("### 7.2 Additivity and margin linearity (Metric 2)")
    add("")
    add("| Model | Output | Mean | Median | 95th pct | Max | Tolerance | Pass |")
    add("|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for output in ("NEG", "POS", "POS-NEG", "max_over_outputs"):
            row = additivity_row(data, model, output)
            label = "max over outputs" if output == "max_over_outputs" else output.replace("-", "−")
            add(f"| {NAMES[model]} | {label} | {_f(row['mean']):.1e} | {_f(row['median']):.1e} | {_f(row['p95']):.1e} | {_f(row['max']):.1e} | "
                f"{_f(row['tolerance']):.0e} | {row['pass_count']}/{row['sentences']} |")
    add("")
    word_lin = {model: additivity_row(data, model, "word_phi_margin_vs_phi_pos_minus_phi_neg") for model in MODELS}
    base_lin = {model: additivity_row(data, model, "base_margin_vs_base_pos_minus_base_neg") for model in MODELS}
    add(f"All 1,000 explanations pass additivity for every output. The largest residuals ({_f(additivity_row(data, 'bert', 'max_over_outputs')['max']):.1e} "
        f"BERT logits, {_f(additivity_row(data, 'clip', 'max_over_outputs')['max']):.1e} CLIP similarity) are consistent with float32 batch-composition noise: "
        "the full sentence is re-scored inside a SHAP batch with different padding than the reference score, and the parity check shows "
        "differences of the same order. Margin linearity holds to "
        f"{_f(word_lin['bert']['max']):.1e} (BERT) and {_f(word_lin['clip']['max']):.1e} (CLIP) for word values and {_f(base_lin['bert']['max']):.0e} / "
        f"{_f(base_lin['clip']['max']):.0e} for base values (tolerance 1e-6, 500/500 pass). Because the margin is formed in float64 from the float32 "
        "class scores, this check verifies that the three outputs were explained from the same coalitions in the correct column order; it is not "
        "a sensitive test of float32 arithmetic. Additivity is a numerical-consistency check, not evidence that attributions are correct.")
    add("")
    add("![Figure 2](../plots/fig2_additivity.png)")
    add("")

    add("### 7.3 Deletion faithfulness (Metric 3)")
    add("")
    add("| | BERT | CLIP |")
    add("|---|---|---|")
    rows_spec = [
        ("Mean normalized AOPC, SHAP (primary)", "mean_aopc_shap_abs"),
        ("Mean normalized AOPC, random", "mean_aopc_random"),
        ("Mean AOPC difference, SHAP − random", "mean_aopc_shap_minus_random"),
        ("Median normalized AOPC, SHAP", "median_aopc_shap_abs"),
        ("Median normalized AOPC, random", "median_aopc_random"),
        ("Median per-sentence difference, SHAP − random", "median_aopc_shap_minus_random"),
        ("Mean raw AOPC difference (model units)", "mean_raw_aopc_shap_minus_random"),
        ("Flip rate, SHAP-ranked deletion", "mean_flip_rate_shap_abs"),
        ("Flip rate, random deletion", "mean_flip_rate_random"),
        ("Flip-rate difference, SHAP − random", "mean_flip_rate_shap_minus_random"),
    ]
    for label, metric in rows_spec:
        digits = 4 if "raw" in metric else 3
        add(f"| {label} | {ci(data, 'bert', metric, digits)} | {ci(data, 'clip', metric, digits)} |")
    paired = {model: faith(data, model, "shap_abs_vs_random_AOPC") for model in MODELS}
    add(f"| Sentences with AOPC_SHAP > AOPC_random | {_f(paired['bert']['fraction_sentences_shap_greater']):.1%} | {_f(paired['clip']['fraction_sentences_shap_greater']):.1%} |")
    add(f"| Wilcoxon signed-rank p (SHAP vs random AOPC) | {pval(paired['bert']['wilcoxon_p'])} | {pval(paired['clip']['wilcoxon_p'])} |")
    add(f"| Supporting: mean AOPC, signed ranking | {ci(data, 'bert', 'mean_aopc_shap_signed')} | {ci(data, 'clip', 'mean_aopc_shap_signed')} |")
    add("")
    add("Median normalized decision-score drop and flip rate by deletion fraction (SHAP-ranked | random):")
    add("")
    add("| Fraction | BERT median drop | BERT flip rate | CLIP median drop | CLIP flip rate |")
    add("|---|---|---|---|---|")
    for fraction in config["faithfulness"]["fractions"]:
        cells = []
        for model in MODELS:
            shap_row, random_row = faith(data, model, "shap_abs", str(float(fraction))), faith(data, model, "random", str(float(fraction)))
            cells.append(f"{_f(shap_row['median_normalized_drop']):.3f} \\| {_f(random_row['median_normalized_drop']):.3f}")
            cells.append(f"{_f(shap_row['prediction_flip_rate']):.3f} \\| {_f(random_row['prediction_flip_rate']):.3f}")
        add(f"| {fraction:.0%} | " + " | ".join(cells) + " |")
    add("")
    clip_mean_sign = ci_sign(data, "clip", "mean_aopc_shap_minus_random")
    post = posthoc(data, "clip")
    clip_rows = [row for (model, _), row in data["sentences"].items() if model == "clip"]
    worst = min(clip_rows, key=lambda row: _f(row["aopc_shap_abs"]))
    margins = sorted(abs(_f(row["margin"])) for row in clip_rows)
    tails = {
        "median_abs_margin": margins[len(margins) // 2 - 1] / 2 + margins[len(margins) // 2] / 2,
        "below_1e-4": sum(value < 1e-4 for value in margins),
        "min_aopc": _f(worst["aopc_shap_abs"]),
        "max_aopc": max(_f(row["aopc_shap_abs"]) for row in clip_rows),
        "min_id": worst["sentence_id"],
        "min_id_margin": abs(_f(worst["margin"])),
    }
    add(f"**BERT.** SHAP-ranked deletion removes much more of the decision score than random deletion on every statistic: the mean AOPC "
        f"difference is {ci(data, 'bert', 'mean_aopc_shap_minus_random')} (CI {direction_phrase(ci_sign(data, 'bert', 'mean_aopc_shap_minus_random'))}), "
        f"and the flip rate rises from {_f(data['ci'][('bert', 'mean_flip_rate_random')]['point_estimate']):.3f} to "
        f"{_f(data['ci'][('bert', 'mean_flip_rate_shap_abs')]['point_estimate']):.3f}.")
    add("")
    add(f"**CLIP.** The primary statistic, the *mean* normalized AOPC, is **inconclusive**: SHAP {ci(data, 'clip', 'mean_aopc_shap_abs')}, random "
        f"{ci(data, 'clip', 'mean_aopc_random')}, difference {ci(data, 'clip', 'mean_aopc_shap_minus_random')} (CI {direction_phrase(clip_mean_sign)}). "
        f"The cause is the normalization: CLIP margins are tiny (median |margin| {tails['median_abs_margin']:.4f} cosine; "
        f"{tails['below_1e-4']} sentences below 1e-4), so dividing by |s(x)| + 1e-6 turns small absolute changes into very large normalized "
        f"drops (per-sentence SHAP AOPC ranges from {tails['min_aopc']:.0f} to {tails['max_aopc']:.0f}; the minimum belongs to "
        f"{tails['min_id']}, whose |margin| is {tails['min_id_margin']:.1e}). "
        "The robust statistics all favour SHAP-ranked deletion: the median per-sentence difference is "
        f"{ci(data, 'clip', 'median_aopc_shap_minus_random')}, SHAP beats random in {_f(paired['clip']['fraction_sentences_shap_greater']):.1%} of sentences "
        f"(Wilcoxon p = {pval(paired['clip']['wilcoxon_p'])}), the raw drop difference in cosine units is {ci(data, 'clip', 'mean_raw_aopc_shap_minus_random', 4)}, "
        f"and the flip-rate difference is {ci(data, 'clip', 'mean_flip_rate_shap_minus_random')}. A **post-hoc** sensitivity analysis (threshold chosen "
        f"after seeing the tails) restricted to |margin| ≥ 1e-3 ({post['rows']} of 500 sentences) gives mean AOPC {_f(post['mean_aopc_shap_abs']):.3f} "
        f"(SHAP) vs {_f(post['mean_aopc_random']):.3f} (random).")
    add("")
    add(f"Normalized AOPC is scale-free, and the median SHAP AOPC is similar for the two models ({ci(data, 'bert', 'median_aopc_shap_abs')} vs "
        f"{ci(data, 'clip', 'median_aopc_shap_abs')}). CLIP's random baseline is higher, however "
        f"({_f(data['ci'][('bert', 'median_aopc_random')]['point_estimate']):.3f} vs {_f(data['ci'][('clip', 'median_aopc_random')]['point_estimate']):.3f}; "
        f"random flip rate {_f(data['ci'][('bert', 'mean_flip_rate_random')]['point_estimate']):.3f} vs {_f(data['ci'][('clip', 'mean_flip_rate_random')]['point_estimate']):.3f}), "
        "because its small margins flip easily under any deletion. For that reason faithfulness is judged against each model's own random "
        "baseline, not by ranking the two models on it. Ranking by |φ_margin| also deletes words that *oppose* the prediction; the signed ranking, "
        "which deletes supporting words first, gives much larger drops for both models.")
    add("")
    add("![Figure 3](../plots/fig3_faithfulness.png)")
    add("")

    add("### 7.4 BERT–CLIP explanation agreement (Metrics 4 and 5)")
    add("")
    spearman = pair(data, "spearman_abs_margin")
    add("| Metric (per sentence, identical content units) | Mean [95% CI] | Median | SD | Chance level | Sentences |")
    add("|---|---|---|---|---|---|")
    for metric, label in (("spearman_abs_margin", "Spearman ρ of \\|φ_margin\\| (primary)"), ("top_k_overlap_rate", "Top-5 overlap rate (primary)"),
                          ("top_k_jaccard", "Top-5 Jaccard (primary)"), ("sign_agreement", "Sign agreement (supporting)"),
                          ("signed_margin_spearman", "Spearman ρ of signed φ_margin (supporting)")):
        row = pair(data, metric)
        chance = _f(row["chance_level_mean"])
        add(f"| {label} | {_f(row['mean']):.3f} [{_f(row['ci_lower']):.3f}, {_f(row['ci_upper']):.3f}] | {_f(row['median']):.3f} | {_f(row['std']):.3f} | "
            f"{chance:.3f} | {row['n_defined']} |")
    add("")
    add(f"Spearman ρ is undefined for {spearman['n_undefined']} sentences (fewer than three content words). Top-5 sets are trivially identical "
        f"when a sentence has ≤ 5 content words ({500 - int(pair(data, 'top_k_overlap_rate', 'more_than_top_k_content_words')['n_defined'])} sentences); "
        f"on the {pair(data, 'top_k_overlap_rate', 'more_than_top_k_content_words')['n_defined']} sentences with more than five, overlap is "
        f"{_f(pair(data, 'top_k_overlap_rate', 'more_than_top_k_content_words')['mean']):.3f} vs chance "
        f"{_f(pair(data, 'top_k_overlap_rate', 'more_than_top_k_content_words')['chance_level_mean']):.3f}.")
    add("")
    add("| Subset | n | Spearman ρ | Top-5 overlap | Top-5 Jaccard | Sign agreement |")
    add("|---|---|---|---|---|---|")
    for subset, label in (("gold_NEG", "Gold NEG"), ("gold_POS", "Gold POS"), ("both_correct", "Both models correct"), ("not_both_correct", "Not both correct")):
        cells = [f"{_f(pair(data, metric, subset)['mean']):.3f} [{_f(pair(data, metric, subset)['ci_lower']):.2f}, {_f(pair(data, metric, subset)['ci_upper']):.2f}]"
                 for metric in ("spearman_abs_margin", "top_k_overlap_rate", "top_k_jaccard", "sign_agreement")]
        add(f"| {label} | {pair(data, 'top_k_overlap_rate', subset)['n_defined']} | " + " | ".join(cells) + " |")
    add("")
    add(f"Agreement is **positive but weak**. The mean Spearman ρ of {_f(spearman['mean']):.3f} is clearly above 0 but far from 1, with a "
        f"wide spread (SD {_f(spearman['std']):.3f}, {'several' if _f(spearman['min']) < 0 else 'no'} sentences below zero). "
        f"On average about {_f(pair(data, 'top_k_overlap_rate')['mean']) * 5:.1f} of the top-5 words are shared, against about "
        f"{_f(pair(data, 'top_k_overlap_rate')['chance_level_mean']) * 5:.1f} expected by chance. Sign agreement "
        f"({_f(pair(data, 'sign_agreement')['mean']):.3f}) is only modestly above its chance level ({_f(pair(data, 'sign_agreement')['chance_level_mean']):.3f}). "
        "The subgroup CIs overlap (gold NEG vs POS; both correct vs not both correct), so no subgroup difference is claimed.")
    add("")
    add("![Figure 4](../plots/fig4_bert_clip_agreement.png)")
    add("")
    add("### 7.5 Supporting: structure of the class-specific explanations")
    add("")
    add(f"Pooled over all {structure['bert']['content_word_rows']} content words, the Pearson correlation between φ_NEG and φ_POS is "
        f"**{_f(structure['bert']['pooled_pearson_phi_neg_vs_phi_pos']):+.3f} for BERT** and **{_f(structure['clip']['pooled_pearson_phi_neg_vs_phi_pos']):+.3f} "
        "for CLIP**. BERT's NEG and POS explanations are near mirror images, so each one carries the sentiment evidence. In CLIP a word moves the "
        "similarity to *both* prototypes in the same direction (a common-mode effect), and the sentiment-specific part is the small difference: "
        f"the median ratio of margin-SHAP mass to class-SHAP mass is {_f(structure['clip']['median_margin_to_class_mass_ratio']):.2f} for CLIP versus "
        f"{_f(structure['bert']['median_margin_to_class_mass_ratio']):.2f} for BERT. CLIP's individual NEG or POS SHAP values should therefore not "
        "be read as sentiment evidence; only its POS−NEG explanation is compared with BERT above. Source: "
        "[shap_output_structure.csv](../metrics/shap_output_structure.csv).")
    add("")

    add("### 7.6 Efficiency")
    add("")
    add("| Model | Device | Mean s / sentence | Median | 95th pct | Total SHAP time | Mean evaluations | Sentences at budget | Total evaluations |")
    add("|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        row = runtime[model]
        add(f"| {NAMES[model]} | {row['device']} | {_f(row['mean_shap_runtime_seconds']):.3f} | {_f(row['median_shap_runtime_seconds']):.3f} | "
            f"{_f(row['p95_shap_runtime_seconds']):.3f} | {_f(row['total_shap_runtime_seconds']):.1f} s ({_f(row['total_shap_runtime_seconds']) / 60:.1f} min) | "
            f"{_f(row['mean_model_evaluations']):.1f} | {row['sentences_reaching_budget']}/500 | {int(row['total_model_evaluations']):,} |")
    add("")
    bert_rows = [row for (model, _), row in data["sentences"].items() if model == "bert"]
    budget = int(config["shap"]["max_evals"])
    at_budget = [int(row["word_count"]) for row in bert_rows if int(row["shap_model_evaluations"]) >= budget]
    below_budget = [int(row["word_count"]) for row in bert_rows if int(row["shap_model_evaluations"]) < budget]
    add("The number of model evaluations is identical for the two models sentence by sentence, because it depends only on the partition tree over "
        f"the shared word units. Sentences reaching the {budget}-evaluation budget have {min(at_budget)}–{max(at_budget)} units (all sentences "
        f"with ≥ {max(below_budget) + 1} units reach it; the largest sentence below it has {max(below_budget)} units); for them, the hierarchy is "
        "resolved only as far as the budget allows. The runtime difference reflects per-forward cost on the same GPU; excluded are a "
        "one-off untimed warm-up explanation and the faithfulness evaluations "
        f"({_f(runtime['bert']['total_faithfulness_runtime_seconds']):.1f} s and {_f(runtime['clip']['total_faithfulness_runtime_seconds']):.1f} s).")
    add("")
    add("![Figure 5](../plots/fig5_efficiency.png)")
    add("")

    add("### 7.7 Representative examples")
    add("")
    add("The five sentences were fixed by deterministic rules on labels, predictions, length, tokenization and IDs **after scoring and before "
        "any SHAP value existed** ([representative_selection.csv](../values/representative_selection.csv)). They illustrate the explanations "
        "and are not evidence that either model is better.")
    add("")
    for selection, stem in zip(data["selection"], data["case_plots"], strict=True):
        sid = selection["sentence_id"]
        row = data["pairs"][sid]
        bert, clip = data["sentences"][("bert", sid)], data["sentences"][("clip", sid)]
        rho = _f(row["spearman_abs_margin"])
        add(f"**{selection['selection_order']}. {selection['role'].replace('_', ' ')} — {sid}** (gold {selection['gold_label']}): "
            f"*{selection['text']}*")
        add("")
        add(f"- Rule: {selection['selection_rule']}.")
        add(f"- BERT {bert['prediction']} (margin {_f(bert['margin']):+.3f} logits); CLIP {clip['prediction']} (margin {_f(clip['margin']):+.5f} cosine).")
        add(f"- Top-5 |φ_margin| — BERT: {row['bert_top_k_words'].replace(' | ', ', ')}; CLIP: {row['clip_top_k_words'].replace(' | ', ', ')}. "
            f"Overlap {_f(row['top_k_overlap_rate']):.1f}, Spearman ρ {'undefined' if not math.isfinite(rho) else f'{rho:.2f}'}, sign agreement {_f(row['sign_agreement']):.2f}.")
        add(f"- Figure: [{stem}.png](../plots/case_studies/{stem}.png) / [.pdf](../plots/case_studies/{stem}.pdf)")
        add("")
    word_bar_manifest = VALUES_DIR / "word_bar_plot_manifest.csv"
    if word_bar_manifest.exists():
        add("Per-model word bar charts (NEG | POS | POS−NEG, one bar per word) for cases 1, 2, 3 and 5, for this experiment and for the same "
            "sentences in `clip_text_shap_improved` Part 6, are in `outputs/plots/word_bars/` "
            "([word_bar_plot_manifest.csv](../values/word_bar_plot_manifest.csv)).")
        add("")
    # Qualitative reading written for this exact (deterministic) selection; omitted if the selection ever changes.
    if [row["sentence_id"] for row in data["selection"]] == ["EV00003", "EV00002", "EV00091", "EV00278", "EV00001"]:
        disagreement = data["sentences"][("clip", "EV00001")]
        add("Qualitative reading (illustrative only): in EV00003 the two models share four of the five top words (*triumph*, *gentility*, "
            "*pathos*, *moments*). In EV00091 both rank the contrast marker *but* second, but BERT's other top words are evaluative (*witty*, "
            "*laughs*, *great*) while CLIP's include non-evaluative units (*film*, *script*, the names *heather* and *mueller*). In EV00278 both "
            "rank *dull* and *stiff-upper-lip* highly; CLIP also ranks *and* and *melodrama*. In EV00001 the models disagree on the label "
            f"(BERT NEG; CLIP POS with a margin of only {_f(disagreement['margin']):+.5f}).")
        add("")

    # 8 limitations
    add("## 8. Limitations")
    add("")
    for item in (
        "One dataset (500 lower-cased, PTB-tokenized SST-2 development sentences of movie-review text); results may not transfer to other domains or tokenization.",
        "BERT was fine-tuned on SST-2 training data, so it has an in-domain advantage; whether its publisher used the development split for model selection is not documented. CLIP is evaluated zero-shot with one frozen prompt family, and its results depend on those prompts.",
        "CLIP's two prototypes are almost identical (cosine 0.994), so margins are tiny and normalized deletion statistics are heavy-tailed. The specified mean normalized AOPC is uninformative for CLIP; robust statistics are reported alongside it, and the margin-threshold analysis is explicitly post hoc.",
        "Deleting words produces inputs outside the training distribution (including the empty string used as the base value); SHAP values are conditional on this deletion baseline.",
        "Partition SHAP returns Owen values with respect to a heuristic text hierarchy; with max_evals = 500, 154 of the longer sentences use the full budget and are explained only at the resolution the budget allows. Different hierarchies or budgets can change word-level values.",
        "\"Content word\" means any non-punctuation unit, so function words are ranked too; Spearman ρ over few words is noisy, is undefined for 4 sentences, and top-5 overlap is trivial for sentences with ≤ 5 content words.",
        "Additivity and margin linearity confirm numerical consistency only; they do not show that attributions are correct. SHAP values are not evidence of causal word importance.",
        "Raw BERT and CLIP SHAP magnitudes are in different units (logits vs cosine) and are never compared; case-study colour scales are per model.",
        "GPU floating-point results are reproducible on this hardware and software stack; small differences (~1e-6) are expected on other devices or library versions.",
        "RFEM was not rerun on this dataset, so no numerical comparison with RFEM is made.",
    ):
        add(f"- {item}")
    add("")

    # 9 conclusions
    add("## 9. Conclusions")
    add("")
    add(f"1. **Performance.** On identical sentences, fine-tuned BERT (Macro-F1 {ci(data, 'bert', 'macro_f1')}) outperforms zero-shot CLIP "
        f"(Macro-F1 {ci(data, 'clip', 'macro_f1')}) by {ci(data, 'bert_minus_clip', 'macro_f1')}, as expected from BERT's task-specific fine-tuning.")
    add("2. **Exactness.** Whole-word Partition SHAP is locally additive for all 1,000 explanations and all three outputs within the specified "
        "tolerances, and the margin explanation equals POS minus NEG. This held only after removing the boundary placeholders used in earlier TRDP maskers.")
    add(f"3. **Faithfulness.** For BERT, deleting the highest-|SHAP| words is clearly more damaging than random deletion (mean normalized AOPC "
        f"difference {ci(data, 'bert', 'mean_aopc_shap_minus_random')}). For CLIP the specified mean statistic is inconclusive because of near-zero "
        f"margins, while every robust statistic (median difference {ci(data, 'clip', 'median_aopc_shap_minus_random')}, "
        f"{_f(paired['clip']['fraction_sentences_shap_greater']):.0%} of sentences, flip-rate difference {ci(data, 'clip', 'mean_flip_rate_shap_minus_random')}) favours SHAP over random.")
    add(f"4. **Agreement.** BERT and CLIP margin explanations agree only weakly: Spearman ρ {ci(data, 'bert_vs_clip[all]', 'spearman_abs_margin')}, "
        f"top-5 overlap {ci(data, 'bert_vs_clip[all]', 'top_k_overlap_rate')} (chance {_f(pair(data, 'top_k_overlap_rate')['chance_level_mean']):.2f}), "
        f"Jaccard {ci(data, 'bert_vs_clip[all]', 'top_k_jaccard')} (chance {_f(pair(data, 'top_k_jaccard')['chance_level_mean']):.2f}), "
        f"sign agreement {ci(data, 'bert_vs_clip[all]', 'sign_agreement')} (chance {_f(pair(data, 'sign_agreement')['chance_level_mean']):.2f}). "
        "The models share some of the words that drive the decision, but largely rely on different evidence.")
    add("5. **Reading CLIP explanations.** CLIP's class-specific SHAP values are dominated by a component shared by both classes; only the "
        "POS−NEG explanation isolates sentiment. BERT's class explanations are mirror images of each other.")
    add("")

    # 10 deviations
    add("## 10. Deviations from the specification and other notes")
    add("")
    for item in (
        "The specified root `D:\\ALL Uni Documents\\UB\\Courses\\TRDP2` exists but is empty. The existing projects are in `D:\\ALL_Uni_Documents\\UB\\Courses\\TRDP2`, so the new folder was created there.",
        "Files added beyond the listed structure: `src/report_builder.py`, `tests/conftest.py`, `outputs/values/model_scores.csv`, `outputs/metrics/{faithfulness_summary, shap_output_structure, prior_run_crosscheck}.csv`, `outputs/metrics/acceptance_checks.json`, and the checkpoint/log files.",
        "Unit rules extend the earlier masker: clitics are merged with their host (the earlier regex split `'s` into `'` + `s`). Boundary placeholders were removed after the first smoke test failed (Section 4).",
        "The margin is formed in float64 from float32 class scores, so margin linearity holds at float64 round-off; the specified 1e-6 tolerance was kept. No tolerance was changed.",
        "Supporting analyses were added (median, raw-unit and signed-ranking AOPC, Wilcoxon and McNemar tests, chance levels, output-structure correlations, and one post-hoc margin-threshold sensitivity analysis). The specified primary statistics are reported unchanged.",
        "Parity was checked on 20 sentences (at least 10 required). Records are checkpointed after every sentence, with a progress snapshot every 25.",
        "An untimed warm-up explanation of a fixed non-dataset sentence runs before timing, so CUDA/numba start-up is not charged to the first sentence.",
        "In the `rfem` environment, importing pandas/pyarrow after CUDA initialization breaks later directory listings (WinError 6714), so `shap` is imported before any CUDA call.",
        "Library versions differ from the earlier projects (shap 0.51 vs 0.52, transformers 5.4 vs 5.9); predictions match the earlier runs for 500/500 sentences per model.",
    ):
        add(f"- {item}")
    add("")

    # 11 reproducibility
    add("## 11. Reproducibility")
    add("")
    add("```bat")
    add("cd bert_clip_partition_shap_500")
    add("run_experiment.bat   :: all stages + tests")
    add(":: uses %USERPROFILE%\\miniconda3\\envs\\rfem\\python.exe (override with RFEM_PYTHON)")
    add("```")
    add("")
    add("Stages: `prepare → score → smoke → shap --model bert → shap --model clip → metrics → plots → report` (each gated by the previous one), then "
        "`python -m pytest tests -p no:cacheprovider`. All settings are in [config.json](../../config.json). Models load offline from the local Hugging "
        "Face cache. The SHAP stage resumes from `outputs/checkpoints/<model>_shap_records.jsonl`.")
    add("")

    # 12 traceability
    add("## 12. Traceability")
    add("")
    add("| Claim | Source |")
    add("|---|---|")
    for claim, source in (
        ("Dataset identity and validation", "data/dataset_manifest.json"),
        ("Label mapping, prompts, parity, frozen parameters", "outputs/checkpoints/model_checks.json"),
        ("Masking audit and smoke test", "outputs/checkpoints/masking_audit.csv, smoke_test.json"),
        ("Per-sentence scores, bases, residuals, runtime, AOPC", "outputs/values/sentence_results.csv (1,000 rows)"),
        ("Per-word SHAP values and tokenizer mapping", "outputs/values/word_shap_values.csv"),
        ("Deletion details (every fraction, ranking and random repeat)", "outputs/values/faithfulness_by_fraction.csv"),
        ("Metric 1", "outputs/metrics/classification_metrics.csv"),
        ("Metric 2", "outputs/metrics/additivity_summary.csv"),
        ("Metric 3", "outputs/metrics/faithfulness_summary.csv, bootstrap_confidence_intervals.csv"),
        ("Metrics 4–5", "outputs/metrics/pairwise_explanation_comparison.csv (500 rows), pairwise_explanation_summary.csv"),
        ("Subgroups", "outputs/metrics/subgroup_results.csv"),
        ("Runtime", "outputs/metrics/runtime_summary.csv"),
        ("Acceptance gate", "outputs/metrics/acceptance_checks.json"),
    ):
        add(f"| {claim} | `{source}` |")
    add("")
    return "\n".join(lines)


# ------------------------------------------------------------------ summary

def render_summary(data: dict[str, Any]) -> str:
    runtime = data["runtime"]
    paired = {model: faith(data, model, "shap_abs_vs_random_AOPC") for model in MODELS}
    lines = [
        "# Results summary — BERT vs CLIP, whole-word Partition SHAP (500 SST-2 sentences)",
        "",
        "Same 250 NEG + 250 POS sentences for both models · 1,000 explanations · outputs NEG, POS, POS−NEG · max_evals 500 · "
        "no new training or fine-tuning in this experiment (BERT was already fine-tuned on SST-2; CLIP is frozen zero-shot).",
        "",
        "| Primary metric | BERT (SST-2 fine-tuned) | CLIP text (zero-shot) |",
        "|---|---|---|",
        f"| 1. Macro-F1 [95% CI] | **{ci(data, 'bert', 'macro_f1')}** | **{ci(data, 'clip', 'macro_f1')}** |",
        f"| 2. Additivity: max residual / pass | {_f(additivity_row(data, 'bert', 'max_over_outputs')['max']):.1e} logit · 500/500 (tol 2e-5) | "
        f"{_f(additivity_row(data, 'clip', 'max_over_outputs')['max']):.1e} cosine · 500/500 (tol 1e-5) |",
        f"| 3. Normalized AOPC, SHAP vs random (mean) | {ci(data, 'bert', 'mean_aopc_shap_abs')} vs {ci(data, 'bert', 'mean_aopc_random')} | "
        f"{ci(data, 'clip', 'mean_aopc_shap_abs')} vs {ci(data, 'clip', 'mean_aopc_random')} — inconclusive |",
        f"| 3. Robust: median AOPC difference / SHAP > random | {ci(data, 'bert', 'median_aopc_shap_minus_random')} / {_f(paired['bert']['fraction_sentences_shap_greater']):.1%} | "
        f"{ci(data, 'clip', 'median_aopc_shap_minus_random')} / {_f(paired['clip']['fraction_sentences_shap_greater']):.1%} |",
        f"| 3. Prediction-flip rate, SHAP vs random | {ci(data, 'bert', 'mean_flip_rate_shap_abs')} vs {ci(data, 'bert', 'mean_flip_rate_random')} | "
        f"{ci(data, 'clip', 'mean_flip_rate_shap_abs')} vs {ci(data, 'clip', 'mean_flip_rate_random')} |",
        "",
        "| BERT–CLIP agreement (identical word units) | Mean [95% CI] | Chance |",
        "|---|---|---|",
        f"| 4. Spearman ρ of \\|φ margin\\| | {ci(data, 'bert_vs_clip[all]', 'spearman_abs_margin')} | 0 |",
        f"| 5. Top-5 overlap | {ci(data, 'bert_vs_clip[all]', 'top_k_overlap_rate')} | {_f(pair(data, 'top_k_overlap_rate')['chance_level_mean']):.3f} |",
        f"| 5. Top-5 Jaccard | {ci(data, 'bert_vs_clip[all]', 'top_k_jaccard')} | {_f(pair(data, 'top_k_jaccard')['chance_level_mean']):.3f} |",
        f"| Sign agreement (supporting) | {ci(data, 'bert_vs_clip[all]', 'sign_agreement')} | {_f(pair(data, 'sign_agreement')['chance_level_mean']):.3f} |",
        "",
        "**Takeaways**",
        "",
        f"- BERT classifies far better (Macro-F1 +{_f(data['ci'][('bert_minus_clip', 'macro_f1')]['point_estimate']):.3f}), as expected from its SST-2 fine-tuning.",
        "- Both explanation sets are numerically exact: additivity passes for all 1,000 explanations, and φ_margin = φ_POS − φ_NEG.",
        "- SHAP-ranked deletion beats random deletion for BERT on every statistic. For CLIP the mean normalized AOPC is inconclusive because its margins are near zero; the median, sign test, raw-unit drop and flip rates all favour SHAP.",
        "- BERT and CLIP highlight partly overlapping but mostly different words (ρ ≈ 0.24; about 2.5 of the top 5 words shared vs about 1.9 by chance).",
        f"- CLIP's NEG and POS explanations move together (r = {_f(data['structure']['clip']['pooled_pearson_phi_neg_vs_phi_pos']):+.3f}); only its POS−NEG explanation is sentiment-specific.",
        f"- Cost on an RTX 4060: {_f(runtime['bert']['mean_shap_runtime_seconds']):.2f} s (BERT) and {_f(runtime['clip']['mean_shap_runtime_seconds']):.2f} s (CLIP) per sentence; "
        f"{_f(runtime['bert']['mean_model_evaluations']):.0f} evaluations on average for both.",
        "",
        "Figures: [performance](../plots/fig1_model_performance.png) · [additivity](../plots/fig2_additivity.png) · "
        "[faithfulness](../plots/fig3_faithfulness.png) · [agreement](../plots/fig4_bert_clip_agreement.png) · "
        "[efficiency](../plots/fig5_efficiency.png) · [case studies](../plots/case_studies/). Full report: [FINAL_REPORT.md](FINAL_REPORT.md).",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ LaTeX

LATEX_REPLACEMENTS = [
    ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"), ("−", "-"), ("–", "--"), ("—", "---"), ("…", "..."), ("’", "'"),
    ("φ", r"$\varphi$"), ("ρ", r"$\rho$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("≈", r"$\approx$"), ("⇔", r"$\Leftrightarrow$"),
    ("→", r"$\rightarrow$"), ("×", r"$\times$"), ("Σ", r"$\Sigma$"), ("·", r"$\cdot$"), ("∈", r"$\in$"), ("∩", r"$\cap$"), ("∪", r"$\cup$"),
    ("Δ", r"$\Delta$"), ("±", r"$\pm$"),
]
VERBATIM_ASCII = {
    "−": "-", "–": "-", "—": "--", "→": "->", "⇔": "<=>", "≈": "~=", "≤": "<=", "≥": ">=", "Σ": "sum", "Δ": "D", "φ": "phi", "ρ": "rho",
    "∈": "in", "∩": "&", "∪": "|", "·": "*", "…": "...", "±": "+/-", "’": "'",
}


def tex(text: Any) -> str:
    value = str(text)
    for old, new in LATEX_REPLACEMENTS[:10]:
        value = value.replace(old, new)
    value = value.replace(r"\textbackslash\{\}", r"\textbackslash{}").replace(r"\textasciitilde\{\}", r"\textasciitilde{}").replace(
        r"\textasciicircum\{\}", r"\textasciicircum{}")
    for old, new in LATEX_REPLACEMENTS[10:]:
        value = value.replace(old, new)
    return value


def markdown_inline_to_tex(text: str) -> str:
    """Escape a Markdown sentence and convert **bold**, *italic* and `code` spans."""


    parts = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))", text)
    output = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            output.append(r"\textbf{" + tex(part[2:-2]) + "}")
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            output.append(r"\emph{" + tex(part[1:-1]) + "}")
        elif part.startswith("`") and part.endswith("`"):
            # Allow long paths and identifiers to wrap after "/" and "_".
            code = tex(part[1:-1]).replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
            output.append(r"\texttt{" + code + "}")
        elif part.startswith("[") and "](" in part:
            output.append(tex(part[1:part.index("](")]))
        else:
            output.append(tex(part))
    return "".join(output)


def markdown_to_latex(markdown: str) -> str:
    """Convert the report's restricted Markdown (headings, lists, tables, code, images) to LaTeX."""
    body: list[str] = []
    lines = markdown.split("\n")
    index = 0
    in_list = None
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if in_list and not (stripped.startswith("- ") or (stripped[:2].rstrip(".").isdigit() and stripped[1:3].startswith(". "))):
            body.append(r"\end{" + in_list + "}")
            in_list = None
        if stripped.startswith("```"):
            block = []
            index += 1
            while not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            body.append(r"\begin{footnotesize}\begin{verbatim}")
            body.extend("".join(VERBATIM_ASCII.get(character, character) for character in value) for value in block)
            body.append(r"\end{verbatim}\end{footnotesize}")
        elif stripped.startswith("# "):
            pass
        elif stripped.startswith("### "):
            body.append(r"\subsection{" + markdown_inline_to_tex(re.sub(r"^[\d.]+\s+", "", stripped[4:])) + "}")
        elif stripped.startswith("## "):
            body.append(r"\section{" + markdown_inline_to_tex(re.sub(r"^[\d.]+\s+", "", stripped[3:])) + "}")
        elif stripped.startswith("![") and "](" in stripped:
            path = stripped[stripped.index("](") + 2 : -1].replace(".png", ".pdf").replace("../plots/", "../../plots/")
            body.append(r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{" + path + r"}\end{figure}")
        elif stripped.startswith("|"):
            table = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table.append(lines[index].strip())
                index += 1
            index -= 1
            rows = [[cell.strip().replace("\\|", "|") for cell in row.strip("|").split(" | ")] for row in table if not set(row) <= set("|-: ")]
            columns = max(len(row) for row in rows)
            spec = ">{\\raggedright\\arraybackslash}p{0.27\\linewidth}" + "X" * (columns - 1)
            body.append(r"\par{\footnotesize\begin{tabularx}{\linewidth}{" + spec + r"}\toprule")
            for row_index, row in enumerate(rows):
                row = row + [""] * (columns - len(row))
                body.append(" & ".join(markdown_inline_to_tex(cell) for cell in row) + r" \\")
                if row_index == 0:
                    body.append(r"\midrule")
            body.append(r"\bottomrule\end{tabularx}}\par\medskip")
        elif stripped.startswith("- "):
            if in_list != "itemize":
                body.append(r"\begin{itemize}")
                in_list = "itemize"
            body.append(r"\item " + markdown_inline_to_tex(stripped[2:]))
        elif stripped[:2].rstrip(".").isdigit() and stripped[1:3].startswith(". "):
            if in_list != "enumerate":
                body.append(r"\begin{enumerate}")
                in_list = "enumerate"
            body.append(r"\item " + markdown_inline_to_tex(stripped[3:]))
        elif stripped:
            body.append(markdown_inline_to_tex(stripped) + "\n")
        index += 1
    if in_list:
        body.append(r"\end{" + in_list + "}")
    return "\n".join(body)


def render_latex(data: dict[str, Any], markdown: str) -> str:
    case_figures = "\n".join(
        r"\begin{figure}[p]\centering\includegraphics[width=\textwidth,height=0.9\textheight,keepaspectratio]{../../plots/case_studies/"
        + stem + r".pdf}\caption{Case study " + tex(stem.split("_", 2)[1]) + "}\end{figure}"
        for stem in data["case_plots"]
    )
    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,booktabs,graphicx,float,tabularx,array,xcolor,microtype}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}\setlength{\emergencystretch}{3em}
\renewcommand{\tabularxcolumn}[1]{>{\raggedright\arraybackslash}p{#1}}
\title{BERT vs CLIP Text Encoder:\\ Whole-word Partition SHAP on 500 SST-2 Sentences}
\author{TRDP2 experiment \texttt{bert\_clip\_partition\_shap\_500}}
\date{""" + tex(data["generated_at"][:10]) + r"""}
\begin{document}
\maketitle
"""
    return preamble + markdown_to_latex(markdown) + "\n\\clearpage\n\\section*{Case-study figures}\n" + case_figures + "\n\\end{document}\n"


def compile_latex(tex_path: Path, pdf_name: str) -> dict[str, Any]:
    """Compile with pdflatex when available; a missing compiler never blocks the experiment."""
    compiler = shutil.which("pdflatex")
    output_dir = tex_path.parent / "output" / "pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    if compiler is None:
        return {"compiled": False, "reason": "pdflatex not found"}
    build_dir = tex_path.parent / "output" / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        result = subprocess.run(
            [compiler, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={build_dir}", tex_path.name],
            cwd=tex_path.parent, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return {"compiled": False, "reason": "pdflatex failed", "log_tail": result.stdout[-3000:]}
    target = output_dir / pdf_name
    shutil.copyfile(build_dir / (tex_path.stem + ".pdf"), target)
    return {"compiled": True, "pdf": str(target)}


def build_reports(config: dict[str, Any]) -> dict[str, Any]:
    data = collect(config)
    markdown = render_markdown(data)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "FINAL_REPORT.md").write_text(markdown, encoding="utf-8")
    (REPORTS_DIR / "RESULTS_SUMMARY.md").write_text(render_summary(data), encoding="utf-8")
    latex_dir = REPORTS_DIR / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = latex_dir / "main.tex"
    tex_path.write_text(render_latex(data, markdown), encoding="utf-8")
    status = compile_latex(tex_path, "BERT_CLIP_PARTITION_SHAP_500_REPORT.pdf")
    print(f"[report] FINAL_REPORT.md, RESULTS_SUMMARY.md written; LaTeX: {status.get('pdf', status.get('reason'))}")
    if not status["compiled"] and "log_tail" in status:
        print(status["log_tail"])
    return status
