"""Five aggregate figures and five BERT-CLIP case studies, each saved as PNG and PDF.

Colour roles: BERT = categorical slot 1 (blue), CLIP = slot 2 (orange), random
baseline = muted gray. SHAP matrices use a blue-gray-red diverging scale
(red = pushes the output up, blue = pushes it down), normalized separately within
each model: raw BERT-logit and CLIP-cosine magnitudes are not comparable.
"""

from __future__ import annotations

import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb  # noqa: E402

from io_utils import CASE_STUDY_DIR, METRICS_DIR, PLOTS_DIR, VALUES_DIR, load_json, read_csv  # noqa: E402


MODEL_COLORS = {"bert": "#2a78d6", "clip": "#eb6834"}
MODEL_LABELS = {"bert": "BERT (SST-2 fine-tuned)", "clip": "CLIP text (zero-shot)"}
RANDOM_COLOR = "#898781"
INK, INK_2, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c22f2e"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#f0efec", "#9ec5f4", "#3987e5", "#184f95"])
OUTPUT_LABELS = ("NEG", "POS", "POS−NEG")

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    }
)


def save_pair(figure: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, facecolor="white")
    plt.close(figure)
    return paths


def _num(value: Any) -> float:
    return float(value) if value not in ("", None) else float("nan")


def _ci_lookup() -> dict[tuple[str, str], dict[str, float]]:
    return {
        (row["scope"], row["metric"]): {key: _num(row[key]) for key in ("point_estimate", "ci_lower", "ci_upper")}
        for row in read_csv(METRICS_DIR / "bootstrap_confidence_intervals.csv")
    }


def _hgrid(axis: plt.Axes) -> None:
    axis.yaxis.grid(True)
    axis.set_axisbelow(True)


def _bar_labels(axis: plt.Axes, bars: Any, values: Sequence[float], fmt: str, offsets: Sequence[float] | None = None) -> None:
    for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
        top = bar.get_height() if offsets is None else offsets[index]
        axis.annotate(format(value, fmt), (bar.get_x() + bar.get_width() / 2, top), xytext=(0, 3),
                      textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)


def _legend_models(axis: plt.Axes, extra: Sequence[tuple[str, dict[str, Any]]] = ()) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[key]) for key in ("bert", "clip")]
    labels = [MODEL_LABELS[key] for key in ("bert", "clip")]
    for label, style in extra:
        handles.append(plt.Line2D([0], [0], **style))
        labels.append(label)
    axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=len(labels), fontsize=9)


# ------------------------------------------------------------- figure 1

def figure_performance(stem: Path) -> list[Path]:
    metrics = {row["model"]: row for row in read_csv(METRICS_DIR / "classification_metrics.csv")}
    ci = _ci_lookup()
    names = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced accuracy")]
    figure = plt.figure(figsize=(14, 5.2))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.35)
    axis = figure.add_subplot(grid[0])
    x = np.arange(len(names))
    width = 0.34
    for offset, model in ((-width / 2 - 0.01, "bert"), (width / 2 + 0.01, "clip")):
        points = [_num(metrics[model][name]) for name, _ in names]
        lower = [points[i] - ci[(model, name)]["ci_lower"] for i, (name, _) in enumerate(names)]
        upper = [ci[(model, name)]["ci_upper"] - points[i] for i, (name, _) in enumerate(names)]
        bars = axis.bar(x + offset, points, width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        axis.errorbar(x + offset, points, yerr=[lower, upper], fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
        _bar_labels(axis, bars, points, ".3f", [p + u for p, u in zip(points, upper, strict=True)])
    axis.set_xticks(x, [label for _, label in names])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score (500 sentences)")
    axis.set_title("Classification performance with 95% bootstrap CIs")
    axis.axhline(0.5, color=MUTED, linewidth=0.8)
    axis.annotate("chance level for the balanced set = 0.5", (0.01, 0.5), xycoords=("axes fraction", "data"), xytext=(0, 3),
                  textcoords="offset points", ha="left", fontsize=8, color=INK_2)
    _hgrid(axis)
    _legend_models(axis)
    for position, model in ((1, "bert"), (2, "clip")):
        ax = figure.add_subplot(grid[position])
        row = metrics[model]
        matrix = np.asarray([[int(row["true_neg"]), int(row["false_pos"])], [int(row["false_neg"]), int(row["true_pos"])]])
        ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=250)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]}\n({matrix[i, j] / matrix[i].sum():.1%})", ha="center", va="center",
                        fontsize=11, color="white" if matrix[i, j] > 150 else INK, fontweight="semibold")
        ax.set_xticks([0, 1], ["NEG", "POS"])
        ax.set_yticks([0, 1], ["NEG", "POS"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.spines[:].set_visible(False)
        ax.set_title(f"{MODEL_LABELS[model]}\nconfusion matrix (row %)")
    figure.suptitle("Figure 1. Model performance on the same 500 balanced SST-2 sentences", fontsize=13, fontweight="semibold", x=0.02, ha="left")
    return save_pair(figure, stem)


# ------------------------------------------------------------- figure 2

def figure_additivity(stem: Path) -> list[Path]:
    rows = read_csv(METRICS_DIR / "additivity_summary.csv")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    floor, ceiling = 1e-8, 2e-4
    stats = [("mean", "Mean", "#86b6ef"), ("p95", "95th percentile", "#3987e5"), ("max", "Maximum", "#184f95")]
    for axis, model in zip(axes, ("bert", "clip"), strict=True):
        selected = {row["output"]: row for row in rows if row["model"] == model and row["check"] == "additivity_residual"}
        linearity = {("word" if "word" in row["output"] else "base"): row for row in rows if row["model"] == model and row["check"] == "margin_linearity"}
        outputs = [("NEG", "NEG"), ("POS", "POS"), ("POS-NEG", "POS−NEG")]
        x = np.arange(len(outputs))
        width = 0.27
        for offset_index, (key, label, color) in enumerate(stats):
            values = [_num(selected[name][key]) for name, _ in outputs]
            bars = axis.bar(x + (offset_index - 1) * width, [max(value, floor) for value in values], width * 0.92, color=color, label=label)
            for bar, value in zip(bars, values, strict=True):
                axis.annotate("0" if value == 0 else f"{value:.1e}", (bar.get_x() + bar.get_width() / 2, max(value, floor)),
                              xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.3)
        tolerance = _num(selected["NEG"]["tolerance"])
        axis.axhline(tolerance, color="#c22f2e", linewidth=1.2)
        axis.annotate(f"additivity tolerance {tolerance:.0e}", (0.01, tolerance), xycoords=("axes fraction", "data"), xytext=(0, 4),
                      textcoords="offset points", ha="left", fontsize=8.5, color=INK)
        passes = int(selected["max_over_outputs"]["pass_count"])
        total = int(selected["max_over_outputs"]["sentences"])
        axis.set_title(
            f"{MODEL_LABELS[model]}: additivity pass {passes}/{total}\n"
            f"margin linearity max |φ_margin − (φ_POS − φ_NEG)| = {_num(linearity['word']['max']):.1e} (words), "
            f"{_num(linearity['base']['max']):.1e} (base); tol {_num(linearity['word']['tolerance']):.0e}",
            fontsize=10,
        )
        axis.set_xticks(x, [label for _, label in outputs])
        axis.set_yscale("log")
        axis.set_ylim(floor, ceiling)
        _hgrid(axis)
    axes[0].set_ylabel("|score − (base + Σ word SHAP)|  (log scale)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=3, fontsize=9)
    figure.text(0.02, -0.06, "Residuals are in each model's own units (BERT logits, CLIP cosine similarity) and are not comparable across panels; "
                "each panel is judged against its own tolerance.", fontsize=8.5, color=INK_2)
    figure.suptitle("Figure 2. Local additivity of whole-word Partition SHAP (per output, 500 sentences per model)", fontsize=13,
                    fontweight="semibold", x=0.02, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return save_pair(figure, stem)


# ------------------------------------------------------------- figure 3

def figure_faithfulness(stem: Path) -> list[Path]:
    summary = read_csv(METRICS_DIR / "faithfulness_summary.csv")
    ci = _ci_lookup()
    figure, axes = plt.subplots(1, 4, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.05, 1.05], "wspace": 0.38})
    for axis, model in zip(axes[:2], ("bert", "clip"), strict=True):
        for ranking, style, label in (("shap_abs", "-", "SHAP-ranked (|φ margin|)"), ("random", "--", "random order (5 repeats)")):
            rows = sorted((r for r in summary if r["model"] == model and r["ranking"] == ranking), key=lambda r: _num(r["fraction"]))
            fractions = [0.0] + [_num(r["fraction"]) for r in rows]
            means = [0.0] + [_num(r["median_normalized_drop"]) for r in rows]
            color = MODEL_COLORS[model] if ranking == "shap_abs" else RANDOM_COLOR
            axis.plot(fractions, means, style, color=color, linewidth=2, marker="o", markersize=6, label=label,
                      markeredgecolor="white", markeredgewidth=1.5)
            for fraction, value in zip(fractions[1:], means[1:], strict=True):
                axis.annotate(f"{value:.2f}", (fraction, value), xytext=(0, 6 if ranking == "shap_abs" else -12),
                              textcoords="offset points", ha="center", fontsize=8, color=INK)
        axis.axhline(1.0, color=MUTED, linewidth=0.8)
        axis.annotate("drop = whole original decision score", (0.0, 1.0), xytext=(2, 3), textcoords="offset points", fontsize=7.5, color=INK_2)
        axis.set_ylim(-0.05, 1.15)
        axis.set_xlabel("Fraction of content words deleted")
        axis.set_ylabel("Median normalized decision-score drop")
        axis.set_title(f"{MODEL_LABELS[model]}: deletion curve")
        axis.set_xticks([0, 0.1, 0.2, 0.3, 0.5])
        _hgrid(axis)
        axis.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=1, fontsize=8.5)
    # AOPC: mean (primary, bars) and median (robust, diamonds), each with a 95% CI
    axis = axes[2]
    groups = [("bert", "shap_abs"), ("bert", "random"), ("clip", "shap_abs"), ("clip", "random")]
    x = np.arange(len(groups))
    colors = [MODEL_COLORS[m] if r == "shap_abs" else RANDOM_COLOR for m, r in groups]

    def interval(statistic: str) -> tuple[list[float], list[float], list[float]]:
        items = [ci[(m, f"{statistic}_aopc_{r}")] for m, r in groups]
        points_ = [item["point_estimate"] for item in items]
        return points_, [p - item["ci_lower"] for p, item in zip(points_, items, strict=True)], [item["ci_upper"] - p for p, item in zip(points_, items, strict=True)]

    means, mean_low, mean_up = interval("mean")
    medians, median_low, median_up = interval("median")
    axis.bar(x - 0.12, means, 0.42, color=colors, label="mean (primary)")
    axis.errorbar(x - 0.12, means, yerr=[mean_low, mean_up], fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
    axis.errorbar(x + 0.2, medians, yerr=[median_low, median_up], fmt="D", color=INK, markerfacecolor="white", markersize=6,
                  elinewidth=1, capsize=3, label="median (robust)")
    for index in range(len(groups)):
        axis.annotate(f"mean {means[index]:.2f}\nmed. {medians[index]:.2f}", (x[index], max(means[index] + mean_up[index], medians[index] + median_up[index])),
                      xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.8)
    paired = {r["model"]: r for r in summary if r["ranking"] == "shap_abs_vs_random_AOPC"}
    paired_text = "\n".join(
        f"{model.upper()}: SHAP > random in {_num(paired[model]['fraction_sentences_shap_greater']):.1%} of sentences "
        f"(Wilcoxon p = {_num(paired[model]['wilcoxon_p']):.0e})"
        for model in ("bert", "clip")
    )
    axis.axhline(0, color=BASELINE, linewidth=0.8)
    axis.set_xticks(x, ["BERT\nSHAP", "BERT\nrandom", "CLIP\nSHAP", "CLIP\nrandom"])
    axis.set_ylabel("Normalized deletion AOPC (95% CI)")
    axis.set_title(f"Normalized AOPC: SHAP vs random\n{paired_text}", fontsize=9)
    low = min(min(m - l for m, l in zip(means, mean_low, strict=True)), 0)
    high = max(max(m + u for m, u in zip(means, mean_up, strict=True)), max(m + u for m, u in zip(medians, median_up, strict=True)))
    axis.set_ylim(low * 1.15, high * 1.35)
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, fontsize=8.5)
    _hgrid(axis)
    # flip rates
    axis = axes[3]
    points = [ci[(m, f"mean_flip_rate_{r}")]["point_estimate"] for m, r in groups]
    lower = [p - ci[(m, f"mean_flip_rate_{r}")]["ci_lower"] for p, (m, r) in zip(points, groups, strict=True)]
    upper = [ci[(m, f"mean_flip_rate_{r}")]["ci_upper"] - p for p, (m, r) in zip(points, groups, strict=True)]
    bars = axis.bar(x, points, 0.6, color=colors)
    axis.errorbar(x, points, yerr=[lower, upper], fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
    _bar_labels(axis, bars, points, ".3f", [p + u for p, u in zip(points, upper, strict=True)])
    axis.set_xticks(x, ["BERT\nSHAP", "BERT\nrandom", "CLIP\nSHAP", "CLIP\nrandom"])
    axis.set_ylabel("Prediction-flip rate (mean over 4 fractions)")
    axis.set_title("Prediction flips after deletion")
    axis.set_ylim(0, max(p + u for p, u in zip(points, upper, strict=True)) * 1.25)
    _hgrid(axis)
    figure.suptitle("Figure 3. Deletion faithfulness (fractions 10/20/30/50% of content words; drop normalized by |original decision score| + 1e-6)",
                    fontsize=13, fontweight="semibold", x=0.02, ha="left", y=1.02)
    figure.text(0.02, -0.2, "Curves show medians because normalized drops are heavy-tailed when the original margin is near zero "
                "(common for CLIP). AOPC panel: bars = mean (primary metric), diamonds = median; all whiskers are stratified bootstrap 95% CIs.", fontsize=8.5, color=INK_2)
    return save_pair(figure, stem)


# ------------------------------------------------------------- figure 4

def figure_agreement(stem: Path) -> list[Path]:
    summary = read_csv(METRICS_DIR / "pairwise_explanation_summary.csv")
    pairs = read_csv(METRICS_DIR / "pairwise_explanation_comparison.csv")
    lookup = {(row["metric"], row["subset"]): row for row in summary}
    metrics = [("spearman_abs_margin", "Spearman ρ"), ("top_k_overlap_rate", "Top-5 overlap"),
               ("top_k_jaccard", "Top-5 Jaccard"), ("sign_agreement", "Sign agreement")]
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), gridspec_kw={"width_ratios": [1.15, 1.1, 1], "wspace": 0.62})
    # (a) the four agreement means with CIs; chance level drawn as a tick across each bar
    axis = axes[0]
    x = np.arange(len(metrics))
    rows = [lookup[(name, "all")] for name, _ in metrics]
    points = [_num(row["mean"]) for row in rows]
    lower = [p - _num(row["ci_lower"]) for p, row in zip(points, rows, strict=True)]
    upper = [_num(row["ci_upper"]) - p for p, row in zip(points, rows, strict=True)]
    bars = axis.bar(x, points, 0.55, color="#4a3aa7")
    axis.errorbar(x, points, yerr=[lower, upper], fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
    tick_labels = []
    for index, (bar, row) in enumerate(zip(bars, rows, strict=True)):
        axis.annotate(f"{points[index]:.3f}\n[{_num(row['ci_lower']):.3f}, {_num(row['ci_upper']):.3f}]",
                      (bar.get_x() + bar.get_width() / 2, points[index] + upper[index]), xytext=(0, 3), textcoords="offset points",
                      ha="center", va="bottom", fontsize=8)
        chance = _num(row["chance_level_mean"])
        if math.isfinite(chance) and chance > 0:
            axis.plot([bar.get_x(), bar.get_x() + bar.get_width()], [chance, chance], color="white", linewidth=2.2)
        wrapped_name = metrics[index][1].replace(" ", "\n", 1)
        tick_labels.append(f"{wrapped_name}\n(chance {chance:.2f})")
    axis.set_xticks(x, tick_labels)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Mean over sentences (95% stratified bootstrap CI)")
    n_defined = int(lookup[("spearman_abs_margin", "all")]["n_defined"])
    axis.set_title(f"All 500 sentences (Spearman defined for {n_defined})\nwhite tick = expected value under random word rankings", fontsize=10)
    _hgrid(axis)
    # (b) subgroup table: sequential colour encodes the agreement value, every cell labelled
    axis = axes[1]
    subsets = [("all", "All"), ("gold_NEG", "Gold NEG"), ("gold_POS", "Gold POS"), ("both_correct", "Both models correct"),
               ("not_both_correct", "Not both correct"), ("more_than_top_k_content_words", "> 5 content words")]
    matrix = np.asarray([[_num(lookup[(name, subset)]["mean"]) for name, _ in metrics] for subset, _ in subsets])
    axis.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            row = lookup[(metrics[j][0], subsets[i][0])]
            axis.text(j, i, f"{matrix[i, j]:.3f}\n[{_num(row['ci_lower']):.2f}, {_num(row['ci_upper']):.2f}]", ha="center", va="center",
                      fontsize=7.8, color="white" if matrix[i, j] > 0.62 else INK)
    counts = [int(lookup[("top_k_overlap_rate", subset)]["n_defined"]) for subset, _ in subsets]
    axis.set_yticks(np.arange(len(subsets)), [f"{label}\n(n={count})" for (_, label), count in zip(subsets, counts, strict=True)], fontsize=9)
    axis.set_xticks(np.arange(len(metrics)), [label.replace(" ", "\n", 1) for _, label in metrics])
    axis.xaxis.tick_top()
    axis.tick_params(length=0)
    axis.spines[:].set_visible(False)
    axis.set_title("Mean agreement [95% CI] by subgroup", fontsize=10, pad=34)
    # (c) distribution of the per-sentence Spearman correlations
    axis = axes[2]
    values = np.asarray([_num(row["spearman_abs_margin"]) for row in pairs])
    values = values[np.isfinite(values)]
    counts_hist, _, _ = axis.hist(values, bins=np.linspace(-1, 1, 21), color="#86b6ef", edgecolor="white", linewidth=1.5)
    top = counts_hist.max()
    axis.set_ylim(0, top * 1.25)
    for statistic, label, style, height in ((values.mean(), "mean", "-", 1.18), (np.median(values), "median", "--", 1.1)):
        axis.plot([statistic, statistic], [0, top * height], color=INK, linestyle=style, linewidth=1.3)
        axis.annotate(f"{label} {statistic:.3f}", (statistic, top * height), xytext=(4, -2), textcoords="offset points", fontsize=8.5)
    axis.axvline(0, color=MUTED, linewidth=0.8)
    axis.set_xlabel("Per-sentence Spearman ρ, |φ margin|")
    axis.set_ylabel("Sentences")
    axis.set_title(f"Distribution of Spearman ρ (n = {values.size}, sd {values.std(ddof=1):.3f})", fontsize=10)
    _hgrid(axis)
    figure.suptitle("Figure 4. BERT–CLIP explanation agreement on identical whole-word content units (scale-free metrics only)", fontsize=13,
                    fontweight="semibold", x=0.02, ha="left", y=1.04)
    return save_pair(figure, stem)


# ------------------------------------------------------------- figure 5

def figure_efficiency(stem: Path) -> list[Path]:
    runtime = {row["model"]: row for row in read_csv(METRICS_DIR / "runtime_summary.csv")}
    sentences = read_csv(VALUES_DIR / "sentence_results.csv")
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.8), gridspec_kw={"wspace": 0.35})
    panels = [
        ("mean_shap_runtime_seconds", "Mean SHAP runtime per sentence (s)", ".3f"),
        ("mean_model_evaluations", "Mean model evaluations per sentence\n(identical: same whole-word partition tree)", ".1f"),
        ("total_shap_runtime_seconds", "Total SHAP runtime, 500 sentences (s)\n" + " | ".join(
            f"{m.upper()} {_num(runtime[m]['total_shap_runtime_seconds']) / 60:.1f} min" for m in ("bert", "clip")), ".1f"),
    ]
    models = ("bert", "clip")
    for axis, (field, title, fmt) in zip(axes[:3], panels, strict=True):
        values = [_num(runtime[m][field]) for m in models]
        bars = axis.bar([0, 1], values, 0.5, color=[MODEL_COLORS[m] for m in models])
        _bar_labels(axis, bars, values, fmt)
        axis.set_xticks([0, 1], ["BERT", "CLIP"])
        axis.set_title(title, fontsize=10.5)
        axis.set_ylim(0, max(values) * 1.2)
        _hgrid(axis)
    axis = axes[3]
    for model in models:
        rows = [row for row in sentences if row["model"] == model]
        axis.scatter([int(row["word_count"]) + (0.15 if model == "clip" else -0.15) for row in rows],
                     [int(row["shap_model_evaluations"]) for row in rows], s=14, color=MODEL_COLORS[model], alpha=0.55,
                     edgecolors="white", linewidths=0.4, label=MODEL_LABELS[model])
    budget = int(runtime["bert"]["max_evals_budget"])
    axis.axhline(budget, color="#c22f2e", linewidth=1)
    axis.annotate(f"budget {budget}", (1, budget), xytext=(2, -11), textcoords="offset points", fontsize=8, color=INK)
    reached = " | ".join(f"{m.upper()} {runtime[m]['sentences_reaching_budget']} reach budget" for m in models)
    axis.set_title(f"Evaluations vs whole-word units\n({reached})", fontsize=10.5)
    axis.set_xlabel("Whole-word units in sentence")
    axis.set_ylabel("Model evaluations used")
    axis.legend(loc="lower right", fontsize=8)
    _hgrid(axis)
    device = runtime["bert"]["device"]
    figure.suptitle(f"Figure 5. Explanation cost of whole-word Partition SHAP (max_evals = {budget}, device = {device})", fontsize=13,
                    fontweight="semibold", x=0.02, ha="left", y=1.04)
    return save_pair(figure, stem)


# ------------------------------------------------------------- case studies

def _matrix_panel(axis: plt.Axes, values: np.ndarray, words: Sequence[str], content: Sequence[int], title: str) -> Any:
    limit = max(float(np.max(np.abs(values))), 1e-12)
    image = axis.imshow(values, aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            rgb = np.asarray(to_rgb(DIVERGING(0.5 + 0.5 * value / limit)))
            luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            text = f"{value:+.4f}" if limit < 0.1 else f"{value:+.3f}"
            axis.text(column, row, text, ha="center", va="center", fontsize=7.8, color="white" if luminance < 0.45 else INK)
    labels = [word if flag else f"{word}  (punct.)" for word, flag in zip(words, content, strict=True)]
    axis.set_yticks(np.arange(len(words)), labels, fontsize=8.5)
    axis.set_xticks(np.arange(3), OUTPUT_LABELS)
    axis.xaxis.tick_top()
    axis.tick_params(length=0)
    axis.spines[:].set_visible(False)
    axis.set_title(title, fontsize=10.5, pad=24)
    return image


def figure_case_study(stem: Path, selection: dict[str, str], records: dict[str, dict[str, Any]], tolerances: dict[str, float]) -> list[Path]:
    words = records["bert"]["words"]
    height = max(6.5, 0.3 * len(words) + 4.6)
    figure, axes = plt.subplots(1, 2, figsize=(13.5, height), gridspec_kw={"wspace": 0.55})
    for axis, model in zip(axes, ("bert", "clip"), strict=True):
        record = records[model]
        values = np.asarray(record["word_values"], dtype=float)
        unit = "logit" if model == "bert" else "cosine"
        image = _matrix_panel(axis, values, words, record["is_content"], f"{MODEL_LABELS[model]}  ·  SHAP in {unit} units")
        bar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
        bar.ax.tick_params(labelsize=7.5)
        bar.set_label(f"{model.upper()} SHAP (own scale)", fontsize=8)
        scores, bases = record["scores"], record["base_values"]
        reconstructed, residual = record["reconstructed"], record["additivity_residual"]
        lines = [
            f"gold {record['gold_label']}  |  prediction {record['prediction']}  ({'correct' if record['correct'] else 'incorrect'})",
            "              NEG          POS          POS−NEG",
            "score     " + "  ".join(f"{v:+11.5f}" for v in scores),
            "base      " + "  ".join(f"{v:+11.5f}" for v in bases),
            "recon.    " + "  ".join(f"{v:+11.5f}" for v in reconstructed),
            "|resid.|  " + "  ".join(f"{v:11.1e}" for v in residual) + f"   (tol {tolerances[model]:.0e})",
        ]
        axis.text(0.0, -0.02, "\n".join(lines), transform=axis.transAxes, va="top", ha="left", fontsize=8.2, family="monospace", color=INK)
    wrapped = textwrap.fill(selection["text"], 130)
    figure.subplots_adjust(top=1 - 1.45 / height, bottom=1.75 / height, left=0.1, right=0.95)
    figure.suptitle(f"{selection['sentence_id']} · case study: {selection['role'].replace('_', ' ')}  (gold {selection['gold_label']})\n{wrapped}",
                    fontsize=11.5, x=0.02, ha="left", va="top", y=1 - 0.05 / height)
    figure.text(0.02, 0.05 / height, "Red = pushes that output up, blue = pushes it down. Colour limits are set separately per model "
                "(symmetric at each model's max |SHAP|); BERT and CLIP colour intensities are not comparable.\n"
                "recon. = base + Σ word SHAP; |resid.| = |score − recon.|.", fontsize=8.3, color=INK_2, va="bottom")
    return save_pair(figure, stem)


def load_case_records(sentence_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    import json

    from io_utils import CHECKPOINT_DIR

    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model in ("bert", "clip"):
        with (CHECKPOINT_DIR / f"{model}_shap_records.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["sentence_id"] in sentence_ids:
                    result[record["sentence_id"]][model] = record
    return result


def make_all_plots(config: dict[str, Any]) -> list[Path]:
    load_json(METRICS_DIR / "acceptance_checks.json")
    for path in list(PLOTS_DIR.glob("*.png")) + list(PLOTS_DIR.glob("*.pdf")) + list(CASE_STUDY_DIR.glob("*.png")) + list(CASE_STUDY_DIR.glob("*.pdf")):
        path.unlink()
    written: list[Path] = []
    written += figure_performance(PLOTS_DIR / "fig1_model_performance")
    written += figure_additivity(PLOTS_DIR / "fig2_additivity")
    written += figure_faithfulness(PLOTS_DIR / "fig3_faithfulness")
    written += figure_agreement(PLOTS_DIR / "fig4_bert_clip_agreement")
    written += figure_efficiency(PLOTS_DIR / "fig5_efficiency")
    selections = read_csv(VALUES_DIR / "representative_selection.csv")
    records = load_case_records({row["sentence_id"] for row in selections})
    tolerances = {model: float(config["tolerances"]["additivity"][model]) for model in ("bert", "clip")}
    for selection in selections:
        stem = CASE_STUDY_DIR / f"case{selection['selection_order']}_{selection['sentence_id']}_{selection['role']}"
        written += figure_case_study(stem, selection, records[selection["sentence_id"]], tolerances)
    print(f"[plots] wrote {len(written)} files")
    return written
