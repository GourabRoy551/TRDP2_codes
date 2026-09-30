"""Numerically annotated figures for Parts 1 and 2 (always PNG and PDF)."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


OUTPUT_NAMES = ("NEG", "POS", "POS−NEG")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def save_pair(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"))
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def _signed_matrix(
    axis: plt.Axes,
    values: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    fmt: str = "+.4f",
) -> None:
    matrix = np.asarray(values, dtype=float)
    limit = max(float(np.max(np.abs(matrix))) if matrix.size else 0.0, 1e-12)
    axis.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_xticks(np.arange(len(column_labels)), labels=column_labels)
    axis.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            axis.text(
                column,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(value) > 0.58 * limit else "black",
            )


def save_part1_overview(
    stem: Path,
    sentence_id: str,
    text: str,
    gold: str,
    prediction: str,
    words: Sequence[str],
    word_values: np.ndarray,
    scores: Sequence[float],
    bases: Sequence[float],
    shap_sums: Sequence[float],
    reconstructed: Sequence[float],
    residuals: Sequence[float],
) -> None:
    summary_labels = ["Score", "Base", "Σ SHAP", "Reconstructed", "Residual"]
    summary = np.vstack([scores, bases, shap_sums, reconstructed, residuals])
    height = max(7.0, 0.34 * len(words) + 3.8)
    figure, axes = plt.subplots(
        1, 2, figsize=(15, height), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    _signed_matrix(
        axes[0], word_values, words, OUTPUT_NAMES, "Word-level Partition SHAP"
    )
    scaled = summary / np.maximum(np.max(np.abs(summary), axis=1, keepdims=True), 1e-12)
    axes[1].imshow(scaled, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[1].set_xticks(np.arange(3), labels=OUTPUT_NAMES)
    axes[1].set_yticks(np.arange(5), labels=summary_labels)
    axes[1].set_title("Sentence-level numerical checks")
    for row in range(5):
        for column in range(3):
            value = float(summary[row, column])
            axes[1].text(
                column,
                row,
                f"{value:.2e}" if row == 4 else f"{value:+.4f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(scaled[row, column]) > 0.58 else "black",
            )
    figure.suptitle(
        f"{sentence_id}: CLIP text SHAP with discriminative margin\n"
        f"gold={gold} | prediction={prediction}\n{textwrap.fill(text, 110)}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_pair(figure, stem)


def save_part1_aggregate(stem: Path, summaries: list[dict[str, Any]]) -> None:
    values = np.asarray(
        [
            [
                row["negative_score"],
                row["positive_score"],
                row["margin_score"],
                row["max_abs_additivity_residual"],
                row["max_abs_margin_linearity_error"],
            ]
            for row in summaries
        ],
        dtype=float,
    )
    columns = ["NEG score", "POS score", "POS−NEG", "Max add. residual", "Margin error"]
    scaled = values / np.maximum(np.max(np.abs(values), axis=0, keepdims=True), 1e-12)
    labels = [f"{row['sentence_id']} G:{row['gold_label']}/P:{row['prediction']}" for row in summaries]
    figure, axis = plt.subplots(figsize=(13, max(7.5, 0.42 * len(labels) + 2.5)))
    axis.imshow(scaled, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(np.arange(len(columns)), labels=columns, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_title("Part 1: scores, additivity and direct-versus-derived margin checks")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            fmt = ".2e" if column >= 3 else "+.4f"
            axis.text(
                column,
                row,
                format(float(values[row, column]), fmt),
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if abs(scaled[row, column]) > 0.58 else "black",
            )
    figure.tight_layout()
    save_pair(figure, stem)


def save_prompt_metrics(stem: Path, rows: list[dict[str, Any]], selected: str) -> None:
    families = [str(row["family"]) for row in rows]
    x = np.arange(len(families))
    width = 0.25
    figure, axis = plt.subplots(figsize=(13, 7))
    metrics = [
        ("accuracy", "Accuracy", "#4C78A8"),
        ("macro_f1", "Macro-F1", "#F58518"),
        ("balanced_accuracy", "Balanced accuracy", "#54A24B"),
    ]
    for offset, (key, label, colour) in enumerate(metrics):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(x + (offset - 1) * width, values, width, label=label, color=colour)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=8)
    axis.set_ylim(0, 1.08)
    axis.set_xticks(x, labels=[name.replace("_", "\n") for name in families])
    axis.set_ylabel("Validation metric")
    axis.set_title(f"Part 2 prompt validation (selected: {selected})")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    save_pair(figure, stem)


def save_confusion_grid(stem: Path, rows: list[dict[str, Any]], selected: str) -> None:
    columns = 3
    rows_count = math.ceil(len(rows) / columns)
    figure, axes = plt.subplots(rows_count, columns, figsize=(13, 4.2 * rows_count))
    flat = np.asarray(axes, dtype=object).reshape(-1)
    for axis, row in zip(flat, rows, strict=False):
        matrix = np.asarray(
            [[row["true_neg"], row["false_pos"]], [row["false_neg"], row["true_pos"]]],
            dtype=int,
        )
        axis.imshow(matrix, cmap="Blues")
        axis.set_xticks([0, 1], labels=["Pred NEG", "Pred POS"])
        axis.set_yticks([0, 1], labels=["Gold NEG", "Gold POS"])
        title = str(row["family"]).replace("_", " ")
        axis.set_title(("★ " if row["family"] == selected else "") + title)
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=12)
    for axis in flat[len(rows) :]:
        axis.axis("off")
    figure.suptitle("Prompt-family validation confusion matrices", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_pair(figure, stem)


def save_agreement_matrix(
    stem: Path, families: Sequence[str], agreement: np.ndarray
) -> None:
    matrix = np.asarray(agreement, dtype=float)
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    labels = [name.replace("_", "\n") for name in families]
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_title("Prediction agreement between prompt families")
    for row in range(len(families)):
        for column in range(len(families)):
            value = float(matrix[row, column])
            axis.text(column, row, f"{value:.3f}", ha="center", va="center")
    figure.tight_layout()
    save_pair(figure, stem)


def save_margin_distribution(
    stem: Path, prediction_rows: list[dict[str, Any]], selected: str
) -> None:
    families = list(dict.fromkeys(str(row["family"]) for row in prediction_rows))
    figure, axes = plt.subplots(len(families), 1, figsize=(12, 2.7 * len(families)), sharex=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for axis, family in zip(axes, families, strict=True):
        negative = [float(row["margin"]) for row in prediction_rows if row["family"] == family and row["gold_label"] == "NEG"]
        positive = [float(row["margin"]) for row in prediction_rows if row["family"] == family and row["gold_label"] == "POS"]
        axis.boxplot([negative, positive], vert=False, tick_labels=["Gold NEG", "Gold POS"], showmeans=True)
        axis.axvline(0, color="black", linewidth=0.9)
        axis.set_title(("★ " if family == selected else "") + family.replace("_", " "))
        axis.text(0.99, 0.08, f"mean NEG={np.mean(negative):+.4f} | mean POS={np.mean(positive):+.4f}", transform=axis.transAxes, ha="right", fontsize=8)
        axis.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel("POS−NEG cosine-similarity margin")
    figure.suptitle("Prompt-family margin distributions on internal validation", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save_pair(figure, stem)


def save_shap_sensitivity(stem: Path, rows: list[dict[str, Any]], selected: str) -> None:
    shown = [row for row in rows if row["family"] != selected]
    names = [str(row["family"]).replace("_", "\n") for row in shown]
    x = np.arange(len(shown))
    figure, axis = plt.subplots(figsize=(12, 6.5))
    width = 0.34
    spearman = [float(row["mean_spearman_abs_rank"]) for row in shown]
    jaccard = [float(row["mean_top_k_jaccard"]) for row in shown]
    bars1 = axis.bar(x - width / 2, spearman, width, label="Mean Spearman rank", color="#4C78A8")
    bars2 = axis.bar(x + width / 2, jaccard, width, label="Mean top-k Jaccard", color="#F58518")
    axis.bar_label(bars1, labels=[f"{v:.3f}" for v in spearman], padding=2)
    axis.bar_label(bars2, labels=[f"{v:.3f}" for v in jaccard], padding=2)
    axis.set_ylim(-0.05, 1.08)
    axis.set_xticks(x, labels=names)
    axis.set_ylabel("Agreement with selected prompt family")
    axis.set_title(f"Development-only margin-SHAP prompt sensitivity (reference: {selected})")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    save_pair(figure, stem)


def save_part3_word_comparison(
    stem: Path,
    sentence_id: str,
    text: str,
    gold: str,
    words: Sequence[str],
    bpe_aggregated: np.ndarray,
    direct_word: np.ndarray,
    bpe_residual: float,
    word_residual: float,
    rank_correlation: float,
) -> None:
    """Show post-hoc BPE aggregation beside genuine whole-word masking."""
    difference = np.asarray(direct_word) - np.asarray(bpe_aggregated)
    height = max(8.0, 0.34 * len(words) + 3.5)
    figure, axes = plt.subplots(1, 3, figsize=(20, height), sharey=True)
    _signed_matrix(
        axes[0], bpe_aggregated, words, OUTPUT_NAMES, "BPE SHAP aggregated to words"
    )
    _signed_matrix(
        axes[1], direct_word, words, OUTPUT_NAMES, "Direct whole-word Partition SHAP"
    )
    _signed_matrix(
        axes[2], difference, words, OUTPUT_NAMES, "Direct word − BPE aggregate"
    )
    figure.suptitle(
        f"{sentence_id}: BPE versus whole-word hierarchical masking\n"
        f"gold={gold} | margin-rank Spearman={rank_correlation:.3f} | "
        f"residuals: BPE={bpe_residual:.2e}, word={word_residual:.2e}\n"
        f"{textwrap.fill(text, 125)}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_pair(figure, stem)


def save_part3_aggregate(stem: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["sentence_id"]) for row in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(2, 1, figsize=(max(14, 0.42 * len(rows)), 11), sharex=True)
    axes[0].bar(x - 0.2, [row["margin_abs_rank_spearman"] for row in rows], 0.4, label="Spearman", color="#4C78A8")
    axes[0].bar(x + 0.2, [row["top_k_jaccard"] for row in rows], 0.4, label="Top-k Jaccard", color="#F58518")
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].set_ylabel("BPE/word agreement")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    for index, row in enumerate(rows):
        axes[0].text(index - 0.2, row["margin_abs_rank_spearman"] + 0.02, f"{row['margin_abs_rank_spearman']:.2f}", ha="center", fontsize=7)
        axes[0].text(index + 0.2, row["top_k_jaccard"] + 0.02, f"{row['top_k_jaccard']:.2f}", ha="center", fontsize=7)
    bpe_runtime = [float(row["bpe_runtime_seconds"]) for row in rows]
    word_runtime = [float(row["word_runtime_seconds"]) for row in rows]
    axes[1].bar(x - 0.2, bpe_runtime, 0.4, label="BPE", color="#72B7B2")
    axes[1].bar(x + 0.2, word_runtime, 0.4, label="Whole word", color="#E45756")
    axes[1].set_ylabel("Runtime (seconds)")
    axes[1].set_xticks(x, labels=labels, rotation=55, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle("Part 3: explanation agreement and runtime by stability sentence", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    save_pair(figure, stem)


def save_training_curves(stem: Path, rows: list[dict[str, Any]], selected_decay: float) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    decays = sorted({float(row["weight_decay"]) for row in rows})
    for decay in decays:
        selected = [row for row in rows if float(row["weight_decay"]) == decay]
        epochs = [int(row["epoch"]) for row in selected]
        label = f"weight decay={decay:g}" + (" ★" if decay == selected_decay else "")
        axes[0].plot(epochs, [float(row["train_loss"]) for row in selected], label=label)
        axes[1].plot(epochs, [float(row["validation_macro_f1"]) for row in selected], label=label)
    axes[0].set_title("Frozen-embedding training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Validation macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1.02)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Part 4: linear-head regularization and stopping selection", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_pair(figure, stem)


def save_model_comparison(
    stem: Path, zero_shot: dict[str, Any], trained: dict[str, Any]
) -> None:
    metrics = ["accuracy", "macro_f1", "balanced_accuracy"]
    labels = ["Accuracy", "Macro-F1", "Balanced accuracy"]
    x = np.arange(len(metrics))
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    width = 0.34
    zero_values = [float(zero_shot[key]) for key in metrics]
    trained_values = [float(trained[key]) for key in metrics]
    bars1 = axes[0].bar(x - width / 2, zero_values, width, label="Zero-shot prompts", color="#4C78A8")
    bars2 = axes[0].bar(x + width / 2, trained_values, width, label="Frozen CLIP + linear head", color="#F58518")
    axes[0].bar_label(bars1, labels=[f"{v:.3f}" for v in zero_values], padding=2)
    axes[0].bar_label(bars2, labels=[f"{v:.3f}" for v in trained_values], padding=2)
    axes[0].set_xticks(x, labels=labels)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Internal-validation performance")
    axes[0].legend()
    matrices = [
        np.asarray([[zero_shot["true_neg"], zero_shot["false_pos"]], [zero_shot["false_neg"], zero_shot["true_pos"]]]),
        np.asarray([[trained["true_neg"], trained["false_pos"]], [trained["false_neg"], trained["true_pos"]]]),
    ]
    comparison = np.column_stack([matrices[0].reshape(-1), matrices[1].reshape(-1)])
    axes[1].imshow(comparison, cmap="Blues")
    axes[1].set_xticks([0, 1], labels=["Zero-shot", "Linear head"])
    axes[1].set_yticks(np.arange(4), labels=["TN", "FP", "FN", "TP"])
    axes[1].set_title("Confusion-matrix counts")
    for row in range(4):
        for column in range(2):
            axes[1].text(column, row, str(int(comparison[row, column])), ha="center", va="center", fontsize=11)
    figure.suptitle("Part 4: zero-shot versus trained frozen-CLIP head", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_pair(figure, stem)


def save_budget_stability(
    stem: Path,
    rows: list[dict[str, Any]],
    reference_budget: int,
    thresholds: dict[str, float],
) -> None:
    """Plot the three explanation-agreement criteria used to choose a budget."""
    metrics = [
        ("median_abs_rank_spearman", "Median absolute-rank Spearman", thresholds["spearman"]),
        ("median_top_k_overlap_rate", "Median top-k overlap", thresholds["top_k_overlap"]),
        ("median_sign_agreement", "Median sign agreement", thresholds["sign_agreement"]),
    ]
    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    colours = ["#4C78A8", "#F58518"]
    markers = ["o", "s"]
    linestyles = ["--", "-"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
    for axis, (key, title, threshold) in zip(axes, metrics, strict=True):
        for condition_index, (colour, condition) in enumerate(
            zip(colours, conditions, strict=False)
        ):
            selected = sorted(
                [row for row in rows if row["condition"] == condition],
                key=lambda row: int(row["budget"]),
            )
            budgets = [int(row["budget"]) for row in selected]
            values = [float(row[key]) for row in selected]
            axis.plot(
                budgets,
                values,
                marker=markers[condition_index],
                markersize=8 if condition_index == 0 else 5,
                markerfacecolor="white" if condition_index == 0 else colour,
                linestyle=linestyles[condition_index],
                linewidth=2.0,
                label=condition,
                color=colour,
                zorder=4 - condition_index,
            )
            for budget, value in zip(budgets, values, strict=True):
                offset = 10 if condition_index == 0 else -15
                axis.annotate(f"{value:.3f}", (budget, value), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=8, color=colour)
        axis.axhline(threshold, color="#E45756", linestyle="--")
        axis.text(
            0.02,
            threshold - 0.045,
            f"criterion ≥ {threshold:.2f}",
            transform=axis.get_yaxis_transform(),
            color="#C43C39",
            fontsize=8,
        )
        axis.axvline(reference_budget, color="black", linestyle=":", alpha=0.55)
        axis.set_xticks(sorted({int(row["budget"]) for row in rows}))
        axis.set_ylim(-0.03, 1.08)
        axis.set_xlabel("SHAP maximum evaluations")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Agreement with 2,000-evaluation reference")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles[: len(conditions)], labels[: len(conditions)], loc="lower center", ncol=2)
    figure.suptitle("Part 5: whole-word SHAP stability by evaluation budget", fontsize=14)
    figure.tight_layout(rect=(0, 0.09, 1, 0.94))
    save_pair(figure, stem)


def save_budget_efficiency(stem: Path, rows: list[dict[str, Any]]) -> None:
    """Show measured runtime and actual perturbed-text evaluations."""
    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    colours = ["#4C78A8", "#F58518"]
    markers = ["o", "s"]
    linestyles = ["--", "-"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for condition_index, (colour, condition) in enumerate(
        zip(colours, conditions, strict=False)
    ):
        selected = sorted(
            [row for row in rows if row["condition"] == condition],
            key=lambda row: int(row["budget"]),
        )
        budgets = [int(row["budget"]) for row in selected]
        runtime = [float(row["mean_runtime_seconds"]) for row in selected]
        evaluations = [float(row["mean_actual_model_evaluations"]) for row in selected]
        style = {
            "marker": markers[condition_index],
            "markersize": 8 if condition_index == 0 else 5,
            "markerfacecolor": "white" if condition_index == 0 else colour,
            "linestyle": linestyles[condition_index],
            "linewidth": 2,
            "label": condition,
            "color": colour,
            "zorder": 4 - condition_index,
        }
        axes[0].plot(budgets, runtime, **style)
        axes[1].plot(budgets, evaluations, **style)
        for budget, value in zip(budgets, runtime, strict=True):
            offset = 10 if condition_index == 0 else -15
            axes[0].annotate(f"{value:.2f}s", (budget, value), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=8, color=colour)
        for budget, value in zip(budgets, evaluations, strict=True):
            offset = 10 if condition_index == 0 else -15
            axes[1].annotate(f"{value:.0f}", (budget, value), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=8, color=colour)
    axes[0].set_title("Mean runtime per sentence")
    axes[0].set_ylabel("Seconds")
    axes[1].set_title("Mean actual model evaluations")
    axes[1].set_ylabel("Perturbed sentences")
    for axis in axes:
        axis.set_xticks(sorted({int(row["budget"]) for row in rows}))
        axis.set_xlabel("Requested SHAP maximum evaluations")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Part 5: explanation cost by CLIP condition", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_pair(figure, stem)


def save_budget_additivity(stem: Path, rows: list[dict[str, Any]]) -> None:
    """Create an annotated condition-by-budget additivity matrix."""
    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    budgets = sorted({int(row["budget"]) for row in rows})
    lookup = {(str(row["condition"]), int(row["budget"])): row for row in rows}
    matrix = np.asarray(
        [
            [float(lookup[(condition, budget)]["maximum_additivity_residual"]) for budget in budgets]
            for condition in conditions
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    limit = max(float(np.max(matrix)), np.finfo(float).eps)
    axis.imshow(matrix, cmap="Greens_r", vmin=0.0, vmax=limit)
    axis.set_xticks(np.arange(len(budgets)), labels=[str(value) for value in budgets])
    axis.set_yticks(np.arange(len(conditions)), labels=conditions)
    axis.set_xlabel("SHAP maximum evaluations")
    axis.set_title("Maximum absolute additivity residual (all pass tolerance 1×10⁻⁵)")
    for row_index in range(len(conditions)):
        for column_index in range(len(budgets)):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2e}", ha="center", va="center", color="black", fontsize=10)
    figure.suptitle("Part 5: numerical reconstruction check", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_pair(figure, stem)


def save_budget_selection_summary(
    stem: Path,
    rows: list[dict[str, Any]],
    selected_budget: int,
    determinism_rows: list[dict[str, Any]],
) -> None:
    """Summarize pass/fail decisions and deterministic-repeat differences."""
    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    budgets = sorted({int(row["budget"]) for row in rows})
    lookup = {(str(row["condition"]), int(row["budget"])): row for row in rows}
    gate = np.asarray(
        [[int(bool(lookup[(condition, budget)]["all_thresholds_passed"])) for budget in budgets] for condition in conditions],
        dtype=int,
    )
    repeat_lookup: dict[str, float] = {}
    for condition in conditions:
        selected = [row for row in determinism_rows if row["condition"] == condition]
        repeat_lookup[condition] = max(float(row["maximum_absolute_shap_difference"]) for row in selected)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    axes[0].imshow(gate, cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(len(budgets)), labels=[str(value) for value in budgets])
    axes[0].set_yticks(np.arange(len(conditions)), labels=conditions)
    axes[0].set_xlabel("SHAP maximum evaluations")
    axes[0].set_title("All stability and additivity criteria")
    for row_index, condition in enumerate(conditions):
        for column_index, budget in enumerate(budgets):
            word = "PASS" if gate[row_index, column_index] else "FAIL"
            marker = "\nSELECTED" if budget == selected_budget else ""
            axes[0].text(column_index, row_index, word + marker, ha="center", va="center", fontsize=9)
    differences = [repeat_lookup[condition] for condition in conditions]
    bars = axes[1].bar(conditions, differences, color=["#4C78A8", "#F58518"][: len(conditions)])
    display_limit = max(max(differences) * 1.25, 1e-12)
    axes[1].set_ylim(0.0, display_limit)
    for bar, value in zip(bars, differences, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            display_limit * 0.05,
            f"{value:.2e}\nEXACT REPEAT" if value == 0.0 else f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[1].set_title(f"Deterministic repeats at selected budget {selected_budget}")
    axes[1].set_ylabel("Maximum absolute SHAP difference")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle("Part 5: budget-selection decision", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_pair(figure, stem)


def save_part6_classification(stem: Path, metrics: list[dict[str, Any]]) -> None:
    conditions = [str(row["condition"]) for row in metrics]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    x = np.arange(len(conditions))
    width = 0.24
    for offset, (key, label, colour) in enumerate(
        [
            ("accuracy", "Accuracy", "#4C78A8"),
            ("macro_f1", "Macro-F1", "#F58518"),
            ("balanced_accuracy", "Balanced accuracy", "#54A24B"),
        ]
    ):
        values = [float(row[key]) for row in metrics]
        bars = axes[0].bar(x + (offset - 1) * width, values, width, label=label, color=colour)
        axes[0].bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2)
    axes[0].set_xticks(x, labels=conditions)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Final held-out classification metrics")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    matrix = np.asarray(
        [
            [row["true_neg"], row["false_pos"], row["false_neg"], row["true_pos"]]
            for row in metrics
        ],
        dtype=float,
    )
    axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks(np.arange(4), labels=["TN", "FP", "FN", "TP"])
    axes[1].set_yticks(np.arange(len(conditions)), labels=conditions)
    axes[1].set_title("Confusion-matrix counts")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axes[1].text(column_index, row_index, str(int(matrix[row_index, column_index])), ha="center", va="center")
    figure.suptitle("Part 6: frozen CLIP conditions on 500 held-out SST-2 sentences", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_pair(figure, stem)


def save_part6_global_importance(stem: Path, rows: list[dict[str, Any]]) -> None:
    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    figure, axes = plt.subplots(1, len(conditions), figsize=(9 * len(conditions), 8))
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for axis, condition in zip(axes, conditions, strict=True):
        selected = sorted(
            [row for row in rows if row["condition"] == condition],
            key=lambda row: -float(row["mean_absolute_margin_shap"]),
        )[:15]
        words = [str(row["word"]) for row in selected][::-1]
        values = [float(row["mean_absolute_margin_shap"]) for row in selected][::-1]
        bars = axis.barh(words, values, color="#4C78A8" if condition.startswith("zero") else "#F58518")
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=8)
        axis.set_title(condition)
        axis.set_xlabel("Mean absolute margin SHAP")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Part 6: globally prominent sentiment words", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_pair(figure, stem)


def save_part6_faithfulness(stem: Path, rows: list[dict[str, Any]]) -> None:
    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    colours = ["#4C78A8", "#F58518"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for colour, condition in zip(colours, conditions, strict=False):
        fractions = sorted({float(row["fraction"]) for row in rows if row["condition"] == condition})
        top, random, sufficiency = [], [], []
        for fraction in fractions:
            selected = [row for row in rows if row["condition"] == condition and float(row["fraction"]) == fraction]
            top.append(float(np.mean([float(row["comprehensiveness_drop"]) for row in selected])))
            random.append(float(np.mean([float(row["random_deletion_drop"]) for row in selected])))
            sufficiency.append(float(np.mean([float(row["sufficiency_gap"]) for row in selected])))
        axes[0].plot(fractions, top, marker="o", linewidth=2, label=f"{condition}: SHAP-ranked", color=colour)
        axes[0].plot(fractions, random, marker="x", linestyle="--", label=f"{condition}: random", color=colour)
        axes[1].plot(fractions, sufficiency, marker="o", linewidth=2, label=condition, color=colour)
        for axis, values in ((axes[0], top), (axes[1], sufficiency)):
            for fraction, value in zip(fractions, values, strict=True):
                axis.annotate(f"{value:+.3f}", (fraction, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    axes[0].set_title("Comprehensiveness: deletion drop")
    axes[0].set_ylabel("Prediction-direction score drop")
    axes[1].set_title("Sufficiency: retained-word gap")
    axes[1].set_ylabel("Prediction-direction score gap")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Fraction of content words")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Part 6: whole-word SHAP faithfulness versus random deletion", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_pair(figure, stem)


def save_part6_subgroups(stem: Path, rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["group_type"] in {"gold_label", "length_group"}]
    conditions = list(dict.fromkeys(str(row["condition"]) for row in selected))
    groups = list(dict.fromkeys(f"{row['group_type']}:{row['group']}" for row in selected))
    x = np.arange(len(groups))
    width = 0.36
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    for condition_index, condition in enumerate(conditions):
        lookup = {f"{row['group_type']}:{row['group']}": row for row in selected if row["condition"] == condition}
        accuracy = [float(lookup[group]["accuracy"]) for group in groups]
        aopc = [float(lookup[group]["mean_top_deletion_aopc"]) for group in groups]
        offset = (condition_index - (len(conditions) - 1) / 2) * width
        bars1 = axes[0].bar(x + offset, accuracy, width, label=condition)
        bars2 = axes[1].bar(x + offset, aopc, width, label=condition)
        axes[0].bar_label(bars1, labels=[f"{value:.3f}" for value in accuracy], fontsize=7, padding=2)
        axes[1].bar_label(bars2, labels=[f"{value:+.3f}" for value in aopc], fontsize=7, padding=2)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Classification by gold class and sentence length")
    axes[1].set_ylabel("Mean top-deletion AOPC")
    axes[1].set_title("Explanation faithfulness by subgroup")
    axes[1].set_xticks(x, labels=[group.replace(":", "\n") for group in groups])
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Part 6: subgroup results", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_pair(figure, stem)


def save_part6_efficiency(stem: Path, rows: list[dict[str, Any]]) -> None:
    conditions = list(dict.fromkeys(str(row["condition"]) for row in rows))
    figure, axes = plt.subplots(1, 3, figsize=(17, 6))
    summaries = []
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        summaries.append(
            [
                float(np.mean([float(row["shap_runtime_seconds"]) for row in selected])),
                float(np.mean([float(row["shap_model_evaluations"]) for row in selected])),
                float(max(float(row["maximum_absolute_additivity_residual"]) for row in selected)),
            ]
        )
    labels = ["Mean runtime (s)", "Mean evaluations", "Max residual"]
    colours = ["#4C78A8", "#F58518"]
    for metric_index, axis in enumerate(axes):
        values = [summary[metric_index] for summary in summaries]
        bars = axis.bar(conditions, values, color=colours[: len(conditions)])
        fmt = ".2e" if metric_index == 2 else ".2f"
        axis.bar_label(bars, labels=[format(value, fmt) for value in values], padding=3)
        axis.set_title(labels[metric_index])
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Part 6: SHAP efficiency and numerical reconstruction", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_pair(figure, stem)


def save_part6_example_comparison(
    stem: Path,
    selection: dict[str, Any],
    words: Sequence[str],
    condition_results: dict[str, dict[str, Any]],
) -> None:
    conditions = list(condition_results)
    height = max(8.0, 0.36 * len(words) + 4.0)
    figure, axes = plt.subplots(1, len(conditions), figsize=(9 * len(conditions), height), sharey=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    subtitles = []
    for axis, condition in zip(axes, conditions, strict=True):
        result = condition_results[condition]
        _signed_matrix(axis, result["word_values"], words, OUTPUT_NAMES, condition)
        subtitles.append(
            f"{condition}: pred={result['prediction']}, margin={result['scores'][2]:+.4f}, "
            f"max residual={result['maximum_residual']:.2e}"
        )
    figure.suptitle(
        f"{selection['sentence_id']} ({selection['role']}): gold={selection['gold_label']}\n"
        + " | ".join(subtitles)
        + "\n"
        + textwrap.fill(str(selection["text"]), 130),
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    save_pair(figure, stem)


def save_final_model_performance(stem: Path, metrics: list[dict[str, Any]]) -> None:
    conditions = [str(row["condition"]) for row in metrics]
    labels = {
        "zero_shot_prompt": "Zero-shot CLIP",
        "frozen_linear_head": "CLIP + linear head",
        "bert_sst2": "Fine-tuned BERT",
    }
    display = [labels.get(condition, condition) for condition in conditions]
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    x = np.arange(len(conditions))
    width = 0.24
    for offset, (key, label, colour) in enumerate(
        [
            ("accuracy", "Accuracy", "#4C78A8"),
            ("macro_f1", "Macro-F1", "#F58518"),
            ("balanced_accuracy", "Balanced accuracy", "#54A24B"),
        ]
    ):
        values = [float(row[key]) for row in metrics]
        bars = axes[0].bar(x + (offset - 1) * width, values, width, label=label, color=colour)
        axes[0].bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2)
    axes[0].set_xticks(x, labels=display)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Classification metrics")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    matrix = np.asarray(
        [[row["true_neg"], row["false_pos"], row["false_neg"], row["true_pos"]] for row in metrics],
        dtype=float,
    )
    axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks(np.arange(4), labels=["TN", "FP", "FN", "TP"])
    axes[1].set_yticks(np.arange(len(display)), labels=display)
    axes[1].set_title("Confusion-matrix counts")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axes[1].text(column_index, row_index, str(int(matrix[row_index, column_index])), ha="center", va="center")
    figure.suptitle("Part 7: BERT and CLIP on the identical 500-sentence SST-2 set", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_pair(figure, stem)


def save_final_explanation_agreement(stem: Path, rows: list[dict[str, Any]]) -> None:
    pairs = [str(row["pair"]) for row in rows]
    metrics = [
        ("mean_abs_rank_spearman", "Mean Spearman"),
        ("mean_top_k_overlap_rate", "Mean top-k overlap"),
        ("mean_sign_agreement", "Mean sign agreement"),
    ]
    matrix = np.asarray([[float(row[key]) for key, _ in metrics] for row in rows])
    figure, axis = plt.subplots(figsize=(11, 6.5))
    axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(metrics)), labels=[label for _, label in metrics], rotation=15, ha="right")
    axis.set_yticks(np.arange(len(pairs)), labels=pairs)
    axis.set_title("Scale-independent margin-SHAP agreement on identical words")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.3f}", ha="center", va="center")
    figure.suptitle("Part 7: BERT-CLIP explanation comparison", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_pair(figure, stem)


def save_final_faithfulness(stem: Path, rows: list[dict[str, Any]]) -> None:
    models = [str(row["model"]) for row in rows]
    keys = [
        ("mean_normalized_top_deletion_aopc", "Normalized top AOPC"),
        ("mean_normalized_random_deletion_aopc", "Normalized random AOPC"),
        ("mean_top_deletion_flip_rate", "Top-deletion flip rate"),
        ("mean_random_deletion_flip_rate", "Random flip rate"),
    ]
    x = np.arange(len(models))
    width = 0.18
    figure, axis = plt.subplots(figsize=(15, 7))
    colours = ["#4C78A8", "#9ecae9", "#F58518", "#FFBF79"]
    for index, ((key, label), colour) in enumerate(zip(keys, colours, strict=True)):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(x + (index - 1.5) * width, values, width, label=label, color=colour)
        axis.bar_label(bars, labels=[f"{value:+.3f}" for value in values], fontsize=7, padding=2)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels=models)
    axis.set_title("Scale-normalized deletion and prediction-flip faithfulness")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Part 7: faithfulness comparison", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_pair(figure, stem)


def save_final_prediction_agreement(stem: Path, models: Sequence[str], matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=float)
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(models)), labels=models, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(models)), labels=models)
    axis.set_title("Prediction agreement on the same 500 sentences")
    for row_index in range(len(models)):
        for column_index in range(len(models)):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.3f}", ha="center", va="center")
    figure.tight_layout()
    save_pair(figure, stem)


def save_final_efficiency(stem: Path, rows: list[dict[str, Any]]) -> None:
    models = [str(row["model"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    runtime = [float(row["mean_shap_runtime_seconds"]) for row in rows]
    evaluations = [float(row["mean_shap_model_evaluations"]) for row in rows]
    bars1 = axes[0].bar(models, runtime, color=["#4C78A8", "#F58518", "#54A24B"])
    bars2 = axes[1].bar(models, evaluations, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].bar_label(bars1, labels=[f"{value:.2f}s" for value in runtime], padding=3)
    axes[1].bar_label(bars2, labels=[f"{value:.1f}" for value in evaluations], padding=3)
    axes[0].set_title("Mean SHAP runtime per sentence")
    axes[1].set_title("Mean perturbed-text evaluations")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Part 7: explanation efficiency", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_pair(figure, stem)


def save_final_case_study(
    stem: Path,
    selection: dict[str, Any],
    words: Sequence[str],
    model_results: dict[str, dict[str, Any]],
) -> None:
    models = list(model_results)
    height = max(8.0, 0.35 * len(words) + 4.2)
    figure, axes = plt.subplots(1, len(models), figsize=(8 * len(models), height), sharey=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    descriptions = []
    for axis, model in zip(axes, models, strict=True):
        result = model_results[model]
        _signed_matrix(axis, result["word_values"], words, OUTPUT_NAMES, model)
        descriptions.append(f"{model}: {result['prediction']} ({result['scores'][2]:+.3f})")
    figure.suptitle(
        f"{selection['sentence_id']} ({selection['role']}), gold={selection['gold_label']}\n"
        + " | ".join(descriptions)
        + "\n"
        + textwrap.fill(str(selection["text"]), 145),
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    save_pair(figure, stem)
