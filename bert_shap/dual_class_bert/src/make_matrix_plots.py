"""Presentation-ready PDF and PNG plots for dual-class BERT SHAP."""

from __future__ import annotations

import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


CLASS_NAMES = ("NEG", "POS")
INCREASE_COLOR = "#d62728"
DECREASE_COLOR = "#1f77b4"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"))
    figure.savefig(output_stem.with_suffix(".png"))
    plt.close(figure)


def _symmetric_limit(values: np.ndarray) -> float:
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    return maximum if maximum > 1e-12 else 1.0


def _annotate_cells(
    axis: plt.Axes,
    values: np.ndarray,
    fmt: str = "+.3f",
    threshold: float | None = None,
) -> None:
    limit = threshold or _symmetric_limit(values)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            color = "white" if abs(value) > 0.58 * limit else "black"
            axis.text(
                column,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
            )


def _draw_shap_matrix(
    axis: plt.Axes,
    row_labels: Sequence[str],
    values: np.ndarray,
    title: str,
    add_colorbar: bool = False,
) -> Any:
    matrix = np.asarray(values, dtype=float)
    limit = _symmetric_limit(matrix)
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_xticks([0, 1], labels=CLASS_NAMES)
    axis.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    axis.set_xlabel("Explained BERT class logit")
    axis.set_title(title, pad=10)
    _annotate_cells(axis, matrix, threshold=limit)
    if add_colorbar:
        colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
        colorbar.set_label("SHAP value")
    return image


def save_dual_word_bar(
    output_stem: Path,
    sentence_id: str,
    text: str,
    gold_label: str,
    prediction: str,
    words: Sequence[str],
    values: np.ndarray,
    logits: Sequence[float],
    base_values: Sequence[float],
    residuals: Sequence[float],
) -> None:
    matrix = np.asarray(values, dtype=float)
    height = max(6.0, 0.40 * len(words) + 3.5)
    figure, axes = plt.subplots(1, 2, figsize=(15, height), sharey=True)
    limit = 1.18 * _symmetric_limit(matrix)
    positions = np.arange(len(words))

    for class_index, axis in enumerate(axes):
        class_values = matrix[:, class_index]
        colors = [
            INCREASE_COLOR if value >= 0 else DECREASE_COLOR
            for value in class_values
        ]
        bars = axis.barh(positions, class_values, color=colors, alpha=0.88)
        axis.axvline(0.0, color="black", linewidth=1.0)
        axis.set_xlim(-limit, limit)
        axis.set_yticks(positions, labels=words)
        axis.grid(axis="x", alpha=0.22)
        axis.set_xlabel(f"SHAP value for {CLASS_NAMES[class_index]} logit")
        axis.set_title(
            f"Target: {CLASS_NAMES[class_index]}\n"
            f"logit={float(logits[class_index]):+.4f} | "
            f"base={float(base_values[class_index]):+.4f} | "
            f"residual={float(residuals[class_index]):.2e}"
        )
        for bar, value in zip(bars, class_values, strict=True):
            offset = 0.015 * limit
            axis.text(
                float(value) + (offset if value >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{float(value):+.3f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8.5,
            )

    axes[0].invert_yaxis()
    axes[0].set_ylabel("Aggregated word")
    figure.suptitle(
        f"{sentence_id}: BERT dual-class Partition SHAP\n"
        f"gold={gold_label} | prediction={prediction}\n"
        f"{textwrap.fill(text, width=110)}",
        fontsize=14,
        y=1.01,
    )
    figure.text(
        0.5,
        0.006,
        "Red increases the target class logit; blue decreases it. "
        "Both panels use the same scale.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.94))
    save_figure(figure, output_stem)


def save_class_value_matrix(
    output_stem: Path,
    sentence_id: str,
    text: str,
    gold_label: str,
    prediction: str,
    row_labels: Sequence[str],
    values: np.ndarray,
    level_name: str,
) -> None:
    height = max(5.5, 0.38 * len(row_labels) + 2.5)
    figure, axis = plt.subplots(figsize=(8.5, height))
    image = _draw_shap_matrix(
        axis,
        row_labels,
        np.asarray(values, dtype=float),
        f"{sentence_id}: {level_name}-by-class SHAP matrix",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
    colorbar.set_label("SHAP value")
    figure.suptitle(
        f"gold={gold_label} | prediction={prediction}\n"
        f"{textwrap.fill(text, width=96)}",
        fontsize=11,
        y=1.005,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    save_figure(figure, output_stem)


def summary_values(
    logits: Sequence[float],
    probabilities: Sequence[float],
    base_values: Sequence[float],
    shap_values: np.ndarray,
    reconstructed: Sequence[float],
    residuals: Sequence[float],
) -> tuple[list[str], np.ndarray]:
    matrix = np.asarray(shap_values, dtype=float)
    rows = [
        "Logit",
        "Probability",
        "SHAP base value",
        "Sum of SHAP values",
        "Reconstructed logit",
        "Additivity residual",
        "Mean absolute SHAP",
        "Total absolute SHAP",
    ]
    values = np.vstack(
        [
            np.asarray(logits, dtype=float),
            np.asarray(probabilities, dtype=float),
            np.asarray(base_values, dtype=float),
            matrix.sum(axis=0),
            np.asarray(reconstructed, dtype=float),
            np.asarray(residuals, dtype=float),
            np.mean(np.abs(matrix), axis=0),
            np.sum(np.abs(matrix), axis=0),
        ]
    )
    return rows, values


def _row_scaled(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    scales = np.max(np.abs(matrix), axis=1, keepdims=True)
    scales[scales < 1e-12] = 1.0
    return matrix / scales


def save_sentence_summary_matrix(
    output_stem: Path,
    sentence_id: str,
    text: str,
    gold_label: str,
    prediction: str,
    logits: Sequence[float],
    probabilities: Sequence[float],
    base_values: Sequence[float],
    shap_values: np.ndarray,
    reconstructed: Sequence[float],
    residuals: Sequence[float],
) -> None:
    row_labels, raw_values = summary_values(
        logits,
        probabilities,
        base_values,
        shap_values,
        reconstructed,
        residuals,
    )
    scaled = _row_scaled(raw_values)
    figure, axis = plt.subplots(figsize=(8.8, 6.8))
    axis.imshow(scaled, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks([0, 1], labels=CLASS_NAMES)
    axis.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    axis.set_title(f"{sentence_id}: sentence-summary value matrix", pad=11)
    for row in range(raw_values.shape[0]):
        for column in range(2):
            value = raw_values[row, column]
            fmt = ".2e" if row == 5 else ".4f"
            color = "white" if abs(scaled[row, column]) > 0.58 else "black"
            axis.text(
                column,
                row,
                format(float(value), fmt),
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    figure.suptitle(
        f"gold={gold_label} | prediction={prediction}\n"
        f"{textwrap.fill(text, width=92)}",
        fontsize=11,
        y=1.005,
    )
    figure.text(
        0.5,
        0.015,
        "Cell colour is normalized within each metric row; printed values are raw.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(figure, output_stem)


def save_sentence_overview(
    output_stem: Path,
    sentence_id: str,
    text: str,
    gold_label: str,
    prediction: str,
    words: Sequence[str],
    word_values: np.ndarray,
    logits: Sequence[float],
    probabilities: Sequence[float],
    base_values: Sequence[float],
    shap_values: np.ndarray,
    reconstructed: Sequence[float],
    residuals: Sequence[float],
) -> None:
    summary_labels, raw_summary = summary_values(
        logits,
        probabilities,
        base_values,
        shap_values,
        reconstructed,
        residuals,
    )
    scaled_summary = _row_scaled(raw_summary)
    height = max(8.0, 0.34 * max(len(words), len(summary_labels)) + 4.2)
    figure, axes = plt.subplots(
        1, 2, figsize=(14, height), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    word_image = _draw_shap_matrix(
        axes[0], words, word_values, "Word-level SHAP values"
    )
    figure.colorbar(word_image, ax=axes[0], fraction=0.045, pad=0.025).set_label(
        "SHAP value"
    )

    axes[1].imshow(
        scaled_summary, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0
    )
    axes[1].set_xticks([0, 1], labels=CLASS_NAMES)
    axes[1].set_yticks(np.arange(len(summary_labels)), labels=summary_labels)
    axes[1].set_title("Sentence-summary values", pad=10)
    for row in range(raw_summary.shape[0]):
        for column in range(2):
            fmt = ".2e" if row == 5 else ".4f"
            color = "white" if abs(scaled_summary[row, column]) > 0.58 else "black"
            axes[1].text(
                column,
                row,
                format(float(raw_summary[row, column]), fmt),
                ha="center",
                va="center",
                color=color,
                fontsize=8.5,
            )

    figure.suptitle(
        f"{sentence_id}: BERT dual-class SHAP overview\n"
        f"gold={gold_label} | prediction={prediction}\n"
        f"{textwrap.fill(text, width=115)}",
        fontsize=14,
        y=1.01,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, output_stem)


def save_all_sentence_summary_matrix(
    output_stem: Path, summaries: list[dict[str, Any]]
) -> None:
    columns = [
        "NEG logit",
        "POS logit",
        "NEG prob.",
        "POS prob.",
        "NEG base",
        "POS base",
        "NEG residual",
        "POS residual",
    ]
    values = np.asarray(
        [
            [
                row["negative_logit"],
                row["positive_logit"],
                row["negative_probability"],
                row["positive_probability"],
                row["negative_base_value"],
                row["positive_base_value"],
                row["negative_additivity_residual"],
                row["positive_additivity_residual"],
            ]
            for row in summaries
        ],
        dtype=float,
    )
    column_scale = np.max(np.abs(values), axis=0, keepdims=True)
    column_scale[column_scale < 1e-12] = 1.0
    scaled = values / column_scale
    labels = [
        f"{row['sentence_id']} [G:{row['gold_label']}/P:{row['prediction']}]"
        for row in summaries
    ]
    figure, axis = plt.subplots(figsize=(15, max(8.0, 0.42 * len(labels) + 3.0)))
    axis.imshow(scaled, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(np.arange(len(columns)), labels=columns, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_title("All-sentence BERT dual-class summary matrix", pad=12)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            fmt = ".1e" if column >= 6 else ".3f"
            color = "white" if abs(scaled[row, column]) > 0.58 else "black"
            axis.text(
                column,
                row,
                format(float(values[row, column]), fmt),
                ha="center",
                va="center",
                fontsize=7.2,
                color=color,
            )
    figure.text(
        0.5,
        0.01,
        "Colour is normalized within each column; printed values are raw.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    save_figure(figure, output_stem)


def save_sentence_class_magnitude_matrix(
    output_stem: Path,
    summaries: list[dict[str, Any]],
) -> None:
    values = np.asarray(
        [
            [row["negative_mean_abs_word_shap"], row["positive_mean_abs_word_shap"]]
            for row in summaries
        ],
        dtype=float,
    )
    labels = [row["sentence_id"] for row in summaries]
    maximum = float(values.max()) if values.size and values.max() > 0 else 1.0
    figure, axis = plt.subplots(figsize=(7.5, max(7.0, 0.4 * len(labels) + 2.5)))
    image = axis.imshow(values, aspect="auto", cmap="Purples", vmin=0.0, vmax=maximum)
    axis.set_xticks([0, 1], labels=CLASS_NAMES)
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_title("Mean absolute word-level SHAP by sentence and class", pad=11)
    for row in range(values.shape[0]):
        for column in range(2):
            color = "white" if values[row, column] > 0.58 * maximum else "black"
            axis.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8.5,
            )
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025).set_label(
        "Mean absolute word SHAP"
    )
    figure.tight_layout()
    save_figure(figure, output_stem)


def save_global_word_class_matrix(
    output_stem: Path,
    word_rows: list[dict[str, Any]],
    top_n: int = 20,
) -> None:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"NEG": [], "POS": []}
    )
    for row in word_rows:
        grouped[str(row["word"]).casefold()][str(row["class_name"])].append(
            abs(float(row["shap_value"]))
        )
    ranked: list[tuple[str, list[float]]] = []
    for word, classes in grouped.items():
        values = [
            float(np.mean(classes[name])) if classes[name] else 0.0
            for name in CLASS_NAMES
        ]
        ranked.append((word, values))
    ranked.sort(key=lambda item: max(item[1]), reverse=True)
    ranked = ranked[:top_n]
    words = [item[0] for item in ranked]
    values = np.asarray([item[1] for item in ranked], dtype=float)
    maximum = float(values.max()) if values.size and values.max() > 0 else 1.0
    figure, axis = plt.subplots(figsize=(8.2, max(7.0, 0.42 * len(words) + 2.5)))
    image = axis.imshow(values, aspect="auto", cmap="Purples", vmin=0.0, vmax=maximum)
    axis.set_xticks([0, 1], labels=CLASS_NAMES)
    axis.set_yticks(np.arange(len(words)), labels=words)
    axis.set_title("Global mean absolute word SHAP by target class", pad=11)
    for row in range(values.shape[0]):
        for column in range(2):
            color = "white" if values[row, column] > 0.58 * maximum else "black"
            axis.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8.5,
            )
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025).set_label(
        "Mean absolute word SHAP"
    )
    figure.tight_layout()
    save_figure(figure, output_stem)


def save_additivity_residual_matrix(
    output_stem: Path, summaries: list[dict[str, Any]]
) -> None:
    values = np.asarray(
        [
            [row["negative_additivity_residual"], row["positive_additivity_residual"]]
            for row in summaries
        ],
        dtype=float,
    )
    labels = [row["sentence_id"] for row in summaries]
    limit = _symmetric_limit(values)
    figure, axis = plt.subplots(figsize=(7.5, max(7.0, 0.4 * len(labels) + 2.5)))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_xticks([0, 1], labels=CLASS_NAMES)
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_title("Class-specific SHAP additivity residuals", pad=11)
    for row in range(values.shape[0]):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{values[row, column]:.2e}",
                ha="center",
                va="center",
                fontsize=8.2,
                color="white" if abs(values[row, column]) > 0.58 * limit else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025).set_label(
        "Model logit minus SHAP reconstruction"
    )
    figure.tight_layout()
    save_figure(figure, output_stem)


def save_class_comparison_bars(
    output_stem: Path,
    word_rows: list[dict[str, Any]],
    top_n: int = 10,
) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in word_rows:
        grouped[(str(row["class_name"]), str(row["word"]).casefold())].append(
            float(row["shap_value"])
        )
    means = {
        key: float(np.mean(values))
        for key, values in grouped.items()
        if values
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_items = [
            (word, value)
            for (name, word), value in means.items()
            if name == class_name
        ]
        increasing = sorted(
            (item for item in class_items if item[1] > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
        decreasing = sorted(
            (item for item in class_items if item[1] < 0),
            key=lambda item: item[1],
        )[:top_n]
        for column, (items, direction) in enumerate(
            [(increasing, "Increase"), (decreasing, "Decrease")]
        ):
            axis = axes[class_index, column]
            shown = list(reversed(items))
            words = [word for word, _ in shown]
            values = [value for _, value in shown]
            color = INCREASE_COLOR if column == 0 else DECREASE_COLOR
            bars = axis.barh(np.arange(len(words)), values, color=color, alpha=0.88)
            axis.set_yticks(np.arange(len(words)), labels=words)
            axis.axvline(0.0, color="black", linewidth=0.9)
            axis.margins(x=0.12)
            axis.grid(axis="x", alpha=0.2)
            axis.set_title(f"{direction} {class_name} logit")
            axis.set_xlabel("Mean signed word-level SHAP value")
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    value,
                    bar.get_y() + bar.get_height() / 2,
                    f" {value:+.3f}" if value >= 0 else f"{value:+.3f} ",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=8,
                )
    figure.suptitle(
        "Words with the strongest mean class-specific SHAP contributions",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, output_stem)
