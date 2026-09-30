"""Figures and the Markdown run report for CLIP Vision Partition SHAP.

Visual conventions follow ``bert_shap/src/make_comparison_plots.py``:
SHAP red (#ff0051) = positive, SHAP blue (#008bfb) = negative, purple for
global magnitude summaries, every figure saved as PNG and PDF. Heatmaps are
drawn on the exact 224x224 crop the model receives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

POSITIVE_COLOR = "#ff0051"
NEGATIVE_COLOR = "#008bfb"
GLOBAL_COLOR = "#7e57c2"

# Blue -> transparent -> red, as in shap.image_plot, so a value of 0 shows the image.
_colors = []
for alpha in np.linspace(1, 0, 100):
    _colors.append((0.0, 0.5451, 0.9843, alpha))
for alpha in np.linspace(0, 1, 100):
    _colors.append((1.0, 0.0, 0.3176, alpha))
RED_TRANSPARENT_BLUE = LinearSegmentedColormap.from_list("red_transparent_blue", _colors)
RED_BLUE = LinearSegmentedColormap.from_list("red_blue", [NEGATIVE_COLOR, "#ffffff", POSITIVE_COLOR])


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _limit(values: np.ndarray) -> float:
    limit = float(np.nanmax(np.abs(values))) if np.size(values) else 0.0
    return limit if limit > 0 else 1e-12


def _decimals(limit: float) -> int:
    """Enough decimals to show two significant digits of the largest value."""
    if limit <= 0 or not np.isfinite(limit):
        return 3
    return int(min(6, max(3, np.ceil(-np.log10(limit)) + 1)))


def _grayscale(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image, dtype=np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return 0.25 + 0.6 * gray  # lighten so red/blue stays readable


def upsample_grid(grid: np.ndarray, size: int) -> np.ndarray:
    """Nearest-patch upsampling: each patch value fills its own 32x32 square."""
    rows, cols = grid.shape
    return np.kron(grid, np.ones((size // rows, size // cols)))


def _title(summary: dict[str, Any]) -> str:
    return (
        f"{summary['image_id']} - CLIP Vision Partition SHAP\n"
        f"score={summary['score']:.4f} | base={summary['base_value']:.4f} | "
        f"additivity residual={summary['additivity_residual']:.2e}"
    )


def save_original_and_input(image_id: str, original: np.ndarray, model_input: np.ndarray,
                            crop_box: Sequence[float], stem: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [original.shape[1] / original.shape[0], 1]})
    axes[0].imshow(original)
    left, top, right, bottom = crop_box
    axes[0].add_patch(Rectangle((left, top), right - left, bottom - top, fill=False,
                                edgecolor="#f5c400", linewidth=2.5, linestyle="--"))
    axes[0].set_title(f"{image_id} original ({original.shape[1]}x{original.shape[0]})\ndashed box = region CLIP keeps", fontsize=13)
    axes[1].imshow(np.clip(model_input, 0, 1))
    axes[1].set_title("Model input (224x224 centre crop)", fontsize=13)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    save_figure(figure, stem)


def save_patch_overlay(summary: dict[str, Any], model_input: np.ndarray, grid: np.ndarray,
                       stem: Path, with_values: bool) -> None:
    size = model_input.shape[0]
    rows, cols = grid.shape
    patch = size // cols
    limit = _limit(grid)
    figure, axis = plt.subplots(figsize=(8.6, 8))
    axis.imshow(_grayscale(model_input), cmap="gray", vmin=0, vmax=1)
    shown = axis.imshow(upsample_grid(grid, size), cmap=RED_TRANSPARENT_BLUE, vmin=-limit, vmax=limit)
    for k in range(1, cols):
        axis.axvline(k * patch - 0.5, color="white", linewidth=0.6, alpha=0.6)
        axis.axhline(k * patch - 0.5, color="white", linewidth=0.6, alpha=0.6)
    if with_values:
        decimals = _decimals(limit)
        for r in range(rows):
            for c in range(cols):
                value = grid[r, c]
                axis.text(c * patch + patch / 2, r * patch + patch / 2, f"{value:+.{decimals}f}",
                          ha="center", va="center", fontsize=7.5 if decimals <= 4 else 6.5, color="black",
                          bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.65, "linewidth": 0})
    axis.set_xticks([c * patch + patch / 2 - 0.5 for c in range(cols)], [str(c) for c in range(cols)])
    axis.set_yticks([r * patch + patch / 2 - 0.5 for r in range(rows)], [str(r) for r in range(rows)])
    axis.set_xlabel("patch column")
    axis.set_ylabel("patch row")
    axis.set_title(_title(summary), fontsize=13)
    bar = figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.03)
    bar.set_label("Patch SHAP value (7x7 ViT-B/32 patches)")
    save_figure(figure, stem)


def save_pixel_map(summary: dict[str, Any], model_input: np.ndarray, pixels: np.ndarray, stem: Path) -> None:
    limit = _limit(pixels)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(np.clip(model_input, 0, 1))
    axes[0].set_title("Model input", fontsize=13)
    axes[1].imshow(_grayscale(model_input), cmap="gray", vmin=0, vmax=1)
    shown = axes[1].imshow(pixels, cmap=RED_TRANSPARENT_BLUE, vmin=-limit, vmax=limit)
    axes[1].set_title("Pixel-level SHAP (colour channels summed)", fontsize=13)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    bar = figure.colorbar(shown, ax=axes[1], fraction=0.046, pad=0.03)
    bar.set_label("SHAP value per pixel")
    figure.suptitle(_title(summary), fontsize=13)
    save_figure(figure, stem)


def save_patch_bar(summary: dict[str, Any], grid: np.ndarray, stem: Path, top_n: int = 12) -> None:
    rows, cols = grid.shape
    flat = grid.reshape(-1)
    order = np.argsort(-np.abs(flat), kind="stable")[:top_n]
    order = order[np.argsort(-flat[order], kind="stable")]
    labels = [f"patch ({i // cols},{i % cols})" for i in order]
    values = flat[order]
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]
    limit = _limit(values)
    figure, axis = plt.subplots(figsize=(10, max(4.5, 0.55 * len(values) + 1.8)))
    positions = np.arange(len(values))
    axis.barh(positions, values, color=colors)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(-1.15 * limit, 1.15 * limit)
    for position, value in zip(positions, values):
        axis.text(value + (0.02 * limit if value >= 0 else -0.02 * limit), position, f"{value:+.{_decimals(limit) + 1}f}",
                  va="center", ha="left" if value >= 0 else "right", fontsize=10)
    axis.grid(axis="x", alpha=0.3)
    axis.set_xlabel("SHAP value for image-embedding similarity")
    axis.set_title(_title(summary) + f"\ntop {len(values)} patches by |SHAP|", fontsize=13)
    figure.tight_layout()
    save_figure(figure, stem)


def save_global_patch_plot(grids: dict[str, np.ndarray], stem: Path) -> None:
    stacked = np.stack([np.abs(g) for g in grids.values()])
    mean_grid = stacked.mean(axis=0)
    per_image = {image_id: float(np.abs(g).sum()) for image_id, g in grids.items()}
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.1]})
    shown = axes[0].imshow(mean_grid, cmap="Purples")
    rows, cols = mean_grid.shape
    decimals = _decimals(float(mean_grid.max()))
    for r in range(rows):
        for c in range(cols):
            axes[0].text(c, r, f"{mean_grid[r, c]:.{decimals}f}", ha="center", va="center", fontsize=8,
                         color="white" if mean_grid[r, c] > 0.6 * mean_grid.max() else "black")
    axes[0].set_xticks(range(cols))
    axes[0].set_yticks(range(rows))
    axes[0].set_xlabel("patch column")
    axes[0].set_ylabel("patch row")
    axes[0].set_title("Mean |patch SHAP| by position over images", fontsize=13)
    figure.colorbar(shown, ax=axes[0], fraction=0.046, pad=0.03)
    ids = sorted(per_image, key=per_image.get, reverse=True)
    axes[1].barh(range(len(ids)), [per_image[i] for i in ids], color=GLOBAL_COLOR)
    axes[1].set_yticks(range(len(ids)), ids)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].set_xlabel("Sum of |patch SHAP| per image")
    axes[1].set_title("Total attribution magnitude per image", fontsize=13)
    figure.suptitle("CLIP Vision ViT-B/32 - Global comparison over selected images", fontsize=15)
    figure.tight_layout()
    save_figure(figure, stem)


def save_overview_gallery(inputs: dict[str, np.ndarray], grids: dict[str, np.ndarray], stem: Path) -> None:
    ids = list(inputs)
    cols = 5
    rows = int(np.ceil(len(ids) / cols))
    figure, axes = plt.subplots(rows * 2, cols, figsize=(3.3 * cols, 6.6 * rows))
    axes = np.atleast_2d(axes)
    for index, image_id in enumerate(ids):
        r, c = divmod(index, cols)
        top, bottom = axes[2 * r, c], axes[2 * r + 1, c]
        top.imshow(np.clip(inputs[image_id], 0, 1))
        top.set_title(image_id, fontsize=12)
        grid = grids[image_id]
        limit = _limit(grid)
        bottom.imshow(_grayscale(inputs[image_id]), cmap="gray", vmin=0, vmax=1)
        bottom.imshow(upsample_grid(grid, inputs[image_id].shape[0]), cmap=RED_TRANSPARENT_BLUE, vmin=-limit, vmax=limit)
        bottom.set_title(f"max |patch SHAP| = {limit:.{_decimals(limit)}f}", fontsize=10)
        for axis in (top, bottom):
            axis.set_xticks([])
            axis.set_yticks([])
    for index in range(len(ids), rows * cols):
        r, c = divmod(index, cols)
        axes[2 * r, c].axis("off")
        axes[2 * r + 1, c].axis("off")
    figure.suptitle("CLIP Vision Partition SHAP - model inputs (top) and 7x7 patch SHAP (bottom)\n"
                    "red = region preserves CLIP's image embedding, blue = region pushes it away; "
                    "colour scale per image", fontsize=14)
    figure.tight_layout()
    save_figure(figure, stem)


def save_faithfulness_plot(rows: list[dict[str, Any]], stem: Path) -> None:
    fractions = sorted({float(r["fraction"]) for r in rows})

    def mean_of(key: str) -> list[float]:
        return [float(np.mean([float(r[key]) for r in rows if float(r["fraction"]) == f])) for f in fractions]

    x = [100 * f for f in fractions]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(x, mean_of("comprehensiveness_drop"), "o-", color=POSITIVE_COLOR, label="blur top SHAP patches")
    axes[0].plot(x, mean_of("random_deletion_drop"), "s--", color="grey", label="blur random patches")
    axes[0].set_title("Comprehensiveness (higher = more faithful)")
    axes[0].set_ylabel("Mean score drop (1 - similarity)")
    axes[1].plot(x, mean_of("sufficiency_gap"), "o-", color=NEGATIVE_COLOR, label="keep only top SHAP patches")
    axes[1].set_title("Sufficiency gap (lower = more faithful)")
    axes[1].set_ylabel("Mean score gap (1 - similarity)")
    for axis in axes:
        axis.set_xlabel("% of the 49 patches selected")
        axis.set_xticks(x)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("CLIP Vision Partition SHAP - patch deletion faithfulness (mean over images)", fontsize=14)
    figure.tight_layout()
    save_figure(figure, stem)


def build_run_report(config: dict[str, Any], summaries: list[dict[str, Any]],
                     faithfulness_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# CLIP Vision SHAP Run Report",
        "",
        "## Experiment",
        "",
        f"- Model: `{config['model_name']}` (vision encoder + projection only)",
        "- Explained output: cosine similarity between the CLIP image embedding of the",
        "  masked image and that of the original image (image-only, no text)",
        f"- Explainer: SHAP PartitionExplainer with a blur masker (box blur {config['blur_kernel']}x{config['blur_kernel']})",
        f"- Images: {len(summaries)}",
        f"- Maximum SHAP evaluations per image: {config['max_evals']}",
        f"- Maximum absolute additivity residual: `{max(abs(s['additivity_residual']) for s in summaries):.3e}`",
        f"- Maximum patch-aggregation conservation error: `{max(abs(s['patch_conservation_error']) for s in summaries):.3e}`",
        "",
        "## Image-level results",
        "",
        "| ID | Size | CLIP keeps | Base (fully blurred) | Sum SHAP | Top + patch | Top - patch | Residual | Model evals |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['image_id']} | {s['original_width']}x{s['original_height']} | {100 * s['crop_fraction_kept']:.0f}% "
            f"| {s['base_value']:.4f} | {s['sum_shap_values']:+.4f} | {s['top_positive_patch']} "
            f"| {s['top_negative_patch']} | {s['additivity_residual']:.2e} | {s['shap_model_evaluations']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The unmasked image always scores 1. A positive patch value means the region helps",
        "preserve CLIP's representation of the whole image; a negative value means showing it",
        "moves the embedding away from the original (relative to the blurred baseline). Values",
        "are local to each image and conditional on blurring representing a missing region.",
        "",
    ]
    if faithfulness_rows:
        fractions = sorted({float(r["fraction"]) for r in faithfulness_rows})
        lines += [
            "## Faithfulness summary",
            "",
            "| Patches selected | Comprehensiveness drop (SHAP) | Random-deletion drop | SHAP - random | Sufficiency gap |",
            "|---:|---:|---:|---:|---:|",
        ]
        for f in fractions:
            sel = [r for r in faithfulness_rows if float(r["fraction"]) == f]
            lines.append(
                f"| {100 * f:.0f}% | {np.mean([r['comprehensiveness_drop'] for r in sel]):.4f} "
                f"| {np.mean([r['random_deletion_drop'] for r in sel]):.4f} "
                f"| {np.mean([r['shap_minus_random_drop'] for r in sel]):+.4f} "
                f"| {np.mean([r['sufficiency_gap'] for r in sel]):.4f} |"
            )
        lines += ["", "Higher comprehensiveness drop is better. Lower sufficiency gap is better.", ""]
    if stability_rows:
        lines += [
            "## Budget stability",
            "",
            "| ID | Budgets | Spearman | Top-5 overlap | Sign agreement | Max abs difference | Same top patch |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in stability_rows:
            lines.append(
                f"| {r['image_id']} | {r['reference_max_evals']} vs {r['rerun_max_evals']} | {r['spearman']:.3f} "
                f"| {r['top_5_overlap']:.2f} | {r['sign_agreement']:.2f} | {r['max_abs_patch_difference']:.4f} "
                f"| {'yes' if r['same_top_patch'] else 'no'} |"
            )
        lines.append("")
    lines += [
        "## Output locations",
        "",
        f"- Values: `{config['values_dir']}`",
        f"- Plots: `{config['plots_dir']}`",
        f"- Evaluation: `{config['evaluation_dir']}`",
        "",
    ]
    return "\n".join(lines)
