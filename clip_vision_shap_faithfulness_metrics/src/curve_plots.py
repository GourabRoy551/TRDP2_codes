"""Deletion and insertion curve figures (PNG + PDF) for the report.

Colours come from the validated reference palette (slot 1 blue = deletion, slot 2
orange = insertion; both pass the categorical checks on the light surface). Individual
images are drawn as thin muted lines behind the mean.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT_LINE = "#c3c2b7"
DELETION = "#2a78d6"
INSERTION = "#eb6834"

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": SURFACE,
    }
)


def _save(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _axes_format(axis: plt.Axes, y_limits: tuple[float, float]) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(*y_limits)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axis.tick_params(length=0)


def _y_limits(curves: dict[str, dict[str, np.ndarray]]) -> tuple[float, float]:
    low = min(float(np.min(curve[kind])) for curve in curves.values() for kind in ("deletion", "insertion"))
    return (max(0.0, np.floor(low * 10) / 10 - 0.05), 1.03)


def save_mean_curves(curves: dict[str, dict[str, np.ndarray]], summary: dict[str, float], stem: Path) -> None:
    """Two panels (deletion | insertion): each image as a thin line, the mean in colour."""
    x = np.linspace(0.0, 1.0, len(next(iter(curves.values()))["deletion"]))
    limits = _y_limits(curves)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    panels = (
        ("deletion", DELETION, "Deletion curve", "Top-SHAP patches removed (blurred)", f"DAUC = {summary['deletion_auc']:.4f}  (lower is better)"),
        ("insertion", INSERTION, "Insertion curve", "Top-SHAP patches restored on the blurred image", f"IAUC = {summary['insertion_auc']:.4f}  (higher is better)"),
    )
    for axis, (kind, colour, title, xlabel, label) in zip(axes, panels, strict=True):
        for curve in curves.values():
            axis.plot(x, curve[kind], color=CONTEXT_LINE, linewidth=1.0, solid_capstyle="round", zorder=2)
        mean = np.mean([curve[kind] for curve in curves.values()], axis=0)
        axis.fill_between(x, limits[0], mean, color=colour, alpha=0.10, linewidth=0, zorder=1)
        axis.plot(x, mean, color=colour, linewidth=2.2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        axis.plot([x[-1]], [mean[-1]], "o", markersize=6, color=colour, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4, clip_on=False)
        _axes_format(axis, limits)
        axis.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=22)
        axis.text(0.0, 1.02, label, transform=axis.transAxes, fontsize=9.5, color=INK_SECONDARY, va="bottom")
        axis.set_xlabel(xlabel)
    axes[0].set_ylabel("Similarity to the original embedding")
    figure.text(
        0.5, -0.04,
        f"CLIP ViT-B/32 image encoder, patch-level Partition SHAP, images I1–I{len(curves)}. "
        "Coloured line: mean over images; grey lines: individual images; shaded area: area under the mean curve.",
        ha="center", fontsize=8.5, color=MUTED,
    )
    figure.tight_layout()
    _save(figure, stem)


def save_pointing_game(
    image_paths: dict[str, Path],
    crops: dict[str, list[float]],
    boxes: dict[str, list[tuple[float, float, float, float]]],
    points: dict[str, tuple[float, float]],
    rows: dict[str, dict],
    tolerance: float,
    stem: Path,
) -> None:
    """Each image with its target box, the CLIP crop and the highest-SHAP patch (hit or miss)."""
    from matplotlib import patches
    from PIL import Image

    good, critical, box_colour = "#0ca30c", "#d03b3b", "#ffd400"
    columns = 5
    ids = list(image_paths)
    figure, axes = plt.subplots(int(np.ceil(len(ids) / columns)), columns, figsize=(16, 7.4))
    for axis, image_id in zip(axes.reshape(-1), ids):
        axis.imshow(Image.open(image_paths[image_id]).convert("RGB"))
        left, top, right, bottom = crops[image_id]
        axis.add_patch(patches.Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor="white", linewidth=1.2, linestyle=(0, (4, 3))))
        for x0, y0, x1, y1 in boxes.get(image_id, []):
            for width, colour in ((3.4, "black"), (1.8, box_colour)):
                axis.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colour, linewidth=width))
        row = rows[image_id]
        scale = (right - left) / 224.0
        x, y = points[image_id]
        axis.add_patch(patches.Rectangle((x - 16 * scale, y - 16 * scale), 32 * scale, 32 * scale, fill=False, edgecolor="white", linewidth=1.6))
        hit = row["pointing_game_hit"]
        colour = good if hit == 1 else critical
        axis.plot([x], [y], "o", markersize=9, color=colour, markeredgecolor="white", markeredgewidth=2)
        if hit == "NA":
            verdict = "excluded"
        elif hit == 1:
            verdict = "✓ hit" if row["pointing_game_hit_no_tolerance"] == 1 else f"✓ hit (within {tolerance:g}-px tolerance)"
        else:
            verdict = "✗ miss"
        axis.set_title(f"{image_id} · {row['pointing_game_target']}", loc="left", fontsize=10.5, fontweight="semibold", color=INK, pad=14)
        axis.text(0.0, 1.01, verdict, transform=axis.transAxes, fontsize=9.5, color=INK_SECONDARY, va="bottom")
        axis.set_axis_off()
    for axis in axes.reshape(-1)[len(ids):]:
        axis.set_visible(False)
    hits = sum(1 for image_id in ids if rows[image_id]["pointing_game_hit"] == 1)
    evaluated = sum(1 for image_id in ids if rows[image_id]["pointing_game_hit"] != "NA")
    figure.suptitle(f"Pointing game: {hits} of {evaluated} images hit the target object", x=0.01, ha="left",
                    fontsize=13, fontweight="semibold", color=INK)
    figure.text(0.01, -0.01,
                "Yellow: author-annotated target box. Dashed white: centre crop CLIP sees. White square and dot: the highest-SHAP patch "
                f"and its centre (green = hit, red = miss; {tolerance:g}-pixel tolerance).",
                fontsize=8.5, color=MUTED)
    figure.tight_layout()
    _save(figure, stem)


def save_per_image_curves(curves: dict[str, dict[str, np.ndarray]], metrics: dict[str, dict[str, float]], stem: Path) -> None:
    """Small multiples: deletion and insertion for each image, one shared legend."""
    x = np.linspace(0.0, 1.0, len(next(iter(curves.values()))["deletion"]))
    limits = _y_limits(curves)
    columns = 5
    rows = int(np.ceil(len(curves) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3.0 * rows + 0.6), sharex=True, sharey=True)
    for axis, (image_id, curve) in zip(axes.reshape(-1), curves.items()):
        axis.plot(x, curve["deletion"], color=DELETION, linewidth=2.0, solid_capstyle="round", label="Deletion")
        axis.plot(x, curve["insertion"], color=INSERTION, linewidth=2.0, solid_capstyle="round", label="Insertion")
        _axes_format(axis, limits)
        axis.set_title(image_id, loc="left", fontsize=11, fontweight="semibold", pad=16)
        axis.text(0.0, 1.02, f"DAUC {metrics[image_id]['deletion_auc']:.3f} · IAUC {metrics[image_id]['insertion_auc']:.3f}",
                  transform=axis.transAxes, fontsize=8.5, color=INK_SECONDARY, va="bottom")
    for axis in axes.reshape(-1)[len(curves):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Patches removed / restored")
    for axis in axes[:, 0]:
        axis.set_ylabel("Similarity")
    handles, labels = axes.reshape(-1)[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(1.0, 1.02), labelcolor=INK_SECONDARY)
    figure.suptitle("Deletion and insertion curves per image", x=0.01, ha="left", fontsize=13, fontweight="semibold", color=INK)
    figure.tight_layout()
    _save(figure, stem)
