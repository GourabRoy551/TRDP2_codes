"""Per-sentence word bar charts (NEG | POS | POS-NEG) for a few report examples.

Same chart type as ``bert_shap/dual_class_bert`` ``*_dual_class_word_bar``: one
horizontal bar per whole-word unit, red = increases the target output, blue =
decreases it, value on every bar, score / base / residual in each panel title.
A third panel shows the POS-NEG margin, which is the only sentiment-specific
output for CLIP.

Sources (read only):
* this experiment: ``outputs/values/sentence_results.csv`` + ``word_shap_values.csv``;
* ``clip_text_shap_improved`` Part 6: ``values/sentence_results.csv`` + ``word_shap_values.csv``.
Figures are written under ``outputs/plots/word_bars/``; nothing is written to the
other experiment's folder.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import plotting  # noqa: F401  (shared rcParams, Agg backend)
from io_utils import PLOTS_DIR, VALUES_DIR, project_path, read_csv, write_csv
from plotting import INK, INK_2, MUTED, plt, save_pair


WORD_BAR_DIR = PLOTS_DIR / "word_bars"
REPORT_SENTENCES = ("EV00003", "EV00002", "EV00091", "EV00001")
INCREASE, DECREASE = "#e34948", "#2a78d6"
IMPROVED_ROOT = "../clip_text_shap_improved/outputs/part6_full_evaluation/values"
SOURCES = {
    "bert_clip_partition_shap_500": {
        "title": "BERT vs CLIP experiment",
        "models": {
            "bert": ("BERT (SST-2 fine-tuned)", "logit"),
            "clip": ("CLIP text encoder, zero-shot", "cosine"),
        },
    },
    "clip_text_shap_improved": {
        "title": "Improved CLIP Text SHAP experiment (Part 6)",
        "models": {
            "zero_shot_prompt": ("CLIP text encoder, zero-shot prompts", "cosine"),
            "frozen_linear_head": ("CLIP text encoder + frozen linear head", "logit"),
        },
    },
}


def _load_this_experiment() -> dict[tuple[str, str], dict[str, Any]]:
    sentences = {(row["model"], row["sentence_id"]): row for row in read_csv(VALUES_DIR / "sentence_results.csv")}
    words: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(VALUES_DIR / "word_shap_values.csv"):
        if row["sentence_id"] in REPORT_SENTENCES:
            words[(row["model"], row["sentence_id"])].append(row)
    result = {}
    for key, rows in words.items():
        sentence = sentences[key]
        rows.sort(key=lambda row: int(row["word_index"]))
        result[key] = {
            "text": sentence["text"], "gold": sentence["gold_label"], "prediction": sentence["prediction"],
            "words": [row["word"] for row in rows], "is_content": [row["is_content"] == "1" for row in rows],
            "values": np.asarray([[float(row[k]) for k in ("neg_shap", "pos_shap", "margin_shap")] for row in rows]),
            "scores": np.asarray([float(sentence[k]) for k in ("neg_score", "pos_score", "margin")]),
            "bases": np.asarray([float(sentence[k]) for k in ("neg_base_value", "pos_base_value", "margin_base_value")]),
            "stored_residual": float(sentence["max_additivity_residual"]),
            "source": "outputs/values/sentence_results.csv; outputs/values/word_shap_values.csv",
        }
    return result


def _load_improved_experiment() -> dict[tuple[str, str], dict[str, Any]]:
    root = project_path(IMPROVED_ROOT)
    if not (root / "word_shap_values.csv").exists():
        return {}
    sentences = {(row["condition"], row["sentence_id"]): row for row in read_csv(root / "sentence_results.csv")}
    words: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(root / "word_shap_values.csv"):
        if row["sentence_id"] in REPORT_SENTENCES:
            words[(row["condition"], row["sentence_id"])].append(row)
    result = {}
    for key, rows in words.items():
        sentence = sentences[key]
        rows.sort(key=lambda row: int(row["word_index"]))
        result[key] = {
            "text": sentence["text"], "gold": sentence["gold_label"], "prediction": sentence["prediction"],
            "words": [row["word"] for row in rows], "is_content": [row["is_content"] == "1" for row in rows],
            "values": np.asarray([[float(row[k]) for k in ("negative_shap", "positive_shap", "margin_shap")] for row in rows]),
            "scores": np.asarray([float(sentence[k]) for k in ("negative_score", "positive_score", "margin_score")]),
            "bases": np.asarray([float(sentence[k]) for k in ("negative_base", "positive_base", "margin_base")]),
            "stored_residual": float(sentence["maximum_absolute_additivity_residual"]),
            "source": f"{IMPROVED_ROOT}/sentence_results.csv; {IMPROVED_ROOT}/word_shap_values.csv",
        }
    return result


def _limit(values: np.ndarray) -> float:
    return 1.2 * max(float(np.max(np.abs(values))) if values.size else 0.0, 1e-12)


def _panel(axis: Any, values: np.ndarray, limit: float, digits: int) -> None:
    positions = np.arange(len(values))
    bars = axis.barh(positions, values, height=0.72, color=[INCREASE if value >= 0 else DECREASE for value in values])
    axis.axvline(0.0, color=INK, linewidth=0.9)
    axis.set_xlim(-limit, limit)
    axis.xaxis.grid(True)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        offset = 0.015 * limit
        axis.text(value + (offset if value >= 0 else -offset), bar.get_y() + bar.get_height() / 2, f"{value:+.{digits}f}",
                  va="center", ha="left" if value >= 0 else "right", fontsize=8, color=INK)


def save_word_bar(stem: Path, item: dict[str, Any], experiment: str, model: str, sentence_id: str, role: str) -> dict[str, Any]:
    label, unit = SOURCES[experiment]["models"][model]
    values, scores, bases = item["values"], item["scores"], item["bases"]
    residual = np.abs(scores - (bases + values.sum(axis=0)))
    digits = 3 if unit == "logit" else 4
    height = max(5.5, 0.34 * len(item["words"]) + 3.4)
    figure, axes = plt.subplots(1, 3, figsize=(17, height), sharey=True, gridspec_kw={"wspace": 0.12})
    shared = _limit(values[:, :2])
    limits = (shared, shared, _limit(values[:, 2:]))
    names = ("NEG", "POS", "POS−NEG")
    for index, axis in enumerate(axes):
        _panel(axis, values[:, index], limits[index], digits)
        axis.set_title(f"Target: {names[index]}\nscore = {scores[index]:+.{digits + 1}f} | base = {bases[index]:+.{digits + 1}f} | "
                       f"residual = {residual[index]:.1e}", fontsize=10)
        scale = "own scale" if index == 2 else "shared NEG/POS scale"
        axis.set_xlabel(f"SHAP value for {names[index]} ({unit}; {scale})")
    content = item["is_content"]
    axes[0].set_yticks(np.arange(len(content)), [w if flag else f"{w} (punct.)" for w, flag in zip(item["words"], content, strict=True)])
    for tick, flag in zip(axes[0].get_yticklabels(), content, strict=True):
        tick.set_color(INK_2 if flag else MUTED)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Whole-word unit")
    margin = scores[2]
    role_label = role.replace("bert_clip", "BERT–CLIP").replace("_", " ")
    figure.suptitle(
        f"{sentence_id} ({role_label}) · {label}: whole-word Partition SHAP\n"
        f"{SOURCES[experiment]['title']}  |  gold = {item['gold']}  |  prediction = {item['prediction']} (POS−NEG = {margin:+.{digits + 1}f})\n"
        f"{textwrap.fill(item['text'], 125)}",
        fontsize=11.5, x=0.02, ha="left", va="top", y=1 - 0.05 / height,
    )
    note = ("Red increases the target output; blue decreases it. NEG and POS panels share one x-scale; the POS−NEG panel has its own. "
            f"Values are in {unit} units and are comparable only within this model.  residual = |score − (base + Σ bars)|.")
    if experiment == "clip_text_shap_improved":
        note += ("\nUnits follow that experiment's segmentation (e.g. 's appears as ' + s). When its 500-evaluation budget was exhausted, "
                 "part of the SHAP credit went to two boundary placeholders (not shown); that mass appears here as the residual.")
    figure.text(0.02, 0.05 / height, note, fontsize=8.3, color=INK_2, va="bottom")
    figure.subplots_adjust(top=1 - 1.35 / height, bottom=(1.35 if experiment == "clip_text_shap_improved" else 1.1) / height, left=0.1, right=0.98)
    save_pair(figure, stem)
    return {
        "experiment": experiment, "model_or_condition": model, "sentence_id": sentence_id, "role": role,
        "gold_label": item["gold"], "prediction": item["prediction"], "words": len(item["words"]),
        **{f"{name}_score": scores[i] for i, name in enumerate(("neg", "pos", "margin"))},
        **{f"{name}_base": bases[i] for i, name in enumerate(("neg", "pos", "margin"))},
        **{f"{name}_word_level_residual": residual[i] for i, name in enumerate(("neg", "pos", "margin"))},
        "stored_max_additivity_residual": item["stored_residual"],
        "figure": str(stem.relative_to(PLOTS_DIR.parent.parent)).replace("\\", "/") + ".png/.pdf",
        "source_files": item["source"],
    }


def make_word_bar_plots(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    roles = {row["sentence_id"]: row["role"] for row in read_csv(VALUES_DIR / "representative_selection.csv")}
    loaded = {"bert_clip_partition_shap_500": _load_this_experiment(), "clip_text_shap_improved": _load_improved_experiment()}
    for directory in WORD_BAR_DIR.glob("*"):
        for path in list(directory.glob("*.png")) + list(directory.glob("*.pdf")):
            path.unlink()
    manifest = []
    for experiment, items in loaded.items():
        for order, sentence_id in enumerate(REPORT_SENTENCES, start=1):
            for model in SOURCES[experiment]["models"]:
                if (model, sentence_id) not in items:
                    continue
                stem = WORD_BAR_DIR / experiment / f"{order}_{sentence_id}_{model}_word_bar"
                manifest.append(save_word_bar(stem, items[(model, sentence_id)], experiment, model, sentence_id, roles[sentence_id]))
    write_csv(VALUES_DIR / "word_bar_plot_manifest.csv", manifest)
    print(f"[wordbars] wrote {len(manifest)} figure pairs to {WORD_BAR_DIR}")
    return manifest
