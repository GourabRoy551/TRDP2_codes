"""Create per-sentence and cross-sentence SHAP visualizations.

The figures use one color convention everywhere: red values push the BERT
margin toward POS, while blue values push it toward NEG.  This module does not
run the model; it only turns already-computed values into readable artifacts.
"""

from __future__ import annotations

import html
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def save_word_bar(
    output_stem: Path,
    sentence_id: str,
    gold_label: str,
    prediction: str,
    text: str,
    words: Sequence[str],
    values: Sequence[float],
    margin: float,
    base_value: float,
    residual: float,
) -> None:
    """Save a signed word-level SHAP bar chart as both PDF and PNG."""
    if not words:
        return

    height = max(5.0, 0.42 * len(words) + 2.8)
    figure, axis = plt.subplots(figsize=(11, height))
    positions = np.arange(len(words))
    colors = ["#ff0051" if value >= 0 else "#008bfb" for value in values]
    bars = axis.barh(positions, values, color=colors, alpha=0.88)
    axis.set_yticks(positions)
    axis.set_yticklabels(words)
    axis.invert_yaxis()
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("SHAP value for POS logit - NEG logit")
    axis.set_title(
        f"{sentence_id} [{gold_label}] - BERT Partition SHAP\n"
        f"prediction={prediction} | margin={margin:.4f} | base={base_value:.4f} | "
        f"additivity residual={residual:.2e}\n{textwrap.fill(text, width=88)}",
        pad=14,
    )
    axis.grid(axis="x", alpha=0.22)

    # Symmetric limits make POS and NEG evidence visually comparable.
    maximum = max(abs(float(value)) for value in values) or 1.0
    axis.set_xlim(-1.20 * maximum, 1.20 * maximum)
    for bar, value in zip(bars, values, strict=True):
        offset = 0.018 * maximum
        x_position = float(value) + (offset if value >= 0 else -offset)
        axis.text(
            x_position,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
        )

    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".pdf"))
    figure.savefig(output_stem.with_suffix(".png"))
    plt.close(figure)


def save_text_html(
    path: Path,
    sentence_id: str,
    gold_label: str,
    prediction: str,
    tokens: Sequence[str],
    values: Sequence[float],
    margin: float,
    base_value: float,
) -> None:
    """Save an interactive token-color explanation with values in tooltips."""
    maximum = max((abs(float(value)) for value in values), default=1.0) or 1.0
    spans: list[str] = []
    for token, value in zip(tokens, values, strict=True):
        strength = min(0.85, 0.12 + 0.73 * abs(float(value)) / maximum)
        if value >= 0:
            color = f"rgba(255,0,81,{strength:.3f})"
        else:
            color = f"rgba(0,139,251,{strength:.3f})"
        spans.append(
            '<span class="token" '
            f'title="{html.escape(f"SHAP {float(value):+.6f}")}" '
            f'style="background:{color}">{html.escape(str(token))}</span>'
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(sentence_id)} BERT SHAP</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1050px; margin: 42px auto; padding: 0 24px; color: #18212b; }}
h1 {{ font-size: 24px; }}
.meta {{ color: #445; margin-bottom: 22px; }}
.sentence {{ font-size: 28px; line-height: 2.0; white-space: pre-wrap; }}
.token {{ padding: 5px 2px; border-radius: 4px; }}
.legend {{ margin-top: 26px; display: flex; gap: 22px; }}
.swatch {{ display: inline-block; width: 20px; height: 14px; margin-right: 7px; vertical-align: middle; }}
</style>
</head>
<body>
<h1>{html.escape(sentence_id)} - BERT Partition SHAP</h1>
<div class="meta">Gold: {html.escape(gold_label)} | Prediction: {html.escape(prediction)} | Margin: {margin:.6f} | Base: {base_value:.6f}</div>
<div class="sentence">{''.join(spans)}</div>
<div class="legend">
<div><span class="swatch" style="background:#ff0051"></span>Pushes toward POS</div>
<div><span class="swatch" style="background:#008bfb"></span>Pushes toward NEG</div>
</div>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def save_global_word_plot(
    path_stem: Path,
    word_rows: list[dict[str, Any]],
    top_n: int = 20,
) -> None:
    """Compare mean absolute word importance across all selected sentences."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in word_rows:
        grouped[str(row["word"]).lower()].append(abs(float(row["shap_value"])))

    ranked = sorted(
        ((word, float(np.mean(values))) for word, values in grouped.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    if not ranked:
        return

    words = [word for word, _ in ranked][::-1]
    scores = [score for _, score in ranked][::-1]
    figure, axis = plt.subplots(
        figsize=(10, max(6.0, 0.38 * len(words) + 2.2))
    )
    axis.barh(np.arange(len(words)), scores, color="#6f42c1", alpha=0.9)
    axis.set_yticks(np.arange(len(words)))
    axis.set_yticklabels(words)
    axis.set_xlabel("Mean absolute word-level SHAP value")
    axis.set_title("BERT SST-2 - Global comparison over selected sentences")
    axis.grid(axis="x", alpha=0.22)
    figure.tight_layout()
    figure.savefig(path_stem.with_suffix(".pdf"))
    figure.savefig(path_stem.with_suffix(".png"))
    plt.close(figure)


def top_word(words: Sequence[str], values: Sequence[float], positive: bool) -> str:
    """Format the most positive or most negative word for the summary table."""
    if not words:
        return ""
    pairs = list(zip(words, values, strict=True))
    if positive:
        word, value = max(pairs, key=lambda item: item[1])
    else:
        word, value = min(pairs, key=lambda item: item[1])
    return f"{word} ({float(value):+.4f})"


def build_run_report(
    summaries: list[dict[str, Any]],
    faithfulness_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    """Build a concise Markdown report for one experiment run."""
    correct = sum(int(row["correct"]) for row in summaries)
    max_residual = max(abs(float(row["additivity_residual"])) for row in summaries)
    lines = [
        "# BERT SHAP Run Report",
        "",
        "## Experiment",
        "",
        f"- Model: `{config['model_name']}`",
        "- Explained output: `positive logit - negative logit`",
        "- Explainer: SHAP PartitionExplainer with the BERT text masker",
        f"- Sentences: {len(summaries)}",
        f"- Correct predictions: {correct}/{len(summaries)}",
        f"- Maximum absolute additivity residual: `{max_residual:.3e}`",
        f"- Maximum SHAP evaluations per sentence: {config['max_evals']}",
        "",
        "## Sentence-level results",
        "",
        "| ID | Gold | Prediction | Margin | Top POS word | Top NEG word | Additivity residual |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {sentence_id} | {gold_label} | {prediction} | {margin:.4f} | "
            "{top_positive_word} | {top_negative_word} | "
            "{additivity_residual:.2e} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Positive attributions push the classifier toward POS; negative "
            "attributions push it toward NEG. Values are local to each sentence "
            "and conditional on `[MASK]` representing a missing feature.",
        ]
    )

    if faithfulness_rows:
        mean_comp = float(
            np.mean(
                [float(row["comprehensiveness_drop"]) for row in faithfulness_rows]
            )
        )
        mean_sufficiency_gap = float(
            np.mean([float(row["sufficiency_gap"]) for row in faithfulness_rows])
        )
        lines.extend(
            [
                "",
                "## Faithfulness summary",
                "",
                f"- Mean decision-score comprehensiveness drop: `{mean_comp:.4f}`",
                f"- Mean decision-score sufficiency gap: `{mean_sufficiency_gap:.4f}`",
                "",
                "Higher comprehensiveness drop is better. Lower sufficiency gap "
                "is better.",
            ]
        )

    lines.extend(
        [
            "",
            "## Output locations",
            "",
            f"- Values: `{config['values_dir']}`",
            f"- Plots: `{config['plots_dir']}`",
            f"- Evaluation: `{config['evaluation_dir']}`",
            "",
        ]
    )
    return "\n".join(lines)
