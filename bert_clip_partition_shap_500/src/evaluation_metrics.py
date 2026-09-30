"""Classification, additivity, cross-model explanation agreement and bootstrap metrics.

Cross-model comparisons use only scale-independent quantities (ranks, top-k sets,
signs and normalized deletion drops). Raw BERT logits and CLIP cosine similarities
are never compared by magnitude.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import binomtest, hypergeom, spearmanr, wilcoxon


# ---------------------------------------------------------------- classification

def classification_metrics(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, float]:
    gold_array = np.asarray(gold)
    predicted_array = np.asarray(predicted)
    result: dict[str, float] = {"sentences": int(gold_array.size)}
    counts = {
        "true_neg": int(np.sum((gold_array == "NEG") & (predicted_array == "NEG"))),
        "false_pos": int(np.sum((gold_array == "NEG") & (predicted_array == "POS"))),
        "false_neg": int(np.sum((gold_array == "POS") & (predicted_array == "NEG"))),
        "true_pos": int(np.sum((gold_array == "POS") & (predicted_array == "POS"))),
    }
    f1_values, recalls = [], []
    for name, tp, fp, fn in (
        ("neg", counts["true_neg"], counts["false_neg"], counts["false_pos"]),
        ("pos", counts["true_pos"], counts["false_pos"], counts["false_neg"]),
    ):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.update({f"{name}_precision": precision, f"{name}_recall": recall, f"{name}_f1": f1, f"{name}_support": tp + fn})
        f1_values.append(f1)
        recalls.append(recall)
    result["accuracy"] = float(np.mean(gold_array == predicted_array))
    result["macro_f1"] = float(np.mean(f1_values))
    result["balanced_accuracy"] = float(np.mean(recalls))
    result.update(counts)
    return result


def mcnemar_exact(left_correct: Sequence[bool], right_correct: Sequence[bool]) -> dict[str, float]:
    """Exact McNemar test on discordant pairs (supporting evidence only)."""
    left = np.asarray(left_correct, dtype=bool)
    right = np.asarray(right_correct, dtype=bool)
    left_only = int(np.sum(left & ~right))
    right_only = int(np.sum(~left & right))
    discordant = left_only + right_only
    p_value = float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {"left_only_correct": left_only, "right_only_correct": right_only, "exact_mcnemar_p": p_value}


# ------------------------------------------------------------- distribution helpers

def describe(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "p95": np.nan, "max": np.nan, "min": np.nan}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "min": float(array.min()),
    }


# ------------------------------------------------------ pairwise explanation agreement

def _top_set(values: np.ndarray, k: int) -> list[int]:
    return [int(index) for index in np.argsort(-np.abs(values), kind="stable")[:k]]


def expected_random_jaccard(n: int, k: int) -> float:
    """E[|A∩B| / |A∪B|] for two independent uniformly random k-subsets of n items."""
    if k <= 0 or n <= 0:
        return np.nan
    overlaps = np.arange(max(0, 2 * k - n), k + 1)
    probabilities = hypergeom(n, k, k).pmf(overlaps)
    return float(np.sum(probabilities * overlaps / (2 * k - overlaps)))


def compare_explanations(
    bert_margin: np.ndarray, clip_margin: np.ndarray, content_mask: np.ndarray, top_k: int
) -> dict[str, Any]:
    """Scale-free agreement between two margin explanations over identical content words."""
    mask = np.asarray(content_mask, dtype=bool)
    bert = np.asarray(bert_margin, dtype=np.float64)[mask]
    clip = np.asarray(clip_margin, dtype=np.float64)[mask]
    positions = np.flatnonzero(mask)
    n = int(mask.sum())
    abs_bert, abs_clip = np.abs(bert), np.abs(clip)
    if n < 3:
        spearman, status = np.nan, "undefined_fewer_than_3_content_words"
    elif np.ptp(abs_bert) == 0.0 or np.ptp(abs_clip) == 0.0:
        spearman, status = np.nan, "undefined_constant_ranking"
    else:
        spearman, status = float(spearmanr(abs_bert, abs_clip).statistic), "defined"
    signed_spearman = (
        float(spearmanr(bert, clip).statistic) if n >= 3 and np.ptp(bert) > 0 and np.ptp(clip) > 0 else np.nan
    )
    k = min(int(top_k), n)
    bert_top = _top_set(bert, k)
    clip_top = _top_set(clip, k)
    intersection = set(bert_top) & set(clip_top)
    union = set(bert_top) | set(clip_top)
    sign_bert, sign_clip = np.sign(bert), np.sign(clip)
    positive_bert = float(np.mean(sign_bert > 0)) if n else 0.0
    positive_clip = float(np.mean(sign_clip > 0)) if n else 0.0
    return {
        "content_words": n,
        "spearman_abs_margin": spearman,
        "spearman_status": status,
        "signed_margin_spearman": signed_spearman,
        "top_k": k,
        "bert_top_k_word_indices": " ".join(str(int(positions[index])) for index in bert_top),
        "clip_top_k_word_indices": " ".join(str(int(positions[index])) for index in clip_top),
        "top_k_overlap_count": len(intersection),
        "top_k_overlap_rate": len(intersection) / k if k else np.nan,
        "top_k_jaccard": len(intersection) / len(union) if union else np.nan,
        "top_k_overlap_chance": k / n if n else np.nan,
        "top_k_jaccard_chance": expected_random_jaccard(n, k),
        "top_k_is_trivial": int(n <= int(top_k)),
        "sign_agreement": float(np.mean(sign_bert == sign_clip)) if n else np.nan,
        "sign_agreement_chance": positive_bert * positive_clip + (1 - positive_bert) * (1 - positive_clip),
        "bert_positive_sign_fraction": positive_bert,
        "clip_positive_sign_fraction": positive_clip,
    }


# ---------------------------------------------------------------------- bootstrap

def stratified_bootstrap_indices(labels: Sequence[str], iterations: int, seed: int) -> np.ndarray:
    """(iterations, n) resample indices; NEG and POS are resampled separately in each replicate."""
    labels_array = np.asarray(labels)
    rng = np.random.default_rng(int(seed))
    groups = [np.flatnonzero(labels_array == label) for label in ("NEG", "POS")]
    samples = np.empty((int(iterations), labels_array.size), dtype=np.int64)
    for iteration in range(int(iterations)):
        samples[iteration] = np.concatenate([rng.choice(group, size=group.size, replace=True) for group in groups])
    return samples


def bootstrap_classification(gold: np.ndarray, predicted: np.ndarray, samples: np.ndarray) -> dict[str, np.ndarray]:
    g, p = gold[samples], predicted[samples]
    tn = np.sum((g == "NEG") & (p == "NEG"), axis=1)
    fp = np.sum((g == "NEG") & (p == "POS"), axis=1)
    fn = np.sum((g == "POS") & (p == "NEG"), axis=1)
    tp = np.sum((g == "POS") & (p == "POS"), axis=1)

    def f1(tp_: np.ndarray, fp_: np.ndarray, fn_: np.ndarray) -> np.ndarray:
        denominator = 2 * tp_ + fp_ + fn_
        return np.divide(2 * tp_, denominator, out=np.zeros(tp_.shape, dtype=float), where=denominator > 0)

    recall_neg = tn / np.maximum(tn + fp, 1)
    recall_pos = tp / np.maximum(tp + fn, 1)
    return {
        "accuracy": (tn + tp) / g.shape[1],
        "macro_f1": (f1(tn, fn, fp) + f1(tp, fp, fn)) / 2,
        "balanced_accuracy": (recall_neg + recall_pos) / 2,
    }


def bootstrap_mean(values: Sequence[float], samples: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.nanmean(array[samples], axis=1)


def bootstrap_median(values: Sequence[float], samples: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.nanmedian(array[samples], axis=1)


def confidence_row(
    scope: str, metric: str, point: float, replicates: np.ndarray, iterations: int, seed: int, level: float
) -> dict[str, Any]:
    tail = (1 - level) / 2 * 100
    return {
        "scope": scope,
        "metric": metric,
        "point_estimate": float(point),
        "ci_lower": float(np.percentile(replicates, tail)),
        "ci_upper": float(np.percentile(replicates, 100 - tail)),
        "bootstrap_mean": float(np.mean(replicates)),
        "bootstrap_std": float(np.std(replicates, ddof=1)),
        "confidence_level": level,
        "iterations": iterations,
        "seed": seed,
        "method": "stratified (NEG/POS resampled separately) paired percentile bootstrap",
    }


def paired_wilcoxon(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    left_array, right_array = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    difference = left_array - right_array
    try:
        p_value = float(wilcoxon(left_array, right_array, zero_method="wilcox").pvalue)
    except ValueError:
        p_value = np.nan
    return {
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "fraction_left_greater": float(np.mean(difference > 0)),
        "wilcoxon_p": p_value,
    }


# ------------------------------------------------------------------ additivity

def residual_summary(model: str, output: str, values: Sequence[float], tolerance: float, kind: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    stats = describe(array)
    passed = int(np.sum(array <= tolerance))
    return {
        "model": model,
        "check": kind,
        "output": output,
        "sentences": int(array.size),
        "mean": stats["mean"],
        "median": stats["median"],
        "p95": stats["p95"],
        "p99": float(np.percentile(array, 99)) if array.size else np.nan,
        "max": stats["max"],
        "tolerance": tolerance,
        "pass_count": passed,
        "fail_count": int(array.size - passed),
        "all_pass": bool(passed == array.size),
    }


def group_means(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {"sentences": len(rows)}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[f"mean_{field}"] = float(finite.mean()) if finite.size else np.nan
    return result


def subgroup_table(
    scope: str,
    rows: Sequence[dict[str, Any]],
    groupers: dict[str, Callable[[dict[str, Any]], str]],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    table = []
    for group_type, key in groupers.items():
        for group in sorted({key(row) for row in rows}):
            selected = [row for row in rows if key(row) == group]
            table.append({"scope": scope, "group_type": group_type, "group": group, **group_means(selected, fields)})
    return table
