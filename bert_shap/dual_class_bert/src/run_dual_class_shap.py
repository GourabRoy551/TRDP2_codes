"""Run BERT Partition SHAP for the NEG and POS logits simultaneously."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    import shap
except ImportError:
    shap = None

from make_matrix_plots import (
    save_additivity_residual_matrix,
    save_all_sentence_summary_matrix,
    save_class_comparison_bars,
    save_class_value_matrix,
    save_dual_word_bar,
    save_global_word_class_matrix,
    save_sentence_class_magnitude_matrix,
    save_sentence_overview,
    save_sentence_summary_matrix,
)
from model_wrapper_dual import (
    BertSentimentDualLogits,
    CLASS_NAMES,
    choose_device,
    load_model_and_tokenizer,
    resolve_label_indices,
)
from prepare_dataset import prepare
from token_aggregation_dual import (
    aggregate_wordpieces_dual,
    bert_feature_tokens,
    calculate_additivity_dual,
    content_feature_indices,
    margin_components,
    normalized_token,
    unpack_dual_explanation,
)


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The top-level configuration must be a JSON object.")
    return config


def experiment_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else EXPERIMENT_DIR / path


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = pre_parser.parse_known_args()
    settings = load_config(known.config.resolve())

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Explain both BERT sentiment logits with Partition SHAP.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=experiment_path(settings["input_path"]),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=experiment_path(settings["output_root"]),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--sentence-ids", nargs="+", default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--model-name", default=settings["model_name"])
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=settings.get("device", "auto"),
    )
    parser.add_argument(
        "--max-length", type=int, default=int(settings.get("max_length", 128))
    )
    parser.add_argument(
        "--max-evals", type=int, default=int(settings.get("max_evals", 500))
    )
    parser.add_argument(
        "--batch-size", type=int, default=int(settings.get("batch_size", 32))
    )
    parser.add_argument(
        "--collapse-mask-token",
        action="store_true",
        default=bool(settings.get("collapse_mask_token", False)),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=bool(settings.get("local_files_only", True)),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def read_examples(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sentence_id", "text", "gold_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        examples: list[dict[str, str]] = []
        for row in reader:
            label = row["gold_label"].strip().upper()
            if label not in CLASS_NAMES:
                raise ValueError(f"Invalid label: {label!r}")
            examples.append(
                {
                    "sentence_id": row["sentence_id"].strip(),
                    "text": row["text"].strip(),
                    "gold_label": label,
                    "source": (row.get("source") or "").strip(),
                    "source_row_id": (row.get("source_row_id") or "").strip(),
                }
            )
    return examples


def select_examples(
    examples: list[dict[str, str]],
    sentence_ids: Sequence[str] | None,
    max_sentences: int | None,
) -> list[dict[str, str]]:
    selected = examples
    if sentence_ids:
        requested = {value.upper() for value in sentence_ids}
        selected = [
            row for row in examples if row["sentence_id"].upper() in requested
        ]
        found = {row["sentence_id"].upper() for row in selected}
        if missing := requested - found:
            raise ValueError(f"Unknown sentence IDs: {sorted(missing)}")
    if max_sentences is not None:
        if max_sentences < 1:
            raise ValueError("--max-sentences must be at least one.")
        selected = selected[:max_sentences]
    if not selected:
        raise RuntimeError("No sentences selected.")
    return selected


def create_output_dirs(root: Path, run_name: str | None) -> dict[str, Path]:
    name = run_name or datetime.now().strftime("bert_dual_%Y%m%d_%H%M%S")
    directories = {
        category: root / category / name
        for category in ("values", "plots", "reports")
    }
    if existing := [path for path in directories.values() if path.exists()]:
        raise FileExistsError(
            "The requested run already exists: "
            + ", ".join(str(path) for path in existing)
        )
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=False)
    return directories


def create_explainer(
    model_function: BertSentimentDualLogits,
    tokenizer: Any,
    collapse_mask_token: bool,
) -> tuple[Any, Any]:
    if shap is None:
        raise RuntimeError("SHAP is unavailable in the selected environment.")
    if tokenizer.mask_token is None:
        raise RuntimeError("The BERT tokenizer has no mask token.")
    masker = shap.maskers.Text(
        tokenizer,
        mask_token=tokenizer.mask_token,
        collapse_mask_token=collapse_mask_token,
    )
    explainer = shap.Explainer(
        model_function,
        masker,
        algorithm="partition",
        output_names=list(CLASS_NAMES),
    )
    return explainer, masker


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def top_word(
    words: Sequence[str], values: np.ndarray, class_index: int, largest: bool
) -> str:
    if not words:
        return ""
    pairs = list(zip(words, values[:, class_index], strict=True))
    word, value = (
        max(pairs, key=lambda item: item[1])
        if largest
        else min(pairs, key=lambda item: item[1])
    )
    return f"{word} ({float(value):+.4f})"


def read_existing_margin_values(path: Path) -> dict[str, dict[int, float]]:
    if not path.exists():
        return {}
    result: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            result[row["sentence_id"]][int(row["feature_index"])] = float(
                row["shap_value"]
            )
    return dict(result)


def build_report(
    summaries: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    correct = sum(int(row["correct"]) for row in summaries)
    max_neg = max(abs(float(row["negative_additivity_residual"])) for row in summaries)
    max_pos = max(abs(float(row["positive_additivity_residual"])) for row in summaries)
    lines = [
        "# BERT Dual-Class SHAP Run Report",
        "",
        "## Experiment",
        "",
        f"- Model: {config['model_name']}",
        "- Explained outputs: NEG logit and POS logit",
        "- Explainer: Partition SHAP with the BERT text masker",
        f"- Sentences: {len(summaries)}",
        f"- Correct predictions: {correct}/{len(summaries)}",
        f"- Maximum NEG additivity residual: {max_neg:.3e}",
        f"- Maximum POS additivity residual: {max_pos:.3e}",
        f"- Maximum evaluations per sentence: {config['max_evals']}",
        "",
        "## Sentence results",
        "",
        "| ID | Gold | Pred. | NEG logit | POS logit | Top NEG increase | Top POS increase |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        lines.append(
            "| {sentence_id} | {gold_label} | {prediction} | "
            "{negative_logit:.4f} | {positive_logit:.4f} | "
            "{top_negative_increasing_word} | {top_positive_increasing_word} |".format(
                **row
            )
        )
    if comparison_rows:
        maximum = max(
            abs(float(row["maximum_absolute_difference"]))
            for row in comparison_rows
        )
        lines.extend(
            [
                "",
                "## Existing margin comparison",
                "",
                "For S1-S10, POS SHAP minus NEG SHAP was compared with the saved "
                "scalar POS-logit-minus-NEG-logit explanation.",
                "",
                f"- Maximum absolute token difference: {maximum:.3e}",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive value in the NEG column increases the NEG logit; a "
            "positive value in the POS column increases the POS logit. A "
            "negative value decreases the corresponding target-class logit.",
            "",
            "All matrix cells display raw numerical values. Sentence-summary "
            "matrix colours are normalized within a row or column only to make "
            "different metric scales readable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.max_evals < 3:
        raise ValueError("--max-evals must be at least three.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    settings = load_config(args.config.resolve())
    if not args.input.exists():
        prepare(args.config.resolve(), args.input)
    examples = select_examples(
        read_examples(args.input), args.sentence_ids, args.max_sentences
    )
    output_dirs = create_output_dirs(args.output_root, args.run_name)
    values_dir = output_dirs["values"]
    plots_dir = output_dirs["plots"]
    reports_dir = output_dirs["reports"]

    start_time = time.time()
    torch.manual_seed(int(settings.get("seed", 42)))
    np.random.seed(int(settings.get("seed", 42)))
    device = choose_device(args.device)
    print(f"Loading {args.model_name} on {device}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, device, local_files_only=args.local_files_only
    )
    negative_index, positive_index, label_source = resolve_label_indices(model)
    model_function = BertSentimentDualLogits(
        model,
        tokenizer,
        device,
        args.max_length,
        negative_index,
        positive_index,
    )
    explainer, masker = create_explainer(
        model_function, tokenizer, args.collapse_mask_token
    )

    existing_margin_path = experiment_path(
        settings.get(
            "existing_margin_values_path",
            "../outputs/values/modular_s1_s10/token_shap_values.csv",
        )
    )
    existing_margin = read_existing_margin_values(existing_margin_path)

    run_config: dict[str, Any] = {
        **vars(args),
        "config": str(args.config.resolve()),
        "input": str(args.input.resolve()),
        "output_root": str(args.output_root.resolve()),
        "values_dir": str(values_dir.resolve()),
        "plots_dir": str(plots_dir.resolve()),
        "reports_dir": str(reports_dir.resolve()),
        "device_resolved": str(device),
        "negative_label_index": negative_index,
        "positive_label_index": positive_index,
        "label_index_source": label_source,
        "output_names": list(CLASS_NAMES),
        "mask_token": tokenizer.mask_token,
        "masker": type(masker).__name__,
        "existing_margin_values_path": str(existing_margin_path.resolve()),
        "versions": {
            "python": sys.version.split()[0],
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "shap": package_version("shap"),
            "numpy": package_version("numpy"),
            "matplotlib": package_version("matplotlib"),
        },
    }
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in run_config.items()
    }
    (values_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    summary_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []
    wide_word_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    print(
        f"Examples: {len(examples)} | max_evals={args.max_evals} | "
        f"batch_size={args.batch_size}"
    )
    for item_index, item in enumerate(examples, start=1):
        sentence_start = time.time()
        sentence_id = item["sentence_id"]
        text = item["text"]
        gold_label = item["gold_label"]
        print(f"[{item_index}/{len(examples)}] {sentence_id}: {text}")
        details = model_function.prediction_details(text)
        evals_before = model_function.evaluated_texts
        explanation = explainer(
            [text],
            max_evals=args.max_evals,
            batch_size=args.batch_size,
            silent=True,
        )
        feature_texts, shap_values, base_values = unpack_dual_explanation(
            explanation
        )
        model_tokens = bert_feature_tokens(
            tokenizer,
            text,
            args.max_length,
            expected_count=len(feature_texts),
        )
        logits = np.asarray(
            [details["negative_logit"], details["positive_logit"]], dtype=float
        )
        probabilities = np.asarray(
            [details["negative_probability"], details["positive_probability"]],
            dtype=float,
        )
        reconstructed, residuals = calculate_additivity_dual(
            logits, base_values, shap_values
        )
        words, word_values = aggregate_wordpieces_dual(model_tokens, shap_values)
        margin_values, margin_base = margin_components(shap_values, base_values)
        reconstructed_margin = float(margin_base + margin_values.sum())
        margin_residual = float(details["margin"] - reconstructed_margin)
        content_indices = set(content_feature_indices(model_tokens))

        for feature_index, (feature_text, model_token) in enumerate(
            zip(feature_texts, model_tokens, strict=True)
        ):
            for class_index, class_name in enumerate(CLASS_NAMES):
                value = float(shap_values[feature_index, class_index])
                token_rows.append(
                    {
                        "sentence_id": sentence_id,
                        "text": text,
                        "gold_label": gold_label,
                        "prediction": details["prediction"],
                        "class_index": class_index,
                        "class_name": class_name,
                        "feature_index": feature_index,
                        "feature_text": feature_text,
                        "model_token": model_token,
                        "normalized_feature": normalized_token(model_token),
                        "shap_value": value,
                        "absolute_shap_value": abs(value),
                        "increases_target_logit": int(value > 0),
                        "is_content": int(feature_index in content_indices),
                    }
                )

        for word_index, word in enumerate(words):
            wide_word_rows.append(
                {
                    "sentence_id": sentence_id,
                    "gold_label": gold_label,
                    "prediction": details["prediction"],
                    "word_index": word_index,
                    "word": word,
                    "negative_shap_value": float(word_values[word_index, 0]),
                    "positive_shap_value": float(word_values[word_index, 1]),
                    "margin_shap_value": float(
                        word_values[word_index, 1] - word_values[word_index, 0]
                    ),
                }
            )
            for class_index, class_name in enumerate(CLASS_NAMES):
                value = float(word_values[word_index, class_index])
                word_rows.append(
                    {
                        "sentence_id": sentence_id,
                        "text": text,
                        "gold_label": gold_label,
                        "prediction": details["prediction"],
                        "class_index": class_index,
                        "class_name": class_name,
                        "word_index": word_index,
                        "word": word,
                        "shap_value": value,
                        "absolute_shap_value": abs(value),
                        "increases_target_logit": int(value > 0),
                    }
                )

        if sentence_id in existing_margin:
            differences = [
                float(margin_values[index]) - existing_margin[sentence_id][index]
                for index in range(len(margin_values))
                if index in existing_margin[sentence_id]
            ]
            comparison_rows.append(
                {
                    "sentence_id": sentence_id,
                    "compared_feature_count": len(differences),
                    "maximum_absolute_difference": max(
                        (abs(value) for value in differences), default=float("nan")
                    ),
                    "mean_absolute_difference": float(
                        np.mean(np.abs(differences))
                    )
                    if differences
                    else float("nan"),
                }
            )

        summary = {
            "sentence_id": sentence_id,
            "text": text,
            "gold_label": gold_label,
            "prediction": details["prediction"],
            "correct": int(details["prediction"] == gold_label),
            "source": item["source"],
            "source_row_id": item["source_row_id"],
            "negative_logit": float(logits[0]),
            "positive_logit": float(logits[1]),
            "negative_probability": float(probabilities[0]),
            "positive_probability": float(probabilities[1]),
            "negative_base_value": float(base_values[0]),
            "positive_base_value": float(base_values[1]),
            "negative_sum_shap_values": float(shap_values[:, 0].sum()),
            "positive_sum_shap_values": float(shap_values[:, 1].sum()),
            "negative_reconstructed_logit": float(reconstructed[0]),
            "positive_reconstructed_logit": float(reconstructed[1]),
            "negative_additivity_residual": float(residuals[0]),
            "positive_additivity_residual": float(residuals[1]),
            "margin": float(details["margin"]),
            "margin_base_value_from_dual": margin_base,
            "margin_sum_shap_values_from_dual": float(margin_values.sum()),
            "margin_reconstructed_from_dual": reconstructed_margin,
            "margin_additivity_residual_from_dual": margin_residual,
            "feature_count": len(model_tokens),
            "content_feature_count": len(content_indices),
            "content_word_count": len(words),
            "negative_mean_abs_word_shap": float(
                np.mean(np.abs(word_values[:, 0]))
            ),
            "positive_mean_abs_word_shap": float(
                np.mean(np.abs(word_values[:, 1]))
            ),
            "top_negative_increasing_word": top_word(
                words, word_values, 0, largest=True
            ),
            "top_negative_decreasing_word": top_word(
                words, word_values, 0, largest=False
            ),
            "top_positive_increasing_word": top_word(
                words, word_values, 1, largest=True
            ),
            "top_positive_decreasing_word": top_word(
                words, word_values, 1, largest=False
            ),
            "shap_model_evaluations": (
                model_function.evaluated_texts - evals_before
            ),
            "runtime_seconds": time.time() - sentence_start,
        }
        summary_rows.append(summary)

        if not args.no_plots:
            sentence_dir = plots_dir / sentence_id
            token_labels = [
                f"{index}: {token}" for index, token in enumerate(model_tokens)
            ]
            word_labels = [
                f"{index}: {word}" for index, word in enumerate(words)
            ]
            common = {
                "sentence_id": sentence_id,
                "text": text,
                "gold_label": gold_label,
                "prediction": str(details["prediction"]),
            }
            save_dual_word_bar(
                sentence_dir / f"{sentence_id}_dual_class_word_bar",
                words=words,
                values=word_values,
                logits=logits,
                base_values=base_values,
                residuals=residuals,
                **common,
            )
            save_class_value_matrix(
                sentence_dir / f"{sentence_id}_token_class_matrix",
                row_labels=token_labels,
                values=shap_values,
                level_name="Token",
                **common,
            )
            save_class_value_matrix(
                sentence_dir / f"{sentence_id}_word_class_matrix",
                row_labels=word_labels,
                values=word_values,
                level_name="Word",
                **common,
            )
            save_sentence_summary_matrix(
                sentence_dir / f"{sentence_id}_summary_matrix",
                logits=logits,
                probabilities=probabilities,
                base_values=base_values,
                shap_values=shap_values,
                reconstructed=reconstructed,
                residuals=residuals,
                **common,
            )
            save_sentence_overview(
                sentence_dir / f"{sentence_id}_presentation_overview",
                words=words,
                word_values=word_values,
                logits=logits,
                probabilities=probabilities,
                base_values=base_values,
                shap_values=shap_values,
                reconstructed=reconstructed,
                residuals=residuals,
                **common,
            )

    write_csv(values_dir / "sentence_summary.csv", summary_rows)
    write_csv(values_dir / "token_shap_values.csv", token_rows)
    write_csv(values_dir / "word_shap_values.csv", word_rows)
    write_csv(values_dir / "word_shap_values_wide.csv", wide_word_rows)
    write_csv(values_dir / "margin_comparison.csv", comparison_rows)

    if not args.no_plots:
        save_all_sentence_summary_matrix(
            plots_dir / "all_sentences_summary_matrix", summary_rows
        )
        save_sentence_class_magnitude_matrix(
            plots_dir / "sentence_class_shap_magnitude_matrix", summary_rows
        )
        save_global_word_class_matrix(
            plots_dir / "global_word_class_importance_matrix", word_rows
        )
        save_additivity_residual_matrix(
            plots_dir / "class_additivity_residual_matrix", summary_rows
        )
        save_class_comparison_bars(
            plots_dir / "global_class_comparison_bars", word_rows
        )

    run_config["total_runtime_seconds"] = time.time() - start_time
    run_config["total_model_text_evaluations"] = model_function.evaluated_texts
    run_config["total_forward_batches"] = model_function.forward_batches
    (values_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    report = build_report(summary_rows, comparison_rows, run_config)
    (reports_dir / "run_report.md").write_text(report, encoding="utf-8")

    maximum_residual = max(
        max(
            abs(float(row["negative_additivity_residual"])),
            abs(float(row["positive_additivity_residual"])),
        )
        for row in summary_rows
    )
    print(f"Done: {values_dir.name}")
    print(f"Values: {values_dir.resolve()}")
    print(f"Plots: {plots_dir.resolve()}")
    print(f"Report: {reports_dir.resolve()}")
    print(f"Maximum class additivity residual: {maximum_residual:.3e}")


if __name__ == "__main__":
    main()
