"""Command-line entry point for the TRDP BERT Partition-SHAP experiment.

This file coordinates the experiment.  Model details, token processing,
faithfulness metrics, and visualization code live in separate modules so each
part can be read and tested independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

try:
    import shap
except ImportError:  # Produce a clearer project-specific error in main().
    shap = None

from evaluate_faithfulness import compute_faithfulness
from make_comparison_plots import (
    build_run_report,
    save_global_word_plot,
    save_text_html,
    save_word_bar,
    top_word,
)
from model_wrapper import (
    BertSentimentMargin,
    choose_device,
    load_model_and_tokenizer,
    resolve_label_indices,
)
from token_aggregation import (
    aggregate_wordpieces,
    bert_feature_tokens,
    calculate_additivity,
    content_feature_indices,
    normalized_token,
    unpack_explanation,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config.json"
LEGACY_REFERENCE_PROJECT = Path(r"D:\Research\TRDP_Study Note\TRDP1\trdp")


def load_config(path: Path) -> dict[str, Any]:
    """Read the human-editable experiment defaults from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("The top level of config.json must be a JSON object.")
    return config


def project_path(value: str | Path) -> Path:
    """Resolve paths in config.json relative to the bert_shap directory."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def parse_args() -> argparse.Namespace:
    """Parse CLI options, using config.json as the source of defaults."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = config_parser.parse_known_args()
    settings = load_config(known.config)

    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Explain the TRDP BERT SST-2 predictions with Partition SHAP.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_path(settings.get("input_path", "data/sentences.csv")),
        help="CSV containing sentence_id, text, and gold_label columns.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_path(settings.get("output_root", "outputs")),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Shared subfolder name under values, plots, and evaluation.",
    )
    parser.add_argument(
        "--model-name",
        default=settings.get("model_name", "textattack/bert-base-uncased-SST-2"),
    )
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
    parser.add_argument("--sentence-ids", nargs="+", default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument(
        "--faithfulness-fractions",
        type=float,
        nargs="+",
        default=settings.get("faithfulness_fractions", [0.1, 0.2, 0.3, 0.5]),
    )
    parser.add_argument("--no-faithfulness", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument(
        "--collapse-mask-token",
        action="store_true",
        default=bool(settings.get("collapse_mask_token", False)),
        help="Collapse adjacent [MASK] tokens (disabled by default for parity).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=bool(settings.get("local_files_only", False)),
    )
    return parser.parse_args()


def load_examples(path: Path) -> list[dict[str, str]]:
    """Load and validate the sentence ID, text, and gold label columns."""
    if not path.exists():
        raise FileNotFoundError(f"Sentence file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sentence_id", "text", "gold_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        examples: list[dict[str, str]] = []
        for row in reader:
            sentence_id = row["sentence_id"].strip()
            text = row["text"].strip()
            label = row["gold_label"].strip().upper()
            if not sentence_id or not text:
                continue
            if label not in {"POS", "NEG"}:
                raise ValueError(
                    f"{sentence_id}: gold_label must be POS or NEG, got {label!r}."
                )
            examples.append(
                {"sentence_id": sentence_id, "text": text, "gold_label": label}
            )
    return examples


def select_examples(
    examples: list[dict[str, str]],
    sentence_ids: Sequence[str] | None,
    max_sentences: int | None,
) -> list[dict[str, str]]:
    """Apply optional sentence-ID and count filters without changing order."""
    selected = examples
    if sentence_ids:
        requested = {value.upper() for value in sentence_ids}
        selected = [
            item for item in selected if item["sentence_id"].upper() in requested
        ]
        found = {item["sentence_id"].upper() for item in selected}
        if missing := requested - found:
            raise ValueError(f"Unknown sentence IDs: {sorted(missing)}")

    if max_sentences is not None:
        if max_sentences < 1:
            raise ValueError("--max-sentences must be at least 1.")
        selected = selected[:max_sentences]
    if not selected:
        raise RuntimeError("No sentences were selected.")
    return selected


def make_output_dirs(output_root: Path, run_name: str | None) -> dict[str, Path]:
    """Create separate value, plot, and evaluation folders for one run."""
    resolved_name = run_name or datetime.now().strftime("bert_shap_%Y%m%d_%H%M%S")
    directories = {
        category: output_root / category / resolved_name
        for category in ("values", "plots", "evaluation")
    }
    existing = [path for path in directories.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Output for this run already exists: "
            + ", ".join(str(path) for path in existing)
        )
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=False)
    return directories


def create_explainer(
    model_function: BertSentimentMargin,
    tokenizer: Any,
    collapse_mask_token: bool,
) -> tuple[Any, Any]:
    """Build Partition SHAP with BERT's own mask token."""
    if shap is None:
        raise RuntimeError(
            "SHAP is not installed. Run: python -m pip install -r requirements.txt"
        )
    if tokenizer.mask_token is None:
        raise RuntimeError("The tokenizer does not define a mask token.")

    masker = shap.maskers.Text(
        tokenizer,
        mask_token=tokenizer.mask_token,
        collapse_mask_token=collapse_mask_token,
    )
    explainer = shap.Explainer(model_function, masker, algorithm="partition")
    return explainer, masker


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV while preserving their first-seen column order."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def package_version(name: str) -> str:
    """Return a package version for reproducibility metadata."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def reference_project() -> Path:
    """Prefer a sibling TRDP folder and fall back to the original study copy."""
    sibling = PROJECT_DIR.parent / "TRDP"
    return sibling if sibling.exists() else LEGACY_REFERENCE_PROJECT


def main() -> None:
    """Run SHAP, write numerical values, create plots, and evaluate faithfulness."""
    args = parse_args()
    if shap is None:
        raise RuntimeError(
            "SHAP is not installed. Run: python -m pip install -r requirements.txt"
        )
    if args.max_evals < 3:
        raise ValueError("--max-evals must be at least 3.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    start_time = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    examples = select_examples(
        load_examples(args.input), args.sentence_ids, args.max_sentences
    )
    output_dirs = make_output_dirs(args.output_root, args.run_name)
    values_dir = output_dirs["values"]
    plots_dir = output_dirs["plots"]
    evaluation_dir = output_dirs["evaluation"]
    device = choose_device(args.device)

    print(f"Loading tokenizer/model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device,
        local_files_only=args.local_files_only,
    )
    negative_index, positive_index, label_source = resolve_label_indices(model)
    model_function = BertSentimentMargin(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        negative_index=negative_index,
        positive_index=positive_index,
    )
    explainer, masker = create_explainer(
        model_function, tokenizer, args.collapse_mask_token
    )

    run_config = {
        **vars(args),
        "config": str(args.config.resolve()),
        "input": str(args.input.resolve()),
        "output_root": str(args.output_root.resolve()),
        "values_dir": str(values_dir.resolve()),
        "plots_dir": str(plots_dir.resolve()),
        "evaluation_dir": str(evaluation_dir.resolve()),
        "device_resolved": str(device),
        "negative_label_index": negative_index,
        "positive_label_index": positive_index,
        "label_index_source": label_source,
        "explained_output": "positive_logit_minus_negative_logit",
        "mask_token": tokenizer.mask_token,
        "reference_project": str(reference_project()),
        "versions": {
            "python": sys.version.split()[0],
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "shap": package_version("shap"),
            "numpy": package_version("numpy"),
            "matplotlib": package_version("matplotlib"),
        },
    }
    # argparse stores a few Path objects; convert those before JSON serialization.
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in run_config.items()
    }
    (values_dir / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    summary_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []
    faithfulness_rows: list[dict[str, Any]] = []

    print(
        f"Device: {device} | examples: {len(examples)} | "
        f"max_evals: {args.max_evals} | mask: {tokenizer.mask_token}"
    )
    for item_index, item in enumerate(examples, start=1):
        sentence_id = item["sentence_id"]
        text = item["text"]
        gold_label = item["gold_label"]
        sentence_plot_dir = plots_dir / sentence_id
        sentence_plot_dir.mkdir(parents=True, exist_ok=False)

        print(f"[{item_index}/{len(examples)}] {sentence_id}: {text}")
        sentence_start = time.time()
        details = model_function.prediction_details(text)
        evals_before = model_function.evaluated_texts

        explanation = explainer(
            [text],
            max_evals=args.max_evals,
            batch_size=args.batch_size,
            silent=True,
        )
        feature_texts, shap_values, base_value = unpack_explanation(explanation)
        model_tokens = bert_feature_tokens(
            tokenizer,
            text,
            args.max_length,
            expected_count=len(feature_texts),
        )
        reconstructed_margin, residual = calculate_additivity(
            float(details["margin"]), base_value, shap_values
        )
        words, word_values = aggregate_wordpieces(model_tokens, shap_values)
        direction = 1.0 if details["prediction"] == "POS" else -1.0
        content_indices = set(content_feature_indices(model_tokens))

        for feature_index, (feature_text, model_token, value) in enumerate(
            zip(feature_texts, model_tokens, shap_values, strict=True)
        ):
            token_rows.append(
                {
                    "sentence_id": sentence_id,
                    "gold_label": gold_label,
                    "prediction": details["prediction"],
                    "feature_index": feature_index,
                    "feature_text": feature_text,
                    "model_token": model_token,
                    "normalized_feature": normalized_token(model_token),
                    "shap_value": float(value),
                    "absolute_shap_value": abs(float(value)),
                    "support_for_prediction": direction * float(value),
                    "is_content": int(feature_index in content_indices),
                }
            )

        for word_index, (word, value) in enumerate(
            zip(words, word_values, strict=True)
        ):
            word_rows.append(
                {
                    "sentence_id": sentence_id,
                    "gold_label": gold_label,
                    "prediction": details["prediction"],
                    "word_index": word_index,
                    "word": word,
                    "shap_value": float(value),
                    "absolute_shap_value": abs(float(value)),
                    "support_for_prediction": direction * float(value),
                }
            )

        if not args.no_figures:
            save_word_bar(
                sentence_plot_dir / f"{sentence_id}_SHAP_word_bar",
                sentence_id,
                gold_label,
                str(details["prediction"]),
                text,
                words,
                word_values,
                float(details["margin"]),
                base_value,
                residual,
            )
        if not args.no_html:
            save_text_html(
                sentence_plot_dir / f"{sentence_id}_SHAP_text.html",
                sentence_id,
                gold_label,
                str(details["prediction"]),
                feature_texts,
                shap_values,
                float(details["margin"]),
                base_value,
            )
        if not args.no_faithfulness:
            faithfulness_rows.extend(
                compute_faithfulness(
                    sentence_id,
                    gold_label,
                    str(details["prediction"]),
                    text,
                    float(details["margin"]),
                    model_tokens,
                    shap_values,
                    args.faithfulness_fractions,
                    masker,
                    model_function,
                )
            )

        summary_rows.append(
            {
                "sentence_id": sentence_id,
                "text": text,
                "gold_label": gold_label,
                "prediction": details["prediction"],
                "correct": int(details["prediction"] == gold_label),
                "negative_probability": details["negative_probability"],
                "positive_probability": details["positive_probability"],
                "negative_logit": details["negative_logit"],
                "positive_logit": details["positive_logit"],
                "margin": details["margin"],
                "base_value": base_value,
                "sum_shap_values": float(shap_values.sum()),
                "reconstructed_margin": reconstructed_margin,
                "additivity_residual": residual,
                "feature_count": len(feature_texts),
                "content_feature_count": len(content_indices),
                "content_word_count": len(words),
                "shap_model_evaluations": (
                    model_function.evaluated_texts - evals_before
                ),
                "top_positive_word": top_word(words, word_values, positive=True),
                "top_negative_word": top_word(words, word_values, positive=False),
                "runtime_seconds": time.time() - sentence_start,
            }
        )

    write_csv(values_dir / "sentence_summary.csv", summary_rows)
    write_csv(values_dir / "token_shap_values.csv", token_rows)
    write_csv(values_dir / "word_shap_values.csv", word_rows)
    write_csv(evaluation_dir / "faithfulness.csv", faithfulness_rows)

    if not args.no_figures:
        save_global_word_plot(plots_dir / "mean_absolute_word_shap", word_rows)

    run_config["total_runtime_seconds"] = time.time() - start_time
    run_config["total_model_text_evaluations"] = model_function.evaluated_texts
    run_config["total_forward_batches"] = model_function.forward_batches
    (values_dir / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    (evaluation_dir / "run_report.md").write_text(
        build_run_report(summary_rows, faithfulness_rows, run_config),
        encoding="utf-8",
    )

    maximum_residual = max(
        abs(float(row["additivity_residual"])) for row in summary_rows
    )
    print(f"Done. Run name: {values_dir.name}")
    print(f"Values: {values_dir.resolve()}")
    print(f"Plots: {plots_dir.resolve()}")
    print(f"Evaluation: {evaluation_dir.resolve()}")
    print(f"Maximum absolute additivity residual: {maximum_residual:.3e}")


if __name__ == "__main__":
    main()
