"""Part 4: train and verify a linear sentiment head on frozen CLIP embeddings."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch

from classifier_head import (
    CLASS_NAMES,
    FrozenClipLinearScorer,
    classification_metrics,
    linear_logits,
    train_linear_head,
)
from clip_backend import choose_device, encode_in_batches, load_clip
from io_utils import load_json, project_path, read_csv, save_json, sha256_file, write_csv
from plotting import save_model_comparison, save_part1_overview, save_training_curves
from prompting import OUTPUT_NAMES, build_prototypes
from shap_utils import additivity, create_text_masker, unpack_explanation
from word_masking import create_word_masker, split_word_units


DEFAULT_CONFIG = project_path("configs/part4_frozen_head.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def model_parameter_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(memoryview(array))
    return digest.hexdigest()


def labels_from_rows(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([0 if row["gold_label"] == "NEG" else 1 for row in rows], dtype=np.int64)


def cached_embeddings(
    split_name: str,
    source_path: Path,
    rows: list[dict[str, str]],
    cache_dir: Path,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    model_name: str,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = cache_dir / f"{split_name}_embeddings.npy"
    labels_path = cache_dir / f"{split_name}_labels.npy"
    metadata_path = cache_dir / f"{split_name}_metadata.json"
    expected = {
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "rows": len(rows),
        "model_name": model_name,
        "max_length": max_length,
    }
    if embeddings_path.exists() and labels_path.exists() and metadata_path.exists():
        metadata = load_json(metadata_path)
        if all(metadata.get(key) == value for key, value in expected.items()):
            return np.load(embeddings_path), np.load(labels_path), True
    embeddings = (
        encode_in_batches(
            model,
            tokenizer,
            [row["text"] for row in rows],
            device,
            max_length,
            batch_size,
        )
        .numpy()
        .astype(np.float32, copy=False)
    )
    labels = labels_from_rows(rows)
    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    save_json(
        metadata_path,
        {
            **expected,
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "class_order": list(CLASS_NAMES),
        },
    )
    return embeddings, labels, False


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    part3 = load_json(project_path(config["part3_checks_path"]))
    if not part3.get("gate_passed"):
        raise RuntimeError("Part 3 gate has not passed; Part 4 cannot start.")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_path = project_path(config["training_path"])
    validation_path = project_path(config["validation_path"])
    stability_path = project_path(config["stability_path"])
    train_rows = read_csv(train_path)
    validation_rows = read_csv(validation_path)
    stability_rows = read_csv(stability_path)
    train_hashes = {row["normalized_text_sha256"] for row in train_rows}
    validation_hashes = {row["normalized_text_sha256"] for row in validation_rows}
    leakage_count = len(train_hashes & validation_hashes)

    output_root = project_path(config["output_dir"])
    values_dir, plots_dir, reports_dir = (
        output_root / "values",
        output_root / "plots",
        output_root / "reports",
    )
    for directory in (values_dir, plots_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    cache_dir = project_path(config["embedding_cache_dir"])

    started = time.time()
    device = choose_device(str(config["device"]))
    print(f"Part 4: loading {config['model_name']} on {device}")
    model, tokenizer = load_clip(
        str(config["model_name"]), device, bool(config["local_files_only"])
    )
    model_hash_before = model_parameter_hash(model)
    print(f"Encoding/loading {len(train_rows):,} frozen training embeddings")
    train_embeddings, train_labels, train_cache_hit = cached_embeddings(
        "training",
        train_path,
        train_rows,
        cache_dir,
        model,
        tokenizer,
        device,
        str(config["model_name"]),
        int(config["max_length"]),
        int(config["embedding_batch_size"]),
    )
    print(f"Encoding/loading {len(validation_rows):,} frozen validation embeddings")
    validation_embeddings, validation_labels, validation_cache_hit = cached_embeddings(
        "validation",
        validation_path,
        validation_rows,
        cache_dir,
        model,
        tokenizer,
        device,
        str(config["model_name"]),
        int(config["max_length"]),
        int(config["embedding_batch_size"]),
    )

    candidate_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for candidate_index, weight_decay in enumerate(config["weight_decays"]):
        print(f"Training linear head: weight_decay={weight_decay}")
        result, curves = train_linear_head(
            train_embeddings,
            train_labels,
            validation_embeddings,
            validation_labels,
            float(weight_decay),
            float(config["learning_rate"]),
            int(config["training_batch_size"]),
            int(config["maximum_epochs"]),
            int(config["early_stopping_patience"]),
            seed + candidate_index,
            bool(config["balanced_class_weights"]),
        )
        candidates.append(result)
        curve_rows.extend(curves)
        candidate_rows.append(
            {
                "weight_decay": result["weight_decay"],
                "best_epoch": result["best_epoch"],
                "epochs_run": result["epochs_run"],
                **result["metrics"],
            }
        )
        print(
            f"  best epoch={result['best_epoch']} | "
            f"macro-F1={result['metrics']['macro_f1']:.4f} | "
            f"accuracy={result['metrics']['accuracy']:.4f}"
        )
    winner = sorted(
        candidates,
        key=lambda item: (
            -float(item["metrics"]["macro_f1"]),
            -float(item["metrics"]["accuracy"]),
            float(item["weight_decay"]),
        ),
    )[0]
    weights = np.asarray(winner["weights"], dtype=np.float32)
    bias = np.asarray(winner["bias"], dtype=np.float32)

    # Repeat the winning training configuration from the same initial seed used in
    # its grid position, proving that saved training is deterministic.
    winner_index = list(map(float, config["weight_decays"])).index(float(winner["weight_decay"]))
    repeated, _ = train_linear_head(
        train_embeddings,
        train_labels,
        validation_embeddings,
        validation_labels,
        float(winner["weight_decay"]),
        float(config["learning_rate"]),
        int(config["training_batch_size"]),
        int(config["maximum_epochs"]),
        int(config["early_stopping_patience"]),
        seed + winner_index,
        bool(config["balanced_class_weights"]),
    )
    reproducibility_error = max(
        float(np.max(np.abs(weights - repeated["weights"]))),
        float(np.max(np.abs(bias - repeated["bias"]))),
    )

    head_path = values_dir / "clip_linear_head.npz"
    np.savez(
        head_path,
        weights=weights,
        bias=bias,
        class_names=np.asarray(CLASS_NAMES),
        model_name=np.asarray([config["model_name"]]),
    )
    reloaded = np.load(head_path)
    validation_logits = linear_logits(validation_embeddings, weights, bias)
    reloaded_logits = linear_logits(
        validation_embeddings, reloaded["weights"], reloaded["bias"]
    )
    reload_error = float(np.max(np.abs(validation_logits - reloaded_logits)))
    trained_metrics = classification_metrics(validation_labels, validation_logits)

    frozen_prompts = load_json(project_path(config["frozen_prompt_manifest_path"]))
    family = {
        "NEG": frozen_prompts["selected_prompts"]["NEG"],
        "POS": frozen_prompts["selected_prompts"]["POS"],
    }
    prototypes, _ = build_prototypes(
        model, tokenizer, family, device, int(config["max_length"])
    )
    zero_shot_scores = validation_embeddings @ prototypes.cpu().numpy().T
    zero_shot_metrics = classification_metrics(validation_labels, zero_shot_scores)

    prediction_rows = []
    zero_predictions = zero_shot_scores.argmax(axis=1)
    trained_predictions = validation_logits.argmax(axis=1)
    for index, row in enumerate(validation_rows):
        prediction_rows.append(
            {
                "sentence_id": row["sentence_id"],
                "source_row_id": row["source_row_id"],
                "gold_label": row["gold_label"],
                "zero_shot_negative_score": float(zero_shot_scores[index, 0]),
                "zero_shot_positive_score": float(zero_shot_scores[index, 1]),
                "zero_shot_prediction": CLASS_NAMES[int(zero_predictions[index])],
                "trained_negative_logit": float(validation_logits[index, 0]),
                "trained_positive_logit": float(validation_logits[index, 1]),
                "trained_margin": float(validation_logits[index, 1] - validation_logits[index, 0]),
                "trained_prediction": CLASS_NAMES[int(trained_predictions[index])],
                "trained_correct": int(trained_predictions[index] == validation_labels[index]),
            }
        )

    # Explain trained logits for five predefined development examples using the
    # whole-word masking method approved in Part 3.
    selected_ids = set(config["shap_sentence_ids"])
    shap_examples = [row for row in stability_rows if row["sentence_id"] in selected_ids]
    scorer = FrozenClipLinearScorer(
        model, tokenizer, weights, bias, device, int(config["max_length"])
    )
    word_masker = create_word_masker()
    explainer = shap.Explainer(
        scorer, word_masker, algorithm="partition", output_names=list(OUTPUT_NAMES)
    )
    shap_summary_rows: list[dict[str, Any]] = []
    shap_word_rows: list[dict[str, Any]] = []
    for item_index, row in enumerate(shap_examples, start=1):
        sentence_id, text, gold = row["sentence_id"], row["text"], row["gold_label"]
        print(f"Trained-head SHAP [{item_index}/{len(shap_examples)}] {sentence_id}")
        scores = scorer([text])[0]
        explanation = explainer(
            [text],
            max_evals=int(config["shap_max_evals"]),
            batch_size=int(config["shap_batch_size"]),
            silent=True,
        )
        _, values, bases = unpack_explanation(explanation, 3)
        units = split_word_units(text)
        direct_values = values[1:-1]
        if len(direct_values) != len(units):
            raise ValueError(f"{sentence_id}: whole-word SHAP feature alignment failed.")
        calculation = additivity(scores, bases, values)
        prediction = "POS" if scores[1] > scores[0] else "NEG"
        shap_summary_rows.append(
            {
                "sentence_id": sentence_id,
                "text": text,
                "gold_label": gold,
                "prediction": prediction,
                "negative_logit": float(scores[0]),
                "positive_logit": float(scores[1]),
                "margin": float(scores[2]),
                "negative_base": float(bases[0]),
                "positive_base": float(bases[1]),
                "margin_base": float(bases[2]),
                "negative_residual": float(calculation["residual"][0]),
                "positive_residual": float(calculation["residual"][1]),
                "margin_residual": float(calculation["residual"][2]),
                "maximum_absolute_residual": float(np.max(np.abs(calculation["residual"]))),
            }
        )
        for word_index, unit in enumerate(units):
            shap_word_rows.append(
                {
                    "sentence_id": sentence_id,
                    "word_index": word_index,
                    "word": unit.text,
                    "negative_logit_shap": float(direct_values[word_index, 0]),
                    "positive_logit_shap": float(direct_values[word_index, 1]),
                    "margin_shap": float(direct_values[word_index, 2]),
                }
            )
        save_part1_overview(
            plots_dir / "trained_head_shap" / f"{sentence_id}_trained_logit_shap",
            sentence_id,
            text,
            gold,
            prediction,
            [unit.text for unit in units],
            direct_values,
            scores,
            bases,
            calculation["shap_sum"],
            calculation["reconstructed"],
            calculation["residual"],
        )

    maximum_shap_residual = max(
        float(row["maximum_absolute_residual"]) for row in shap_summary_rows
    )
    model_hash_after = model_parameter_hash(model)
    model_unchanged = model_hash_before == model_hash_after
    performance_improved = trained_metrics["macro_f1"] > zero_shot_metrics["macro_f1"]

    write_csv(values_dir / "regularization_candidates.csv", candidate_rows)
    write_csv(values_dir / "training_curves.csv", curve_rows)
    write_csv(values_dir / "validation_predictions.csv", prediction_rows)
    write_csv(values_dir / "trained_head_shap_summary.csv", shap_summary_rows)
    write_csv(values_dir / "trained_head_word_shap.csv", shap_word_rows)
    save_training_curves(
        plots_dir / "linear_head_training_curves",
        curve_rows,
        float(winner["weight_decay"]),
    )
    save_model_comparison(
        plots_dir / "zero_shot_vs_trained_head", zero_shot_metrics, trained_metrics
    )

    checks = {
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "training_validation_normalized_overlap": leakage_count,
        "class_order": list(CLASS_NAMES),
        "selected_weight_decay": float(winner["weight_decay"]),
        "selected_epoch": int(winner["best_epoch"]),
        "zero_shot_validation_metrics": zero_shot_metrics,
        "trained_head_validation_metrics": trained_metrics,
        "macro_f1_improvement": float(trained_metrics["macro_f1"] - zero_shot_metrics["macro_f1"]),
        "model_parameter_hash_before": model_hash_before,
        "model_parameter_hash_after": model_hash_after,
        "reload_max_absolute_logit_difference": reload_error,
        "reproducibility_max_absolute_parameter_difference": reproducibility_error,
        "maximum_trained_head_shap_additivity_residual": maximum_shap_residual,
        "embedding_cache": {
            "training_cache_hit": train_cache_hit,
            "validation_cache_hit": validation_cache_hit,
            "training_shape": list(train_embeddings.shape),
            "validation_shape": list(validation_embeddings.shape),
        },
        "final_evaluation_used": False,
        "tests": {
            "clip_parameters_unchanged": model_unchanged,
            "saved_head_reload_parity": reload_error <= float(config["reload_tolerance"]),
            "training_reproducibility": reproducibility_error <= float(config["reload_tolerance"]),
            "no_split_leakage": leakage_count == 0,
            "class_order_neg_pos": list(CLASS_NAMES) == ["NEG", "POS"],
            "trained_logit_shap_additivity": maximum_shap_residual <= float(config["additivity_tolerance"]),
            "improves_over_zero_shot": performance_improved,
        },
        "runtime_seconds": time.time() - started,
    }
    checks["gate_passed"] = all(checks["tests"].values())
    save_json(values_dir / "part4_checks.json", checks)
    save_json(
        values_dir / "classifier_metadata.json",
        {
            "model_name": config["model_name"],
            "encoder_frozen": True,
            "embedding_normalization": "L2-normalized CLIP get_text_features output",
            "class_order": list(CLASS_NAMES),
            "weights_shape": list(weights.shape),
            "bias_shape": list(bias.shape),
            "selected_weight_decay": float(winner["weight_decay"]),
            "selected_epoch": int(winner["best_epoch"]),
            "seed": seed,
            "training_source_sha256": sha256_file(train_path),
            "validation_source_sha256": sha256_file(validation_path),
        },
    )
    save_json(values_dir / "run_config.json", {**config, "device_resolved": str(device)})
    report = f"""# Part 4 Results — Frozen CLIP Linear Sentiment Head

The CLIP Text Encoder remained frozen. A two-logit linear head was trained on
{len(train_rows):,} cached CLIP text embeddings and selected on {len(validation_rows):,}
internal-validation examples. The final 500-sentence evaluation set was not used.

## Selected head

- Weight decay: `{winner['weight_decay']}`
- Selected epoch: `{winner['best_epoch']}`
- Zero-shot validation macro-F1: `{zero_shot_metrics['macro_f1']:.4f}`
- Trained-head validation macro-F1: `{trained_metrics['macro_f1']:.4f}`
- Macro-F1 improvement: `{checks['macro_f1_improvement']:+.4f}`
- Trained-head validation accuracy: `{trained_metrics['accuracy']:.4f}`

## Gate checks

- CLIP parameter hashes unchanged: `{model_unchanged}`
- Saved/reloaded maximum logit difference: `{reload_error:.3e}`
- Repeated-training maximum parameter difference: `{reproducibility_error:.3e}`
- Training/validation normalized-text overlap: `{leakage_count}`
- Maximum trained-logit SHAP residual: `{maximum_shap_residual:.3e}`
- Gate passed: **{checks['gate_passed']}**

Five predefined development examples were explained with the approved whole-word
Partition-SHAP masker. Part 5 has not been started.
"""
    (reports_dir / "PART4_RESULTS.md").write_text(report, encoding="utf-8")
    print(f"Part 4 gate passed: {checks['gate_passed']}")
    print(
        f"Validation macro-F1: zero-shot={zero_shot_metrics['macro_f1']:.4f}, "
        f"trained={trained_metrics['macro_f1']:.4f}"
    )
    if not checks["gate_passed"]:
        raise RuntimeError("Part 4 gate failed; Part 5 must not proceed.")


if __name__ == "__main__":
    main()
