"""Frozen SST-2 BERT scorer returning ``[NEG logit, POS logit, POS-NEG margin]``."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from model_outputs import CLASS_NAMES, as_text_list, assemble_outputs, freeze


def load_bert(config: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    from transformers import BertForSequenceClassification, BertTokenizer

    settings = config["models"]["bert"]
    local = bool(config["local_files_only"])
    tokenizer = BertTokenizer.from_pretrained(settings["name_or_path"], local_files_only=local)
    # The cached TextAttack checkpoint stores PyTorch weights (pytorch_model.bin).
    model = BertForSequenceClassification.from_pretrained(
        settings["name_or_path"], local_files_only=local, use_safetensors=False
    ).to(device)
    return freeze(model), tokenizer


def resolve_label_mapping(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Read NEG/POS indices from ``id2label``; otherwise use the documented fallback.

    The fallback is never used silently: it is recorded with its source and must
    pass :func:`verify_label_mapping` before any evaluation sentence is scored.
    """
    id2label = {int(index): str(label) for index, label in (model.config.id2label or {}).items()}
    lowered = {index: label.lower() for index, label in id2label.items()}
    negative = [index for index, label in lowered.items() if "neg" in label]
    positive = [index for index, label in lowered.items() if "pos" in label]
    if int(model.config.num_labels) != 2:
        raise ValueError("The BERT checkpoint is not a binary classifier.")
    if len(negative) == 1 and len(positive) == 1:
        return {
            "NEG": negative[0],
            "POS": positive[0],
            "config_id2label": id2label,
            "resolution": "model.config.id2label contains explicit NEG/POS names",
            "config_is_ambiguous": False,
        }
    fallback = config["models"]["bert"]["label_mapping_fallback"]
    return {
        "NEG": int(fallback["NEG"]),
        "POS": int(fallback["POS"]),
        "config_id2label": id2label,
        "resolution": "config id2label is generic (ambiguous); documented fallback applied",
        "fallback_source": fallback["source"],
        "config_is_ambiguous": True,
    }


class BertThreeOutputScorer:
    """Callable used by SHAP: raw BERT logits reordered as NEG, POS plus the margin."""

    def __init__(
        self, model: Any, tokenizer: Any, device: torch.device, max_length: int, mapping: dict[str, Any]
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = int(max_length)
        self.class_indices = [int(mapping[name]) for name in CLASS_NAMES]
        self.text_evaluations = 0
        self.forward_batches = 0

    def logits(self, texts: Sequence[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            return self.model(**encoded).logits

    def __call__(self, texts: Any) -> np.ndarray:
        batch = as_text_list(texts)
        logits = self.logits(batch)
        self.text_evaluations += len(batch)
        self.forward_batches += 1
        return assemble_outputs(logits[:, self.class_indices])

    def token_count(self, text: str) -> int:
        """WordPiece tokens including [CLS]/[SEP] after truncation."""
        return len(self.tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"])


def verify_label_mapping(scorer: BertThreeOutputScorer, config: dict[str, Any]) -> dict[str, Any]:
    """Check the mapping on independent hand-written sentences (not in the 500-sentence set)."""
    items = config["models"]["bert"]["label_mapping_check_sentences"]
    scores = scorer([item["text"] for item in items])
    rows = []
    for item, score in zip(items, scores, strict=True):
        prediction = "POS" if score[2] > 0 else "NEG"
        swapped = "NEG" if prediction == "POS" else "POS"
        rows.append(
            {
                "id": item["id"],
                "gold_label": item["gold_label"],
                "neg_logit": float(score[0]),
                "pos_logit": float(score[1]),
                "prediction": prediction,
                "correct_under_mapping": prediction == item["gold_label"],
                "correct_under_swapped_mapping": swapped == item["gold_label"],
            }
        )
    correct = sum(row["correct_under_mapping"] for row in rows)
    swapped_correct = sum(row["correct_under_swapped_mapping"] for row in rows)
    return {
        "sentences": len(rows),
        "correct_under_mapping": correct,
        "correct_under_swapped_mapping": swapped_correct,
        "rows": rows,
        "passed": correct == len(rows) and swapped_correct == 0,
    }
