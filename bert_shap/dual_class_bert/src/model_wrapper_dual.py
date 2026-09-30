"""Two-output BERT adapter for explaining NEG and POS logits together."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer


BERT_SHAP_DIR = Path(__file__).resolve().parents[2]
EXISTING_SRC = BERT_SHAP_DIR / "src"
if str(EXISTING_SRC) not in sys.path:
    sys.path.insert(0, str(EXISTING_SRC))

from model_wrapper import (  # noqa: E402
    choose_device,
    load_model_and_tokenizer,
    resolve_label_indices,
)


CLASS_NAMES = ("NEG", "POS")


class BertSentimentDualLogits:
    """Map text batches to logits ordered as NEG then POS."""

    def __init__(
        self,
        model: BertForSequenceClassification,
        tokenizer: BertTokenizer,
        device: torch.device,
        max_length: int,
        negative_index: int,
        positive_index: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.negative_index = negative_index
        self.positive_index = positive_index
        self.evaluated_texts = 0
        self.forward_batches = 0

    @staticmethod
    def _as_text_list(texts: Any) -> list[str]:
        if isinstance(texts, str):
            return [texts]
        values = np.asarray(texts, dtype=object).reshape(-1).tolist()
        return [str(value) for value in values]

    def raw_logits(self, texts: Any) -> torch.Tensor:
        text_list = self._as_text_list(texts)
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs, output_attentions=False).logits
        self.evaluated_texts += len(text_list)
        self.forward_batches += 1
        return logits

    def __call__(self, texts: Any) -> np.ndarray:
        logits = self.raw_logits(texts)
        ordered = logits[:, [self.negative_index, self.positive_index]]
        return ordered.detach().cpu().numpy()

    def prediction_details(self, text: str) -> dict[str, Any]:
        ordered_logits = self([text])[0]
        probabilities = torch.softmax(
            torch.as_tensor(ordered_logits, dtype=torch.float32), dim=-1
        ).numpy()
        prediction_index = int(np.argmax(probabilities))
        return {
            "negative_logit": float(ordered_logits[0]),
            "positive_logit": float(ordered_logits[1]),
            "negative_probability": float(probabilities[0]),
            "positive_probability": float(probabilities[1]),
            "prediction_index": prediction_index,
            "prediction": CLASS_NAMES[prediction_index],
            "margin": float(ordered_logits[1] - ordered_logits[0]),
        }


__all__ = [
    "BertSentimentDualLogits",
    "CLASS_NAMES",
    "choose_device",
    "load_model_and_tokenizer",
    "resolve_label_indices",
]
