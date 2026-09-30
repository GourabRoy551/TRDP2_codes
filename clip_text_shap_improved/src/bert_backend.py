"""Frozen BERT SST-2 backend ordered as NEG, POS and POS-minus-NEG."""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import torch


def load_bert(
    model_name: str, device: torch.device, local_files_only: bool
) -> tuple[Any, Any, int, int, str]:
    from transformers import BertForSequenceClassification, BertTokenizer

    if local_files_only:
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    tokenizer = BertTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        use_safetensors=False,
    ).to(device)
    model.eval()
    id2label = {
        int(index): str(label).lower()
        for index, label in (model.config.id2label or {}).items()
    }
    negative = next((index for index, label in id2label.items() if "neg" in label), None)
    positive = next((index for index, label in id2label.items() if "pos" in label), None)
    if negative is None or positive is None:
        if int(model.config.num_labels) != 2:
            raise ValueError("BERT checkpoint is not a binary sentiment classifier.")
        negative, positive = 0, 1
        source = "TRDP fallback convention: index 0=NEG, index 1=POS"
    else:
        source = "resolved from model.config.id2label"
    return model, tokenizer, int(negative), int(positive), source


class BertThreeOutputScorer:
    """Return raw BERT `[NEG logit, POS logit, POS-NEG margin]` outputs."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
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
        self.text_evaluations = 0
        self.forward_batches = 0

    def __call__(self, texts: Sequence[str] | np.ndarray) -> np.ndarray:
        batch = [str(value) for value in np.asarray(texts, dtype=object).reshape(-1)]
        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded, output_attentions=False).logits
        ordered = logits[:, [self.negative_index, self.positive_index]]
        margin = ordered[:, 1:2] - ordered[:, 0:1]
        self.text_evaluations += len(batch)
        self.forward_batches += 1
        return (
            torch.cat([ordered, margin], dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
