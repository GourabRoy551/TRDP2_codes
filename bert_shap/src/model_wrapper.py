"""BERT loading and the scalar prediction function explained by SHAP.

The original TRDP classifier has two output logits.  SHAP needs one scalar
output, so this module exposes the margin ``positive logit - negative logit``.
That choice gives every attribution a stable interpretation:

* positive value: evidence toward positive sentiment;
* negative value: evidence toward negative sentiment.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer


DEFAULT_NEGATIVE_INDEX = 0
DEFAULT_POSITIVE_INDEX = 1


def choose_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` to a PyTorch device."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    local_files_only: bool = False,
) -> tuple[BertForSequenceClassification, BertTokenizer]:
    """Load the same BERT checkpoint used by the RFEM experiment."""
    if local_files_only:
        # Transformers otherwise starts a background conversion check that may
        # contact Hugging Face even though the caller explicitly requested an
        # offline load.
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    tokenizer = BertTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        # The cached TRDP checkpoint stores PyTorch weights.  Explicitly using
        # them prevents Transformers from starting an online conversion check.
        use_safetensors=False,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def resolve_label_indices(
    model: BertForSequenceClassification,
    fallback_negative: int = DEFAULT_NEGATIVE_INDEX,
    fallback_positive: int = DEFAULT_POSITIVE_INDEX,
) -> tuple[int, int, str]:
    """Find POS/NEG indices from model metadata, with the TRDP convention as fallback."""
    id2label = {
        int(index): str(label).lower()
        for index, label in (model.config.id2label or {}).items()
    }
    negative = next(
        (index for index, label in id2label.items() if "neg" in label), None
    )
    positive = next(
        (index for index, label in id2label.items() if "pos" in label), None
    )
    if negative is not None and positive is not None:
        return negative, positive, "resolved from model.config.id2label"

    if model.config.num_labels != 2:
        raise ValueError(
            "The checkpoint is not binary and has no explicit POS/NEG labels."
        )
    return (
        fallback_negative,
        fallback_positive,
        "TRDP fallback convention: index 0=NEG, index 1=POS",
    )


class BertSentimentMargin:
    """Callable adapter that maps one or more texts to BERT sentiment margins."""

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

        # These counters make the cost of a SHAP run visible in its config snapshot.
        self.evaluated_texts = 0
        self.forward_batches = 0

    @staticmethod
    def _as_text_list(texts: Any) -> list[str]:
        """Normalize strings, lists, and NumPy arrays to a flat list of strings."""
        if isinstance(texts, str):
            return [texts]
        values = np.asarray(texts, dtype=object).reshape(-1).tolist()
        return [str(value) for value in values]

    def logits(self, texts: Any) -> torch.Tensor:
        """Return raw model logits for a batch of texts."""
        text_list = self._as_text_list(texts)
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs, output_attentions=False).logits

        self.evaluated_texts += len(text_list)
        self.forward_batches += 1
        return logits

    def __call__(self, texts: Any) -> np.ndarray:
        """Return ``positive logit - negative logit`` for SHAP."""
        logits = self.logits(texts)
        margins = logits[:, self.positive_index] - logits[:, self.negative_index]
        return margins.detach().cpu().numpy()

    def prediction_details(self, text: str) -> dict[str, Any]:
        """Return the logits, probabilities, label, and margin for one sentence."""
        logits = self.logits([text]).squeeze(0)
        probabilities = torch.softmax(logits, dim=-1)
        prediction_index = int(probabilities.argmax().item())
        margin = float(
            (logits[self.positive_index] - logits[self.negative_index]).item()
        )
        return {
            "negative_logit": float(logits[self.negative_index].item()),
            "positive_logit": float(logits[self.positive_index].item()),
            "negative_probability": float(probabilities[self.negative_index].item()),
            "positive_probability": float(probabilities[self.positive_index].item()),
            "prediction_index": prediction_index,
            "prediction": (
                "POS" if prediction_index == self.positive_index else "NEG"
            ),
            "margin": margin,
        }
