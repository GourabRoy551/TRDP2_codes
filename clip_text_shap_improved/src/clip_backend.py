"""CLIP Text Encoder loading, batching, and fixed-order sentiment scoring."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from prompting import build_prototypes, encode_texts


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def load_clip(
    model_name: str, device: torch.device, local_files_only: bool
) -> tuple[Any, Any]:
    from transformers import CLIPModel, CLIPTokenizerFast

    tokenizer = CLIPTokenizerFast.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model = CLIPModel.from_pretrained(
        model_name, local_files_only=local_files_only
    ).to(device)
    model.eval()
    return model, tokenizer


class ClipSentimentScorer:
    """Return `[NEG, POS, POS−NEG]` scores for each supplied string."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prototypes: torch.Tensor,
        device: torch.device,
        max_length: int,
    ) -> None:
        if tuple(prototypes.shape[:1]) != (2,):
            raise ValueError("Exactly two prototypes are required in [NEG, POS] order.")
        self.model = model
        self.tokenizer = tokenizer
        self.prototypes = prototypes.to(device)
        self.device = device
        self.max_length = max_length
        self.text_evaluations = 0
        self.forward_batches = 0

    def __call__(self, texts: Sequence[str] | np.ndarray) -> np.ndarray:
        batch = [str(value) for value in np.asarray(texts, dtype=object).reshape(-1)]
        features = encode_texts(
            self.model, self.tokenizer, batch, self.device, self.max_length
        )
        class_scores = features @ self.prototypes.T
        margin = class_scores[:, 1:2] - class_scores[:, 0:1]
        self.text_evaluations += len(batch)
        self.forward_batches += 1
        return (
            torch.cat([class_scores, margin], dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )


class MarginOnlyScorer:
    """One-output view used for prompt-sensitivity SHAP comparisons."""

    def __init__(self, scorer: ClipSentimentScorer) -> None:
        self.scorer = scorer

    def __call__(self, texts: Sequence[str] | np.ndarray) -> np.ndarray:
        return self.scorer(texts)[:, 2:3]


def encode_in_batches(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    batches = []
    for start in range(0, len(texts), batch_size):
        batches.append(
            encode_texts(
                model,
                tokenizer,
                texts[start : start + batch_size],
                device,
                max_length,
            ).cpu()
        )
    return torch.cat(batches, dim=0)


def prototypes_for_families(
    model: Any,
    tokenizer: Any,
    families: dict[str, dict[str, Any]],
    device: torch.device,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    result: dict[str, torch.Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    for family_name, family in families.items():
        prototypes, rows = build_prototypes(
            model, tokenizer, family, device, max_length
        )
        result[family_name] = prototypes.cpu()
        diagnostics.extend({"family": family_name, **row} for row in rows)
    return result, diagnostics
