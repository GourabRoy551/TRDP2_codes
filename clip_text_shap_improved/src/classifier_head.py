"""Deterministic linear sentiment head trained only on frozen CLIP embeddings."""

from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from prompting import encode_texts


CLASS_NAMES = ("NEG", "POS")


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(logits).argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "negative_precision": float(precision[0]),
        "negative_recall": float(recall[0]),
        "negative_f1": float(f1[0]),
        "negative_support": int(support[0]),
        "positive_precision": float(precision[1]),
        "positive_recall": float(recall[1]),
        "positive_f1": float(f1[1]),
        "positive_support": int(support[1]),
        "true_neg": int(matrix[0, 0]),
        "false_pos": int(matrix[0, 1]),
        "false_neg": int(matrix[1, 0]),
        "true_pos": int(matrix[1, 1]),
    }


def linear_logits(
    embeddings: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    return np.asarray(embeddings) @ np.asarray(weights).T + np.asarray(bias)


def train_linear_head(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_labels: np.ndarray,
    weight_decay: float,
    learning_rate: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    seed: int,
    balanced_class_weights: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train one regularization candidate and retain its best validation epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_x = torch.from_numpy(np.asarray(train_embeddings, dtype=np.float32))
    train_y = torch.from_numpy(np.asarray(train_labels, dtype=np.int64))
    validation_x = torch.from_numpy(np.asarray(validation_embeddings, dtype=np.float32))
    head = torch.nn.Linear(train_x.shape[1], 2)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_weights = None
    if balanced_class_weights:
        counts = np.bincount(train_labels, minlength=2)
        values = len(train_labels) / (2.0 * counts)
        loss_weights = torch.tensor(values, dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    stale_epochs = 0
    curves: list[dict[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        head.train()
        order = rng.permutation(len(train_y))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start : start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_x[indices])
            loss = criterion(logits, train_y[indices])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
        head.eval()
        with torch.inference_mode():
            validation_logits = head(validation_x).numpy()
        metrics = classification_metrics(validation_labels, validation_logits)
        curves.append(
            {
                "weight_decay": weight_decay,
                "epoch": epoch,
                "train_loss": total_loss / len(train_y),
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
        score = (metrics["macro_f1"], metrics["accuracy"])
        best_score = (
            (-1.0, -1.0)
            if best_metrics is None
            else (best_metrics["macro_f1"], best_metrics["accuracy"])
        )
        if score > best_score:
            best_state = copy.deepcopy(head.state_dict())
            best_metrics = metrics
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("Linear-head training failed to record a valid epoch.")
    result = {
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "epochs_run": len(curves),
        "metrics": best_metrics,
        "weights": best_state["weight"].numpy().copy(),
        "bias": best_state["bias"].numpy().copy(),
    }
    return result, curves


class FrozenClipLinearScorer:
    """Map text to learned NEG/POS logits and their POS−NEG margin."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        weights: np.ndarray,
        bias: np.ndarray,
        device: torch.device,
        max_length: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
        self.bias = torch.tensor(bias, dtype=torch.float32, device=device)
        self.device = device
        self.max_length = max_length
        self.text_evaluations = 0

    def __call__(self, texts: Sequence[str] | np.ndarray) -> np.ndarray:
        batch = [str(value) for value in np.asarray(texts, dtype=object).reshape(-1)]
        embeddings = encode_texts(
            self.model, self.tokenizer, batch, self.device, self.max_length
        )
        logits = embeddings @ self.weights.T + self.bias
        margin = logits[:, 1:2] - logits[:, 0:1]
        self.text_evaluations += len(batch)
        return (
            torch.cat([logits, margin], dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
