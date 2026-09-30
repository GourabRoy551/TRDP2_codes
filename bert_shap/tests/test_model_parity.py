"""Checks that the SHAP wrapper preserves the original BERT prediction."""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from model_wrapper import (  # noqa: E402
    BertSentimentMargin,
    load_model_and_tokenizer,
    resolve_label_indices,
)


MODEL_NAME = "textattack/bert-base-uncased-SST-2"


class ModelParityTest(unittest.TestCase):
    """Compare wrapper results with a direct forward pass of the same model."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.device = torch.device("cpu")
            cls.model, cls.tokenizer = load_model_and_tokenizer(
                MODEL_NAME, cls.device, local_files_only=True
            )
        except Exception as error:  # Cache availability differs between machines.
            raise unittest.SkipTest(f"Local BERT checkpoint is unavailable: {error}")

        negative_index, positive_index, _ = resolve_label_indices(cls.model)
        cls.negative_index = negative_index
        cls.positive_index = positive_index
        cls.wrapper = BertSentimentMargin(
            cls.model,
            cls.tokenizer,
            cls.device,
            max_length=128,
            negative_index=negative_index,
            positive_index=positive_index,
        )

    def test_sentences_file_matches_previous_s1_to_s10_set(self) -> None:
        with (PROJECT_DIR / "data" / "sentences.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["sentence_id"] for row in rows], [f"S{i}" for i in range(1, 11)])
        self.assertEqual(rows[0]["gold_label"], "POS")
        self.assertIn("Painfully boring", rows[6]["text"])

    def test_wrapper_margin_equals_direct_bert_margin(self) -> None:
        text = "The film is a beautiful and moving portrait of human resilience."
        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = self.model(**inputs, output_attentions=False).logits
        direct_margin = (
            logits[:, self.positive_index] - logits[:, self.negative_index]
        ).cpu().numpy()

        wrapped_margin = self.wrapper([text])
        np.testing.assert_allclose(wrapped_margin, direct_margin, rtol=0, atol=1e-7)

    def test_wrapper_prediction_equals_direct_argmax(self) -> None:
        text = "This movie is an absolute waste of time and money."
        inputs = self.tokenizer(
            [text], truncation=True, max_length=128, return_tensors="pt"
        )
        with torch.inference_mode():
            direct_prediction = int(self.model(**inputs).logits.argmax(dim=-1).item())

        details = self.wrapper.prediction_details(text)
        self.assertEqual(details["prediction_index"], direct_prediction)


if __name__ == "__main__":
    unittest.main()
