"""Compare the dual-output wrapper with a direct BERT forward pass."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from model_wrapper_dual import (  # noqa: E402
    BertSentimentDualLogits,
    load_model_and_tokenizer,
    resolve_label_indices,
)


MODEL_NAME = "textattack/bert-base-uncased-SST-2"


class DualModelParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.device = torch.device("cpu")
            cls.model, cls.tokenizer = load_model_and_tokenizer(
                MODEL_NAME, cls.device, local_files_only=True
            )
        except Exception as error:
            raise unittest.SkipTest(f"Cached BERT checkpoint unavailable: {error}")
        negative, positive, _ = resolve_label_indices(cls.model)
        cls.negative = negative
        cls.positive = positive
        cls.wrapper = BertSentimentDualLogits(
            cls.model,
            cls.tokenizer,
            cls.device,
            max_length=128,
            negative_index=negative,
            positive_index=positive,
        )

    def test_wrapper_returns_neg_then_pos_logits(self) -> None:
        text = "A wonderful and moving film."
        inputs = self.tokenizer(
            [text], truncation=True, max_length=128, return_tensors="pt"
        )
        with torch.inference_mode():
            direct = self.model(**inputs).logits[
                :, [self.negative, self.positive]
            ].numpy()
        wrapped = self.wrapper([text])
        self.assertEqual(wrapped.shape, (1, 2))
        np.testing.assert_allclose(wrapped, direct, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
