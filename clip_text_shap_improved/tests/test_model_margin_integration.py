"""One-model-call integration check for fixed output ordering and margin arithmetic."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from clip_backend import ClipSentimentScorer, load_clip
from prompting import build_prototypes, load_prompt_registry


ROOT = Path(__file__).resolve().parents[1]


class ModelMarginIntegrationTests(unittest.TestCase):
    def test_three_output_scorer(self) -> None:
        device = torch.device("cpu")
        model, tokenizer = load_clip("openai/clip-vit-base-patch32", device, True)
        family = load_prompt_registry(ROOT / "configs" / "prompt_families.json")[
            "baseline_movie_review"
        ]
        prototypes, _ = build_prototypes(model, tokenizer, family, device, 77)
        scorer = ClipSentimentScorer(model, tokenizer, prototypes, device, 77)
        values = scorer(["A beautiful film.", "A terrible film."])
        self.assertEqual(values.shape, (2, 3))
        np.testing.assert_allclose(values[:, 2], values[:, 1] - values[:, 0], atol=0)


if __name__ == "__main__":
    unittest.main()
