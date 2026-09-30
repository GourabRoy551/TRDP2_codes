"""Compare the SHAP wrapper against CLIP's standard preprocessing and forward pass.

Skips cleanly when PyTorch/transformers are missing or the checkpoint is not cached.
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


class ClipVisionParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch  # noqa: F401
            from model_wrapper import ClipImageSimilarity, load_model_and_processor
        except ImportError as error:
            raise unittest.SkipTest(f"PyTorch/transformers unavailable: {error}")
        config = json.loads((PROJECT / "config.json").read_text(encoding="utf-8"))
        cls.torch = torch
        cls.device = torch.device("cpu")
        try:
            cls.model, cls.processor = load_model_and_processor(config["model_name"], cls.device, True)
        except Exception as error:  # noqa: BLE001 - any loading failure means "not cached"
            raise unittest.SkipTest(f"CLIP checkpoint not available offline: {error}")
        cls.scorer = ClipImageSimilarity(cls.model, cls.processor, cls.device)
        with (PROJECT / "data" / "images.csv").open(encoding="utf-8") as handle:
            first = next(csv.DictReader(handle))
        cls.image_path = (PROJECT / first["image_path"]).resolve()

    def test_wrapper_embedding_matches_standard_path(self) -> None:
        from model_wrapper import load_rgb, prepare_model_input

        pil = load_rgb(str(self.image_path))
        standard = self.processor(images=pil, return_tensors="pt")["pixel_values"].to(self.device)
        with self.torch.inference_mode():
            direct = self.model(pixel_values=standard).image_embeds
        direct = self.torch.nn.functional.normalize(direct.float(), dim=-1)[0]
        wrapped = self.scorer.embed(prepare_model_input(pil, self.processor))[0]
        self.assertLess(float((direct - wrapped).abs().max()), 1e-4)

    def test_unmasked_image_scores_one(self) -> None:
        from model_wrapper import load_rgb, prepare_model_input

        image = prepare_model_input(load_rgb(str(self.image_path)), self.processor)
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)
        self.scorer.set_reference(image)
        score = float(self.scorer(image[None])[0, 0])
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_batch_scores_match_single_scores(self) -> None:
        from model_wrapper import load_rgb, prepare_model_input
        from patch_aggregation import box_blur

        image = prepare_model_input(load_rgb(str(self.image_path)), self.processor)
        self.scorer.set_reference(image)
        blurred = box_blur(image, 128)
        batch = self.scorer(np.stack([image, blurred]))[:, 0]
        single = [float(self.scorer(image[None])[0, 0]), float(self.scorer(blurred[None])[0, 0])]
        np.testing.assert_allclose(batch, single, atol=1e-5)
        self.assertLess(single[1], single[0])


if __name__ == "__main__":
    unittest.main()
