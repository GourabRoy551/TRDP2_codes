"""CLIP Vision Encoder wrapper that turns an image into one explainable score.

The explained score is the cosine similarity between

* the CLIP image embedding of a (partially blurred) image, and
* the CLIP image embedding of the original, unmasked image.

It is image-only: no text encoder, prompts, captions or labels are used. For
the unmasked image the score is exactly 1. A positive SHAP value therefore
means that a region helps preserve CLIP's representation of the image.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


def choose_device(requested: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model_and_processor(
    model_name: str, device: torch.device, local_files_only: bool
) -> tuple[Any, Any]:
    """Load the CLIP vision tower with its projection head and the image processor."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    try:
        processor = CLIPImageProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        model = CLIPVisionModelWithProjection.from_pretrained(model_name, local_files_only=local_files_only)
    except OSError:
        if not local_files_only:
            raise
        print(f"{model_name} is not fully cached; downloading it once from Hugging Face.")
        processor = CLIPImageProcessor.from_pretrained(model_name)
        model = CLIPVisionModelWithProjection.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, processor


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB")


def prepare_model_input(image: Image.Image, processor: Any) -> np.ndarray:
    """Return the exact 224x224 crop CLIP sees, as HWC float32 in [0, 1].

    CLIP's own processor is used for resizing and centre cropping; only the
    final mean/std normalisation is postponed so SHAP can mask real pixels.
    """
    encoded = processor(images=image, do_normalize=False, return_tensors="pt")
    chw = encoded["pixel_values"][0].detach().cpu().numpy().astype(np.float32)
    return np.ascontiguousarray(np.transpose(chw, (1, 2, 0)))


def _setting(container: Any, key: str, default: int) -> int:
    """Read a size field from a dict or a transformers SizeDict."""
    try:
        value = container[key]
    except (KeyError, TypeError, IndexError):
        value = getattr(container, key, None)
    return int(value) if value is not None else default


def crop_box_in_original(width: int, height: int, processor: Any) -> tuple[float, float, float, float]:
    """Region of the original image kept by resize-shortest-edge + centre crop."""
    size = _setting(processor.size, "shortest_edge", 224)
    crop_h = _setting(processor.crop_size, "height", 224)
    crop_w = _setting(processor.crop_size, "width", 224)
    short, long = min(width, height), max(width, height)
    new_long = int(size * long / short)
    resized_w, resized_h = (new_long, size) if width >= height else (size, new_long)
    scale_x, scale_y = resized_w / width, resized_h / height
    left = (resized_w - crop_w) // 2
    top = (resized_h - crop_h) // 2
    return (left / scale_x, top / scale_y, (left + crop_w) / scale_x, (top + crop_h) / scale_y)


class ClipImageSimilarity:
    """Map a batch of HWC images in [0, 1] to cosine similarity with a reference."""

    def __init__(self, model: Any, processor: Any, device: torch.device) -> None:
        self.model = model
        self.device = device
        mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.reference: torch.Tensor | None = None
        self.evaluated_images = 0
        self.forward_batches = 0

    def embed(self, images: np.ndarray) -> torch.Tensor:
        """L2-normalised CLIP image embeddings for HWC images in [0, 1]."""
        batch = np.asarray(images, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[None]
        pixels = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).to(self.device)
        pixels = (pixels - self.mean) / self.std
        with torch.inference_mode():
            output = self.model(pixel_values=pixels)
        embeddings = output.image_embeds if hasattr(output, "image_embeds") else output[0]
        self.evaluated_images += int(batch.shape[0])
        self.forward_batches += 1
        return torch.nn.functional.normalize(embeddings.float(), dim=-1)

    def set_reference(self, image: np.ndarray) -> None:
        self.reference = self.embed(image)[0].detach()

    def __call__(self, images: np.ndarray) -> np.ndarray:
        if self.reference is None:
            raise RuntimeError("Call set_reference() before scoring images.")
        similarity = self.embed(images) @ self.reference
        return similarity.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
