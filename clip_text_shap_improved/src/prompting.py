"""Load balanced prompt families and construct normalized CLIP prototypes."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as functional

from io_utils import load_json


CLASS_NAMES = ("NEG", "POS")
OUTPUT_NAMES = ("NEG", "POS", "MARGIN")


def load_prompt_registry(path: Any) -> dict[str, dict[str, Any]]:
    registry = load_json(path)
    if tuple(registry.get("class_order", [])) != CLASS_NAMES:
        raise ValueError(f"Prompt class order must be {CLASS_NAMES}.")
    families = registry.get("families")
    if not isinstance(families, dict) or not families:
        raise ValueError("Prompt registry must contain at least one family.")
    for family_name, family in families.items():
        negative = [str(value).strip() for value in family.get("NEG", [])]
        positive = [str(value).strip() for value in family.get("POS", [])]
        if not negative or len(negative) != len(positive):
            raise ValueError(f"{family_name}: NEG and POS prompt counts must match.")
        prompts = negative + positive
        if any(not value for value in prompts) or len(set(prompts)) != len(prompts):
            raise ValueError(f"{family_name}: prompts must be non-empty and unique.")
        family["NEG"], family["POS"] = negative, positive
    return families


def extract_text_features(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    raise TypeError(f"Unsupported CLIP text output: {type(output).__name__}")


def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        features = extract_text_features(model.get_text_features(**encoded))
    return functional.normalize(features, dim=-1)


def build_prototypes(
    model: Any,
    tokenizer: Any,
    family: dict[str, Any],
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    prototypes: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        prompt_features = encode_texts(
            model, tokenizer, family[class_name], device, max_length
        )
        prototype = functional.normalize(prompt_features.mean(dim=0), dim=-1)
        prototypes.append(prototype)
        for index, (prompt, similarity) in enumerate(
            zip(family[class_name], prompt_features @ prototype, strict=True), start=1
        ):
            diagnostics.append(
                {
                    "class_name": class_name,
                    "prompt_index": index,
                    "prompt": prompt,
                    "cosine_to_prototype": float(similarity.item()),
                }
            )
    return torch.stack(prototypes), diagnostics
