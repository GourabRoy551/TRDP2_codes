"""Frozen zero-shot CLIP text scorer returning ``[NEG sim, POS sim, POS-NEG margin]``.

prototype_c = normalize(mean_j normalize(E(prompt_c,j)))   for c in {NEG, POS}
score_c(x)  = normalize(E(x)) . prototype_c
No classification head is trained; the prompts were frozen before this experiment.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from io_utils import load_json, project_path, sha256_file
from model_outputs import CLASS_NAMES, as_text_list, assemble_outputs, freeze


def load_clip(config: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    from transformers import CLIPModel, CLIPTokenizerFast

    settings = config["models"]["clip"]
    local = bool(config["local_files_only"])
    tokenizer = CLIPTokenizerFast.from_pretrained(settings["name_or_path"], local_files_only=local)
    model = CLIPModel.from_pretrained(settings["name_or_path"], local_files_only=local).to(device)
    return freeze(model), tokenizer


def encode_texts(
    model: Any, tokenizer: Any, texts: Sequence[str], device: torch.device, max_length: int
) -> torch.Tensor:
    """L2-normalized CLIP text embeddings (projection output)."""
    encoded = tokenizer(
        list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        features = model.get_text_features(**encoded)
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    return functional.normalize(features.float(), dim=-1)


def prompt_family(config: dict[str, Any]) -> dict[str, list[str]]:
    prompts = config["models"]["clip"]["prompts"]
    family = {name: [str(value).strip() for value in prompts[name]] for name in CLASS_NAMES}
    if len(family["NEG"]) != len(family["POS"]) or not family["NEG"]:
        raise ValueError("The frozen prompt family must be balanced and non-empty.")
    return family


def verify_prompt_source(config: dict[str, Any]) -> dict[str, Any]:
    """Confirm the configured prompts equal the frozen family selected without the final set."""
    settings = config["models"]["clip"]
    path = project_path(settings["prompt_source_manifest"])
    family = prompt_family(config)
    if not path.exists():
        return {"source_manifest_found": False, "passed": False}
    manifest = load_json(path)
    checks = {
        "family_name_matches": manifest.get("selected_family") == settings["prompt_family"],
        "prompts_identical": manifest.get("selected_prompts") == family,
        "selection_did_not_use_final_evaluation": manifest.get("selection_uses_final_evaluation") is False,
        "balanced_prompt_counts": len(family["NEG"]) == len(family["POS"]),
    }
    return {
        "source_manifest_found": True,
        "source_manifest_path": str(path),
        "source_manifest_sha256": sha256_file(path),
        "prompts_per_class": len(family["NEG"]),
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_prototypes(
    model: Any, tokenizer: Any, family: dict[str, list[str]], device: torch.device, max_length: int
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    prototypes, diagnostics = [], []
    for class_name in CLASS_NAMES:
        features = encode_texts(model, tokenizer, family[class_name], device, max_length)
        prototype = functional.normalize(features.mean(dim=0), dim=-1)
        prototypes.append(prototype)
        for index, (prompt, similarity) in enumerate(
            zip(family[class_name], (features @ prototype).tolist(), strict=True), start=1
        ):
            diagnostics.append(
                {"class_name": class_name, "prompt_index": index, "prompt": prompt, "cosine_to_prototype": similarity}
            )
    return torch.stack(prototypes), diagnostics


class ClipThreeOutputScorer:
    """Callable used by SHAP: cosine similarity to the NEG and POS prototypes plus the margin."""

    def __init__(
        self, model: Any, tokenizer: Any, prototypes: torch.Tensor, device: torch.device, max_length: int
    ) -> None:
        if tuple(prototypes.shape[:1]) != (2,):
            raise ValueError("Exactly two prototypes are required, ordered [NEG, POS].")
        self.model = model
        self.tokenizer = tokenizer
        self.prototypes = prototypes.to(device)
        self.device = device
        self.max_length = int(max_length)
        self.text_evaluations = 0
        self.forward_batches = 0

    def __call__(self, texts: Any) -> np.ndarray:
        batch = as_text_list(texts)
        features = encode_texts(self.model, self.tokenizer, batch, self.device, self.max_length)
        self.text_evaluations += len(batch)
        self.forward_batches += 1
        return assemble_outputs(features @ self.prototypes.T)

    def token_count(self, text: str) -> int:
        """BPE tokens including start/end tokens after truncation."""
        return len(self.tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"])


def create_clip_scorer(config: dict[str, Any], device: torch.device) -> tuple[ClipThreeOutputScorer, dict[str, Any]]:
    model, tokenizer = load_clip(config, device)
    max_length = int(config["models"]["clip"]["max_length"])
    prototypes, diagnostics = build_prototypes(model, tokenizer, prompt_family(config), device, max_length)
    scorer = ClipThreeOutputScorer(model, tokenizer, prototypes, device, max_length)
    info = {
        "prototype_diagnostics": diagnostics,
        "prototype_norms": [float(value) for value in prototypes.norm(dim=-1).tolist()],
        "neg_pos_prototype_cosine": float((prototypes[0] @ prototypes[1]).item()),
    }
    return scorer, info
