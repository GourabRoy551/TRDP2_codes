"""Command-line entry point for the TRDP CLIP Vision Partition-SHAP experiment.

This file coordinates the experiment. Model details, patch aggregation,
faithfulness metrics and visualisation live in separate modules so each part
can be read and tested independently (same layout as ``bert_shap``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import shap
except ImportError:  # Produce a clearer project-specific error in main().
    shap = None

from evaluate_faithfulness import compute_faithfulness, stability_row
from make_plots import (
    build_run_report,
    save_faithfulness_plot,
    save_global_patch_plot,
    save_original_and_input,
    save_overview_gallery,
    save_patch_bar,
    save_patch_overlay,
    save_pixel_map,
)
from model_wrapper import (
    ClipImageSimilarity,
    choose_device,
    crop_box_in_original,
    load_model_and_processor,
    load_rgb,
    prepare_model_input,
)
from patch_aggregation import (
    box_blur,
    calculate_additivity,
    patch_grid,
    patch_label,
    patch_rows,
    pixel_map,
    unpack_image_explanation,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config.json"
REFERENCE_PROJECT = Path(r"D:\Research\TRDP_Study Note\TRDP1\trdp")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("The top level of config.json must be a JSON object.")
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = config_parser.parse_known_args()
    settings = load_config(known.config)

    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Explain CLIP ViT-B/32 image embeddings with Partition SHAP.",
    )
    parser.add_argument("--input", type=Path, default=project_path(settings.get("input_path", "data/images.csv")))
    parser.add_argument("--output-root", type=Path, default=project_path(settings.get("output_root", "outputs")))
    parser.add_argument("--run-name", default=None, help="Shared subfolder name under values, plots, and evaluation.")
    parser.add_argument("--model-name", default=settings.get("model_name", "openai/clip-vit-base-patch32"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=settings.get("device", "auto"))
    parser.add_argument("--max-evals", type=int, default=int(settings.get("max_evals", 500)))
    parser.add_argument("--batch-size", type=int, default=int(settings.get("batch_size", 32)))
    parser.add_argument("--blur-kernel", type=int, default=int(settings.get("blur_kernel", 128)))
    parser.add_argument("--patch-size", type=int, default=int(settings.get("patch_size", 32)))
    parser.add_argument("--image-ids", nargs="+", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--faithfulness-fractions", type=float, nargs="+",
                        default=settings.get("faithfulness_fractions", [0.1, 0.2, 0.3, 0.5]))
    parser.add_argument("--random-repeats", type=int, default=int(settings.get("faithfulness_random_repeats", 10)))
    parser.add_argument("--seed", type=int, default=int(settings.get("seed", 42)))
    parser.add_argument("--stability-image-ids", nargs="*", default=settings.get("stability_image_ids", ["I1"]))
    parser.add_argument("--stability-max-evals", type=int, default=int(settings.get("stability_max_evals", 1000)))
    parser.add_argument("--no-faithfulness", action="store_true")
    parser.add_argument("--no-stability", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=bool(settings.get("local_files_only", True)))
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face downloads (overrides local_files_only).")
    args = parser.parse_args()
    if args.allow_download:
        args.local_files_only = False
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_images(path: Path, check_hashes: bool) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Image list not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {"image_id", "image_path"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        rows = list(reader)
    images = []
    for row in rows:
        image_path = Path(row["image_path"])
        image_path = image_path if image_path.is_absolute() else (path.parent.parent / image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"{row['image_id']}: image not found at {image_path}")
        if check_hashes and row.get("sha256") and sha256_file(image_path) != row["sha256"]:
            raise ValueError(f"{row['image_id']}: file changed since data/images.csv was written ({image_path}).")
        images.append({"image_id": row["image_id"].strip(), "path": image_path, "sha256": row.get("sha256", "")})
    return images


def select_images(images: list[dict[str, Any]], image_ids: Sequence[str] | None, max_images: int | None) -> list[dict[str, Any]]:
    selected = images
    if image_ids:
        requested = {value.upper() for value in image_ids}
        selected = [item for item in selected if item["image_id"].upper() in requested]
        found = {item["image_id"].upper() for item in selected}
        if missing := requested - found:
            raise ValueError(f"Unknown image IDs: {sorted(missing)}")
    if max_images is not None:
        if max_images < 1:
            raise ValueError("--max-images must be at least 1.")
        selected = selected[:max_images]
    if not selected:
        raise RuntimeError("No images were selected.")
    return selected


def make_output_dirs(output_root: Path, run_name: str | None) -> tuple[str, dict[str, Path]]:
    resolved = run_name or datetime.now().strftime("clip_vision_shap_%Y%m%d_%H%M%S")
    directories = {category: output_root / category / resolved for category in ("values", "plots", "evaluation")}
    existing = [p for p in directories.values() if p.exists()]
    if existing:
        raise FileExistsError("Output for this run already exists: " + ", ".join(str(p) for p in existing))
    for p in directories.values():
        p.mkdir(parents=True, exist_ok=False)
    return resolved, directories


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "shap", "numpy", "scipy", "matplotlib", "pillow"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


class BatchedScorer:
    """Split large requests into model-sized batches (used for faithfulness)."""

    def __init__(self, scorer: ClipImageSimilarity, batch_size: int) -> None:
        self.scorer = scorer
        self.batch_size = max(1, batch_size)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        images = np.asarray(images)
        parts = [self.scorer(images[i:i + self.batch_size]) for i in range(0, len(images), self.batch_size)]
        return np.concatenate(parts, axis=0)


def explain_image(scorer: ClipImageSimilarity, model_input: np.ndarray, blurred: np.ndarray,
                  max_evals: int, batch_size: int) -> tuple[np.ndarray, float, int, float]:
    """Run Partition SHAP for one image; returns (HWC values, base, model evals, seconds)."""
    masker = shap.maskers.Image(blurred, shape=model_input.shape)
    explainer = shap.Explainer(scorer, masker, algorithm="partition", output_names=["image_similarity"])
    before = scorer.evaluated_images
    start = time.perf_counter()
    explanation = explainer(model_input[None], max_evals=max_evals, batch_size=batch_size, silent=True)
    seconds = time.perf_counter() - start
    values, base = unpack_image_explanation(explanation, model_input.shape)
    return values, base, scorer.evaluated_images - before, seconds


def main() -> None:
    args = parse_args()
    if shap is None:
        raise RuntimeError("SHAP is not installed. Run: python -m pip install -r requirements.txt")
    import torch

    run_start = time.perf_counter()
    images = select_images(load_images(args.input, not args.skip_hash_check), args.image_ids, args.max_images)
    run_name, dirs = make_output_dirs(args.output_root, args.run_name)
    device = choose_device(args.device)
    print(f"Loading {args.model_name} on {device} (local_files_only={args.local_files_only})")
    torch.manual_seed(args.seed)
    model, processor = load_model_and_processor(args.model_name, device, args.local_files_only)
    scorer = ClipImageSimilarity(model, processor, device)
    batched = BatchedScorer(scorer, args.batch_size)

    summaries: list[dict[str, Any]] = []
    all_patch_rows: list[dict[str, Any]] = []
    faithfulness_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    inputs: dict[str, np.ndarray] = {}
    grids: dict[str, np.ndarray] = {}
    arrays: dict[str, np.ndarray] = {}
    stability_ids = {value.upper() for value in (args.stability_image_ids or [])}

    print(f"Images: {len(images)} | max_evals={args.max_evals} | blur={args.blur_kernel} | run={run_name}")
    for item in images:
        image_id = item["image_id"]
        pil = load_rgb(str(item["path"]))
        original = np.asarray(pil)
        model_input = prepare_model_input(pil, processor)
        blurred = box_blur(model_input, args.blur_kernel)
        scorer.set_reference(model_input)
        score = float(scorer(model_input[None])[0, 0])
        blurred_score = float(scorer(blurred[None])[0, 0])

        values, base, evaluations, seconds = explain_image(scorer, model_input, blurred, args.max_evals, args.batch_size)
        pixels = pixel_map(values)
        grid = patch_grid(pixels, args.patch_size)
        additivity = calculate_additivity(score, base, values)
        conservation = float(grid.sum() - pixels.sum())
        cols = grid.shape[1]
        flat = grid.reshape(-1)
        crop_box = crop_box_in_original(pil.width, pil.height, processor)
        kept = ((crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])) / (pil.width * pil.height)

        summary = {
            "image_id": image_id,
            "image_path": str(item["path"]),
            "sha256": item["sha256"],
            "original_width": pil.width,
            "original_height": pil.height,
            "crop_box_original": ";".join(f"{v:.2f}" for v in crop_box),
            "crop_fraction_kept": kept,
            "score": score,
            "base_value": base,
            "fully_blurred_score": blurred_score,
            "base_minus_fully_blurred": base - blurred_score,
            **additivity,
            "patch_conservation_error": conservation,
            "positive_patch_sum": float(flat[flat > 0].sum()),
            "negative_patch_sum": float(flat[flat < 0].sum()),
            "top_positive_patch": f"{patch_label(int(np.argmax(flat)), cols)} ({flat.max():+.4f})",
            "top_negative_patch": (f"{patch_label(int(np.argmin(flat)), cols)} ({flat.min():+.4f})"
                                   if flat.min() < 0 else "none (all patches >= 0)"),
            "shap_model_evaluations": evaluations,
            "max_evals": args.max_evals,
            "runtime_seconds": seconds,
        }
        summaries.append(summary)
        all_patch_rows.extend(patch_rows(image_id, grid, args.patch_size))
        inputs[image_id], grids[image_id] = model_input, grid
        arrays[f"{image_id}_pixel_shap"] = pixels.astype(np.float32)
        arrays[f"{image_id}_patch_shap"] = grid
        arrays[f"{image_id}_base_value"] = np.array(base)
        print(f"  {image_id}: score={score:.4f} base={base:.4f} sum={additivity['sum_shap_values']:+.4f} "
              f"residual={additivity['additivity_residual']:.2e} evals={evaluations} ({seconds:.1f}s)")

        if not args.no_faithfulness:
            faithfulness_rows.extend(compute_faithfulness(
                image_id, model_input, blurred, grid, score, args.faithfulness_fractions,
                args.patch_size, batched, args.random_repeats, args.seed))

        if not args.no_stability and image_id.upper() in stability_ids:
            rerun_values, _, rerun_evals, rerun_seconds = explain_image(
                scorer, model_input, blurred, args.stability_max_evals, args.batch_size)
            rerun_grid = patch_grid(pixel_map(rerun_values), args.patch_size)
            row = stability_row(image_id, grid, rerun_grid, (args.max_evals, args.stability_max_evals))
            row.update({"rerun_model_evaluations": rerun_evals, "rerun_runtime_seconds": rerun_seconds})
            stability_rows.append(row)
            arrays[f"{image_id}_patch_shap_max_evals_{args.stability_max_evals}"] = rerun_grid
            print(f"    stability {args.max_evals} vs {args.stability_max_evals}: spearman={row['spearman']:.3f} "
                  f"top5={row['top_5_overlap']:.2f}")

        if not args.no_figures:
            folder = dirs["plots"] / image_id
            save_original_and_input(image_id, original, model_input, crop_box, folder / f"{image_id}_original_and_input")
            save_patch_overlay(summary, model_input, grid, folder / f"{image_id}_SHAP_patch_overlay", with_values=False)
            save_patch_overlay(summary, model_input, grid, folder / f"{image_id}_SHAP_patch_overlay_values", with_values=True)
            save_pixel_map(summary, model_input, pixels, folder / f"{image_id}_SHAP_pixel_map")
            save_patch_bar(summary, grid, folder / f"{image_id}_SHAP_patch_bar")

    write_csv(dirs["values"] / "image_summary.csv", summaries)
    write_csv(dirs["values"] / "patch_shap_values.csv", all_patch_rows)
    np.savez_compressed(dirs["values"] / "shap_arrays.npz", **arrays)
    if faithfulness_rows:
        write_csv(dirs["evaluation"] / "faithfulness.csv", faithfulness_rows)
    if stability_rows:
        write_csv(dirs["evaluation"] / "stability.csv", stability_rows)
    if not args.no_figures:
        save_global_patch_plot(grids, dirs["plots"] / "mean_absolute_patch_shap")
        save_overview_gallery(inputs, grids, dirs["plots"] / "all_images_patch_shap_overview")
        if faithfulness_rows:
            save_faithfulness_plot(faithfulness_rows, dirs["evaluation"] / "faithfulness_curves")

    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update({
        "run_name": run_name,
        "values_dir": str(dirs["values"]),
        "plots_dir": str(dirs["plots"]),
        "evaluation_dir": str(dirs["evaluation"]),
        "device_resolved": str(device),
        "explained_output": "cosine(CLIP image embedding of masked image, CLIP image embedding of original image)",
        "masker": f"shap.maskers.Image with precomputed box blur ({args.blur_kernel}x{args.blur_kernel}, mirror border)",
        "model_input": "CLIPImageProcessor resize shortest edge + centre crop 224, CLIP mean/std normalisation inside the wrapper",
        "reference_project": str(REFERENCE_PROJECT),
        "versions": package_versions(),
        "total_runtime_seconds": time.perf_counter() - run_start,
        "total_model_image_evaluations": scorer.evaluated_images,
        "total_forward_batches": scorer.forward_batches,
    })
    (dirs["values"] / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (dirs["evaluation"] / "run_report.md").write_text(
        build_run_report(config, summaries, faithfulness_rows, stability_rows), encoding="utf-8")
    worst = max(abs(s["additivity_residual"]) for s in summaries)
    print(f"Done in {config['total_runtime_seconds']:.1f}s. Max additivity residual {worst:.2e}.")
    print(f"Report: {dirs['evaluation'] / 'run_report.md'}")


if __name__ == "__main__":
    sys.exit(main())
