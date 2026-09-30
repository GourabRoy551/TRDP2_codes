"""Deletion AUC, insertion AUC, AOPC and pointing game for the existing CLIP vision patch SHAP.

Steps (every gate must pass before any metric file is written):
  1. load the stored 7x7 patch SHAP maps (shap_arrays.npz), check them against
     patch_shap_values.csv, and rebuild the |SHAP| ranking; it must equal the stored
     abs_rank and the patches the source faithfulness evaluation selected;
  2. rebuild each 224x224 input and its blurred background with the source code, load the
     frozen CLIP vision encoder, and require the original and fully blurred scores to
     reproduce the stored ones;
  3. score the 50-step deletion and insertion sequences; their end points and their
     10/20/30/50 % points must reproduce the stored scores;
  4. compute DAUC, IAUC and AOPC per image; look for ground-truth boxes / masks;
  5. write the per-image CSV, curve points, summary table and curve figures.
No SHAP value is recomputed, CLIP is neither trained nor modified, and nothing inside the
source folder or the image folder is written (both are hashed before and after).
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from source_link import OUTPUT_DIR, SOURCE_ROOT, file_hashes, load_own_config  # noqa: E402
from vision_metrics import (  # noqa: E402
    aopc,
    clip_box,
    curve_auc,
    deletion_images,
    fraction_count,
    insertion_images,
    pointing_hit,
    rank_patches,
)
import run_shap  # noqa: E402  (source orchestration module: load_config, load_images, BatchedScorer)
from model_wrapper import ClipImageSimilarity, choose_device, load_model_and_processor, load_rgb, prepare_model_input  # noqa: E402
from patch_aggregation import box_blur  # noqa: E402


LOG_PATH = OUTPUT_DIR / "run.log"
PLOTS_DIR = OUTPUT_DIR / "plots"
IMAGES_DIR = SOURCE_ROOT.parent / "clip_vision_images"
REVIEW_STATUSES = {"approved", "excluded"}
TABLE_ROWS = ("Deletion AUC ↓", "Insertion AUC ↑", "AOPC ↑", "Pointing/Localization Accuracy ↑")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: Any) -> None:
    """JSON with NumPy scalars converted to Python values."""
    path.write_text(json.dumps(value, indent=2, default=lambda item: item.item()), encoding="utf-8")


def fraction_label(fraction: float) -> str:
    return f"f{int(round(fraction * 100)):02d}"


# ------------------------------------------------------------------ pointing game

def find_localization_annotations(own: dict[str, Any], source_config: dict[str, Any]) -> dict[str, Any]:
    """Look for ground-truth boxes / masks in the image list and next to the images."""
    settings = own["pointing_game"]
    image_list = SOURCE_ROOT / source_config["input_path"]
    with image_list.open("r", encoding="utf-8-sig", newline="") as stream:
        columns = list(csv.DictReader(stream).fieldnames or [])
    keyword_columns = [c for c in columns if any(k in c.lower() for k in settings["annotation_keywords"])]
    candidates = []
    for folder in (IMAGES_DIR, SOURCE_ROOT / "data"):
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in settings["annotation_extensions"] and path != image_list:
                candidates.append(str(path))
    return {
        "image_list_columns": columns,
        "annotation_like_columns": keyword_columns,
        "annotation_like_files": candidates,
        "folders_searched": [str(IMAGES_DIR), str(SOURCE_ROOT / "data")],
        "available": bool(keyword_columns or candidates),
    }


def load_author_boxes(own: dict[str, Any], image_ids: list[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Approved author-annotated boxes per image; refuses to run while any box is unreviewed."""
    from source_link import ROOT

    path = ROOT / own["pointing_game"]["boxes_path"]
    rows = read_csv(path)
    statuses = sorted({row["status"] for row in rows})
    unknown_images = sorted({row["image_id"] for row in rows} - set(image_ids))
    boxes: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "approved":
            boxes.setdefault(row["image_id"], []).append(
                {"target": row["target_object"], "box": tuple(float(row[key]) for key in ("x0", "y0", "x1", "y1"))}
            )
    excluded = sorted({row["image_id"] for row in rows if row["status"] == "excluded"} - set(boxes))
    info = {
        "boxes_path": str(path),
        "boxes_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "box_rows": len(rows),
        "statuses": statuses,
        "approved_images": sorted(boxes, key=image_ids.index),
        "excluded_images": excluded,
        "all_rows_reviewed": set(statuses) <= REVIEW_STATUSES,
        "every_image_reviewed": set(boxes) | set(excluded) == set(image_ids) and not unknown_images,
        "one_target_per_image": all(len({item["target"] for item in items}) == 1 for items in boxes.values()),
    }
    return boxes, info


# ------------------------------------------------------------------ outputs

def write_tables(summary: dict[str, Any], n_images: int) -> None:
    values = {
        "Deletion AUC ↓": f"{summary['deletion_auc']:.4f}",
        "Insertion AUC ↑": f"{summary['insertion_auc']:.4f}",
        "AOPC ↑": f"{summary['aopc']:.4f}",
        "Pointing/Localization Accuracy ↑": f"{summary['pointing_game']:.4f}",
    }
    write_csv(OUTPUT_DIR / "summary_table.csv", [
        {"Metric": "Deletion AUC ↓", "CLIP Vision": summary["deletion_auc"]},
        {"Metric": "Insertion AUC ↑", "CLIP Vision": summary["insertion_auc"]},
        {"Metric": "AOPC ↑", "CLIP Vision": summary["aopc"]},
        {"Metric": "Pointing/Localization Accuracy ↑", "CLIP Vision": summary["pointing_game"]},
    ])
    hits, pointed, tolerance = summary["pointing_game_hits"], summary["pointing_game_images"], summary["pointing_game_tolerance_px"]
    markdown = [
        "# CLIP Vision: patch-level SHAP explanation metrics",
        "",
        f"CLIP ViT-B/32 image encoder, images I1–I{n_images}, stored 7×7 patch-level Partition SHAP values.",
        "Values are means over the images.",
        "",
        "| Metric | CLIP Vision |",
        "|---|---|",
        *[f"| {name} | {values[name]} |" for name in TABLE_ROWS],
        "",
        "**Target score.** f = cosine similarity between the CLIP image embedding of the perturbed 224×224 input and",
        "that of the original input (the explained output of the source experiment; the original scores 1).",
        "**Ranking.** The 49 patches (32×32 px) by |SHAP| descending, ties by patch index.",
        "**Masking.** A removed patch is replaced by the 128×128 box-blurred image, the background SHAP itself used.",
        "",
        "- **Deletion AUC**: blur the top-ranked patches one at a time (k = 0…49) and take the trapezoidal area under",
        "  f against the fraction of patches removed (0–1). Starts at f(original) = 1, ends at f(fully blurred).",
        "- **Insertion AUC**: start from the fully blurred image and restore the top-ranked patches one at a time; area",
        "  under f against the fraction restored. Starts at f(fully blurred), ends at 1.",
        "- **AOPC**: mean of f(original) − f(top-k patches blurred) at k = 10/20/30/50 % of the patches",
        "  (5, 10, 15, 25 patches), the same cut-offs as the text AOPC.",
        f"- **Pointing/Localization Accuracy**: {hits} of {pointed} images are hits. A hit means the centre of the highest-|SHAP|",
        f"  patch falls inside a box of the image's target object (clipped to the CLIP crop), with the standard {tolerance:g}-pixel",
        f"  tolerance (without tolerance: {summary['pointing_game_no_tolerance']:.4f}). The images have no dataset boxes or masks;",
        "  the boxes were annotated by the authors without looking at the SHAP maps. The explained score is image-only,",
        "  so this measures whether the most important patch lies on the main object, not class-specific localization.",
        "",
        f"Fully blurred scores range {summary['fully_blurred_min']:.3f}–{summary['fully_blurred_max']:.3f} "
        f"(mean {summary['fully_blurred_mean']:.3f}), so both curves stay inside [{summary['fully_blurred_min']:.3f}, 1].",
        "",
    ]
    (OUTPUT_DIR / "summary_table.md").write_text("\n".join(markdown), encoding="utf-8")
    latex = [
        "% Generated by clip_vision_shap_faithfulness_metrics/src/run_vision_metrics.py",
        "\\begin{tabular}{lr}",
        "\\hline",
        "Metric & CLIP Vision \\\\",
        "\\hline",
        f"Deletion AUC $\\downarrow$ & {values['Deletion AUC ↓']} \\\\",
        f"Insertion AUC $\\uparrow$ & {values['Insertion AUC ↑']} \\\\",
        f"AOPC $\\uparrow$ & {values['AOPC ↑']} \\\\",
        f"Pointing/Localization Accuracy $\\uparrow$ & {values['Pointing/Localization Accuracy ↑']} \\\\",
        "\\hline",
        "\\end{tabular}",
        "",
    ]
    (OUTPUT_DIR / "summary_table.tex").write_text("\n".join(latex), encoding="utf-8")


# ------------------------------------------------------------------ main

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    started = time.perf_counter()
    own = load_own_config()
    source_config = run_shap.load_config(SOURCE_ROOT / "config.json")
    values_dir = SOURCE_ROOT / "outputs" / "values" / own["source_run_name"]
    evaluation_dir = SOURCE_ROOT / "outputs" / "evaluation" / own["source_run_name"]
    stored_config = json.loads((values_dir / "config.json").read_text(encoding="utf-8"))
    source_before, images_before = file_hashes(SOURCE_ROOT), file_hashes(IMAGES_DIR)
    tolerance = float(own["rescore_tolerance"])
    fractions = [float(value) for value in own["aopc_fractions"]]
    patch_size, blur_kernel = int(stored_config["patch_size"]), int(stored_config["blur_kernel"])
    log(f"source run: {values_dir} ({len(source_before)} source files, {len(images_before)} image files hashed)")

    checks: dict[str, bool] = {
        "source_config_matches_stored_run": all(
            source_config[key] == stored_config[key] for key in ("model_name", "blur_kernel", "patch_size", "max_evals", "faithfulness_fractions")
        ),
        "aopc_fractions_equal_source_faithfulness_fractions": fractions == [float(v) for v in stored_config["faithfulness_fractions"]],
    }

    # ---- stored SHAP maps, ranking and the source faithfulness selections
    images = run_shap.load_images(SOURCE_ROOT / source_config["input_path"], check_hashes=True)
    arrays = np.load(values_dir / "shap_arrays.npz")
    summaries = {row["image_id"]: row for row in read_csv(values_dir / "image_summary.csv")}
    patch_table: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(values_dir / "patch_shap_values.csv"):
        patch_table.setdefault(row["image_id"], []).append(row)
    stored_faith = {(row["image_id"], float(row["fraction"])): row for row in read_csv(evaluation_dir / "faithfulness.csv")}
    checks["ten_images_with_stored_shap"] = len(images) == 10 and all(f"{item['image_id']}_patch_shap" in arrays for item in images)

    grids, orders, diagnostics = {}, {}, {}
    for item in images:
        image_id = item["image_id"]
        grid = np.asarray(arrays[f"{image_id}_patch_shap"], dtype=np.float64)
        rows = sorted(patch_table[image_id], key=lambda row: int(row["patch_index"]))
        order = rank_patches(grid)
        stored_abs_rank = [int(row["abs_rank"]) for row in rows]
        my_rank = [0] * grid.size
        for position, index in enumerate(order):
            my_rank[index] = position + 1
        selections_match = all(
            [int(v) for v in stored_faith[(image_id, fraction)]["selected_patches"].split()] == order[: fraction_count(fraction, grid.size)]
            for fraction in fractions
        )
        grids[image_id], orders[image_id] = grid, order
        diagnostics[image_id] = {
            "npz_equals_csv": float(np.max(np.abs(grid.reshape(-1) - np.asarray([float(row["shap_value"]) for row in rows])))) == 0.0,
            "ranking_equals_stored_abs_rank": my_rank == stored_abs_rank,
            "ranking_equals_stored_faithfulness_selection": selections_match,
            "all_patch_shap_nonnegative": bool(np.all(grid >= 0)),
        }
    for name in ("npz_equals_csv", "ranking_equals_stored_abs_rank", "ranking_equals_stored_faithfulness_selection"):
        checks[f"all_images_{name}"] = all(result[name] for result in diagnostics.values())
    log("stored SHAP maps and ranking: " + ", ".join(f"{k}={v}" for k, v in checks.items() if k.startswith("all_images")))

    pointing = find_localization_annotations(own, source_config)
    if pointing["available"]:
        raise RuntimeError(f"Possible localization annotations found {pointing}: implement the pointing game against them.")
    log(f"pointing game: the image dataset has no boxes or masks (columns {pointing['image_list_columns']}); using author-annotated boxes")
    author_boxes, box_info = load_author_boxes(own, [item["image_id"] for item in images])
    pointing["author_boxes"] = box_info
    for name in ("all_rows_reviewed", "every_image_reviewed", "one_target_per_image"):
        checks[f"pointing_boxes_{name}"] = box_info[name]
    tolerance_px = float(own["pointing_game"]["tolerance_px"])
    if not all(checks.values()):
        save_json(OUTPUT_DIR / "checks.json", {"gate_passed": False, "checks": checks, "images": diagnostics})
        raise RuntimeError("Pre-scoring gate failed; see outputs/checks.json.")

    # ---- frozen CLIP vision encoder, source input preparation and blur background
    import torch

    device = choose_device(str(source_config.get("device", "auto")))
    torch.manual_seed(int(source_config["seed"]))
    model, processor = load_model_and_processor(source_config["model_name"], device, local_files_only=True)
    checks["model_in_eval_mode"] = not model.training
    parameters_before = sum(float(parameter.detach().double().sum()) for parameter in model.parameters())
    scorer = ClipImageSimilarity(model, processor, device)
    batched = run_shap.BatchedScorer(scorer, int(source_config["batch_size"]))

    curves: dict[str, dict[str, np.ndarray]] = {}
    per_image, curve_rows, errors = [], [], {}
    crop_boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    points: dict[str, tuple[float, float]] = {}
    for item in images:
        image_id = item["image_id"]
        pil = load_rgb(str(item["path"]))
        model_input = prepare_model_input(pil, processor)
        blurred = box_blur(model_input, blur_kernel)
        scorer.set_reference(model_input)
        score = float(scorer(model_input[None])[0, 0])
        blurred_score = float(scorer(blurred[None])[0, 0])
        grid, order = grids[image_id], orders[image_id]
        deletion = batched(deletion_images(model_input, blurred, order, grid.shape, patch_size)).reshape(-1)
        insertion = batched(insertion_images(model_input, blurred, order, grid.shape, patch_size)).reshape(-1)
        stored = summaries[image_id]
        ks = [fraction_count(fraction, grid.size) for fraction in fractions]
        errors[image_id] = {
            "original_vs_stored": abs(score - float(stored["score"])),
            "fully_blurred_vs_stored": abs(blurred_score - float(stored["fully_blurred_score"])),
            "deletion_start_vs_original": abs(deletion[0] - score),
            "deletion_end_vs_fully_blurred": abs(deletion[-1] - blurred_score),
            "insertion_start_vs_fully_blurred": abs(insertion[0] - blurred_score),
            "insertion_end_vs_original": abs(insertion[-1] - score),
            "deletion_vs_stored_comprehensiveness": max(
                abs(deletion[k] - float(stored_faith[(image_id, f)]["comprehensiveness_score"])) for k, f in zip(ks, fractions, strict=True)
            ),
            "insertion_vs_stored_sufficiency": max(
                abs(insertion[k] - float(stored_faith[(image_id, f)]["sufficiency_score"])) for k, f in zip(ks, fractions, strict=True)
            ),
        }
        curves[image_id] = {"deletion": deletion, "insertion": insertion}
        aopc_value, drops = aopc(score, deletion, fractions)
        top = order[0]
        top_row, top_col = divmod(top, grid.shape[1])
        centre_x, centre_y = (top_col + 0.5) * patch_size, (top_row + 0.5) * patch_size
        crop = [float(v) for v in stored["crop_box_original"].split(";")]
        left, top_edge, right, bottom = crop
        side = grid.shape[0] * patch_size
        point = (left + centre_x * (right - left) / side, top_edge + centre_y * (bottom - top_edge) / side)
        evaluated = image_id in author_boxes
        visible = [box for box in (clip_box(item["box"], crop) for item in author_boxes.get(image_id, [])) if box is not None]
        crop_boxes[image_id], points[image_id] = visible, point
        if evaluated and not visible:
            checks[f"pointing_{image_id}_box_inside_clip_crop"] = False
        pointing_fields = {
            "pointing_game_target": author_boxes[image_id][0]["target"] if evaluated else "",
            "pointing_game_boxes": len(visible),
            "pointing_game_hit": int(pointing_hit(point, visible, tolerance_px)) if evaluated else "NA",
            "pointing_game_hit_no_tolerance": int(pointing_hit(point, visible, 0.0)) if evaluated else "NA",
            "pointing_game_status": "evaluated (author-annotated box)" if evaluated else "excluded by the author",
        }
        per_image.append(
            {
                "image_id": image_id,
                "original_score": score,
                "fully_blurred_score": blurred_score,
                "patches": grid.size,
                "deletion_auc": curve_auc(deletion),
                "insertion_auc": curve_auc(insertion),
                "aopc": aopc_value,
                **{f"aopc_drop_{fraction_label(f)}": drop for f, drop in zip(fractions, drops, strict=True)},
                **{f"patches_removed_{fraction_label(f)}": k for f, k in zip(fractions, ks, strict=True)},
                "max_shap_patch_index": top,
                "max_shap_patch_row": top_row,
                "max_shap_patch_col": top_col,
                "max_shap_patch_value": float(grid.reshape(-1)[top]),
                "max_shap_centre_x_model_input": centre_x,
                "max_shap_centre_y_model_input": centre_y,
                "max_shap_centre_x_original_image": point[0],
                "max_shap_centre_y_original_image": point[1],
                **pointing_fields,
            }
        )
        for step in range(grid.size + 1):
            changed = order[step - 1] if step else None
            curve_rows.append(
                {
                    "image_id": image_id,
                    "step": step,
                    "fraction_of_patches": step / grid.size,
                    "patch_changed_at_step": "" if changed is None else changed,
                    "patch_abs_shap": "" if changed is None else float(abs(grid.reshape(-1)[changed])),
                    "deletion_score": float(deletion[step]),
                    "insertion_score": float(insertion[step]),
                }
            )
        log(f"{image_id}: DAUC {per_image[-1]['deletion_auc']:.4f} IAUC {per_image[-1]['insertion_auc']:.4f} "
            f"AOPC {aopc_value:.4f} pointing hit {pointing_fields['pointing_game_hit']} | "
            f"max rescoring error {max(errors[image_id].values()):.1e}")

    parameters_after = sum(float(parameter.detach().double().sum()) for parameter in model.parameters())
    checks["model_parameters_unchanged"] = parameters_before == parameters_after
    for name in next(iter(errors.values())):
        checks[f"{name}_within_{tolerance:g}"] = max(result[name] for result in errors.values()) <= tolerance
    checks["source_experiment_unchanged"] = file_hashes(SOURCE_ROOT) == source_before
    checks["image_folder_unchanged"] = file_hashes(IMAGES_DIR) == images_before

    pointed = [row for row in per_image if row["pointing_game_hit"] != "NA"]
    summary = {
        "images": len(per_image),
        "deletion_auc": float(np.mean([row["deletion_auc"] for row in per_image])),
        "insertion_auc": float(np.mean([row["insertion_auc"] for row in per_image])),
        "aopc": float(np.mean([row["aopc"] for row in per_image])),
        "pointing_game": float(np.mean([row["pointing_game_hit"] for row in pointed])),
        "pointing_game_hits": int(sum(row["pointing_game_hit"] for row in pointed)),
        "pointing_game_images": len(pointed),
        "pointing_game_no_tolerance": float(np.mean([row["pointing_game_hit_no_tolerance"] for row in pointed])),
        "pointing_game_tolerance_px": tolerance_px,
        "fully_blurred_mean": float(np.mean([row["fully_blurred_score"] for row in per_image])),
        "fully_blurred_min": float(np.min([row["fully_blurred_score"] for row in per_image])),
        "fully_blurred_max": float(np.max([row["fully_blurred_score"] for row in per_image])),
    }
    gate = all(checks.values())
    report = {
        "gate_passed": gate,
        "checks": checks,
        "summary": summary,
        "max_abs_errors": {name: max(result[name] for result in errors.values()) for name in next(iter(errors.values()))},
        "per_image_errors": errors,
        "per_image_stored_checks": diagnostics,
        "pointing_game": pointing,
        "source_run": str(values_dir),
        "model_evaluations": scorer.evaluated_images,
        "device": str(device),
        "versions": run_shap.package_versions(),
        "runtime_seconds": time.perf_counter() - started,
        "completed_at_utc": utc_now(),
    }
    save_json(OUTPUT_DIR / "checks.json", report)
    if not gate:
        raise RuntimeError(f"Gate failed ({[k for k, v in checks.items() if not v]}); metric files were not written.")

    write_csv(OUTPUT_DIR / "per_image_metrics.csv", per_image)
    write_csv(OUTPUT_DIR / "curve_points.csv", curve_rows)
    write_tables(summary, len(per_image))
    from curve_plots import save_mean_curves, save_per_image_curves, save_pointing_game

    save_mean_curves(curves, summary, PLOTS_DIR / "deletion_insertion_curves_mean")
    save_per_image_curves(curves, {row["image_id"]: row for row in per_image}, PLOTS_DIR / "deletion_insertion_curves_per_image")
    save_pointing_game(
        {item["image_id"]: item["path"] for item in images},
        {row["image_id"]: [float(v) for v in summaries[row["image_id"]]["crop_box_original"].split(";")] for row in per_image},
        crop_boxes, points, {row["image_id"]: row for row in per_image}, tolerance_px, PLOTS_DIR / "pointing_game",
    )
    log(f"Deletion AUC {summary['deletion_auc']:.4f} | Insertion AUC {summary['insertion_auc']:.4f} | AOPC {summary['aopc']:.4f} | "
        f"Pointing game {summary['pointing_game']:.4f} ({summary['pointing_game_hits']}/{summary['pointing_game_images']})")
    log(f"all {len(checks)} checks passed; outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
