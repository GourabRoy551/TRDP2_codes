"""Completeness and internal consistency of the saved outputs."""

from __future__ import annotations

import numpy as np
import pytest

from source_link import OUTPUT_DIR
from vision_metrics import curve_auc


def test_all_gates_passed(checks):
    assert checks["gate_passed"] is True
    assert all(checks["checks"].values())
    assert checks["checks"]["source_experiment_unchanged"] and checks["checks"]["image_folder_unchanged"]
    assert checks["pointing_game"]["available"] is False  # the image dataset itself has no boxes
    assert checks["pointing_game"]["author_boxes"]["all_rows_reviewed"] is True


def test_ten_images_with_fifty_curve_points_each(per_image, curve_points):
    ids = [f"I{i}" for i in range(1, 11)]
    assert [row["image_id"] for row in per_image] == ids
    for image_id in ids:
        steps = [int(row["step"]) for row in curve_points if row["image_id"] == image_id]
        assert steps == list(range(50))


def test_aucs_recompute_from_curve_points(per_image, curve_points):
    for row in per_image:
        points = [p for p in curve_points if p["image_id"] == row["image_id"]]
        deletion = [float(p["deletion_score"]) for p in points]
        insertion = [float(p["insertion_score"]) for p in points]
        assert float(row["deletion_auc"]) == pytest.approx(curve_auc(deletion), abs=1e-12)
        assert float(row["insertion_auc"]) == pytest.approx(curve_auc(insertion), abs=1e-12)
        drops = [float(row["original_score"]) - deletion[int(row[f"patches_removed_{label}"])] for label in ("f10", "f20", "f30", "f50")]
        assert float(row["aopc"]) == pytest.approx(np.mean(drops), abs=1e-12)


def test_curves_run_between_original_and_fully_blurred(per_image, curve_points):
    for row in per_image:
        points = [p for p in curve_points if p["image_id"] == row["image_id"]]
        original, blurred = float(row["original_score"]), float(row["fully_blurred_score"])
        assert float(points[0]["deletion_score"]) == pytest.approx(original, abs=1e-5)
        assert float(points[-1]["deletion_score"]) == pytest.approx(blurred, abs=1e-5)
        assert float(points[0]["insertion_score"]) == pytest.approx(blurred, abs=1e-5)
        assert float(points[-1]["insertion_score"]) == pytest.approx(original, abs=1e-5)
        changed = [int(p["patch_changed_at_step"]) for p in points[1:]]
        assert sorted(changed) == list(range(49))
        abs_values = [float(p["patch_abs_shap"]) for p in points[1:]]
        assert abs_values == sorted(abs_values, reverse=True)


def test_summary_table_is_mean_of_per_image(per_image, summary_table):
    assert list(summary_table) == ["Deletion AUC ↓", "Insertion AUC ↑", "AOPC ↑", "Pointing/Localization Accuracy ↑"]
    for metric, column in (("Deletion AUC ↓", "deletion_auc"), ("Insertion AUC ↑", "insertion_auc"), ("AOPC ↑", "aopc")):
        assert float(summary_table[metric]) == pytest.approx(np.mean([float(row[column]) for row in per_image]), abs=1e-12)
    pointed = [int(row["pointing_game_hit"]) for row in per_image if row["pointing_game_hit"] != "NA"]
    assert float(summary_table["Pointing/Localization Accuracy ↑"]) == pytest.approx(np.mean(pointed), abs=1e-12)


def test_pointing_game_recomputes_from_approved_boxes(per_image, checks):
    import csv

    from source_link import ROOT
    from vision_metrics import clip_box, pointing_hit

    with (ROOT / "annotations" / "pointing_game_boxes.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["status"] for row in rows} <= {"approved", "excluded"}
    summaries = {}
    with (ROOT.parent / "clip_vision_shap" / "outputs" / "values" / "clip_vision_i1_i10" / "image_summary.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            summaries[row["image_id"]] = [float(v) for v in row["crop_box_original"].split(";")]
    tolerance = float(checks["summary"]["pointing_game_tolerance_px"])
    for row in per_image:
        boxes = [tuple(float(box[k]) for k in ("x0", "y0", "x1", "y1")) for box in rows
                 if box["image_id"] == row["image_id"] and box["status"] == "approved"]
        if not boxes:
            assert row["pointing_game_hit"] == "NA"
            continue
        visible = [b for b in (clip_box(box, summaries[row["image_id"]]) for box in boxes) if b is not None]
        point = (float(row["max_shap_centre_x_original_image"]), float(row["max_shap_centre_y_original_image"]))
        assert int(row["pointing_game_hit"]) == int(pointing_hit(point, visible, tolerance))
        assert int(row["pointing_game_hit_no_tolerance"]) == int(pointing_hit(point, visible, 0.0))
        if int(row["pointing_game_hit_no_tolerance"]):
            assert int(row["pointing_game_hit"]) == 1


def test_curve_figures_exist():
    for stem in ("deletion_insertion_curves_mean", "deletion_insertion_curves_per_image", "pointing_game"):
        for suffix in (".png", ".pdf"):
            assert (OUTPUT_DIR / "plots" / f"{stem}{suffix}").stat().st_size > 10_000
