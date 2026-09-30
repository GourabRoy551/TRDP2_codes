"""Metric definitions and perturbation sequences on small synthetic inputs."""

from __future__ import annotations

import numpy as np
import pytest

from vision_metrics import aopc, curve_auc, deletion_images, fraction_count, insertion_images, rank_patches


def test_ranking_by_absolute_value_ties_by_index():
    grid = np.array([[0.1, -0.5], [0.5, 0.0]])
    assert rank_patches(grid) == [1, 2, 0, 3]


def test_fraction_counts_match_source_rule():
    assert [fraction_count(f, 49) for f in (0.1, 0.2, 0.3, 0.5)] == [5, 10, 15, 25]
    assert fraction_count(0.01, 49) == 1


def test_deletion_and_insertion_sequences_are_complements():
    image = np.ones((4, 4, 3), dtype=np.float32)
    blurred = np.zeros((4, 4, 3), dtype=np.float32)
    order = [3, 0, 2, 1]
    deletion = deletion_images(image, blurred, order, (2, 2), 2)
    insertion = insertion_images(image, blurred, order, (2, 2), 2)
    assert deletion.shape == insertion.shape == (5, 4, 4, 3)
    np.testing.assert_array_equal(deletion[0], image)
    np.testing.assert_array_equal(deletion[-1], blurred)
    np.testing.assert_array_equal(insertion[0], blurred)
    np.testing.assert_array_equal(insertion[-1], image)
    np.testing.assert_array_equal(deletion[1][2:, 2:], 0)  # patch 3 (bottom right) removed first
    np.testing.assert_array_equal(deletion[1][:2, :], 1)
    for k in range(5):
        np.testing.assert_array_equal(deletion[k] + insertion[k], image)


def test_curve_auc_trapezoid():
    assert curve_auc([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert curve_auc([1.0, 0.0]) == pytest.approx(0.5)
    assert curve_auc([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert curve_auc(np.linspace(1, 0.4, 50)) == pytest.approx(np.trapezoid(np.linspace(1, 0.4, 50), np.linspace(0, 1, 50)))


def test_pointing_hit_with_clipping_and_tolerance():
    from vision_metrics import clip_box, pointing_hit

    crop = (80.0, 0.0, 400.0, 320.0)
    assert clip_box((10, 10, 100, 100), crop) == (80.0, 10.0, 100, 100)
    assert clip_box((410, 10, 500, 100), crop) is None
    box = [(100.0, 100.0, 200.0, 200.0)]
    assert pointing_hit((150, 150), box, 0.0)
    assert not pointing_hit((210, 150), box, 0.0)
    assert pointing_hit((210, 150), box, 15.0)
    assert not pointing_hit((216, 150), box, 15.0)
    assert not pointing_hit((150, 150), [], 15.0)


def test_aopc_uses_fraction_cut_offs():
    scores = np.linspace(1.0, 0.51, 50)  # score after removing k patches = 1 - 0.01 k
    value, drops = aopc(1.0, scores, [0.1, 0.2, 0.3, 0.5])
    assert drops == pytest.approx([0.05, 0.10, 0.15, 0.25])
    assert value == pytest.approx(0.1375)
