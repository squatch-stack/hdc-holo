"""Seeded CPU checks; focused localization is distinct from the full ladder."""

import numpy as np
import pytest

from bench import change_detection as change
from holo.capture import render_mip, save_spz
from holo.spectral import SplatScene, sample_frequencies


def _easy():
    points = np.array([[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]], np.float32)
    before = SplatScene(points, np.tile(np.eye(3, dtype=np.float32) * 0.04**2,
                                      (2, 1, 1)), np.ones((2, 1), np.float32))
    after = change._take(before, np.array([1]))
    bbox = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]], np.float32)
    freqs = sample_frequencies(2048, 3, 10, np.random.default_rng(0))
    return before, after, bbox, freqs, change._grid(13)


def test_removed_cluster_is_localised():
    before, after, bbox, freqs, grid = _easy()
    # The cluster's annotation spans 0.4 units, so sigma_rec = cluster / 4.
    row, _ = change._evaluate(before, after, bbox, freqs, grid, (0.1, 0.01, 1))
    assert row["iou_bundle"] >= 0.5


def test_drift_below_half_sigma_gives_little_false_change():
    before, _, _, freqs, grid = _easy()
    clean = render_mip(before, 0.1)
    observed = change.drift(clean, np.random.default_rng(1), 0.02, 0.2, 0.1)
    spectrum = change._encode(observed, freqs)
    threshold = change.null_threshold(spectrum, clean, freqs, grid,
                                      np.random.default_rng(100), 0.02, 0.2, 0.1)
    other = change.drift(clean, np.random.default_rng(200), 0.02, 0.2, 0.1)
    field = change.change_map(change.difference_bundle(
        spectrum, change._encode(other, freqs)), freqs, grid)
    # Held-out unchanged capture: false surviving mass / original field mass.
    reference = change.change_map(spectrum, freqs, grid)
    assert field[field > threshold].sum() / reference.sum() < 0.1


def test_add_and_remove_have_opposite_sign():
    before, after, _, freqs, _ = _easy()
    obj = change._take(before, np.array([0]))
    restored, bbox = change.insert(after, obj, [0, 0, 0])
    a, b = change._encode(before, freqs), change._encode(after, freqs)
    added = change.difference_bundle(b, change._encode(restored, freqs))
    removed = change.difference_bundle(a, b)
    np.testing.assert_allclose(added, -removed, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(bbox, np.tile(obj.mu, (2, 1)), atol=1e-7)


def _save(path, points):
    points = np.asarray(points, dtype=np.float32)
    count = len(points)
    save_spz(path, points, np.full((count, 3), 0.02, np.float32),
             np.ones((count, 4), np.float32),
             np.tile(np.array([0, 0, 0, 1], np.float32), (count, 1)))


def test_non_subset_crop_is_refused(tmp_path):
    parent, crop = tmp_path / "parent.spz", tmp_path / "crop.spz"
    _save(parent, [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
    # Outside the chosen frame: verification must precede cube filtering.
    _save(crop, [[0.2, 0.3, 0.4], [5, 5, 5]])
    with pytest.raises(ValueError, match="not a subset"):
        change.remove_subset(parent, crop, np.zeros(3), 1, 1e-5)


def test_exact_subset_uses_parent_frame(tmp_path):
    parent, crop = tmp_path / "parent.spz", tmp_path / "crop.spz"
    _save(parent, [[2, 3, 4], [6, 7, 8], [3, 4, 5]])
    _save(crop, [[2, 3, 4], [3, 4, 5]])
    after, bbox, overlap = change.remove_subset(parent, crop, np.ones(3), 10,
                                               1e-5)
    assert overlap == 1
    np.testing.assert_allclose(after.mu, [[0.5, 0.6, 0.7]], atol=3e-5)
    np.testing.assert_allclose(bbox, [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                               atol=3e-5)


def test_baseline_and_bundle_agree_on_the_easy_case():
    before, after, _, freqs, _ = _easy()
    # A coarse Cartesian grid contains both isolated primitive centers. Both methods
    # use a held-out drift null on these same points and the same RNG draws.
    grid = np.stack(np.meshgrid(*([np.array([0.3, 0.7], np.float32)] * 3),
                                indexing="ij"), -1)
    bbox = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]], np.float32)
    row, _ = change._evaluate(before, after, bbox, freqs, grid, (0.1, 0.0, 1))
    assert row["iou_bundle"] == row["iou_baseline"] == 1.0


def test_output_is_deterministic_for_a_seed():
    scenes = change.synthetic_fixture(3)
    settings = (8.0, 128, 3, 5, (0.25,), (0.05,), None)
    assert change._ladder(scenes, *settings) == change._ladder(scenes, *settings)


def test_split_preserves_spectrum():
    before, _, _, freqs, _ = _easy()
    split = change.drift(before, np.random.default_rng(0), 0, 0, 1)
    assert split.n == before.n * 2
    np.testing.assert_allclose(change._encode(split, freqs),
                               change._encode(before, freqs), rtol=2e-6, atol=1e-9)


def test_distance_bins_match_brute_force():
    rng = np.random.default_rng(4)
    points, query = rng.normal(size=(40, 3)), rng.normal(size=(30, 3))
    distances, indices = change._nearest(points, query, 0.7)
    expected = np.linalg.norm(query[:, None] - points, axis=-1).min(axis=1)
    np.testing.assert_allclose(distances, np.minimum(expected, 0.7), atol=1e-12)
    assert np.array_equal(indices >= 0, expected < 0.7)


def test_real_cli_removal_and_insertion(tmp_path):
    parent, crop = tmp_path / "parent.spz", tmp_path / "crop.spz"
    _save(parent, [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8], [0.4, 0.5, 0.6]])
    _save(crop, [[0.2, 0.3, 0.4], [0.4, 0.5, 0.6]])
    common = [str(tmp_path / "result.json"), str(parent), "--dim", "32",
              "--grid", "5", "--drift", "0", "--sigma-units", "0.1"]
    removal = change.main([*common, "--remove", str(crop)])
    assert removal["overlap"] == 1.0
    insertion = change.main([*common, "--insert", str(crop),
                             "--at", "0.5", "0.5", "0.5"])
    assert insertion["has_truth"]
    plain = change.main([*common, str(parent)])
    assert plain["ladder"][0]["iou_bundle"] is None
