"""Seeded checks of shift sign, search normalization, and control limitations."""

import json

import numpy as np
import pytest

from bench import place_recognition as place
from holo import accel
from holo.capture import build_scene, save_spz
from holo.spectral import SplatScene, sample_frequencies, translate_bundle


@pytest.fixture
def setup_scene(monkeypatch):
    monkeypatch.setattr(accel, "active", lambda: False)
    rng = np.random.default_rng(4)
    scene = place.synthetic_scene(rng)
    freqs = sample_frequencies(4096, 3, 40, rng)
    return scene, freqs, rng


def test_translated_copy_is_found_with_its_offset(setup_scene):
    scene, freqs, _ = setup_scene
    fp = place.fingerprint(scene, freqs, 0.025)
    t = np.array([0.047, -0.032, 0.021], np.float32)
    shifted = translate_bundle(fp[None, :], freqs, t)[0]
    score, offset = place.correlate(fp, shifted, freqs, place.translation_grid(9, 0.12))
    assert score >= 0.9
    assert np.linalg.norm(offset - t) <= 0.025 / 2


def test_unrelated_and_scrambled_scenes_score_below_a_third_of_the_match(setup_scene):
    scene, freqs, rng = setup_scene
    fp = place.fingerprint(scene, freqs, 0.025)
    grid = place.translation_grid(9, 0.12)
    match, _ = place.correlate(fp, fp, freqs, grid)
    for other in (place.synthetic_scene(rng), place.scramble(scene, rng)):
        score, _ = place.correlate(fp, place.fingerprint(other, freqs, 0.025),
                                   freqs, grid)
        assert score < match / 3


def test_magnitude_descriptor_is_exactly_translation_invariant(setup_scene):
    scene, freqs, _ = setup_scene
    shifted = SplatScene(scene.mu + np.array([0.08, -0.04, 0.03], np.float32),
                          scene.cov, scene.amp)
    a = place.radial_power(place.fingerprint(scene, freqs, 0.025), freqs)
    b = place.radial_power(place.fingerprint(shifted, freqs, 0.025), freqs)
    np.testing.assert_allclose(a, b, rtol=2e-5, atol=float(a.max()) * 2e-6)


def test_yawed_copy_is_found_at_the_right_angle(setup_scene):
    scene, freqs, _ = setup_scene
    angles = np.arange(8) * (2 * np.pi / 8)
    k = 3
    query = place.fingerprint(place.yaw_scene(scene, angles[k]), freqs, 0.025)
    grid = np.zeros((1, 3), np.float32)
    scores = [place.correlate(place.fingerprint(place.yaw_scene(scene, angle),
                                                freqs, 0.025), query, freqs, grid)[0]
              for angle in angles]
    assert np.argmax(scores) == k
    assert scores[k] > 0.999
    assert scores[k] - max(scores[:k] + scores[k + 1:]) > 0.4


def test_matrix_is_deterministic_for_a_seed(setup_scene):
    _, freqs, _ = setup_scene
    grid = place.translation_grid(3, 0.06)

    def run():
        rng = np.random.default_rng(7)
        scene = place.synthetic_scene(rng)
        fps = [place.fingerprint(s, freqs, 0.025)
               for s in (scene, place.scramble(scene, rng))]
        return place.similarity_matrix(fps, freqs, grid)

    a, b = run(), run()
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_allclose(np.diag(a[0]), 1, atol=1e-6)


def test_fixed_box_matches_capture_steps(tmp_path):
    rng = np.random.default_rng(2)
    pos = rng.uniform(-2, 2, (40, 3))
    scale = rng.uniform(0.001, 0.2, (40, 3))
    rgba = rng.uniform(0, 1, (40, 4))
    quat = rng.normal(size=(40, 4))
    path = tmp_path / "scene.spz"
    save_spz(path, pos, scale, rgba, quat)
    # Reconstruct the exact crop used by the ordinary loader, then compare
    # every output. This catches alpha ordering, quaternion and RGBA mistakes.
    pos, _, rgba, _ = place.load_scene_file(path)
    keep = rgba[:, 3] >= place.ALPHA_MIN
    pos, alpha = pos[keep], rgba[keep, 3]
    from holo.capture import weighted_quantile

    center = np.array([weighted_quantile(pos[:, i], alpha, 0.5) for i in range(3)])
    radius = weighted_quantile(np.abs(pos - center).max(axis=1), alpha, 0.75)
    fixed = place.build_scene_fixed(path, center - 1.2 * radius, 2.4 * radius)
    ordinary = build_scene(path, verbose=False)
    for name in ("mu", "cov", "amp"):
        np.testing.assert_allclose(getattr(fixed[0], name), getattr(ordinary[0], name))
    np.testing.assert_allclose(fixed[1], ordinary[1])
    np.testing.assert_array_equal(fixed[2], ordinary[2])
    with pytest.raises(ValueError, match="no splats"):
        place.build_scene_fixed(path, [100, 100, 100], 1)


def test_uniform_splats_are_unchanged_by_scramble(setup_scene):
    scene, freqs, rng = setup_scene
    scene.cov[:] = np.eye(3, dtype=np.float32) * 0.01**2
    scene.amp[:] = 1
    a = place.fingerprint(scene, freqs, 0.025)
    b = place.fingerprint(place.scramble(scene, rng), freqs, 0.025)
    np.testing.assert_allclose(a, b, atol=float(np.abs(a).max()) * 1e-6)


def test_yaw_matrix_offsets_and_dispatch(setup_scene, monkeypatch):
    scene, freqs, _ = setup_scene
    fp = place.fingerprint(scene, freqs, 0.025)
    t = np.array([0.03, 0, 0], np.float32)
    moved = translate_bundle(fp[None, :], freqs, t)[0]
    calls = []

    def decode(bundle, freqs, weights, points):
        calls.append(len(points))
        return (np.exp(1j * (points @ freqs.T)) @ (bundle * weights).T).real

    monkeypatch.setattr(accel, "active", lambda: True)
    monkeypatch.setattr(accel, "decode", decode)
    scores, offsets = place.similarity_matrix(
        [fp, moved], freqs, place.translation_grid(3, 0.03),
        np.array([[fp], [moved]]))
    np.testing.assert_allclose(scores, 1, atol=1e-6)
    np.testing.assert_allclose(offsets[0, 1], t, atol=1e-6)
    np.testing.assert_allclose(offsets[1, 0], -t, atol=1e-6)
    assert len(calls) == 16  # coarse + three refinements for every ordered pair


def test_cli_synthetic_calibration(tmp_path, capsys):
    path = tmp_path / "place.json"
    result = place.main([str(path), "--synthetic", "3", "--numpy", "--dim", "64",
                         "--grid", "2", "--yaws", "1", "--scrambles", "2"])
    assert json.loads(path.read_text()) == result
    assert np.shape(result["scores"]) == (12, 12)
    assert np.shape(result["offsets"]) == (12, 12, 3)
    assert result["noise"]["count"] == 2
    samples = result["noise"]["samples"]
    assert result["noise"]["sigma"] == np.std(samples)
    assert result["noise"]["p95"] == np.percentile(samples, 95)
    assert len(result["retrieval"]) == 9
    assert "Radial control" in capsys.readouterr().out


def test_retrieval_excludes_self_and_other_known_positives():
    scores = np.array([[1, 0.9, 0.8, 0.2], [0.9, 1, 0.7, 0.3],
                       [0.8, 0.7, 1, 0.1], [0.2, 0.3, 0.1, 1]])
    reports = place._retrieval(scores, [0, 0, 0, 1], [], 0.1)
    assert len(reports) == 3
    assert all(r["rank1_hit"] for r in reports)
    assert reports[0]["separation_sigma"] == pytest.approx(7)
    explicit = place._retrieval(scores, list(range(4)), [(0, 1)], 0)
    assert len(explicit) == 2
    assert explicit[0]["separation_sigma"] is None


def test_phase_surrogate_keeps_radial_power_and_breaks_correlation():
    rng = np.random.default_rng(5)
    scene = place.synthetic_scene(rng)
    freqs = sample_frequencies(512, 3, 1 / 0.05, rng)
    grid = place.translation_grid(5)
    fp = place.fingerprint(scene, freqs, 0.05)
    surrogate = place.phase_surrogate(fp, rng)
    np.testing.assert_allclose(np.abs(surrogate), np.abs(fp), rtol=1e-5)
    np.testing.assert_allclose(place.radial_power(surrogate, freqs),
                               place.radial_power(fp, freqs), rtol=1e-4)
    match = place.correlate(fp, fp, freqs, grid)[0]
    assert place.correlate(fp, surrogate, freqs, grid)[0] < match / 3


def test_frame_from_reference_matches_build_scene(tmp_path):
    rng = np.random.default_rng(9)
    n = 60
    pos = rng.uniform(-3, 3, (n, 3)).astype(np.float32)
    scale = rng.uniform(0.01, 0.05, (n, 3)).astype(np.float32)
    rgba = rng.uniform(0.2, 1, (n, 4)).astype(np.float32)
    quat = rng.normal(size=(n, 4)).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    path = tmp_path / "ref.spz"
    save_spz(path, pos, scale, rgba, quat)
    lo, extent = place.crop_box(path)
    framed, _, _ = place.build_scene_fixed(path, lo, extent)
    stock, _, _ = build_scene(path, verbose=False)
    np.testing.assert_allclose(framed.mu, stock.mu, atol=1e-6)
    np.testing.assert_allclose(framed.cov, stock.cov, rtol=1e-5)
    np.testing.assert_allclose(framed.amp, stock.amp)
