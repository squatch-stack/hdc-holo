"""Seeded descriptor checks, including counterexamples to exact yaw invariance."""

import json

import numpy as np
import pytest

from bench import shape_descriptor as shape
from holo.spectral import SplatScene, sample_frequencies


def _fixture():
    return shape.normalise_scene(shape.synthetic_shapes(
        np.random.default_rng(4), kinds=("rod",), per_kind=1)[0][1])


def test_descriptors_are_invariant_to_translation_yaw_and_scale():
    raw = shape.synthetic_shapes(np.random.default_rng(4), per_kind=1)[1][1]
    moved = SplatScene(raw.mu * 2.3 + [4, -2, 1], raw.cov * 2.3**2, raw.amp)
    a, b = shape.normalise_scene(raw), shape.normalise_scene(moved)
    np.testing.assert_allclose(a.mu, b.mu, atol=1e-7)
    np.testing.assert_allclose(a.cov, b.cov, rtol=1e-6)
    freqs = sample_frequencies(512, 3, 40, np.random.default_rng(3))
    for fn in (shape.radial_profile,):
        np.testing.assert_allclose(fn(a), fn(b), atol=1e-7)
    np.testing.assert_allclose(shape.spectral_radial(a, freqs, 0.025),
                               shape.spectral_radial(b, freqs, 0.025),
                               rtol=2e-5, atol=1e-5)
    for other in (b, shape.yaw_scene(a, 0.61)):
        np.testing.assert_allclose(shape.radial_profile(a),
                                   shape.radial_profile(other), atol=1e-7)
        np.testing.assert_allclose(
            shape.d2_histogram(a, np.random.default_rng(8), 5000),
            shape.d2_histogram(other, np.random.default_rng(8), 5000), atol=1e-7)
    # Exact yaw lattice match and inverse, with normalization already applied.
    yawed = shape.yaw_scene(b, np.pi / 2)
    candidates = [shape.fingerprint(shape.yaw_scene(yawed, t), freqs, 0.025)
                  for t in np.arange(4) * np.pi / 2]
    score = shape.whitened_similarity(shape.fingerprint(a, freqs, 0.025),
                                      candidates, freqs, np.zeros((1, 3)))
    assert score == pytest.approx(1, abs=1e-2)


def test_exact_yaw_invariance_has_two_counterexamples():
    a = _fixture()
    freqs = sample_frequencies(512, 3, 40, np.random.default_rng(3))
    b = shape.yaw_scene(a, 0.61)
    # Finite iid frequency bins are not an analytic azimuthal average.
    pa = shape.spectral_radial(a, freqs, 0.025)
    pb = shape.spectral_radial(b, freqs, 0.025)
    assert shape.similarity(pa, pb, "spectral") < 0.99
    # Recomputing an axis-aligned cube after yaw also changes radial and D2.
    reframed = shape.normalise_scene(b)
    assert not np.allclose(shape.radial_profile(a), shape.radial_profile(reframed))
    da = shape.d2_histogram(a, np.random.default_rng(8), 5000)
    db = shape.d2_histogram(reframed, np.random.default_rng(8), 5000)
    assert not np.allclose(da, db)


def test_d2_histogram_is_normalised_and_deterministic():
    scene = _fixture()
    a = shape.d2_histogram(scene, np.random.default_rng(10))
    b = shape.d2_histogram(scene, np.random.default_rng(10))
    assert a.shape == (64,)
    np.testing.assert_array_equal(a, b)
    assert a.sum() == pytest.approx(1, abs=1e-12)
    # Two weighted atoms: replacement sampling includes self pairs with mass
    # 0.9**2 + 0.1**2; uniform sampling would incorrectly give 0.5.
    atoms = SplatScene(np.array([[0, 0, 0], [1, 0, 0]]),
                      np.zeros((2, 3, 3)), np.array([[0.9], [0.1]]))
    h = shape.d2_histogram(atoms, np.random.default_rng(9), 20000)
    assert h[0] == pytest.approx(0.82, abs=0.01)


def test_synthetic_classes_separate(tmp_path, capsys):
    output = tmp_path / "scores.json"
    result = shape.main([str(output), "--synthetic", "--dim", "128",
                         "--yaws", "4", "--grid", "1", "--pairs", "5000",
                         "--per-kind", "3"])
    assert json.loads(output.read_text()) == result
    assert "The hypothesis under test" in capsys.readouterr().out
    # This smaller CPU fixture also exposes failures; do not tune the seed.
    expected = {"radial": 7 / 9, "spectral": 1.0, "whitened": 1 / 9, "d2": 1.0}
    for kind in shape.KINDS:
        assert np.shape(result["matrices"][kind]) == (9, 9)
        assert result["reports"][kind]["loo_accuracy"] == pytest.approx(expected[kind])


def test_class_report_shapes():
    sims = np.array([[1, 0.8, 0.2], [0.9, 1, 0.3], [0.1, 0.4, 1]])
    report = shape.class_report(sims, ["a", "a", "b"])
    assert report["within_mean"] == pytest.approx(0.85)
    assert report["between_mean"] == pytest.approx(0.25)
    assert report["loo_accuracy"] == pytest.approx(2 / 3)
    assert report["neighbors"] == [1, 0, 1]
    assert report["within_pairs"] == 2
    assert report["between_pairs"] == 4
    assert shape.class_report(sims, ["a"] * 3)["between_mean"] is None
    assert shape.class_report(sims, ["a", "b", "c"])["within_mean"] is None
    with pytest.raises(ValueError):
        shape.class_report(np.eye(2), ["a"])


def test_core_box_and_capture_cli(tmp_path, monkeypatch, capsys):
    raw = shape.synthetic_shapes(np.random.default_rng(12), per_kind=1)[0][1]
    rgba = np.column_stack([np.ones((len(raw.mu), 3)), raw.amp[:, 0]])
    scale = np.full_like(raw.mu, 0.008)
    quat = np.tile([1, 0, 0, 0], (len(raw.mu), 1))

    def load(path):
        return raw.mu, scale, rgba, quat

    monkeypatch.setattr(shape, "load_scene_file", load)
    monkeypatch.setattr("bench.place_recognition.load_scene_file", load)
    lo, extent, radius = shape.core_box("fixture.spz")
    assert extent == pytest.approx(6 * radius)
    actual = shape.build_scene_fixed("fixture.spz", lo, extent)[0]
    np.testing.assert_allclose(actual.mu, shape.normalise_scene(raw).mu, atol=1e-7)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"a.spz": "sphere", "b.spz": "sphere"}))
    result = shape.main([str(tmp_path / "real.json"), "a.spz", "b.spz",
                         "--labels", str(labels), "--dim", "32", "--grid", "1",
                         "--yaws", "1", "--pairs", "100"])
    assert result["hypothesis"] == {"a.spz": "sphere", "b.spz": "sphere"}
    assert "The hypothesis under test" in capsys.readouterr().out
    with pytest.raises(ValueError):
        shape.core_box("fixture.spz", margin=0)
