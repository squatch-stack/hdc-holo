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


def test_whitened_correlation_finds_the_translated_copy_and_ignores_envelope():
    rng = np.random.default_rng(11)
    scene = place.synthetic_scene(rng)
    freqs = sample_frequencies(1024, 3, 1 / 0.05, rng)
    grid = place.translation_grid(9, 0.25)
    fp = place.fingerprint(scene, freqs, 0.05)
    shift = np.array([0.1, -0.05, 0.05], np.float32)
    moved = place.fingerprint(SplatScene(scene.mu + shift, scene.cov, scene.amp),
                              freqs, 0.05)
    score, t_hat = place.correlate(fp, moved, freqs, grid, whiten=1.0)
    assert abs(place.correlate(fp, fp, freqs, grid, whiten=1.0)[0] - 1) < 1e-4
    assert score > 0.5 and np.abs(t_hat - shift).max() <= 0.025
    # An envelope-only twin (same magnitudes, random phases) scores like noise
    # under whitening even though its raw correlation shares the blob.
    twin = place.phase_surrogate(fp, rng)
    assert place.correlate(fp, twin, freqs, grid, whiten=1.0)[0] < 0.2
    with pytest.raises(ValueError):
        place.correlate(fp, fp, freqs, grid, whiten=2.0)


def test_tile_lattice_covers_the_cube_with_overlap():
    lo = np.array([-1, 2, 3])
    corners = place.tile_lattice(lo, 2.3, 1, 0.5)
    np.testing.assert_allclose(corners[0], lo)
    np.testing.assert_allclose(corners[-1] + 1, lo + 2.3)
    for axis in range(3):
        starts = np.unique(np.array(corners)[:, axis])
        assert np.all(np.diff(starts) <= 0.5 + 1e-12)
    points = np.random.default_rng(3).uniform(lo, lo + 2.3, (100, 3))
    assert all(any(np.all((p >= c) & (p <= c + 1)) for c in corners) for p in points)
    assert len(place.tile_lattice(lo, 0.5, 1, 0)) == 1
    for overlap in (-0.1, 1, np.nan):
        with pytest.raises(ValueError):
            place.tile_lattice(lo, 2, 1, overlap)


def test_tiling_is_deterministic_and_drops_low_mass_tiles(tmp_path, monkeypatch):
    path = tmp_path / "tiles.spz"
    pos = np.array([[0.2, 0.2, 0.2], [0.4, 0.3, 0.3], [1.5, 0.3, 0.3]])
    rgba = np.ones((3, 4))
    rgba[-1, 3] = 0.15
    save_spz(path, pos, np.full((3, 3), 0.01), rgba,
             np.tile([1, 0, 0, 0], (3, 1)))
    monkeypatch.setattr(place, "crop_box", lambda *_a: (np.zeros(3), 2))
    a = place.tile_scenes(path, 1, 0, 0.01)
    b = place.tile_scenes(path, 1, 0, 0.01)
    assert len(a) == len(b) == 2
    for first, second in zip(a, b):
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1].mu, second[1].mu)
        assert first[2] == second[2]
    retained = place.tile_scenes(path, 1, 0, 0.1)
    assert len(retained) == 1
    assert retained[0][2] > 0.9


def test_sigma_units_give_one_codebook_for_two_extents(monkeypatch):
    rng = np.random.default_rng(1)
    scene = place.synthetic_scene(rng)
    original_tiles = place.tile_scenes
    monkeypatch.setattr(place, "tile_scenes", lambda *_a: [(np.zeros(3), scene, 1)])
    calls = []
    original = place.sample_frequencies

    def sample(dim, dim_space, sigma_rho, rng):
        calls.append(sigma_rho)
        return original(dim, dim_space, sigma_rho, rng)

    monkeypatch.setattr(place, "sample_frequencies", sample)
    args = place._parser().parse_args([
        "/tmp/unused.json", "small.spz", "large.spz", "--tile", "6",
        "--sigma-units", "0.15", "--dim", "1024", "--grid", "2", "--limit",
        "0.01", "--yaws", "1", "--scrambles", "1", "--numpy"])
    result = place._run(args)
    np.testing.assert_allclose(calls, [40], rtol=1e-7)
    assert result["settings"]["sigma_box"] == pytest.approx(0.025)
    monkeypatch.setattr(place, "tile_scenes", original_tiles)
    # Actual loaders with different enclosing cubes still get identical tiles.
    monkeypatch.setattr(place, "crop_box", lambda p, *_a: (np.zeros(3), float(p)))
    data = (np.array([[0.2, 0.3, 0.4]]), np.full((1, 3), 0.01),
            np.ones((1, 4)), np.array([[1, 0, 0, 0]]))
    monkeypatch.setattr(place, "load_scene_file", lambda _p: data)
    small = place.tile_scenes("2", 1, 0, 0.01)
    large = place.tile_scenes("3", 1, 0, 0.01)
    freqs = original(1024, 3, 40, rng)
    np.testing.assert_array_equal(
        place.tile_fingerprints(small, freqs, 0.025, [0]),
        place.tile_fingerprints(large, freqs, 0.025, [0]))


def _small_tile_fixture():
    rng = np.random.default_rng(0)
    captures, truth = place._synthetic_tiles(rng, 1, 0, 0.01)
    pair = truth["pairs"][0]
    selected = [captures[0][pair["reference_tile"]],
                captures[1][pair["query_tile"] - len(captures[0])],
                captures[2][0]]
    freqs = sample_frequencies(1024, 3, 40, rng)
    angles = np.arange(4) * np.pi / 2
    fps = place.tile_fingerprints(selected, freqs, 0.025, angles)
    return fps, freqs, pair, rng


def test_translated_world_is_found_tile_by_tile():
    fps, freqs, pair, rng = _small_tile_fixture()
    grid = place.translation_grid(3, 0.06)
    result = place.tile_matrix(fps[:2], [0, 1], freqs, grid, 1, 0, rng)
    scores = result["tile_scores"]
    assert np.argmax(scores[:, 0]) == 1
    assert np.argmax(scores[:, 1]) == 0
    for query in (0, 1):
        unrelated, _, _ = place._best_yaw(fps[2, 0], fps[query], freqs, grid, 1)
        assert scores[1 - query, query] > unrelated
    assert scores[0, 1] > 0.99
    np.testing.assert_allclose(result["tile_offsets"][0, 1], pair["offset_box"],
                               atol=0.03, rtol=0)
    assert result["tile_yaws"][0, 1] == 3
    assert np.isneginf(np.diag(scores)).all()
    assert result["pairs_scored"] == result["pairs_possible"] == 2


def test_unrelated_world_scores_inside_the_null():
    fps, freqs, _, rng = _small_tile_fixture()
    result = place.tile_matrix(fps[:, :1], [0, 1, 2], freqs,
                               place.translation_grid(3, 0.06), 1, 0, rng)
    noise = result["noise"]
    unrelated = result["tile_scores"][[0, 1], 2].max()
    assert unrelated <= noise["mean"] + 3 * noise["sigma"]


def test_tile_prefilter_masks_unscored_pairs_and_preserves_own_capture_mask():
    fps, freqs, _, rng = _small_tile_fixture()
    result = place.tile_matrix(fps, [0, 1, 2], freqs,
                               np.zeros((1, 3), np.float32), 1, 1, rng)
    assert result["pairs_scored"] == 3
    assert result["pairs_possible"] == 6
    assert np.isfinite(result["tile_scores"]).sum() == 3
    assert np.isneginf(np.diag(result["tile_scores"])).all()
    assert result["noise"]["count"] == 3
    json.dumps(place._nullable(result["tile_scores"]), allow_nan=False)


# Frozen pre-lane computation: keep the legacy JSON contract independent.
def _legacy_place_run(args):
    rng = np.random.default_rng(args.seed)
    scenes, labels, groups = place._inputs(args, rng)
    freqs = place.sample_frequencies(args.dim, 3, 1 / args.sigma, rng)
    grid = place.translation_grid(args.grid, args.limit)
    angles = np.arange(args.yaws) * (2 * np.pi / args.yaws)
    fps = np.stack([place.fingerprint(s, freqs, args.sigma) for s in scenes])
    yaw_fps = np.stack([place._yaw_fingerprints(s, angles, freqs, args.sigma)
                        for s in scenes])
    scores, offsets = place.similarity_matrix(fps, freqs, grid, yaw_fps, args.whiten)
    power = np.stack([place.radial_power(fp, freqs) for fp in fps])
    power /= np.maximum(np.linalg.norm(power, axis=1, keepdims=True), 1e-30)
    noise = place._calibrate(scenes, fps, yaw_fps, angles, freqs, grid, args, rng)
    return {"labels": labels, "scores": scores.tolist(), "offsets": offsets.tolist(),
            "radial": (power @ power.T).tolist(), "noise": noise,
            "retrieval": place._retrieval(scores, groups, args.partner, noise["sigma"]),
            "settings": {"seed": args.seed, "dim": args.dim, "sigma": args.sigma,
                         "grid": args.grid, "limit": args.limit,
                         "yaws": args.yaws, "angles": angles.tolist(),
                         "lo": args.lo, "extent": args.extent,
                         "frame": args.frame, "null": args.null,
                         "whiten": args.whiten,
                         "numpy": args.numpy, "synthetic": not bool(args.paths)},
            "control_caveat": "Translation exact to rounding; finite-sample yaw "
                              "invariance approximate. Permuting equal splats "
                              "does not change a scene."}



def test_no_tile_flag_leaves_output_unchanged(tmp_path, monkeypatch):
    argv = [str(tmp_path / "legacy.json"), "--synthetic", "1", "--numpy",
            "--dim", "64", "--grid", "2", "--yaws", "1", "--scrambles", "2"]
    args = place._parser().parse_args(argv)
    golden = _legacy_place_run(args)

    def forbidden(*args, **kwargs):
        pytest.fail("legacy branch called tile code")

    for name in ("_run_tiles", "tile_scenes", "tile_fingerprints", "tile_matrix",
                 "_display_tiles", "_validate_tiles"):
        monkeypatch.setattr(place, name, forbidden)
    actual = place.main(argv)
    assert actual == golden
    assert json.loads((tmp_path / "legacy.json").read_text()) == golden
