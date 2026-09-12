"""Seeded controls for spectral correlation and honest resonator failures."""

import json

import numpy as np
import pytest

from bench import resonator_capture
from bench.place_recognition import correlate, translation_grid
from bench.resonator_capture import (
    _jittered_instances,
    _metrics,
    _query,
    composite,
    one_shot_baseline,
    position_codebooks,
    prototype,
    prototype_search,
    synthetic_fixture,
    what_is_where,
)
from holo.fhrr import FHRR
from holo.spectral import translate_bundle


@pytest.fixture(scope="module")
def fixture():
    return synthetic_fixture(dim=4096, seed=0)


def _assert_recovered(found, truth, step):
    for item in truth:
        matches = [r for r in found if r["k_hat"] == item["k"] and r["converged"]]
        assert matches, f"identity {item['k']} not recovered: {found}"
        error = min(np.linalg.norm(np.array(r["t_hat"]) - item["t"]) for r in matches)
        assert error <= step / 2, f"identity {item['k']}: {error / step} fine steps"


@pytest.fixture(scope="module")
def correlation_fixture():
    # Preserve the diagnosed geometry; a deterministic frequency subset keeps
    # CPU tests below three seconds. The report runs all 4096 frequencies.
    full = synthetic_fixture(dim=4096, seed=0, extras=4)
    return {
        **full,
        "S": full["S"][::2],
        "identity": full["identity"][:, ::2],
        "freqs": full["freqs"][::2],
    }


@pytest.mark.parametrize("band_floor", [0.0, 0.3])
def test_whitened_baseline_recovers_three_objects_at_grid_32(
    correlation_fixture, band_floor
):
    f = correlation_fixture
    rows = one_shot_baseline(
        f["S"],
        f["identity"],
        f["freqs"],
        translation_grid(32, 0.5) + 0.5,
        band_floor=band_floor,
        sigma=f["sigma"],
    )
    selected = sorted(rows, key=lambda r: -r["score"])[:3]
    assert {r["k_hat"] for r in selected} == {0, 1, 2}
    for row in selected:
        error = np.linalg.norm(np.array(row["t_hat"]) - f["truth"][row["k_hat"]]["t"])
        assert error <= f["sigma"] / 2
        assert row["what_at"] == row["k_hat"]


def test_grid_16_misses_the_whitened_peak_and_the_tool_says_so(correlation_fixture):
    f = correlation_fixture
    one = composite(f["identity"][:1], [f["truth"][0]["t"]], f["freqs"])
    rows = [
        one_shot_baseline(
            one, f["identity"][:1], f["freqs"], translation_grid(size, 0.5) + 0.5
        )[0]
        for size in (16, 32)
    ]
    assert rows[0]["score"] < 0.1 and rows[0]["weak_peak"]
    assert rows[1]["score"] >= 0.9 and not rows[1]["weak_peak"]


def test_prototype_of_two_instances_beats_the_foreign_prototype(correlation_fixture):
    f = correlation_fixture
    rng = np.random.default_rng(200)
    training = _jittered_instances(f["crops"][0], f["freqs"], f["sigma"], 0.05, rng)
    foreign = _jittered_instances(f["crops"][3], f["freqs"], f["sigma"], 0.05, rng)
    rows = prototype_search(
        f["S"],
        training,
        f["freqs"],
        translation_grid(32, 0.5) + 0.5,
        f["sigma"],
        foreign,
    )
    assert rows["prototype"]["score"] >= 2 * rows["foreign"]["score"]
    error = np.linalg.norm(np.array(rows["prototype"]["t_hat"]) - f["truth"][0]["t"])
    assert error <= f["sigma"] / 2
    assert rows["prototype"]["error_box"] is None  # Search has no held-out truth.


def test_resonator_reports_nonconvergence_honestly(fixture, monkeypatch, tmp_path):
    def synthetic(*args, **kwargs):
        # Use the specified widest re-blur diagnostic; native refined fixed
        # points vary even though stage 1 consistently fails at this load.
        return {
            "rows": [
                _query(
                    fixture, phase=True, values=8, baseline_size=8, coarse_sigma=1 / 8
                )
            ]
        }

    monkeypatch.setattr(resonator_capture, "run_synthetic", synthetic)
    output = tmp_path / "failure.json"
    resonator_capture.main([str(output), "--synthetic", "--dim", "4096"])
    row = json.loads(output.read_text())["rows"][0]
    assert any(not r["converged"] for r in row["found"])
    assert row["converged_rate"] < 1
    assert row["tolerance_rule"] == "max(fine_step_box, sigma_box / 2)"


def test_identity_codewords_are_orthogonal_only_phase_only(correlation_fixture):
    codes = correlation_fixture["identity"]
    raw = codes / np.linalg.norm(codes, axis=1)[:, None]
    phase = FHRR.normalize(codes) / np.sqrt(codes.shape[1])
    off = ~np.eye(len(codes), dtype=bool)
    assert np.abs(raw @ raw.conj().T)[off].max() >= 0.4
    assert np.abs(phase @ phase.conj().T)[off].max() <= 0.2


def _single_object():
    one = synthetic_fixture(objects=1)
    shift = [0.3125, 0.5625, 0.6875]
    one["truth"] = [{"k": 0, "t": shift}]
    one["S"] = composite(one["identity"], [shift], one["freqs"])
    return one


def test_baseline_and_resonator_agree_on_one_object():
    one = _single_object()
    result = _query(one, values=17, baseline_size=17)
    _assert_recovered(result["found"], one["truth"], 1 / 128)
    assert result["baseline_joint_rate"] == 1


@pytest.mark.parametrize("seed", range(30))
def test_seeded_comparison_trial(seed):
    result = _query(
        synthetic_fixture(4096, seed, 2, extras=4), values=8, baseline_size=8
    )
    # Keep all independent seeds; old constants described the old blur/metric.
    for method in ("resonator", "baseline"):
        identity = result[method + "_identity_rate"]
        position = result[method + "_position_rate"]
        joint = result[method + "_joint_rate"]
        assert 0 <= joint <= min(identity, position) <= 1
    failures = [i for i, row in enumerate(result["found"]) if not row["converged"]]
    assert not failures or failures == [len(result["found"]) - 1]


def test_composite_keeps_both_shifted_components(fixture):
    truth = fixture["truth"][:2]
    signal = composite(
        fixture["identity"][:2], [t["t"] for t in truth], fixture["freqs"]
    )
    expected = sum(
        translate_bundle(code[None], fixture["freqs"], item["t"])[0]
        for code, item in zip(fixture["identity"][:2], truth)
    )
    np.testing.assert_allclose(signal, expected, rtol=2e-6, atol=1e-9)


def test_sign_convention_matches_translate_bundle(fixture):
    code = fixture["identity"][:1]
    shift = np.array([0.3125, 0.5625, 0.6875])
    shifted = translate_bundle(code, fixture["freqs"], shift)[0]
    found = what_is_where(shifted, code, fixture["freqs"], 1, values=17)
    _assert_recovered(found, [{"k": 0, "t": shift}], 1 / 128)
    books = position_codebooks(fixture["freqs"], shift[:, None])
    np.testing.assert_allclose(
        books[0][0] * books[1][0] * books[2][0] * code[0], shifted, rtol=2e-5, atol=1e-9
    )


def test_capacity_cliff_is_reported_not_hidden():
    fixture = synthetic_fixture(1024, 0, 6)
    result = _query(fixture, values=16, baseline_size=8)
    assert result["resonator_joint_rate"] < 0.5
    assert any(not row["coarse_converged"] for row in result["found"])


def test_phase_only_identity_changes_scores_not_positions():
    one = _single_object()
    raw = _query(one, values=17, baseline_size=8)
    phase = _query(one, phase=True, values=17, baseline_size=8)
    for result in (raw, phase):
        _assert_recovered(result["found"], one["truth"], 1 / 128)
    np.testing.assert_allclose(
        raw["found"][0]["t_hat"], phase["found"][0]["t_hat"], atol=1 / 112
    )
    assert not np.isclose(raw["found"][0]["score"], phase["found"][0]["score"])


def test_prototype_and_composite_follow_phasor_algebra(fixture):
    codes = fixture["identity"]
    summed = codes[:2].sum(axis=0)
    np.testing.assert_allclose(
        prototype(codes[:2]), summed / np.abs(summed), rtol=2e-6, atol=2e-6
    )
    original = codes.copy()
    result = composite(codes, np.zeros((3, 3)), fixture["freqs"])
    np.testing.assert_allclose(result, codes.sum(axis=0), rtol=2e-6, atol=1e-9)
    np.testing.assert_array_equal(codes, original)


def test_baseline_reverse_query_and_positive_bounds(fixture):
    code = fixture["identity"][:1]
    shift = np.array([0.2, 0.4, 0.6], np.float32)
    signal = translate_bundle(code, fixture["freqs"], shift)[0]
    baseline = one_shot_baseline(signal, code, fixture["freqs"], shift[None])
    np.testing.assert_allclose(baseline[0]["t_hat"], shift, atol=1e-7)
    assert baseline[0]["what_at"] == 0
    assert baseline[0]["score"] > 0.999


def test_stage_loads_and_scene_units(fixture):
    result = what_is_where(
        fixture["S"], fixture["identity"], fixture["freqs"], 1, values=16, extent=40
    )[0]
    assert result["stage_loads"] == {
        "coarse": 0.375,
        "fine_per_cell": 3.0,
        "fine_total": 12.0,
    }
    np.testing.assert_allclose(
        result["t_hat_scene"], np.array(result["t_hat"]) * 40, atol=1e-6
    )


def test_empty_energy_and_invalid_grid_rejected(fixture):
    with pytest.raises(ValueError):
        what_is_where(fixture["S"], fixture["identity"] * 0, fixture["freqs"], 1)
    with pytest.raises(ValueError):
        what_is_where(fixture["S"], fixture["identity"], fixture["freqs"], 1, top_r=0)


def test_capture_cli_uses_shared_frames_and_writes_queries(
    fixture, monkeypatch, tmp_path
):
    def fixed(path, lo, extent):
        np.testing.assert_allclose(lo, [0, 0, 0], atol=1e-7)
        assert extent == 40
        scene = fixture["parent"] if "parent" in path else fixture["crops"][0]
        return scene, None, None

    def box(path):
        return np.zeros(3), 40.0

    monkeypatch.setattr(resonator_capture, "build_scene_fixed", fixed)
    monkeypatch.setattr(resonator_capture, "crop_box", box)
    output = tmp_path / "queries.json"
    resonator_capture.main(
        [
            str(output),
            "parent.spz",
            "--crop",
            "crop.spz",
            "--dim",
            "4096",
            "--sigma-units",
            "1",
            "--grid",
            "8",
            "--values",
            "4",
            "--coarse",
            "4",
            "--top-r",
            "2",
            "--distractor",
            "other-parent.spz",
            "--candidate",
            "candidate-parent.spz",
            "candidate-crop.spz",
            "--instance",
            "source-parent.spz",
            "source-crop.spz",
            "--instance",
            "second-parent.spz",
            "second-crop.spz",
        ]
    )
    result = json.loads(output.read_text())
    assert [r["distractors"] for r in result["rows"]] == [0, 1, 2, 4, 8]
    assert result["prototype"]["prototype"]
    assert result["composite"]
    assert result["settings"]["sigma_units"] == 1
    item = result["rows"][0]["found"][0]
    np.testing.assert_allclose(item["error_scene"], item["error_box"] * 40, atol=1e-6)


@pytest.mark.parametrize("whiten", [0.0, 0.5, 1.0])
def test_batched_baseline_matches_existing_correlate(fixture, whiten):
    grid = translation_grid(6, 0.5) + 0.5
    codes = fixture["identity"][:2]
    rows = one_shot_baseline(fixture["S"], codes, fixture["freqs"], grid, whiten)
    for code, row in zip(codes, rows):
        score, point = correlate(code, fixture["S"], fixture["freqs"], grid, whiten)
        np.testing.assert_allclose(row["score"], score, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(row["t_hat"], point, rtol=0, atol=1e-7)


def test_metrics_separate_identity_position_and_convergence():
    truth = [{"k": 0, "t": [0.0, 0.0, 0.0]}]
    row = {
        "k_hat": 1,
        "t_hat": [0.007, 0.0, 0.0],
        "score": 1.0,
        "converged": True,
        "coarse_converged": False,
        "n_iters": 2,
        "stage_loads": {},
    }
    metrics = _metrics([row], [row], truth, 0.008, 0.012, coarse=16)
    assert metrics["resonator_identity_rate"] == 0
    assert metrics["resonator_position_rate"] == 1
    assert metrics["resonator_joint_rate"] == 0
    assert metrics["converged_rate"] == 1
    np.testing.assert_allclose(metrics["position_tolerance_box"], 0.008, atol=1e-9)
    finer = _metrics([row], [row], truth, 0.002, 0.012)
    np.testing.assert_allclose(finer["position_tolerance_box"], 0.006, atol=1e-9)
    assert finer["resonator_position_rate"] == 0


@pytest.mark.parametrize("coarse_sigma", [1 / 32, 1 / 16, 1 / 8])
def test_coarse_reblur_retains_failed_attempts(fixture, coarse_sigma):
    rows = what_is_where(
        fixture["S"],
        fixture["identity"],
        fixture["freqs"],
        3,
        values=8,
        coarse_sigma=coarse_sigma,
        sigma=fixture["sigma"],
    )
    assert rows
    assert all(r["coarse_sigma"] == coarse_sigma for r in rows)
    assert any(not r["coarse_converged"] for r in rows)


def test_band_mask_and_empty_signal(fixture):
    f = fixture
    grid = np.array([f["truth"][0]["t"]], np.float32)
    with pytest.raises(ValueError, match="sigma"):
        one_shot_baseline(f["S"], f["identity"], f["freqs"], grid, band_floor=0.3)
    rows = one_shot_baseline(
        f["S"] * 0, f["identity"], f["freqs"], grid, band_floor=0.3, sigma=f["sigma"]
    )
    assert all(r["score"] == 0 and r["weak_peak"] for r in rows)
    expected = np.exp(-0.5 * f["sigma"] ** 2 * np.sum(f["freqs"] ** 2, axis=1)) >= 0.3
    assert rows[0]["retained_frequencies"] == int(expected.sum())


def test_parent_frame_instances_cli(fixture, monkeypatch, tmp_path):
    def box(path):
        return np.zeros(3), 40.0

    def fixed(path, lo, extent):
        np.testing.assert_allclose(lo, np.zeros(3), atol=1e-9)
        assert extent == 40.0
        return (
            fixture["parent"] if path == "parent.spz" else fixture["crops"][0],
            None,
            None,
        )

    monkeypatch.setattr(resonator_capture, "crop_box", box)
    monkeypatch.setattr(resonator_capture, "build_scene_fixed", fixed)
    output = tmp_path / "prototype.json"
    resonator_capture.main(
        [
            str(output),
            "parent.spz",
            "--crop",
            "crop.spz",
            "--instances",
            "first.spz",
            "second.spz",
            "--frame-of",
            "parent",
            "--prototype",
            "--sigma-units",
            "0.5",
            "--dim",
            "128",
            "--grid",
            "4",
            "--values",
            "4",
            "--coarse",
            "4",
            "--top-r",
            "2",
            "--band-floor",
            "0.3",
            "--coarse-sigma",
            "0.0625",
        ]
    )
    result = json.loads(output.read_text())
    assert result["settings"]["baseline_size"] == 4
    assert result["settings"]["band_floor"] == 0.3
    assert result["prototype"]["prototype"]["error_box"] is not None
    assert result["prototype"]["single"]["error_box"] is not None
