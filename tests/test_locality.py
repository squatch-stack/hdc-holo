"""Separate spike concentration from Gaussian and constant-magnitude nulls."""

import json
from math import exp, pi, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from holo.locality import (
    LocalityReport,
    enrichment,
    error_shares,
    locality_report,
    null_share,
    sweep_row_fields,
)


def test_one_spike_on_a_faithful_field_reports_over_90pct_in_worst_1pct():
    rng = np.random.default_rng(7)
    truth = np.ones(10_000)
    recon = truth + rng.normal(0, 0.001, truth.size)
    recon[42] += 10
    report = locality_report(truth, recon)
    assert report.shares[0.01] > 0.9
    assert report.reading == "spike"
    assert report.cells is None


def test_constant_magnitude_noise_reports_share_equal_to_fraction():
    resid = np.random.default_rng(8).choice([-1.0, 1.0], 10_000)
    assert error_shares(resid) == {0.01: 0.01, 0.001: 0.001}
    assert locality_report(np.ones(resid.size), 1 + resid).reading == "distributed"


def _gaussian_share_se(frac, n):
    # Influence function for the empirical top-fraction ratio:
    # (X-q)+ - share*X, centred, with X distributed as chi-square(1).
    z = -NormalDist().inv_cdf(frac / 2)
    q = z * z
    share = null_share(frac, n)
    tail_second = 3 * frac + 2 * (z**3 + 3 * z) * exp(-q / 2) / sqrt(2 * pi)
    positive_second = tail_second - 2 * q * share + q * q * frac
    cross = tail_second - q * share
    variance = positive_second - 2 * share * cross + 3 * share**2 - (q * frac)**2
    return sqrt(variance / n)


def test_gaussian_noise_matches_the_chi2_null_share():
    n = 50_000
    shares = error_shares(np.random.default_rng(19).normal(size=n))
    assert null_share(0.01, n) == pytest.approx(0.084491659, abs=1e-9)
    for frac, observed in shares.items():
        assert abs(observed - null_share(frac, n)) < 4 * _gaussian_share_se(frac, n)


def test_enrichment_is_unity_under_uniform_error_and_high_for_the_spiked_cell():
    labels = np.repeat([4, 9, 12, 20], 100)
    rows = enrichment(np.ones(400), labels)
    assert [row["cell"] for row in rows] == [4, 9, 12, 20]
    assert all(row["enrichment"] == 1 for row in rows)
    resid2 = np.ones(400)
    resid2[110] = 100_000
    rows = enrichment(resid2, labels)
    assert rows[0]["cell"] == 9
    assert rows[0]["enrichment"] > 3.9


def test_overlapping_membership_uses_shared_credit():
    rows = enrichment(np.array([1., 9., 0., 0.]),
                      [np.array([0, 1, 1]), np.array([1, 2]), np.array([])])
    assert rows == [
        {"cell": 0, "share": 1., "px_frac": .5, "enrichment": 2.},
        {"cell": 1, "share": .9, "px_frac": .5, "enrichment": 1.8},
    ]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sweep_row_fields_match_the_inline_formula(dtype):
    rng = np.random.default_rng(31)
    truth = rng.uniform(0, 2, 5017).astype(dtype)
    recon = truth + rng.normal(0, .1, truth.size).astype(dtype)
    key = "top_down"
    nt = float(np.linalg.norm(truth))
    resid = recon - truth
    row = {}
    row[f"norm_{key}"] = round(nt, 3)
    row[f"fill_{key}"] = round(float(np.mean(truth > 0.01 * truth.max())), 4)
    sq = np.sort(resid.astype(np.float64) ** 2)[::-1]
    tot = float(sq.sum()) or 1.0
    for frac, tag in ((0.01, "1pct"), (0.001, "01pct")):
        k = max(1, int(len(sq) * frac))
        row[f"err_share_{tag}_{key}"] = round(float(sq[:k].sum()) / tot, 4)
    row[f"peak_ratio_{key}"] = round(float(recon.max() / max(truth.max(), 1e-9)), 2)
    report = locality_report(truth, recon)
    assert sweep_row_fields(report, key) == row
    assert report.rel_l2 == float(np.linalg.norm(resid) / nt)


def test_zero_error_zero_truth_and_custom_fractions():
    report = locality_report(np.zeros(10), np.zeros(10), [np.arange(10)])
    assert report.rel_l2 == report.fill == report.peak_ratio == 0
    assert report.shares == {0.01: 0., 0.001: 0.}
    assert report.cells == []
    assert report.reading == "distributed"
    report = locality_report(np.zeros(10), np.ones(10), fracs=(.5,))
    assert report.rel_l2 == float("inf")
    assert set(report.shares) == {.01, .5}
    assert error_shares(np.ones(3)) == {.01: 1 / 3, .001: 1 / 3}
    assert null_share(1, 3) == 1


def test_localised_reading_and_scattered_spikes():
    resid = np.zeros(10_000)
    resid[:400] = 1
    assert locality_report(np.ones(10_000), 1 + resid).reading == "localised"
    resid[:] = 0
    resid[::100] = np.random.default_rng(4).choice([-10., 10.], 100)
    report = locality_report(np.ones(10_000), 1 + resid, np.repeat(np.arange(10), 1000))
    assert report.reading == "spike"
    assert all(row["enrichment"] == 1 for row in report.cells)


@pytest.mark.parametrize("truth,recon", [([], []), ([1], [1, 2]),
                                        ([[1]], [[1]]), ([np.nan], [0])])
def test_invalid_fields_are_rejected(truth, recon):
    with pytest.raises(ValueError):
        locality_report(truth, recon)


def test_invalid_fractions_and_membership_are_rejected():
    for frac in (0, -1, 1.1, np.nan):
        with pytest.raises(ValueError):
            error_shares([1.], (frac,))
        with pytest.raises(ValueError):
            null_share(frac, 10)
    with pytest.raises(ValueError):
        null_share(.01, 0)
    for membership in ([0], [0., 1.], [np.array([2])], [np.array([-1])]):
        with pytest.raises(ValueError):
            enrichment(np.ones(2), membership)
    with pytest.raises(ValueError):
        enrichment([-1., 1.], [0, 1])


def test_sweep_row_fields_round_trip_recorded_gpu_row():
    """Pin serialized diagnostic keys and digits without capture files.

    The JSON stores rounded diagnostics, not probe arrays; arithmetic precision
    is covered separately by test_sweep_row_fields_match_the_inline_formula.
    """
    path = Path(__file__).resolve().parents[1] / "results" / "gpu_sweep.json"
    row = json.loads(path.read_text())[0]
    for key in ("top_down", "side"):
        report = LocalityReport(
            rel_l2=row["err"][key],
            norm_truth=row[f"norm_{key}"],
            fill=row[f"fill_{key}"],
            peak_ratio=row[f"peak_ratio_{key}"],
            shares={0.01: row[f"err_share_1pct_{key}"],
                    0.001: row[f"err_share_01pct_{key}"]},
            null_shares={}, cells=None, reading="localised",
        )
        actual = sweep_row_fields(report, key)
        expected = {f"{prefix}_{key}": row[f"{prefix}_{key}"] for prefix in (
            "norm", "fill", "err_share_1pct", "err_share_01pct", "peak_ratio")}
        assert json.dumps(actual, sort_keys=True) == json.dumps(
            expected, sort_keys=True)
