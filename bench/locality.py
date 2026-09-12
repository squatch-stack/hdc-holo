"""Diagnose error concentration independently of the field representation.

Wilson's Creek has 91% of squared error in its worst 1% of pixels, whereas
cannon's side slice has 16%, despite both exceeding 150% relative L2
(results/gpu_sweep.md). Enrichment divides error share by image coverage:
raw share otherwise ranks band reach, as bench/find_bad_cell.py demonstrated.
Overlapping cells each receive full credit; their shares may exceed one.

Salt-and-pepper outliers can have high worst-1% share but enrichment near
one everywhere. Share alone cannot distinguish one blown cell from scattered
outliers; membership is required for a cause reading, and even then is evidence,
not proof. Shares depend on pixel count: we select max(1, int(n * frac)) pixels
and normalise coverage by n. Report n when comparing resolutions. The Gaussian
null is a population tail mass, not an exact finite-n order-statistic mean.

The abstract of arXiv:2510.08394, Spectral Prefiltering of Neural Fields
(https://arxiv.org/abs/2510.08394), checked online, describes filtering Fourier
features using the filter frequency response and learning filtered signals.
This is a neural-field analogue of evaluating against a matched filtered
reference; it does not establish the cause of this repository's residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sqrt
from statistics import NormalDist

import numpy as np


@dataclass
class LocalityReport:
    """Keep magnitude and concentration together so neither hides the other."""

    rel_l2: float
    norm_truth: float
    fill: float
    peak_ratio: float
    shares: dict[float, float]
    null_shares: dict[float, float]
    cells: list[dict] | None
    reading: str


def _vector(value):
    array = np.asarray(value)
    if array.ndim != 1 or not array.size:
        raise ValueError("expected a nonempty one-dimensional array")
    if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise ValueError("expected finite real values")
    return array


def _fraction(frac):
    if not 0 < frac <= 1:
        raise ValueError("fractions must be in (0, 1]")


def error_shares(resid, fracs=(0.01, 0.001)) -> dict[float, float]:
    """Match the sweep's top-k selection and float64 squared-error summation.

    Constant magnitudes give k/n, exactly frac when n*frac is integral.
    A zero residual gives zero shares rather than an undefined ratio.
    """
    sq = np.sort(_vector(resid).astype(np.float64) ** 2)[::-1]
    total = float(sq.sum()) or 1.0
    shares = {}
    for frac in fracs:
        _fraction(frac)
        k = max(1, int(len(sq) * frac))
        shares[frac] = float(sq[:k].sum()) / total
    return shares


def null_share(frac, n) -> float:
    """Return the chi-square(1) population upper-tail mass, about .0845 at 1%.

    For z = Phi^-1(1-frac/2), integration by parts gives frac + 2*z*phi(z).
    The brief's .055 sanity estimate understates this analytic value. n must
    be positive but does not change a population quantile. For rounded pixel
    selection, callers can supply max(1, int(n*frac))/n as frac; neither value
    is the exact finite-sample expected ratio of order statistics.
    """
    _fraction(frac)
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    z = -NormalDist().inv_cdf(frac / 2)
    return float(frac + 2 * z * exp(-z * z / 2) / sqrt(2 * pi))


def _cell_indices(membership, n):
    if isinstance(membership, np.ndarray) or (
        isinstance(membership, (list, tuple))
        and membership and np.isscalar(membership[0])
    ):
        labels = np.asarray(membership)
        if labels.shape != (n,) or labels.dtype.kind not in "iu":
            raise ValueError("labels must be one integer per pixel")
        return [(int(cell), np.flatnonzero(labels == cell))
                for cell in np.unique(labels)]
    cells = []
    for cell, members in enumerate(membership):
        indices = np.asarray(members)
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
            raise ValueError("cell indices must be one-dimensional integers")
        if np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("cell index outside the pixel array")
        cells.append((cell, np.unique(indices.astype(np.intp))))
    return cells


def enrichment(resid2, membership) -> list[dict]:
    """Credit each covering cell fully, then rank error per unit image area.

    Like _score_cells, omit empty cells and cells with no error. Duplicate
    indices within one cell count once, as they would in a boolean reach mask.
    Integer labels retain their IDs; index-array lists use their list positions.
    """
    resid2 = _vector(resid2)
    if np.any(resid2 < 0):
        raise ValueError("squared residuals must be nonnegative")
    total = float(resid2.sum())
    rows = []
    for cell, indices in _cell_indices(membership, len(resid2)):
        mass = float(resid2[indices].sum())
        if mass <= 0:
            continue
        share = mass / total
        px_frac = len(indices) / len(resid2)
        rows.append({"cell": cell, "share": share, "px_frac": px_frac,
                     "enrichment": share / px_frac})
    return sorted(rows, key=lambda row: -row["enrichment"])


def locality_report(truth, recon, membership=None, *, floor=0.01,
                    fracs=(0.01, 0.001), spike_share=0.5) -> LocalityReport:
    """Preserve sweep arithmetic, including float32 subtraction before squaring.

    Always include 1% for the reading, even with custom fractions. Zero truth
    gives relative L2 zero for exact reconstruction and infinity otherwise.
    Reading thresholds describe concentration, not a spatial cause.
    """
    truth, recon = _vector(truth), _vector(recon)
    if truth.shape != recon.shape:
        raise ValueError("truth and recon must cover the same pixels")
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("floor must be finite and nonnegative")
    _fraction(spike_share)
    resid = recon - truth
    # The same expression as sweep_scenes._slices, so the quotient is taken
    # in the same precision on every NumPy (NEP 50 keeps a float32 norm
    # divided by a Python float in float32; a float64 quotient differed in
    # the eighth digit on Linux CI).
    norm_truth = float(np.linalg.norm(truth))
    norm_resid = np.linalg.norm(resid)
    rel_l2 = float(norm_resid / norm_truth) if norm_truth else (
        0.0 if norm_resid == 0 else float("inf"))
    shares = error_shares(resid, dict.fromkeys((*fracs, 0.01)))
    nulls = {frac: null_share(frac, len(truth)) for frac in shares}
    reading = "localised"
    if shares[0.01] >= spike_share:
        reading = "spike"
    elif shares[0.01] <= 2 * nulls[0.01]:
        reading = "distributed"
    return LocalityReport(
        rel_l2, norm_truth,
        float(np.mean(truth > floor * truth.max())),
        float(recon.max() / max(truth.max(), 1e-9)), shares, nulls,
        enrichment(resid.astype(np.float64) ** 2, membership)
        if membership is not None else None,
        reading,
    )


def sweep_row_fields(report, key) -> dict:
    """Emit the five diagnostic fields with the sweep's exact names and rounding.

    Reports must include both default fractions to export a complete sweep row.
    """
    return {
        f"norm_{key}": round(report.norm_truth, 3),
        f"fill_{key}": round(report.fill, 4),
        f"err_share_1pct_{key}": round(report.shares[0.01], 4),
        f"err_share_01pct_{key}": round(report.shares[0.001], 4),
        f"peak_ratio_{key}": round(report.peak_ratio, 2),
    }
