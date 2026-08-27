"""Analytic L2 projection of capture cells (issue #2).

Forward encoding ACCUMULATES a cell's splats. This SOLVES for the
coefficients instead, with zero samples, by projecting the exact mixture
onto the codebook. `spectral_bundle` already computes the projection's
right-hand side — it is the mixture's Fourier transform — so the whole
difference from forward encoding is replacing the diagonal importance
weighting with a Gram solve.

Two objectives are compared, and which one wins depends entirely on
where the splats sit:

    box     minimise over the cell's box. The best method here when it
            is valid — 5.4x better than forward on interior splats —
            but its right-hand side is only correct for splats well
            INSIDE the cell. The exact box-restricted transform of an
            anisotropic Gaussian needs the complex error function, does
            not separate for non-diagonal covariance, and is not in
            numpy. On real cells, whose splats sit at the boundaries,
            that approximation costs more than the method wins.

    window  minimise under a Gaussian weight centred on the cell. Its
            right-hand side stays EXACT for any splat position, because
            a Gaussian window times a Gaussian splat is another
            Gaussian. Slightly worse than the box at its best, and far
            more robust: 2.5x better than forward on whole real cells,
            and stable at full rank, so it needs no truncation tuning.

`INTERIOR=1` restricts splats to the cell interior. That is the control
separating "the box objective is bad" from "the box right-hand side is
approximate here" — it is the latter, and the control is what shows it.

Both Grams factorise as `G_c = D G0 D^H` with `D` a unitary diagonal of
cell-centre phases, so ONE eigendecomposition per band serves every
cell. This driver relies on that: it decomposes once and reuses.

Usage:
    python -m examples.run_analytic_projection data/scan-tucson.spz
    INTERIOR=1 python -m examples.run_analytic_projection ...

Run it as a MODULE. From a worktree, running by path puts sys.path[0]
at examples/ and silently imports the shared checkout instead.
"""

import os
import sys

import numpy as np

from holo.capture import BANDS, S_LO, band_of, build_scene
from holo.spectral import (
    SplatScene,
    decode_field,
    eval_scene_exact,
    sample_frequencies,
    spectral_bundle,
)

INTERIOR_ONLY = os.environ.get("INTERIOR") == "1"
DIM = 2048                     # not the production 8192; all methods share it
KEEPS = (128, 256, 512, 1024, 1536, 2048)
CELLS = 6
MIN_SPLATS = 40


def cells_for_band(scene, smax, band_index, cell):
    """The most populated cells of one band, as (key, member ids)."""
    bands = band_of(smax)
    per_cell = {}
    for i in np.flatnonzero(bands == band_index):
        per_cell.setdefault(tuple((scene.mu[i] // cell).astype(int)), []).append(i)
    ranked = sorted(per_cell.items(), key=lambda kv: -len(kv[1]))
    return [(k, np.array(v)) for k, v in ranked
            if len(v) >= MIN_SPLATS][:CELLS]


def cell_scene(scene, ids, centre, half):
    """The cell's splats in cell-local coordinates, optionally trimmed
    to the interior so the box right-hand side is valid."""
    sub = SplatScene(mu=(scene.mu[ids] - centre).astype(np.float32),
                     cov=scene.cov[ids], amp=scene.amp[ids][:, :1])
    if not INTERIOR_ONLY:
        return sub
    inside = np.flatnonzero(np.abs(sub.mu).max(axis=1) < 0.55 * half)
    if inside.size < 20:
        return None
    return SplatScene(mu=sub.mu[inside], cov=sub.cov[inside],
                      amp=sub.amp[inside])


def eigen(gram):
    """One decomposition per Gram, reused for every cell and every
    truncation — the point of the G_c = D G0 D^H factorisation."""
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(np.abs(values))[::-1]
    return values[order], vectors[:, order]


def solve(eig, rhs, keep):
    values, vectors = eig
    basis = vectors[:, :keep]
    return basis @ ((basis.conj().T @ rhs) / values[:keep])


def box_gram(freqs, half):
    delta = freqs[:, None, :] - freqs[None, :, :]
    return np.prod(2 * half * np.sinc(delta * half / np.pi),
                   axis=2).astype(np.complex128)


def window_gram(freqs, sigma):
    delta = freqs[:, None, :] - freqs[None, :, :]
    return ((2 * np.pi * sigma ** 2) ** 1.5
            * np.exp(-0.5 * sigma ** 2
                     * (delta ** 2).sum(2))).astype(np.complex128)


def window_rhs(scene, freqs, sigma):
    """Fourier transform of (Gaussian window x mixture), window at the
    origin. Exact for anisotropic covariance and any splat position:
    the product of two Gaussians is a third, with shrunk covariance and
    a mean pulled toward the window's centre."""
    out = np.zeros(freqs.shape[0], np.complex128)
    eye = np.eye(3) / sigma ** 2
    for k in range(scene.n):
        prec = np.linalg.inv(scene.cov[k].astype(np.float64))
        joint = prec + eye
        shrunk = np.linalg.inv(joint)
        mu = scene.mu[k].astype(np.float64)
        pulled = shrunk @ (prec @ mu)
        scale = np.exp(-0.5 * (mu @ prec @ mu - pulled @ joint @ pulled))
        volume = (2 * np.pi) ** 1.5 * np.sqrt(np.linalg.det(shrunk))
        envelope = np.exp(-0.5 * np.einsum("di,ij,dj->d", freqs, shrunk, freqs))
        out += (float(scene.amp[k, 0]) * scale * volume) * envelope \
            * np.exp(-1j * (freqs @ pulled))
    return out


def measure_cell(scene, freqs, rho, half, grams, rng):
    """Relative error of every method on one cell."""
    span = 0.7 if INTERIOR_ONLY else 1.0
    points = (rng.uniform(-span, span, (2000, 3)) * half).astype(np.float32)
    truth = eval_scene_exact(scene, points)[:, 0]
    norm = np.linalg.norm(truth)
    if norm == 0:
        return None
    basis = np.exp(1j * (points.astype(np.float64) @ freqs.T))
    bundle = spectral_bundle(scene, freqs.astype(np.float32))
    out = {"forward": np.linalg.norm(
        decode_field(bundle, freqs.astype(np.float32), rho, points)[:, 0]
        - truth) / norm}
    for label, eig, rhs in grams(scene, bundle):
        for keep in KEEPS:
            est = np.real(basis @ solve(eig, rhs, keep))
            out["%s@%d" % (label, keep)] = np.linalg.norm(est - truth) / norm
    return out


def main(path):
    scene, smax, _box = build_scene(path, verbose=False)
    _name, cap, cell = BANDS[0]                       # xfine
    ncomp = 3 + max(2, round(np.log2(cap / S_LO)))
    rho = list(1.0 / np.geomspace(S_LO, cap, ncomp))
    freqs = sample_frequencies(DIM, 3, rho,
                               np.random.default_rng(7)).astype(np.float64)
    half = cell / 2
    eig_box = eigen(box_gram(freqs, half))
    eig_win = {s: eigen(window_gram(freqs, s)) for s in (half / 2, half)}

    def grams(sub, bundle):
        yield "box", eig_box, bundle[0].astype(np.complex128)
        for sigma, eig in eig_win.items():
            yield ("win%.2gh" % (sigma / half), eig,
                   window_rhs(sub, freqs, sigma))

    rng = np.random.default_rng(0)
    totals = {}
    picked = cells_for_band(scene, smax, 0, cell)
    print("%s — xfine band, %d cells, d=%d%s"
          % (os.path.basename(path), len(picked), DIM,
             "  [splats restricted to cell interior]" if INTERIOR_ONLY else ""))
    for key, ids in picked:
        sub = cell_scene(scene, ids, (np.array(key) + 0.5) * cell, half)
        if sub is None:
            continue
        rec = measure_cell(sub, freqs, rho, half, grams, rng)
        if rec:
            for name, value in rec.items():
                totals.setdefault(name, []).append(value)
    median = {k: float(np.median(v)) for k, v in totals.items()}
    print("\n  median relative error over cells (lower is better)")
    print("  %-10s %s" % ("method", "  ".join("%9d" % k for k in KEEPS)))
    print("  %-10s %9.4f  (no solve)" % ("forward", median["forward"]))
    for label in ("box", "win0.5h", "win1h"):
        row = ["%9.4f" % median["%s@%d" % (label, k)] for k in KEEPS]
        print("  %-10s %s" % (label, "  ".join(row)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(os.path.dirname(os.path.dirname(
             os.path.abspath(__file__))), "data", "scan-tucson.spz"))
