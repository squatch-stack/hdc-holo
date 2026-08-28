"""Analytic projection through the WHOLE pipeline (issue #2).

`run_analytic_projection.py` measures one cell's reconstruction. This
encodes EVERY cell analytically, decodes the same evidence slices, and
compares against the same exact-mixture referee — the quantity every
other number in this repo is reported in. Measured:

    saguaro   forward 0.3501 / 0.2132  ->  0.2170 / 0.1367   +38% / +36%
    train     forward 0.9591 / 0.4948  ->  0.3765 / 0.1716   +61% / +65%

Biggest on the DENSE scene, which is the one more dimension could not
help (issue #3) and where orthogonal coupling bought 1.9%.

Three things make this affordable, and the third is a trap:

1. The windowed right-hand side IS `spectral_bundle` applied to a
   modified scene — covariance shrunk by the window, mean pulled toward
   the cell centre, amplitude scaled. Verified to 9e-7 against a direct
   per-splat loop, so the existing fast path does the work.
2. `G_c = D G0 D^H`, so ONE eigendecomposition serves a whole band:
   63 s at d=8192, amortised over thousands of cells.
3. TRUNCATION IS MANDATORY AT PRODUCTION d. The window Gram's condition
   number is 1.6e20 to 3.2e20 at d=8192 — past what float64 can invert
   — and solving at full rank returns garbage of order 1e5, not a
   degraded answer. Keeping 25% of the spectrum is safe across all four
   bands and is close to the ~3,300 space-bandwidth DOF a cell of this
   size actually supports. An earlier per-cell study at d=2048 found
   full rank stable and concluded no truncation was needed; that was an
   artifact of the smaller d.

Cost: encoding is ~15x slower than forward (770 s against 51 s on
train). Decode and storage are unchanged — the output is an ordinary
bundle.

Usage: python -m examples.run_projection_pipeline data/train.splat [keep_frac]
"""
import sys
import time

import numpy as np

from holo.capture import (
    BANDS,
    band_codebooks,
    build_scene,
    decode_slice,
    encode_bands,
    exact_slice,
    mass_mode,
    slice_grid,
)
from holo.spectral import SplatScene, spectral_bundle

CELL_CHUNK = 256          # cells per batched solve; ~0.13 GB peak


def eigen(G):
    ev, V = np.linalg.eigh(G)
    o = np.argsort(np.abs(ev))[::-1]
    return ev[o], V[:, o]

def window_bundle(scene, ids, centre, s, freqs):
    """RHS of the windowed projection for one cell, all channels."""
    cov = scene.cov[ids].astype(np.float64)
    mu = (scene.mu[ids] - centre).astype(np.float64)
    prec = np.linalg.inv(cov)
    joint = prec + np.eye(3) / s**2
    shrunk = np.linalg.inv(joint)
    pulled = np.einsum("nij,njk,nk->ni", shrunk, prec, mu)
    scale = np.exp(-0.5 * (np.einsum("ni,nij,nj->n", mu, prec, mu)
                           - np.einsum("ni,nij,nj->n", pulled, joint, pulled)))
    mod = SplatScene(mu=pulled.astype(np.float32),
                     cov=shrunk.astype(np.float32),
                     amp=(scene.amp[ids] * scale[:, None]).astype(np.float32))
    return spectral_bundle(mod, freqs)          # (C, d)

def band_operator(gram, keep_frac, tikhonov):
    """The per-band solve operator, and a label for it.

    TSVD needs the eigendecomposition (O(d^3), 106 s at d=8192 and 98%
    of the fixed cost). Tikhonov needs none: an explicit inverse is ~6x
    cheaper and leaves the per-cell cost a matvec either way.
    """
    if tikhonov is None:
        ev, vec = eigen(gram)
        keep = max(1, round(keep_frac * len(ev)))
        op = (vec[:, :keep] / ev[:keep][None, :]) @ vec[:, :keep].T
        return op, "keep=%d" % keep
    lam = tikhonov * float(np.abs(gram).max())
    op = np.linalg.inv(gram + lam * np.eye(gram.shape[0]))
    return op, "tikhonov lam=%.0e" % tikhonov


def main(path, keep_frac=1.0, tikhonov=None, also_shrink=False):
    t0 = time.time()
    scene, smax, box = build_scene(path, verbose=False)
    books = band_codebooks(np.random.default_rng(42))
    fwd_bundles, members = encode_bands(scene, smax, books, verbose=False)

    w = scene.amp[:, 0]
    slices = [("top-down", slice_grid((0, box[0]), (0, box[2]), "y",
                                      mass_mode(scene.mu[:, 1], w, box[1]))),
              ("side", slice_grid((0, box[2]), (0, box[1]), "x",
                                  mass_mode(scene.mu[:, 0], w, box[0])))]
    truth = {n: exact_slice(pts, scene, members) for n, (pts, _) in slices}

    def err(bundles):
        return [float(np.linalg.norm(decode_slice(pts, bundles, books)[:, 0]
                                     - truth[n][:, 0])
                      / np.linalg.norm(truth[n][:, 0]))
                for n, (pts, _) in slices]

    base = err(fwd_bundles)
    print("%s  |  forward encoding: %.4f / %.4f  (%.0fs)"
          % (path.split("/")[-1], base[0], base[1], time.time() - t0))

    ana = {}
    for name, _cap, cell in BANDS:
        if not fwd_bundles.get(name):
            ana[name] = {}
            continue
        freqs, _rho, weights = books[name]
        fd = freqs.astype(np.float64)
        half = cell / 2
        s = half / 2                              # the width that won per-cell
        # |w_j - w_k|^2 via one gemm instead of materialising a
        # (d, d, 3) array — 1.6 GB at d=8192, which was the wall.
        sq = (fd**2).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (fd @ fd.T)
        np.maximum(d2, 0.0, out=d2)
        # the Gaussian Gram is REAL symmetric: no complex storage, and
        # the real symmetric eigensolver rather than the Hermitian one
        G = (2*np.pi*s**2)**1.5 * np.exp(-0.5*(s**2)*d2)
        del d2
        M, how = band_operator(G, keep_frac, tikhonov)
        del G
        print("  %-7s %d cells, d=%d, %s  (%.0fs)"
              % (name, len(fwd_bundles[name]), freqs.shape[0], how,
                 time.time() - t0))
        # Solve the cells in CHUNKS rather than one at a time. Identical
        # arithmetic — the same M applied to the same right-hand sides —
        # but one BLAS-3 matmul instead of hundreds of BLAS-2 matvecs,
        # measured 7.1x on this shape. Peak extra memory at 256 cells is
        # ~0.13 GB, which is why it is chunked rather than done at once.
        out = {}
        keys = list(members[name].keys())
        for lo in range(0, len(keys), CELL_CHUNK):
            batch = keys[lo:lo + CELL_CHUNK]
            centres = [(np.array(k, dtype=np.float64) + 0.5) * cell
                       for k in batch]
            rhs = np.concatenate(
                [window_bundle(scene, members[name][k], c0, s, freqs)
                 for k, c0 in zip(batch, centres)], axis=0)      # (B*C, d)
            sol = (M @ rhs.astype(np.complex128).T).T            # (B*C, d)
            nch = rhs.shape[0] // len(batch)
            for i, (k, c0) in enumerate(zip(batch, centres)):
                c = sol[i * nch:(i + 1) * nch]
                # cell-local -> world phase, then pre-divide so
                # decode_slice's weight multiply cancels exactly
                c = c * np.exp(-1j * (fd @ c0))[None, :]
                out[k] = (c / weights[None, :]).astype(np.complex64)
        ana[name] = out
        del M
    a = err(ana)
    print("  analytic (window s=h/2): %.4f / %.4f" % (a[0], a[1]))
    if also_shrink:
        # Does shrinkage add anything to an ALREADY-SOLVED bundle?
        # Prediction: little or negative — the solve is L2-optimal on the
        # window and already regularised, so shrinkage moves away from
        # the optimum rather than removing crosstalk it left behind.
        from holo.denoise import percentile_threshold, shrink
        for pct in (10, 25):
            sh = {b: {k: shrink(v, percentile_threshold(v, pct))
                      for k, v in cells.items()} for b, cells in ana.items()}
            e = err(sh)
            print("    + shrink p%-2d          : %.4f / %.4f  (%+.1f%% / %+.1f%%)"
                  % (pct, e[0], e[1], 100*(a[0]-e[0])/a[0],
                     100*(a[1]-e[1])/a[1]))
    print("  change: top-down %+.1f%%   side %+.1f%%   (%.0fs total)"
          % (100*(base[0]-a[0])/base[0], 100*(base[1]-a[1])/base[1],
             time.time() - t0))

if __name__ == "__main__":
    argv = sys.argv[1:]
    tik = None
    if "--tikhonov" in argv:
        i = argv.index("--tikhonov")
        tik = float(argv[i + 1])
        del argv[i:i + 2]
    shr = "--shrink" in argv
    argv = [a for a in argv if not a.startswith("--")]
    main(argv[0], float(argv[1]) if len(argv) > 1 else 1.0, tik, shr)
