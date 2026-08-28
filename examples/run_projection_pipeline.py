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

SWEEP IN ONE PROCESS, NOT ONE PROCESS PER SETTING. The Gram depends
only on (codebook, cell size, window width) — not on the regulariser
and not on the scene — so every setting in a sweep can share it, and
every TSVD truncation can share one eigendecomposition. Running the
settings as concurrent processes instead pays N times for the same
537 MB matrix AND N times for the same O(d^3), which is how this
pipeline OOM-killed the machine twice.

Usage:
    python -m examples.run_projection_pipeline data/train.splat [keep_frac]
    python -m examples.run_projection_pipeline data/train.splat \
        --sweep tikhonov=1e-6,1e-3,1e-1 --sweep keep=0.25
"""
import sys
import time
from collections import namedtuple

import numpy as np

from holo import runlog
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

#: Bytes a batched solve may hold in its right-hand side and solution.
#: Cells per chunk falls out of this and the actual (channels, d) shape,
#: so a wider capture takes smaller chunks instead of more memory.
CHUNK_BUDGET = 0.25 * (1 << 30)


def cell_chunk(channels, dim):
    """How many cells fit CHUNK_BUDGET. rhs and sol are both
    (chunk*channels, dim) complex128 — 16 bytes each, twice over."""
    per_cell = 2 * channels * dim * 16
    return max(1, int(CHUNK_BUDGET // per_cell))


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

def build_gram(fd, s):
    """The band's Gaussian-window Gram, in ONE d x d buffer.

    G_jk = (2 pi s^2)^{3/2} exp(-s^2 |w_j - w_k|^2 / 2), real symmetric.
    The readable form holds three 537 MB arrays at once at d=8192 and
    allocates six in all; this holds two and allocates two, by doing
    every step after the two products in place.

    BIT-IDENTICAL, and that is a requirement rather than a bonus. The
    tempting version folds the gemm into one buffer — G = fd @ fd.T,
    then *= -2, += sq[:,None], += sq[None,:] — which reassociates
    (a + b) - 2c into (-2c + a) + b and moves the Gram by one ulp. That
    is 5e-15 relative on G, and harmless-looking, but this solve is
    ill-conditioned by construction (1.6e20 at d=8192, which is why
    truncation is mandatory) and the truncated pseudo-inverse amplified
    that ulp to 2.8e-8 on the operator at d=1024 alone. Identical
    arithmetic is what lets every downstream number stand unre-derived.
    """
    sq = (fd ** 2).sum(1)
    G = sq[:, None] + sq[None, :]         # buffer 1
    prod = fd @ fd.T                      # buffer 2
    prod *= 2.0                           # exact: a power of two
    G -= prod                             # (a + b) - 2c, in that order
    del prod
    np.maximum(G, 0.0, out=G)             # |w_j - w_k|^2, clipped
    G *= -0.5 * s ** 2
    np.exp(G, out=G)
    G *= (2 * np.pi * s ** 2) ** 1.5
    return G


class BandSolver:
    """One band's Gram, reused across every setting in a sweep.

    The Gram depends on (codebook, cell size, window width) alone — not
    on the regulariser, not on the scene — so a sweep builds it once.
    TSVD additionally shares ONE eigendecomposition across every
    truncation, because truncating is just taking fewer columns of a
    spectrum already computed. That is the whole reason a sweep belongs
    in one process: N processes pay N times for both.
    """

    def __init__(self, G):
        self.G = G
        self.n = G.shape[0]
        # G is exp(negative) times a positive constant, so it is strictly
        # positive and max() is abs().max() without the 537 MB abs copy.
        self.scale = float(G.max())
        self.diag0 = G.diagonal().copy()
        self._eig = None

    def _restore(self):
        """Tikhonov writes lambda into the diagonal in place; every
        other use needs the original back. Restoring on entry rather
        than on exit means a sweep can interleave the two in any order."""
        self.G.flat[::self.n + 1] = self.diag0

    def eigen(self):
        if self._eig is None:
            self._restore()
            self._eig = eigen(self.G)
        return self._eig

    def operator(self, setting):
        """The per-band solve operator for one setting, and a label.

        TSVD needs the eigendecomposition (O(d^3), 106 s at d=8192 and
        98% of the fixed cost). Tikhonov needs none: an explicit inverse
        is ~6x cheaper and leaves the per-cell cost a matvec either way.
        """
        kind, val = setting
        if kind == "keep":
            ev, vec = self.eigen()
            keep = max(1, round(val * len(ev)))
            op = (vec[:, :keep] / ev[:keep][None, :]) @ vec[:, :keep].T
            return op, "keep=%d" % keep
        self._restore()
        # in place: `G + lam * np.eye(d)` allocates a 537 MB identity AND
        # a 537 MB sum for a change that touches d of d*d entries
        self.G.flat[::self.n + 1] = self.diag0 + val * self.scale
        return np.linalg.inv(self.G), "tikhonov lam=%.0e" % val

    def close(self):
        self._eig = None
        self.G = None


#: Everything about a band that the per-cell solve needs, so the solve
#: takes a geometry rather than eight loose positional arguments.
BandGeom = namedtuple("BandGeom", "cell s freqs fd weights")


def solve_band(M, scene, members_band, geom, chunk):
    """Every cell of one band through one operator, in batches.

    Identical arithmetic to solving cells one at a time — the same M
    against the same right-hand sides — but one BLAS-3 matmul instead of
    hundreds of BLAS-2 matvecs, measured 7.1x on this shape and verified
    bit-identical including an uneven final chunk.
    """
    out = {}
    keys = list(members_band.keys())
    for lo in range(0, len(keys), chunk):
        batch = keys[lo:lo + chunk]
        centres = [(np.array(k, dtype=np.float64) + 0.5) * geom.cell
                   for k in batch]
        rhs = np.concatenate(
            [window_bundle(scene, members_band[k], c0, geom.s, geom.freqs)
             for k, c0 in zip(batch, centres)], axis=0)          # (B*C, d)
        sol = (M @ rhs.astype(np.complex128).T).T                # (B*C, d)
        nch = rhs.shape[0] // len(batch)
        for i, (k, c0) in enumerate(zip(batch, centres)):
            c = sol[i * nch:(i + 1) * nch]
            # cell-local -> world phase, then pre-divide so decode_slice's
            # weight multiply cancels exactly
            c = c * np.exp(-1j * (geom.fd @ c0))[None, :]
            out[k] = (c / geom.weights[None, :]).astype(np.complex64)
    return out


def label_of(setting):
    kind, val = setting
    return "keep=%.2f" % val if kind == "keep" else "tikhonov=%.0e" % val


def estimate_gb(n_settings):
    """What this run will need, for the headroom guard.

    Deliberately a loose UPPER bound. The guard's failure direction is
    under-protecting — an estimate that is too low reads as a check
    while providing none — and report_peak prints the real peak against
    this on every run, so being wrong is visible and cheap.

    Measured on saguaro at d=8192: 5.28 GB for one setting, 5.00 GB for
    a three-setting sweep. The peak is the eigendecomposition (the Gram,
    its eigenvectors and LAPACK's workspace), which the setting count
    does not move, because it lands on the first band before any solved
    bundles have accumulated. Settings still earn a term: on a denser
    capture each one's bundles are larger (train holds 1.3 GB against
    saguaro's 0.65 GB) and eventually outgrow the eigendecomposition.
    """
    return 5.5 + 0.7 * (n_settings - 1)


def report_setting(lab, cells, base, err, also_shrink, run):
    """One setting's slice error, and optionally what shrinkage adds."""
    a = err(cells)
    print("  %-16s analytic (window s=h/2): %.4f / %.4f"
          "   top-down %+.1f%%  side %+.1f%%"
          % (lab, a[0], a[1], 100 * (base[0] - a[0]) / base[0],
             100 * (base[1] - a[1]) / base[1]))
    if run is not None:
        run.result(**{lab: {"top_down": round(a[0], 4),
                            "side": round(a[1], 4),
                            "vs_forward_pct": round(
                                100 * (base[0] - a[0]) / base[0], 1)}})
    if not also_shrink:
        return
    # Does shrinkage add anything to an ALREADY-SOLVED bundle? Prediction
    # was "little or negative" — the solve is L2-optimal on the window and
    # already regularised. Measured +6.4% / +3.6%: the objective (windowed
    # L2 per cell) is not the evaluation (slice error with cross-cell
    # contributions).
    from holo.denoise import percentile_threshold, shrink
    for pct in (10, 25):
        sh = {b: {k: shrink(v, percentile_threshold(v, pct))
                  for k, v in band.items()} for b, band in cells.items()}
        e = err(sh)
        del sh
        print("    + shrink p%-2d       : %.4f / %.4f  (%+.1f%% / %+.1f%%)"
              % (pct, e[0], e[1], 100 * (a[0] - e[0]) / a[0],
                 100 * (a[1] - e[1]) / a[1]))


def main(path, settings, also_shrink=False, run=None):
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
    if run is not None:
        run.result(forward={"top_down": round(base[0], 4),
                            "side": round(base[1], 4)})
        run.stage("forward encode", time.time() - t0)
    print("%s  |  forward encoding: %.4f / %.4f  (%.0fs)"
          % (path.split("/")[-1], base[0], base[1], time.time() - t0),
          flush=True)
    # the forward bundles have now been scored and are never read again;
    # only their cell counts are. At capture scale they are 0.6-1.3 GB.
    counts = {n: len(c) for n, c in fwd_bundles.items()}
    del fwd_bundles

    labels = [label_of(st) for st in settings]
    ana = {lab: {} for lab in labels}
    for name, _cap, cell in BANDS:
        if not counts.get(name):
            for lab in labels:
                ana[lab][name] = {}
            continue
        freqs, _rho, weights = books[name]
        fd = freqs.astype(np.float64)
        s = (cell / 2) / 2                       # the width that won per-cell
        geom = BandGeom(cell, s, freqs, fd, weights)
        solver = BandSolver(build_gram(fd, s))
        chunk = cell_chunk(scene.channels, freqs.shape[0])
        for i, (st, lab) in enumerate(zip(settings, labels)):
            M, how = solver.operator(st)
            if i == len(settings) - 1:
                # Release the Gram and its eigenvectors BEFORE the solves.
                # BandSolver retains them so a sweep can share them, which
                # is the whole point — but on the last setting that is
                # 537 MB per retained array held through every per-cell
                # solve for nothing. Measured on saguaro: retaining them
                # cost 5.63 GB peak against main's 5.05 GB, and releasing
                # here recovers 0.35 of that 0.58 — landing at 5.28 GB,
                # still 4.5% above main. The per-process peak is NOT where
                # this file wins; a 3-setting sweep peaks at 5.00 GB in one
                # process where three processes cost about 15 GB.
                solver.close()
            print("  %-7s %d cells, d=%d, %s, chunk=%d  (%.0fs)"
                  % (name, counts[name], freqs.shape[0], how, chunk,
                     time.time() - t0), flush=True)
            t_band = time.time()
            ana[lab][name] = solve_band(M, scene, members[name], geom, chunk)
            if run is not None:
                # recorded per band per setting, so a killed run's last
                # stage says exactly how far it got
                run.stage("%s/%s" % (name, lab), time.time() - t_band,
                          cells=counts[name], operator=how)
            del M
        del solver

    for lab in labels:
        report_setting(lab, ana[lab], base, err, also_shrink, run)
    print("  total %.0fs" % (time.time() - t0))


def parse_settings(argv):
    """Back-compatible: a positional keep_frac and --tikhonov still work.
    --sweep keep=a,b / --sweep tikhonov=a,b add settings that SHARE the
    band Gram instead of each needing its own process."""
    settings, rest = [], []
    i = 0
    while i < len(argv):
        if argv[i] == "--tikhonov":
            settings.append(("tikhonov", float(argv[i + 1])))
            i += 2
        elif argv[i] == "--sweep":
            kind, _, vals = argv[i + 1].partition("=")
            if kind not in ("keep", "tikhonov"):
                raise SystemExit("--sweep takes keep=... or tikhonov=...")
            settings += [(kind, float(v)) for v in vals.split(",")]
            i += 2
        elif not argv[i].startswith("--"):
            rest.append(argv[i])
            i += 1
        else:
            i += 1
    if not settings:
        settings = [("keep", float(rest[1]) if len(rest) > 1 else 1.0)]
    return rest[0], settings


if __name__ == "__main__":
    argv = sys.argv[1:]
    path, settings = parse_settings(argv)
    with runlog.record(path.split("/")[-1],
                       need_gb=estimate_gb(len(settings)),
                       force="--force-memory" in argv) as run:
        run.result(forward=None)          # replaced once the baseline lands
        main(path, settings, also_shrink="--shrink" in argv, run=run)
