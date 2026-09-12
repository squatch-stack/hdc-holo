"""Opt-in analytic Gaussian-window L2 projection of spatial cells.

The solver arithmetic is lifted unchanged from the projection pipeline.
``eps`` is a fixed threshold on eigenvalue magnitude relative to the largest
magnitude, not a fraction of retained ranks. See docs/projection.md for the
measured failures and the limited capture evidence behind the defaults.
"""
from collections import namedtuple

import numpy as np

from .budget import require_headroom
from .capture import BANDS, decode_slice, exact_slice, mass_mode, slice_grid
from .spectral import SplatScene, spectral_bundle

DIVERGENCE_RATIO = 1.2
SIGNAL_FLOOR = 1e-3
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
        if kind in ("keep", "eps"):
            ev, vec = self.eigen()
            keep = self.rank(ev, kind, val)
            # UNCHANGED ARITHMETIC. Both truncations differ only in how
            # many columns they take; the operator built from those
            # columns is the same expression it always was, so every
            # keep= number in docs/projection.md still stands unre-derived.
            op = (vec[:, :keep] / ev[:keep][None, :]) @ vec[:, :keep].T
            return op, ("keep=%d" % keep if kind == "keep"
                        else "eps=%.0e -> keep=%d" % (val, keep))
        self._restore()
        # in place: `G + lam * np.eye(d)` allocates a 537 MB identity AND
        # a 537 MB sum for a change that touches d of d*d entries
        self.G.flat[::self.n + 1] = self.diag0 + val * self.scale
        return np.linalg.inv(self.G), "tikhonov lam=%.0e" % val

    @staticmethod
    def rank(ev, kind, val):
        """How many eigenvalues survive, by rank fraction or by threshold.

        These are NOT the same knob. `keep` takes the largest `val*d` of
        them and says nothing about how small the smallest survivor is —
        and the operator divides by that survivor. Each band's Gram
        decays at its own rate, so one rank fraction lands at a wildly
        different eigenvalue per band: at d=8192, keep=0.55 cuts `fine`
        at 2.47e-12 and `coarse` at 1.79e-03, a factor of 7e8 in what it
        actually regularises. `eps` cuts at the level instead, so it
        adapts to each band's spectrum by construction — which is what
        the Fourier-extension literature does, with accuracy going as
        sqrt(eps). See docs/projection.md and `--spectrum`.
        """
        if kind == "keep":
            return max(1, round(val * len(ev)))
        a = np.abs(ev)
        return max(1, int((a > val * a[0]).sum()))

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


def divergence_ratio(solved, forward):
    """Median ratio of solved to forward bundle norm, over shared cells.

    The catastrophic failure of an over-loose truncation is silent in
    the bundle — it decodes to garbage rather than raising — but it is
    LOUD in the norm, orders of magnitude before it is subtle anywhere
    else. Cheap enough to run always.
    """
    keys = [k for k in solved if k in forward]
    if not keys:
        return None
    num = np.median([np.linalg.norm(solved[k]) for k in keys])
    den = np.median([np.linalg.norm(forward[k]) for k in keys])
    return float(num / den) if den else None


class Diverged(ValueError):
    """A band's solved bundles are past the truncation cliff."""


def check_divergence(solved, forward, band, label, allow=False,
                     limit=DIVERGENCE_RATIO, forward_errors=None):
    """GATE, not a warning: refuse a band whose solve has diverged.

    This used to print and carry on, and that is exactly how a corrupted
    band reaches a caller unnoticed. On Red Rock at keep=0.55 every cell
    of the `fine` band sits at 1030x the forward bundle norm while the
    SLICE ERROR IMPROVES — because that band holds 0.7% of the splats,
    so destroying it barely moves the metric. Whoever reads those
    bundles for a render, a `what_is_at` query or storage gets garbage,
    and the number they were shown said the run was the best of four.

    `allow=True` (the --allow-divergence flag) is for sweeps that
    deliberately explore past the cliff, which is the only reason to
    want a diverged band at all.
    """
    ratio = divergence_ratio(solved, forward)
    if forward_errors is not None and not gate_eligible(forward_errors):
        return ratio
    if ratio is None or (np.isfinite(ratio) and ratio <= limit):
        return ratio
    message = (
        "%s band diverged at %s: solved bundles are %.3fx the forward norm "
        "(limit %.3f). This truncation is past the cliff and the band is "
        "garbage — the slice error will NOT necessarily show it, because a "
        "band holding a small share of the splats can be destroyed without "
        "moving it. Use a smaller keep, or --allow-divergence if you are "
        "sweeping past the cliff on purpose."
        % (band, label, ratio, limit))
    if not allow:
        raise Diverged(message)
    print("    !! %s" % message, flush=True)
    return ratio



def rel_err(got, want, floor=0.0):
    """Relative error, or None when there is nothing to be relative to.

    Two ways that happens, and only the first was handled before. A band
    with no splats near a slice gives an exactly zero referee. A band
    with a FEW distant splats gives a nearly zero one, and dividing by
    it produces a number that looks like a catastrophic failure and is
    actually an absence of signal. Both are missing measurements, not
    scores of zero and not scores of 4e6.
    """
    scale = float(np.linalg.norm(want))
    if scale <= floor:
        return None
    return float(np.linalg.norm(got - want) / scale)


#: What a setting is scored against: the two error functions and the
#: forward-encoding baselines each is measured against. Bundled because
#: report_setting took six positional arguments and adding per-band
#: scoring to it would have taken eight.
Referee = namedtuple("Referee", "err band_err base band_base")


def band_errors(bundles, books, slices, band_truth, bands=None):
    """Each band scored against its OWN ground truth.

    The aggregate is dominated by whichever band holds the splats — on
    Red Rock `xfine` holds 542,122 of 546,638 — so a band can be
    destroyed by three orders of magnitude while the aggregate
    IMPROVES. That is not hypothetical: it is keep=0.55 on Red Rock, and
    it is why slice error alone was never an acceptance test.

    `decode_slice` and `exact_slice` both already restrict to a band
    list, and the bands partition the splats, so this costs one extra
    decode pass in total rather than one per band.

    A band too faint at these points to score is reported as absent
    rather than as a number — see SIGNAL_FLOOR.
    """
    bands = BANDS if bands is None else bands
    out = {b[0]: [] for b in bands if bundles.get(b[0])}
    for n, (pts, _) in slices:
        # the bands partition the splats, so their truths sum to the
        # whole field at these points — which is what "a share of the
        # field" is measured against
        floor = SIGNAL_FLOOR * float(np.linalg.norm(
            sum(band_truth[n][b[0]][:, 0] for b in bands)))
        for b in bands:
            if b[0] not in out:
                continue
            out[b[0]].append(
                rel_err(decode_slice(pts, bundles, books, bands=[b])[:, 0],
                        band_truth[n][b[0]][:, 0], floor))
    return out



def gate_eligible(errors):
    """Gate bands with signal that forward reconstructs better than zeros.

    A band must pass on every slice where it has signal. An absent band
    or a pre-existing forward failure cannot establish projection damage.
    Nonfinite errors remain eligible so they cannot silently disable a gate.
    """
    present = [error for error in errors if error is not None]
    return bool(present) and not any(error >= 1.0 for error in present)


def evidence_slices(scene):
    """The driver's two mass-mode planes in an enclosing integer box.

    Normalized captures use the original unit box and pixel spacing.
    Other coordinates use the enclosing box, with the same relative spacing.
    """
    lo = np.floor(scene.mu.min(axis=0))
    hi = np.maximum(np.ceil(scene.mu.max(axis=0)), lo + 1)
    extent = hi - lo
    w = scene.amp[:, 0]
    planes = []
    for name, axis, u, v in (("top-down", 1, 0, 2), ("side", 0, 2, 1)):
        at = lo[axis] + mass_mode(scene.mu[:, axis] - lo[axis], w, extent[axis])
        planes.append((name, slice_grid(
            (lo[u], hi[u]), (lo[v], hi[v]), "y" if axis == 1 else "x", at,
            pix=float(extent.max()) / 224)))
    return planes


def _forward_cells(scene, members, freqs):
    return {key: spectral_bundle(
        SplatScene(scene.mu[ids], scene.cov[ids], scene.amp[ids]),
        freqs, chunk=2048) for key, ids in members.items()}


def _validate(setting, window, limit):
    kind, value = setting
    if kind not in ("eps", "keep", "tikhonov"):
        raise ValueError("setting must be eps, keep or tikhonov")
    if not np.isfinite(value) or value <= 0 or (kind != "tikhonov" and value > 1):
        raise ValueError("setting must be positive; eps and keep must be <= 1")
    if not np.isfinite(window) or window <= 0:
        raise ValueError("window must be finite and positive")
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("limit must be finite and positive")


def project_cells(scene, members, books, bands=None,
                  setting=("eps", 1e-3), window=0.5,
                  limit=DIVERGENCE_RATIO, allow_divergence=False):
    """Project cells, returning ``{band: {cell: (channels, d)}}`` bundles.

    ``members`` and ``books`` come from ``encode_bands``. ``bands`` must
    describe that same fixed lattice. The Gaussian window has standard
    deviation ``window * cell / 2``. Forward encoding remains unchanged.

    Check memory headroom before allocating Grams. Gate each band against
    its forward norm (first 64 cells, preserving the capture measurement).
    Forward errors on the two mass-mode planes determine gate eligibility;
    absent signal and forward errors at least one are excluded. Exclusion
    is not an assertion that the returned band is accurate. ``Diverged``
    refuses the entire result unless explicitly allowed. See the docs for
    the limitations of this empirical detector and adaptive lattices.
    """
    _validate(setting, window, limit)
    bands = BANDS if bands is None else bands
    active = [b for b in bands if members.get(b[0])]
    out = {b[0]: {} for b in bands}
    if not active:
        return out
    dim = max(len(books[b[0]][0]) for b in active)
    stored = sum(len(members[b[0]]) * scene.channels * len(books[b[0]][0]) * 8
                 for b in active) / (1 << 30)
    require_headroom(6.0 * (dim / 8192) ** 2 + 2 * stored + 0.25)
    slices = evidence_slices(scene)
    truth = {n: {b[0]: exact_slice(pts, scene, members, bands=[b])
                 for b in active} for n, (pts, _) in slices}
    for band in active:
        name, _cap, cell = band
        freqs, _rho, weights = books[name]
        forward = _forward_cells(scene, members[name], freqs)
        errors = band_errors({name: forward}, books, slices, truth, active)[name]
        reference = dict(list(forward.items())[:64])
        del forward
        fd = freqs.astype(np.float64)
        s = window * (cell / 2)
        solver = BandSolver(build_gram(fd, s))
        operator, label = solver.operator(setting)
        solver.close()
        solved = solve_band(operator, scene, members[name],
                            BandGeom(cell, s, freqs, fd, weights),
                            cell_chunk(scene.channels, len(freqs)))
        check_divergence(solved, reference, name, label, allow_divergence,
                         limit, errors)
        out[name] = solved
        del operator
    return out
