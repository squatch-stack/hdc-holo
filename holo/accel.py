"""GPU acceleration for the holographic kernels (Apple Metal via MLX).

Everything hot in this codebase is two primitives — evaluate cos/sin of
a big phase matrix, then real GEMMs — so the whole pipeline runs on any
backend that has matmul + trig. Complex arithmetic is deliberately
avoided on the device: Metal frameworks have patchy complex64 support,
so phasors are carried as cos/sin planes and every complex GEMM becomes
two real ones (same flops as complex, universally supported):

    encode:  S = amp^T @ (norm * env * e^{-i phase})
        ->   S_re = a^T @ (m * cos phase),  S_im = -a^T @ (m * sin phase)
    decode:  out = Re(E @ (S * w)^T),  E = e^{+i phase}
        ->   out  = cos(phase) @ (w * S_re)^T - sin(phase) @ (w * S_im)^T

Backend selection: MLX on the default (GPU) device when importable,
NumPy otherwise; force with HDC_BACKEND=mlx|numpy. Results match the
NumPy path to float32 rounding (~1e-7 relative).

Measured on an M1 Max (24-core GPU, 64 GB unified), d = 32,768:
encode 37x, decode 106x over NumPy/OpenBLAS — a 22-minute real-scene
pipeline's holographic stages drop to seconds. Unified memory means the
arrays are never copied across a bus.
"""

import os

import numpy as np

#: None until something asks. Probing on IMPORT meant that
#: `from holo import HoloMap` initialised MLX and selected the GPU,
#: because most of the package imports this module — so a caller who
#: wanted a hypervector map got GPU exposure they never asked for. That
#: is not hypothetical: a Metal fault raised by an unrelated process on
#: a shared machine killed a run here, as an innocent victim. The
#: backend is still chosen once and cached; it is chosen on first ASK.
_HAVE_MLX = None
mx = None


def _probe():
    """Import MLX on first use and remember the answer."""
    global _HAVE_MLX, mx
    if _HAVE_MLX is None:
        try:
            if os.environ.get("HDC_BACKEND", "").lower() == "numpy":
                raise ImportError("forced off via HDC_BACKEND=numpy")
            import mlx.core as _mx
            mx, _HAVE_MLX = _mx, True
        except ImportError:
            mx, _HAVE_MLX = None, False
    return _HAVE_MLX


def active():
    """True when the MLX/Metal backend will be used."""
    return _probe()


def backend_name():
    return "mlx-gpu" if _probe() else "numpy"


def _cov_pairs(dim):
    return [(i, j) for i in range(dim) for j in range(i, dim)]


def spectral_bundle(scene, freqs, chunk=16384):
    """GPU drop-in for hdc_splat.spectral_bundle: identical semantics,
    real-formulation kernels, returns (C, d) complex64 on the host."""
    _probe()
    d, dim = freqs.shape
    pairs = _cov_pairs(dim)
    wq = np.stack([freqs[:, i] * freqs[:, j] * (1.0 if i == j else 2.0)
                   for i, j in pairs], axis=1)
    m_freqs = mx.array(freqs)
    m_wq = mx.array(wq)
    br = mx.zeros((scene.channels, d), dtype=mx.float32)
    bi = mx.zeros((scene.channels, d), dtype=mx.float32)
    for lo in range(0, scene.n, chunk):
        mu = scene.mu[lo:lo + chunk]
        cov = scene.cov[lo:lo + chunk]
        amp = scene.amp[lo:lo + chunk]
        norm = ((2 * np.pi) ** (dim / 2)
                * np.sqrt(np.linalg.det(cov.astype(np.float64)))) \
            .astype(np.float32)
        cq = np.stack([cov[:, i, j] for i, j in pairs], axis=1)
        m_amp = mx.array(amp * norm[:, None])
        env = mx.exp(-0.5 * (mx.array(cq) @ m_wq.T))
        phase = mx.array(mu) @ m_freqs.T
        br = br + m_amp.T @ (env * mx.cos(phase))
        bi = bi - m_amp.T @ (env * mx.sin(phase))
        mx.eval(br, bi)
    out = np.array(br).astype(np.float32) \
        + 1j * np.array(bi).astype(np.float32)
    return out.astype(np.complex64)


def ridge_cell_fit(freqs, points, targets, lam, prior=None):
    """Dual ridge regression of a field onto the cos/sin features of one
    codebook, with an optional per-frequency PRIOR: minimize
    ||[cos F, sin F] diag(prior) x' - y||^2 + lam ||prior||^2 ||x'||^2,
    returning x = prior * x'. Without a prior, minimum-norm regression
    spreads energy uniformly across the codebook — including its finest
    frequencies — so the fit memorizes sample points as kernel-width
    bumps and oscillates between them. A prior shaped like the expected
    splat spectrum (e.g. exp(-1/2 sigma_p^2 |w|^2)) makes the implied
    kernel match the content's scale; the data can still override it.

    The Gram and both projections run on the device (A A^T = C C^T +
    S S^T without materializing A); the P x P solve stays float64 on the
    CPU (see holo/fit.py on why the convex problem deserves its closed
    form). Returns the fitted (C, d) complex64 coefficient bundle in
    decode convention: out(p) = Re(e^{i F p} @ x^T) — i.e. the WEIGHTED
    bundle; importance-weighted decoders divide by their weights.
    """
    _probe()
    Y = np.asarray(targets, dtype=np.float64)
    if prior is None:
        prior = np.ones(freqs.shape[0], dtype=np.float32)
    ridge = lam * float((prior.astype(np.float64) ** 2).sum())
    if _HAVE_MLX:
        mp = mx.array(prior)
        ph = mx.array(np.asarray(points, np.float32)) \
            @ mx.array(np.asarray(freqs, np.float32)).T
        Cp, Sp = mx.cos(ph) * mp, mx.sin(ph) * mp
        K = Cp @ Cp.T + Sp @ Sp.T
        mx.eval(K)
        Kn = np.array(K).astype(np.float64)
        Kn[np.diag_indices_from(Kn)] += ridge
        Beta = np.linalg.solve(Kn, Y)
        mB = mx.array(Beta.astype(np.float32))
        Tc, Ts = mp[:, None] * (Cp.T @ mB), mp[:, None] * (Sp.T @ mB)
        mx.eval(Tc, Ts)
        Tc, Ts = np.array(Tc), np.array(Ts)
    else:
        ph = np.asarray(points, np.float32) @ np.asarray(freqs).T
        Cp = np.cos(ph) * prior
        Sp = np.sin(ph) * prior
        Kn = (Cp @ Cp.T + Sp @ Sp.T).astype(np.float64)
        Kn[np.diag_indices_from(Kn)] += ridge
        Beta = np.linalg.solve(Kn, Y).astype(np.float32)
        Tc = prior[:, None] * (Cp.T @ Beta)
        Ts = prior[:, None] * (Sp.T @ Beta)
    return (Tc.T - 1j * Ts.T).astype(np.complex64)


def cell_decode(freqs, points, cells, chunk=8192):
    """Masked multi-cell decode on the GPU: sum over cells of
    Re(E[mask] @ (weighted bundle).T), the inner loop of chunked-scene
    slice decoding and X-ray rendering.

    cells: sequence of (mask over all points, (C, d) complex64 weighted
    bundle). The phase trig is computed once per point-chunk; each cell
    contributes two masked real GEMMs, queued and evaluated as one batch
    per chunk so kernel-launch overhead amortizes. Returns (P, C) float32.
    """
    _probe()
    cells = list(cells)
    if not cells:
        return np.zeros((len(points), 0), dtype=np.float32)
    m_freqs = mx.array(freqs)
    prep = [(np.where(m)[0],
             mx.array(np.ascontiguousarray(b.real.T.astype(np.float32))),
             mx.array(np.ascontiguousarray(b.imag.T.astype(np.float32))))
            for m, b in cells]
    n_pts, n_ch = len(points), cells[0][1].shape[0]
    out = np.zeros((n_pts, n_ch), dtype=np.float32)
    for lo in range(0, n_pts, chunk):
        ph = mx.array(points[lo:lo + chunk]) @ m_freqs.T
        cph, sph = mx.cos(ph), mx.sin(ph)
        pend = []
        for idx, wr, wi in prep:
            sel = idx[(idx >= lo) & (idx < lo + chunk)] - lo
            if len(sel) == 0:
                continue
            m_sel = mx.array(sel)
            o = mx.take(cph, m_sel, 0) @ wr - mx.take(sph, m_sel, 0) @ wi
            pend.append((sel + lo, o))
        if pend:
            mx.eval(*[o for _, o in pend])
            for gsel, o in pend:
                out[gsel] += np.array(o)
    return out


def decode(bundle, freqs, weights, points, chunk=8192):
    """GPU drop-in for hdc_splat._decode: out[p, c] = Re(E @ (S w)^T)."""
    _probe()
    wr = mx.array((bundle.real * weights[None, :]).T.astype(np.float32))
    wi = mx.array((bundle.imag * weights[None, :]).T.astype(np.float32))
    m_freqs = mx.array(freqs)
    out = np.empty((points.shape[0], bundle.shape[0]), dtype=np.float32)
    for lo in range(0, points.shape[0], chunk):
        phase = mx.array(points[lo:lo + chunk]) @ m_freqs.T
        o = mx.cos(phase) @ wr - mx.sin(phase) @ wi
        mx.eval(o)
        out[lo:lo + chunk] = np.array(o)
    return out


def readout(points, W, S, chunk=8192):
    """The universal field readout, backend-dispatched:

        out = Re( e^{i points @ W.T} @ conj(S).T ) / d
            = cos(phase) @ (Re S / d).T  +  sin(phase) @ (Im S / d).T

    (decode() without weights and WITH the conjugate — the sign flip on
    the sin term is the whole difference). S is one bundle (d,) or c
    channel bundles (c, d); returns float32 (n,) or (n, c). Every eval
    path in the SDK (fields, scenes, fitted holograms, folded view
    bundles) funnels through here, so the MLX/Metal speedup applies
    uniformly; the NumPy fallback computes the identical real
    formulation. Both paths agree to float32 rounding (~1e-6)."""
    _probe()
    points = np.ascontiguousarray(points, dtype=np.float32)
    S2 = np.atleast_2d(S)
    d = W.shape[0]
    wr = np.ascontiguousarray(S2.real.T, dtype=np.float32) / d
    wi = np.ascontiguousarray(S2.imag.T, dtype=np.float32) / d
    out = np.empty((len(points), S2.shape[0]), dtype=np.float32)
    if _HAVE_MLX:
        m_W, m_wr, m_wi = mx.array(W), mx.array(wr), mx.array(wi)
        for lo in range(0, len(points), chunk):
            phase = mx.array(points[lo:lo + chunk]) @ m_W.T
            o = mx.cos(phase) @ m_wr + mx.sin(phase) @ m_wi
            mx.eval(o)
            out[lo:lo + chunk] = np.array(o)
    else:
        for lo in range(0, len(points), chunk):
            phase = points[lo:lo + chunk] @ W.T
            out[lo:lo + chunk] = np.cos(phase) @ wr + np.sin(phase) @ wi
    return out if np.ndim(S) == 2 else out[:, 0]
