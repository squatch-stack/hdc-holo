"""An out-of-tree CUDA backend: the holographic kernels on CuPy.

`holo/accel.py` dispatches NumPy or MLX/Metal. There is no CUDA path,
so a real-scene run on an NVIDIA box falls all the way back to NumPy —
which is what happened on 2026-09-06, when three scenes cost 12, 17 and
44 minutes of laptop CPU. This module supplies the missing path by
patching `holo.accel` at runtime, which the SDK explicitly supports:
`holo/backend.py` and the `hdc/*` shims resolve through a module
`__getattr__` on every access rather than binding accel's function
objects at import, precisely so an out-of-tree backend is picked up by
the facades (docs/backend.md; the import-time-binding bug was issue
#10). Nothing here enters the SDK surface — see SDK.md.

Two families are patched, because the real-scene pipeline spends its
time in both and only one of them is backend-dispatched:

  * the accel kernels — `spectral_bundle` (encode), `cell_decode`
    (the inner loop of `decode_slice` and `render_xray`), `decode`,
    `readout`. `holo.spectral.spectral_bundle` and `holo.capture`
    already route through `_accel.active()`, so patching is enough.
  * the GROUND TRUTH — `capture.exact_slice` and `capture.exact_xray`.
    These are plain NumPy einsum loops with no dispatch hook at all,
    and on a cell-dense scene they dominate. They are ported here
    term-for-term, same formulas, same chunked loop, same masks.

**Precision contract: TF32 is off.** Ampere/Blackwell fp32 matmuls
default to TF32 tensor cores, costing ~2 orders of magnitude of
relative accuracy (1e-7 -> ~1e-5) against this repo's cross-backend
bar — the finding already recorded in docs/backend.md and bench/
RECIPE.md from the first 5090 bring-up. CuPy leaves cuBLAS in default
math mode unless CUPY_TF32=1, so this module refuses to install if that
variable is set rather than quietly producing a weaker number.

Verify before believing any timing: `bench/verify_cuda.py` re-runs
every patched function against the stock NumPy path on the same inputs
and reports max relative deviation.
"""

import os

import numpy as np

#: Module state in one dict rather than three `global` rebinds — the
#: kernels below read `cp` on every call, so it has to be resolvable at
#: call time, and a dict entry is that without the rebinding.
_STATE = {"installed": False, "cupy": None, "vram": None}


class _LazyCupy:
    """`cp.foo` resolves against the imported module on each access, so
    the kernels can be written in the natural `cp.exp(...)` style while
    the import itself stays deferred to `install()`."""

    def __getattr__(self, name):
        mod = _STATE["cupy"]
        if mod is None:
            raise RuntimeError("cuda_backend.install() has not run")
        return getattr(mod, name)


cp = _LazyCupy()


def install(verbose=True):
    """Patch holo.accel and holo.capture in place. Idempotent."""
    if _STATE["installed"]:
        return
    if os.environ.get("CUPY_TF32", "") == "1":
        raise SystemExit(
            "CUPY_TF32=1 is set: fp32 matmuls would run on TF32 tensor "
            "cores and miss the cross-backend accuracy bar by ~2 orders "
            "of magnitude. Unset it (see docs/backend.md).")
    import cupy
    _STATE["cupy"] = cupy

    from holo import accel, capture

    accel.active = lambda: True
    accel.backend_name = lambda: "cupy-cuda"
    accel.spectral_bundle = spectral_bundle
    accel.cell_decode = cell_decode
    accel.decode = decode
    accel.readout = readout

    capture.exact_slice = exact_slice
    capture.exact_xray = exact_xray

    _STATE["installed"] = True
    if verbose:
        props = cp.cuda.runtime.getDeviceProperties(0)
        cc = cp.cuda.Device(0).compute_capability
        free, total = cp.cuda.runtime.memGetInfo()
        print(f"cuda backend: {props['name'].decode()} sm_{cc}, "
              f"{free / 1e9:.1f}/{total / 1e9:.1f} GB free, "
              f"cupy {cp.__version__}, TF32 off")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _vram_budget(budget=0.25):
    """Bytes a chunked temp may occupy, measured once.

    `cudaMemGetInfo` SYNCHRONIZES the device. The GT kernels ask per
    cell, and a real scene has thousands of cells, so querying live
    turned every cell boundary into a pipeline stall — measured as a
    3x slowdown against NumPy on a kernel that should have won. The
    device's capacity does not change during a run; the pool's usage
    does, and that is what `budget` leaves headroom for.
    """
    if _STATE["vram"] is None:
        _STATE["vram"] = cp.cuda.runtime.memGetInfo()[0]
    return _STATE["vram"] * budget


def _rows_for(per_row_bytes, cap):
    """Rows of a chunked temp that fit the VRAM budget.

    The GT kernels build an (n, P, 3) difference tensor; on a
    million-splat scene with a wide slice grid that is the only thing
    in the pipeline big enough to fail, so it is sized against what the
    device actually has rather than a constant. Chunks are allowed to
    grow well past the NumPy default — a bigger einsum is exactly how
    the launch overhead per cell gets amortized.
    """
    if per_row_bytes <= 0:
        return cap
    return max(1, int(_vram_budget() // per_row_bytes))


# ---------------------------------------------------------------------------
# accel kernels
# ---------------------------------------------------------------------------

def _cov_pairs(dim):
    return [(i, j) for i in range(dim) for j in range(i, dim)]


def spectral_bundle(scene, freqs, chunk=16384):
    """CuPy drop-in for accel.spectral_bundle: same real formulation
    (phasors as cos/sin planes), returns (C, d) complex64 on the host."""
    d, dim = freqs.shape
    pairs = _cov_pairs(dim)
    wq = np.stack([freqs[:, i] * freqs[:, j] * (1.0 if i == j else 2.0)
                   for i, j in pairs], axis=1)
    g_freqs = cp.asarray(freqs.astype(np.float32))
    g_wq = cp.asarray(wq.astype(np.float32))
    br = cp.zeros((scene.channels, d), dtype=cp.float32)
    bi = cp.zeros((scene.channels, d), dtype=cp.float32)
    for lo in range(0, scene.n, chunk):
        mu = scene.mu[lo:lo + chunk]
        cov = scene.cov[lo:lo + chunk]
        amp = scene.amp[lo:lo + chunk]
        norm = ((2 * np.pi) ** (dim / 2)
                * np.sqrt(np.linalg.det(cov.astype(np.float64)))) \
            .astype(np.float32)
        cq = np.stack([cov[:, i, j] for i, j in pairs], axis=1)
        g_amp = cp.asarray((amp * norm[:, None]).astype(np.float32))
        env = cp.exp(-0.5 * (cp.asarray(cq.astype(np.float32)) @ g_wq.T))
        phase = cp.asarray(mu.astype(np.float32)) @ g_freqs.T
        br += g_amp.T @ (env * cp.cos(phase))
        bi -= g_amp.T @ (env * cp.sin(phase))
    out = cp.asnumpy(br).astype(np.float32) \
        + 1j * cp.asnumpy(bi).astype(np.float32)
    return out.astype(np.complex64)


def cell_decode(freqs, points, cells, chunk=16384):
    """CuPy drop-in for accel.cell_decode: sum over cells of
    Re(E[mask] @ (weighted bundle).T). Returns (P, C) float32."""
    cells = list(cells)
    if not cells:
        return np.zeros((len(points), 0), dtype=np.float32)
    g_freqs = cp.asarray(freqs.astype(np.float32))
    prep = [(cp.asarray(np.where(m)[0]),
             cp.asarray(np.ascontiguousarray(b.real.T.astype(np.float32))),
             cp.asarray(np.ascontiguousarray(b.imag.T.astype(np.float32))))
            for m, b in cells]
    n_pts, n_ch = len(points), cells[0][1].shape[0]
    g_pts = cp.asarray(np.ascontiguousarray(points, dtype=np.float32))
    out = cp.zeros((n_pts, n_ch), dtype=cp.float32)
    for lo in range(0, n_pts, chunk):
        hi = min(lo + chunk, n_pts)
        ph = g_pts[lo:hi] @ g_freqs.T
        cph, sph = cp.cos(ph), cp.sin(ph)
        for idx, wr, wi in prep:
            sel = idx[(idx >= lo) & (idx < hi)]
            if sel.size == 0:
                continue
            loc = sel - lo
            o = cph[loc] @ wr - sph[loc] @ wi
            out[sel] += o
    return cp.asnumpy(out)


def decode(bundle, freqs, weights, points, chunk=16384):
    """CuPy drop-in for accel.decode: out[p, c] = Re(E @ (S w)^T)."""
    wr = cp.asarray((bundle.real * weights[None, :]).T.astype(np.float32))
    wi = cp.asarray((bundle.imag * weights[None, :]).T.astype(np.float32))
    g_freqs = cp.asarray(freqs.astype(np.float32))
    g_pts = cp.asarray(np.ascontiguousarray(points, dtype=np.float32))
    out = np.empty((points.shape[0], bundle.shape[0]), dtype=np.float32)
    for lo in range(0, points.shape[0], chunk):
        hi = min(lo + chunk, points.shape[0])
        phase = g_pts[lo:hi] @ g_freqs.T
        out[lo:hi] = cp.asnumpy(cp.cos(phase) @ wr - cp.sin(phase) @ wi)
    return out


def readout(points, W, S, chunk=16384):
    """CuPy drop-in for accel.readout: the universal field readout,
    out = cos(phase) @ (Re S / d).T + sin(phase) @ (Im S / d).T."""
    points = np.ascontiguousarray(points, dtype=np.float32)
    S2 = np.atleast_2d(S)
    d = W.shape[0]
    wr = cp.asarray(np.ascontiguousarray(S2.real.T, dtype=np.float32) / d)
    wi = cp.asarray(np.ascontiguousarray(S2.imag.T, dtype=np.float32) / d)
    g_W = cp.asarray(W.astype(np.float32))
    g_pts = cp.asarray(points)
    out = np.empty((len(points), S2.shape[0]), dtype=np.float32)
    for lo in range(0, len(points), chunk):
        hi = min(lo + chunk, len(points))
        phase = g_pts[lo:hi] @ g_W.T
        out[lo:hi] = cp.asnumpy(cp.cos(phase) @ wr + cp.sin(phase) @ wi)
    return np.squeeze(out, axis=1) if np.ndim(S) == 1 else out


# ---------------------------------------------------------------------------
# ground truth — ported term for term from holo/capture.py
# ---------------------------------------------------------------------------

def exact_slice(points, scene, members, bands=None, chunk=2048,
                footprint=0.0):
    """CuPy port of capture.exact_slice. Identical formula, identical
    cell masks and band loop; the splat chunk is resized to free VRAM
    because the (n, P, 3) difference tensor is the one big temp."""
    from holo.capture import BANDS, cell_mask, footprint_blur

    if footprint > 0:
        scene = footprint_blur(scene, footprint)
    inv_cov = np.linalg.inv(scene.cov.astype(np.float64)).astype(np.float32)
    g_mu = cp.asarray(scene.mu.astype(np.float32))
    g_ic = cp.asarray(inv_cov)
    g_amp = cp.asarray(scene.amp.astype(np.float32))
    out = cp.zeros((len(points), scene.channels), dtype=cp.float32)
    g_points = cp.asarray(np.ascontiguousarray(points, dtype=np.float32))
    for name, cap, cell in (bands or BANDS):
        reach = 3.0 * cap
        for k, ids in members[name].items():
            m = cell_mask(points, k, cell, reach)
            if not m.any():
                continue
            g_m = cp.asarray(m)
            pts = g_points[g_m]
            g_ids = cp.asarray(ids)
            acc = cp.zeros((pts.shape[0], scene.channels), dtype=cp.float32)
            step = _rows_for(int(pts.shape[0]) * 3 * 4 * 3, chunk)
            for slo in range(0, len(ids), step):
                sub = g_ids[slo:slo + step]
                diff = pts[None, :, :] - g_mu[sub][:, None, :]
                quad = cp.einsum("npi,nij,npj->np", diff, g_ic[sub], diff)
                acc += cp.exp(-0.5 * quad).T @ g_amp[sub]
            out[g_m] += acc
    return cp.asnumpy(out)


def exact_xray(scene, members, view, center, half, res, bands=None,
               chunk=1024):
    """CuPy port of capture.exact_xray: the analytic full-line integral
    alpha sqrt(2 pi / q) exp(-1/2 (d^T S^-1 d - s^2/q)), same masks.

    The exponent is evaluated as the rank-1 downdate

        d^T S^-1 d - s^2/q  ==  d^T M d,
        M = S^-1 - (S^-1 v)(S^-1 v)^T / q,   q = v^T S^-1 v

    rather than as the difference the formula is written in, because in
    float32 that difference cancels away most of the mantissa. M is
    S^-1 projected orthogonal to the view ray, so it is positive
    semi-definite and the quadratic form is a sum of non-negative
    terms; built once per splat in float64 and cast down, it is both
    exact to float32 rounding and cheaper — one einsum where there were
    two.

    This port is where that was found: measured against a float64
    evaluation of the same formula on the cannon, the difference form
    carried 5.2e-4 relative error against the downdate's 1.9e-7, so the
    "disagreement" between the two backends was the NumPy original
    being wrong. `holo/capture.py` now uses the same identity (#93),
    which is why this is no longer a departure from it — the two agree
    by construction rather than by coincidence, and `verify_cuda.py`
    scores both against a float64 referee rather than against each
    other, so it would still catch either one drifting.
    """
    from holo.capture import (
        BANDS,
        _cell_uv_mask,
        _pixel_grid,
        camera_basis_yup,
    )

    v, u1, u2 = camera_basis_yup(view)
    uv, _ = _pixel_grid(center, v, u1, u2, half, res, 0.0)
    plane = (np.asarray(center, dtype=np.float32)
             + uv[:, :1] * u1 + uv[:, 1:] * u2)
    # the rank-1 downdate, in float64 where the cancellation lives
    ic64 = np.linalg.inv(scene.cov.astype(np.float64))
    v64 = np.asarray(v, dtype=np.float64)
    icv64 = ic64 @ v64                                     # (n, 3)
    q64 = np.einsum("ni,ni->n", icv64, v64[None, :])       # (n,)
    m64 = ic64 - (icv64[:, :, None] * icv64[:, None, :]) / q64[:, None, None]
    g_plane = cp.asarray(np.ascontiguousarray(plane, dtype=np.float32))
    g_mu = cp.asarray(scene.mu.astype(np.float32))
    g_m = cp.asarray(m64.astype(np.float32))
    g_pref = cp.asarray(np.sqrt(2 * np.pi / q64).astype(np.float32))
    g_amp = cp.asarray(scene.amp.astype(np.float32))
    out = cp.zeros((len(plane), scene.channels), dtype=cp.float32)
    for name, cap, cell in (bands or BANDS):
        reach = 3.0 * cap
        for k, ids in members[name].items():
            m = _cell_uv_mask(uv, k, cell, reach, center, u1, u2)
            if not m.any():
                continue
            g_mask = cp.asarray(m)
            pts = g_plane[g_mask]
            g_ids = cp.asarray(ids)
            acc = cp.zeros((pts.shape[0], scene.channels), dtype=cp.float32)
            step = _rows_for(int(pts.shape[0]) * 3 * 4 * 3, chunk)
            for slo in range(0, len(ids), step):
                sub = g_ids[slo:slo + step]
                delta = pts[None, :, :] - g_mu[sub][:, None, :]
                perp = cp.einsum("npi,nij,npj->np", delta, g_m[sub], delta)
                line = g_pref[sub][:, None] * cp.exp(-0.5 * perp)
                acc += line.T @ g_amp[sub]
            out[g_mask] += acc
    return cp.asnumpy(out)
