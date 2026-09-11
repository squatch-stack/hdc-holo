"""Prove the CuPy backend computes what NumPy computes, before any
timing from it is quoted.

Every function `bench/cuda_backend.install()` patches is run twice on
byte-identical inputs — once on the stock NumPy path, once on CUDA —
and the max relative deviation is reported against the repo's
cross-backend bar. The bar is float32 rounding: docs/backend.md pins
both paths to ~1e-6 relative, and the first 5090 bring-up showed
exactly what a violation looks like (TF32 tensor cores silently
degrading fp32 matmuls to ~1e-5), which is why this is a gate and not
a smoke test.

The scene is a real capture, subsampled so the NumPy reference is
affordable — the point is agreement on real geometry (wildly
anisotropic covariances, empty cells, single-member cells), which
synthetic Gaussians do not reproduce.

Usage:
    bench/verify_cuda.py <scene.ply|spz> [--splats 40000] [--res 96]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOL = 2e-6          # cross-backend bar: float32 rounding (docs/backend.md)


def rel(a, b):
    """Max relative deviation, scaled by the reference's own magnitude.

    Promotes through complex128, not float64: `spectral_bundle` returns
    complex64, and casting that to a real dtype discards the imaginary
    half silently — which made an earlier version of this gate check
    only the real part of the encode and call it agreement.
    """
    a, b = np.asarray(a), np.asarray(b)
    dt = np.complex128 if (np.iscomplexobj(a) or np.iscomplexobj(b)) \
        else np.float64
    a, b = a.astype(dt), b.astype(dt)
    denom = max(float(np.abs(b).max()), 1e-30)
    return float(np.abs(a - b).max() / denom)


def _xray_float64(scene, members, view, center, half, res, chunk=1024):
    """capture.exact_xray's formula, evaluated in float64 — the referee
    for a kernel whose float32 form cancels. Deliberately the literal
    formula, cancellation and all, so it tests the two float32 paths
    rather than the algebra."""
    from holo.capture import (
        BANDS,
        _cell_uv_mask,
        _pixel_grid,
        camera_basis_yup,
    )

    v, u1, u2 = camera_basis_yup(view)
    uv, _ = _pixel_grid(center, v, u1, u2, half, res, 0.0)
    plane = (np.asarray(center, dtype=np.float32)
             + uv[:, :1] * u1 + uv[:, 1:] * u2).astype(np.float64)
    ic_all = np.linalg.inv(scene.cov.astype(np.float64))
    mu, amp = scene.mu.astype(np.float64), scene.amp.astype(np.float64)
    vv = np.asarray(v, dtype=np.float64)
    out = np.zeros((len(plane), scene.channels), dtype=np.float64)
    for name, cap_, cell in BANDS:
        reach = 3.0 * cap_
        for k, ids in members[name].items():
            m = _cell_uv_mask(uv, k, cell, reach, center, u1, u2)
            if not m.any():
                continue
            pts = plane[m]
            acc = np.zeros((len(pts), scene.channels), dtype=np.float64)
            for slo in range(0, len(ids), chunk):
                sub = ids[slo:slo + chunk]
                ic = ic_all[sub]
                delta = pts[None, :, :] - mu[sub][:, None, :]
                icv = ic @ vv
                q = (icv @ vv)[:, None]
                s = np.einsum("npi,ni->np", delta, icv)
                quad = np.einsum("npi,nij,npj->np", delta, ic, delta)
                acc += (np.sqrt(2 * np.pi / q)
                        * np.exp(-0.5 * (quad - s * s / q))).T @ amp[sub]
            out[m] += acc
    return out


def _build_case(path, n_splats):
    """A real capture, subsampled deterministically, encoded once.

    Both paths are then run against THIS — same scene, same codebooks,
    same cell membership — so any deviation is the kernel's and not the
    setup's.
    """
    from holo.capture import (
        band_codebooks,
        build_scene,
        encode_bands,
        mass_mode,
        slice_grid,
    )
    from holo.spectral import SplatScene

    full, smax_full, box = build_scene(path, verbose=False)
    rng = np.random.default_rng(0)
    n = min(n_splats, full.n)
    idx = np.sort(rng.choice(full.n, n, replace=False))
    scene = SplatScene(full.mu[idx], full.cov[idx], full.amp[idx])
    print(f"{os.path.basename(path)}: {full.n:,} splats, "
          f"verifying on {scene.n:,}")

    books = band_codebooks(np.random.default_rng(42))
    bundles, members = encode_bands(scene, smax_full[idx], books,
                                    verbose=False)
    y = mass_mode(scene.mu[:, 1], scene.amp[:, 0], box[1])
    pts, _ = slice_grid((0, box[0]), (0, box[2]), "y", y)
    # a slice grid is ~50k px; the NumPy reference for exact_slice is
    # the slow half of this script, so thin it rather than the scene
    pts = np.ascontiguousarray(pts[::4])
    return scene, bundles, members, books, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--splats", type=int, default=40000)
    ap.add_argument("--res", type=int, default=96)
    args = ap.parse_args()

    from holo import capture as cap

    scene, bundles, members, books, pts = _build_case(args.scene, args.splats)
    view, center, half = [1.0, 0.0, 0.25], [0.5, 0.5, 0.5], 0.5
    freqs = books["fine"][0]
    S = bundles["fine"][next(iter(bundles["fine"]))]

    import bench.cuda_backend as cb
    from holo import accel

    assert not accel.active(), (
        "MLX is active; this box should be NumPy-vs-CUDA. "
        "Set HDC_BACKEND=numpy.")

    def measure():
        """Every patched function, on identical inputs, timed."""
        out, times = {}, {}
        for name, fn in [
            ("spectral_bundle",
             lambda: cap.spectral_bundle(scene, freqs, chunk=512)),
            ("exact_slice",
             lambda: cap.exact_slice(pts, scene, members)),
            ("exact_xray",
             lambda: cap.exact_xray(scene, members, view, center, half,
                                    args.res)),
            ("decode_slice",
             lambda: cap.decode_slice(pts, bundles, books)),
            ("readout", lambda: accel.readout(pts, freqs, S)),
        ]:
            t = time.perf_counter()
            out[name] = fn()
            times[name] = time.perf_counter() - t
        return out, times

    def report(label, times):
        total = sum(times.values())
        print(f"{label:16s} {total:5.1f}s  " + ", ".join(
            f"{k.split('_')[-1]} {v:.1f}" for k, v in times.items()))
        return total

    ref, t_ref = measure()
    numpy_total = report("numpy reference:", t_ref)
    cb.install()
    got, t_got = measure()
    cuda_total = report("cuda:", t_got)

    # -- exact_xray is judged against float64, not against stock ------
    # Its exponent, d^T S^-1 d - s^2/q, is a difference of two large
    # nearly-equal float32 quantities, and writing it that way loses
    # most of the mantissa. Both paths now evaluate the equivalent
    # rank-1 downdate instead (#93), so they agree — but "agrees with
    # stock" was never the right question here and still isn't: it
    # passes just as happily when both are wrong together, which is
    # exactly the state this comparison found the code in. Scoring each
    # against an independent float64 evaluation catches that; CUDA must
    # be no worse than stock.
    x64 = _xray_float64(scene, members, view, center, half, args.res)
    d_stock = rel(ref["exact_xray"], x64)
    d_cuda = rel(got["exact_xray"], x64)
    print(f"\nexact_xray vs float64:  stock {d_stock:.2e}, "
          f"cuda {d_cuda:.2e}  "
          f"({'cuda better' if d_cuda <= d_stock else 'CUDA WORSE'})")

    # -- verdict ------------------------------------------------------
    print(f"\n{'kernel':18s} {'max rel dev':>12s} {'numpy':>9s} "
          f"{'cuda':>9s} {'speedup':>9s}  verdict")
    bad = []
    for k, want in ref.items():
        if k == "exact_xray":
            d, ok = d_cuda, d_cuda <= max(d_stock, TOL)
        else:
            d = rel(got[k], want)
            ok = d <= TOL
        tn, tc = t_ref[k], t_got[k]
        if not ok:
            bad.append((k, d))
        print(f"{k:18s} {d:12.2e} {tn:8.2f}s {tc:8.2f}s "
              f"{tn / max(tc, 1e-9):8.1f}x  {'ok' if ok else 'FAIL'}")
    print(f"\ntotal {numpy_total:.1f}s -> {cuda_total:.1f}s "
          f"({numpy_total / max(cuda_total, 1e-9):.1f}x), bar {TOL:.0e}")
    if bad:
        print("\nFAILED: " + ", ".join(f"{k} at {d:.2e}" for k, d in bad))
        return 1
    print("\nall kernels agree with NumPy within the cross-backend bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
