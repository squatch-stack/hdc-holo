"""holo kernel benchmark — one file, four backends, identical math.

Runs the two kernels that dominate every holo pipeline (see
docs/backend.md) on whichever accelerator is present, timing them on a
fixed workload shipped as scene.npz:

    encode:  S_re = (amp*norm)^T @ (env * cos(mu F^T))          (per chunk)
             S_im = -(amp*norm)^T @ (env * sin(mu F^T))
             env  = exp(-0.5 * cov6 @ wq^T)
    readout: out  = cos(P F^T) @ (w*S_re)^T - sin(P F^T) @ (w*S_im)^T

Backends, auto-selected best-first (override with --backend):
    torch-cuda   (NVIDIA)
    cupy         (NVIDIA; the job-runner case when torch is absent)
    mlx          (Apple silicon)
    numpy        (the reference everywhere)

Precision contract: TF32 is DISABLED on the CUDA paths. Blackwell/
Ampere fp32 matmuls default to TF32 tensor cores, which costs ~2 orders
of magnitude of relative error (1e-7 -> 1e-5) — discovered on the
studio's RTX 5090 during recipe bring-up. hdc-holo's cross-backend
agreement bar is float32 rounding, so the benchmark measures true-fp32
clocks; a TF32 mode would be a different (documented) trade.

Written to run inside a locked-down job sandbox: stdlib + numpy required,
everything else optional; reads ./scene.npz; writes ./bench.json. No
network, no paths outside the working directory. Float64 checksums of
the outputs ship in bench.json so cross-hardware runs can verify they
computed the same thing before comparing clocks.

Local use: python bench/holo_bench_job.py --backend mlx|numpy
Payload:   built by examples/run_gpu_bench.py (sizes chosen to fit a 12 GB
           VRAM budget with chunked transients).
"""

import argparse
import json
import sys
import time

import numpy as np


def pick_backend(name):
    if name in (None, "torch-cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = False   # see docstring
                torch.backends.cudnn.allow_tf32 = False
                return "torch-cuda", torch
        except ImportError:
            pass
        if name == "torch-cuda":
            sys.exit("torch with CUDA requested but not available")
    if name in (None, "cupy"):
        try:
            import cupy
            cupy.cuda.Device(0).compute_capability
            return "cupy", cupy
        except Exception:
            if name == "cupy":
                sys.exit("cupy requested but not available")
    if name in (None, "mlx"):
        try:
            import mlx.core as mx
            return "mlx", mx
        except ImportError:
            if name == "mlx":
                sys.exit("mlx requested but not available")
    return "numpy", np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",
                    choices=["torch-cuda", "cupy", "mlx", "numpy"])
    ap.add_argument("--payload", default="scene.npz")
    ap.add_argument("--out", default="bench.json")
    args = ap.parse_args()

    backend, lib = pick_backend(args.backend)
    z = np.load(args.payload)
    mu, cov6, norm, amp = z["mu"], z["cov6"], z["norm"], z["amp"]
    freqs, wq, weights, points = z["freqs"], z["wq"], z["weights"], z["points"]
    chunk = int(z["chunk"])
    reps = int(z["reps_fast"]) if backend != "numpy" else int(z["reps_slow"])
    n, d, c = len(mu), freqs.shape[0], amp.shape[1]
    m_amp = (amp * norm[:, None]).astype(np.float32)

    if backend == "torch-cuda":
        dev = lib.device("cuda")
        T = lambda a: lib.from_numpy(np.ascontiguousarray(a)).to(dev)
        cos, sin, exp = lib.cos, lib.sin, lib.exp
        zeros = lambda s: lib.zeros(s, device=dev)
        sync = lib.cuda.synchronize
        back = lambda t: t.cpu().numpy()
        device_name = lib.cuda.get_device_name(0)
    elif backend == "cupy":
        # cupy fp32 GEMMs use cuBLAS default math mode (no TF32 unless
        # CUPY_TF32=1 is set in the environment — leave it unset)
        T = lib.asarray
        cos, sin, exp = lib.cos, lib.sin, lib.exp
        zeros = lambda s: lib.zeros(s, dtype=lib.float32)
        sync = lib.cuda.Stream.null.synchronize
        back = lib.asnumpy
        props = lib.cuda.runtime.getDeviceProperties(0)
        device_name = props["name"].decode()
    elif backend == "mlx":
        T = lib.array
        cos, sin, exp = lib.cos, lib.sin, lib.exp
        zeros = lib.zeros
        sync = lambda: None                 # eval() below forces
        back = np.array
        device_name = "Apple Metal (MLX)"
    else:
        T = lambda a: a
        cos, sin, exp = np.cos, np.sin, np.exp
        zeros = np.zeros
        sync = lambda: None
        back = lambda a: a
        import platform
        device_name = f"NumPy on {platform.processor() or platform.machine()}"

    mF, mWq = T(freqs), T(wq)

    def encode():
        br, bi = zeros((c, d)), zeros((c, d))
        for lo in range(0, n, chunk):
            ph = T(mu[lo:lo + chunk]) @ mF.T
            env = exp(-0.5 * (T(cov6[lo:lo + chunk]) @ mWq.T))
            ma = T(m_amp[lo:lo + chunk])
            br = br + ma.T @ (env * cos(ph))
            bi = bi - ma.T @ (env * sin(ph))
        if backend == "mlx":
            lib.eval(br, bi)
        sync()
        return br, bi

    def readout(br, bi):
        w = T(weights.astype(np.float32))
        wr, wi = (br * w).T, (bi * w).T
        outs = []
        for lo in range(0, len(points), chunk):
            ph = T(points[lo:lo + chunk]) @ mF.T
            o = cos(ph) @ wr - sin(ph) @ wi
            outs.append(o)
        if backend == "mlx":
            lib.eval(*outs)
        sync()
        return outs

    def clock(fn, *a):
        best = None
        result = None
        for _ in range(reps):
            t0 = time.perf_counter()
            result = fn(*a)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        return best, result

    encode()                                        # warmup / JIT / cache
    t_enc, (br, bi) = clock(encode)
    readout(br, bi)                                 # warmup
    t_read, outs = clock(readout, br, bi)

    check_bundle = float(np.float64(back(br).astype(np.float64).sum())
                         + np.float64(back(bi).astype(np.float64).sum()))
    check_out = float(sum(back(o).astype(np.float64).sum() for o in outs))

    report = {
        "backend": backend,
        "device": device_name,
        "workload": {"splats": n, "points": len(points), "dim": d,
                     "channels": c, "chunk": chunk, "reps": reps},
        "encode_s": round(t_enc, 4),
        "readout_s": round(t_read, 4),
        "encode_msplat_per_s": round(n / t_enc / 1e6, 3),
        "readout_mpoint_per_s": round(len(points) / t_read / 1e6, 3),
        "check_bundle_sum": check_bundle,
        "check_readout_sum": check_out,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
