"""Cross-hardware kernel comparison driver for bench/holo_bench_job.py.

Builds the reference workload (capture-shaped: xfine-band scales,
mixture codebook, premultiplied RGBA channels), runs the SAME job
script on every local backend, and merges any remote bench.json (e.g.
from a sandboxed GPU job runner — see bench/RECIPE.md) into one table.
Checksums must agree across backends before clocks are compared.

    .venv/bin/python examples/run_gpu_bench.py                 # payload + local runs
    .venv/bin/python examples/run_gpu_bench.py --merge gpugate-*/bench.json

Artifacts land in ./gpubench/ (gitignored).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

# repo root: this driver lives in examples/, its assets do not
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "gpubench")
JOB = os.path.join(ROOT, "bench", "holo_bench_job.py")

N, P, D, C, CHUNK = 131072, 262144, 32768, 4, 16384
S_LO, S_CAP = 0.002, 0.004                      # the xfine band's range


def make_payload(path):
    sys.path.insert(0, ROOT)
    from holo.capture import quat_to_rot
    from holo.spectral import decode_weights, sample_frequencies

    rng = np.random.default_rng(20260826)
    mu = rng.uniform(0.05, 0.95, (N, 3)).astype(np.float32)
    q = rng.standard_normal((N, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rots = quat_to_rot(q).astype(np.float32)
    axes = rng.uniform(S_LO, S_CAP, (N, 3)).astype(np.float32)
    cov = np.einsum("nij,nj,nkj->nik", rots, axes**2, rots).astype(np.float32)
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    cov6 = np.stack([cov[:, i, j] for i, j in pairs], 1).astype(np.float32)
    norm = ((2 * np.pi) ** 1.5
            * np.sqrt(np.linalg.det(cov.astype(np.float64)))).astype(np.float32)
    alpha = rng.uniform(0.3, 1.0, (N, 1)).astype(np.float32)
    amp = np.concatenate([alpha, alpha * rng.uniform(0, 1, (N, 3))],
                         axis=1).astype(np.float32)

    rho = list(1.0 / np.geomspace(S_LO, S_CAP, 4))
    freqs = sample_frequencies(D, 3, rho, np.random.default_rng(7))
    weights = decode_weights(freqs, rho)
    wq = np.stack([freqs[:, i] * freqs[:, j] * (1.0 if i == j else 2.0)
                   for i, j in pairs], 1).astype(np.float32)

    near = mu[rng.integers(0, N, P // 2)] \
        + 0.004 * rng.standard_normal((P // 2, 3)).astype(np.float32)
    uniform = rng.uniform(0, 1, (P - P // 2, 3))
    points = np.concatenate([near, uniform]).astype(np.float32)

    np.savez_compressed(path, mu=mu, cov6=cov6, norm=norm, amp=amp,
                        freqs=freqs, wq=wq, weights=weights, points=points,
                        chunk=CHUNK, reps_fast=3, reps_slow=1)
    print(f"payload: {path} ({os.path.getsize(path) / 1048576:.1f} MB) — "
          f"{N:,} splats, {P:,} points, d={D:,}, c={C}")


def run_local(payload, backend):
    out = os.path.join(OUT, f"bench-{backend}.json")
    print(f"running local backend: {backend} ...")
    subprocess.run([sys.executable, JOB, "--backend", backend,
                    "--payload", payload, "--out", out],
                   check=True, stdout=subprocess.DEVNULL)
    return out


def merge(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows.append(json.load(f))
    ref = rows[0]
    scale = abs(ref["check_readout_sum"]) + 1e-9
    for r in rows[1:]:
        drift = abs(r["check_readout_sum"] - ref["check_readout_sum"]) / scale
        assert drift < 1e-3, \
            f"checksum mismatch: {r['backend']} vs {ref['backend']} ({drift:.2e})"
    print("\nchecksums agree across all backends — clocks are comparable\n")
    base = next((r for r in rows if r["backend"] == "numpy"), None)
    print(f"{'backend':>11} {'device':<28} {'encode s':>9} {'readout s':>10} "
          f"{'Msplat/s':>9} {'Mpoint/s':>9} {'vs numpy':>9}")
    for r in sorted(rows, key=lambda r: r["encode_s"]):
        if base is None or r is base:
            speed = "—"
        else:
            ratio = ((base["encode_s"] + base["readout_s"])
                     / (r["encode_s"] + r["readout_s"]))
            speed = f"{ratio:.0f}x"
        print(f"{r['backend']:>11} {r['device'][:28]:<28} "
              f"{r['encode_s']:>9.2f} {r['readout_s']:>10.2f} "
              f"{r['encode_msplat_per_s']:>9.2f} "
              f"{r['readout_mpoint_per_s']:>9.2f} {speed:>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-payload", action="store_true",
                    help="only build gpubench/scene.npz")
    ap.add_argument("--merge", nargs="*", default=[],
                    help="extra bench.json files (e.g. from a remote job)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    payload = os.path.join(OUT, "scene.npz")
    if not os.path.exists(payload):
        make_payload(payload)
    if args.make_payload:
        return

    locals_ = []
    try:
        import mlx.core  # noqa: F401
        locals_.append(run_local(payload, "mlx"))
    except ImportError:
        pass
    locals_.append(run_local(payload, "numpy"))
    merge(locals_ + args.merge)


if __name__ == "__main__":
    main()
