"""One-off diagnostic for results/place_recognition.md.

Run from a checkout with bench/ importable, SCENES pointing at the gallery
.spz files, GPU via bench.cuda_backend.
Is the wide-capture block numerical? Fingerprints three ways (CUDA float32,
NumPy float32, NumPy float64), then two follow-ups: a uniform 2-D sheet cut by
the same crop cube, and apodised box faces."""

import os
import time

import numpy as np

import bench.cuda_backend as cb
from bench import place_recognition as place
from holo import spectral
from holo.capture import build_scene, render_mip
from holo.spectral import SplatScene, sample_frequencies

SC = os.environ.get("SCENES", "scenes") + "/"
cb.install()
rng = np.random.default_rng(0)
sigma = 0.025
freqs = sample_frequencies(8192, 3, 1 / sigma, rng)
grid = place.translation_grid(48, 0.25)


def bundle64(scene, freqs, chunk=4096):
    f = freqs.astype(np.float64)
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    wq = np.stack([f[:, i] * f[:, j] * (1.0 if i == j else 2.0) for i, j in pairs], 1)
    out = np.zeros(len(f), np.complex128)
    for lo in range(0, scene.n, chunk):
        mu = scene.mu[lo : lo + chunk].astype(np.float64)
        cov = scene.cov[lo : lo + chunk].astype(np.float64)
        amp = scene.amp[lo : lo + chunk, 0].astype(np.float64)
        norm = (2 * np.pi) ** 1.5 * np.sqrt(np.linalg.det(cov))
        cq = np.stack([cov[:, i, j] for i, j in pairs], 1)
        env = np.exp(-0.5 * (cq @ wq.T))
        vec = (norm[:, None] * env) * np.exp(-1j * (mu @ f.T))
        out += amp @ vec
    return out


def three_ways(scene):
    alpha = render_mip(SplatScene(scene.mu, scene.cov, scene.amp[:, :1]), sigma)
    t = time.time()
    cuda = spectral.spectral_bundle(alpha, freqs)[0]
    tc = time.time() - t
    saved = spectral._accel
    spectral._accel = None
    t = time.time()
    np32 = spectral.spectral_bundle(alpha, freqs)[0]
    tn = time.time() - t
    spectral._accel = saved
    t = time.time()
    f64 = bundle64(alpha, freqs)
    t64 = time.time() - t
    print(f"    encode s: cuda {tc:.1f} numpy32 {tn:.1f} float64 {t64:.1f}", flush=True)
    return cuda, np32, f64


def whitened_t0(a, b):
    p = np.conj(a.astype(np.complex128)) * b.astype(np.complex128)
    live = np.abs(p) > 1e-6 * np.abs(p).max()
    return abs(np.mean(p[live] / np.abs(p[live])))


def report(name, f32, f64, tag):
    rel = np.abs(f32.astype(np.complex128) - f64) / np.maximum(np.abs(f64), 1e-300)
    mag = np.abs(f64) / np.abs(f64).max()
    edges = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0]
    cells = []
    for hi, lo in zip(edges[:-1], edges[1:]):
        m = (mag <= hi) & (mag > lo)
        cells.append(
            "%g-%g: n=%d med %.1e p90 %.1e"
            % (
                lo,
                hi,
                m.sum(),
                np.median(rel[m]) if m.any() else 0,
                np.percentile(rel[m], 90) if m.any() else 0,
            )
        )
    print(f"  {name} {tag} vs float64 | rel err by |F|/max: " + " | ".join(cells))
    print(
        f"  {name} {tag} vs float64 | frac rel err > 0.1: {np.mean(rel > 0.1):.4f}  "
        f"phase agreement (whitened, t=0): {whitened_t0(f32, f64):.4f}",
        flush=True,
    )


scenes = {}
F = {}
for n in ("wilsons-creek", "redrock", "cannon"):
    scene, _, _ = build_scene(SC + n + ".spz", verbose=False)
    scenes[n] = scene
    print(n, "n=%d" % scene.n, flush=True)
    F[n] = three_ways(scene)
    report(n, F[n][0], F[n][2], "cuda32")
    report(n, F[n][1], F[n][2], "numpy32")

print("\n== wc vs rr, whitened at t=0 (direct) and via the search (score raw/whitened)")
for k, tag in enumerate(("cuda32", "numpy32", "float64")):
    a, b = F["wilsons-creek"][k], F["redrock"][k]
    a32, b32 = a.astype(np.complex64), b.astype(np.complex64)
    print(
        "  %-8s t0-whitened %.3f | search raw %.3f whitened %.3f"
        % (
            tag,
            whitened_t0(a, b),
            place.correlate(a32, b32, freqs, grid, 0)[0],
            place.correlate(a32, b32, freqs, grid, 1)[0],
        ),
        flush=True,
    )
a, b = F["wilsons-creek"][2], F["cannon"][2]
print("  float64 wc vs cannon: t0-whitened %.3f" % whitened_t0(a, b))

print(
    "\n== uniform 2-D sheet cut by the same cube "
    "(own n, cov, amp; height = scene median y)"
)
sheets = {}
for n in ("wilsons-creek", "redrock"):
    s = scenes[n]
    mu = rng.uniform(0, 1, s.mu.shape).astype(np.float32)
    mu[:, 1] = np.median(s.mu[:, 1])
    sheets[n] = place.fingerprint(
        SplatScene(mu, s.cov.copy(), s.amp.copy()), freqs, sigma
    )
    own = F[n][0]
    print(
        "  %-14s sheet vs own scene: raw %.3f whitened %.3f"
        % (
            n,
            place.correlate(own, sheets[n], freqs, grid, 0)[0],
            place.correlate(own, sheets[n], freqs, grid, 1)[0],
        ),
        flush=True,
    )
print(
    "  sheet(wc) vs sheet(rr): raw %.3f whitened %.3f"
    % (
        place.correlate(sheets["wilsons-creek"], sheets["redrock"], freqs, grid, 0)[0],
        place.correlate(sheets["wilsons-creek"], sheets["redrock"], freqs, grid, 1)[0],
    )
)

print("\n== apodised box faces (Tukey 0.15 per axis on alpha) ")


def tukey(x, r=0.15):
    w = np.ones_like(x)
    lo = x < r
    hi = x > 1 - r
    w[lo] = 0.5 * (1 - np.cos(np.pi * x[lo] / r))
    w[hi] = 0.5 * (1 - np.cos(np.pi * (1 - x[hi]) / r))
    return w


apo = {}
for n in ("wilsons-creek", "redrock", "cannon"):
    s = scenes[n]
    w = tukey(s.mu[:, 0]) * tukey(s.mu[:, 1]) * tukey(s.mu[:, 2])
    apo[n] = place.fingerprint(
        SplatScene(s.mu, s.cov, s.amp * w[:, None]), freqs, sigma
    )
    print(
        "  %-14s apodised vs raw self: whitened %.3f"
        % (n, place.correlate(F[n][0], apo[n], freqs, grid, 1)[0]),
        flush=True,
    )
for a, b in (("wilsons-creek", "redrock"), ("wilsons-creek", "cannon")):
    print(
        "  apodised %s vs %s: raw %.3f whitened %.3f"
        % (
            a,
            b,
            place.correlate(apo[a], apo[b], freqs, grid, 0)[0],
            place.correlate(apo[a], apo[b], freqs, grid, 1)[0],
        )
    )
