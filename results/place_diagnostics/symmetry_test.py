"""One-off diagnostic for results/place_recognition.md (run from a checkout with
bench/ importable, SCENES pointing at the gallery .spz files, GPU via bench.cuda_backend).
Is the shared phase the centred envelope (a symmetric blob has a real spectrum)?
Measure the fraction of near-real components per fingerprint, and re-score with
LOW |w| masked (drop small |w|) rather than small horizontal radius."""
import numpy as np
import bench.cuda_backend as cb; cb.install()
from bench import place_recognition as place
from holo.capture import build_scene
from holo.spectral import sample_frequencies, SplatScene
import os
SC = os.environ.get("SCENES", "scenes") + "/"
rng = np.random.default_rng(0); sigma = 0.025
freqs = sample_frequencies(8192, 3, 1 / sigma, rng); grid = place.translation_grid(48, 0.25)
wmag = np.linalg.norm(freqs, axis=1); wmax = wmag.max()
print("|w| quantiles 10/50/90:", np.percentile(wmag, [10, 50, 90]).round(1), "max", wmax.round(1))
F = {}
for n in ("wilsons-creek", "redrock", "research-library", "oak", "cannon", "brookline-station"):
    scene, _, _ = build_scene(SC + n + ".spz", verbose=False); F[n] = place.fingerprint(scene, freqs, sigma)
    uni = SplatScene(rng.uniform(0, 1, scene.mu.shape).astype(np.float32), scene.cov.copy(), scene.amp.copy())
    F["uniform-" + n] = place.fingerprint(uni, freqs, sigma)
for n, f in F.items():
    ph = np.angle(f); live = np.abs(f) > 1e-3 * np.abs(f).max()
    print("%-26s near-real (|phase|<pi/4) %.2f  live %.2f  |F| decay: median|F| at |w|<q50 / >q50 = %.1f" % (
        n, np.mean(np.abs(ph[live]) < np.pi / 4), live.mean(),
        np.median(np.abs(f)[wmag < np.median(wmag)]) / max(np.median(np.abs(f)[wmag >= np.median(wmag)]), 1e-12)), flush=True)
pairs = [("wilsons-creek", "redrock"), ("wilsons-creek", "research-library"), ("oak", "redrock"), ("wilsons-creek", "cannon"), ("uniform-wilsons-creek", "uniform-redrock")]
for frac in (0.0, 0.2, 0.4, 0.6):
    keep = wmag >= frac * wmax
    sc = lambda a, b, w: place.correlate(np.where(keep, a, 0), np.where(keep, b, 0), freqs, grid, w)[0]
    print("drop |w|<%.1f max kept=%5d | " % (frac, keep.sum()) + " ".join("%s-%s %.2f/%.2f" % (a[:7], b[:7], sc(F[a], F[b], 0), sc(F[a], F[b], 1)) for a, b in pairs), flush=True)
