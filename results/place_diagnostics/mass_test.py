"""One-off diagnostic for results/place_recognition.md (run from a checkout with
bench/ importable, SCENES pointing at the gallery .spz files, GPU via bench.cuda_backend)."""
import numpy as np
import bench.cuda_backend as cb; cb.install()
from bench import place_recognition as place
from holo.capture import build_scene
from holo.spectral import sample_frequencies, SplatScene
import os
SC = os.environ.get("SCENES", "scenes") + "/"
rng = np.random.default_rng(0); sigma = 0.025
freqs = sample_frequencies(8192, 3, 1 / sigma, rng); grid = place.translation_grid(48, 0.25)
top = {}
for n in ("wilsons-creek", "redrock", "research-library", "oak", "cannon"):
    scene, smax, _ = build_scene(SC + n + ".spz", verbose=False)
    vol = np.sqrt(np.linalg.det(scene.cov.astype(np.float64))); mass = scene.amp[:, 0] * vol
    order = np.argsort(mass)[::-1]; cum = np.cumsum(mass[order]) / mass.sum()
    k1 = max(1, len(mass) // 100)
    print("%-17s n=%7d  mass share: top10 %.3f top100 %.3f top1%% %.3f | smax==S_HI frac %.4f  top10 mu (box units): %s" % (
        n, len(mass), cum[9], cum[99], cum[k1 - 1], np.mean(smax >= 0.9999 * smax.max()), np.round(scene.mu[order[:5]], 2).tolist()), flush=True)
    keep = order[:k1]
    top[n] = (place.fingerprint(scene, freqs, sigma),
              place.fingerprint(SplatScene(scene.mu[keep], scene.cov[keep], scene.amp[keep]), freqs, sigma),
              place.fingerprint(SplatScene(scene.mu[order[k1:]], scene.cov[order[k1:]], scene.amp[order[k1:]]), freqs, sigma))
for n in top:
    f, t, r = top[n]
    print("%-17s full vs top1%% raw/whit %.2f/%.2f   full vs rest99%% %.2f/%.2f" % (n, *place.correlate(f, t, freqs, grid, 0)[:1], *place.correlate(f, t, freqs, grid, 1)[:1], *place.correlate(f, r, freqs, grid, 0)[:1], *place.correlate(f, r, freqs, grid, 1)[:1]), flush=True)
for a, b in (("wilsons-creek", "redrock"), ("wilsons-creek", "research-library")):
    print("%s-%s  top1%%-top1%% raw/whit %.2f/%.2f   rest-rest %.2f/%.2f" % (a[:3], b[:3], place.correlate(top[a][1], top[b][1], freqs, grid, 0)[0], place.correlate(top[a][1], top[b][1], freqs, grid, 1)[0], place.correlate(top[a][2], top[b][2], freqs, grid, 0)[0], place.correlate(top[a][2], top[b][2], freqs, grid, 1)[0]), flush=True)
