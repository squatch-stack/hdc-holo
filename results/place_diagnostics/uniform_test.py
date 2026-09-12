"""One-off diagnostic for results/place_recognition.md (run from a checkout with
bench/ importable, SCENES pointing at the gallery .spz files, GPU via bench.cuda_backend).
Is the whitened 0.82 between two unrelated wide captures the box? Compare each
wide parent against a uniform fill of its own box carrying its own cov and amp."""
import json, sys
import numpy as np
import bench.cuda_backend as cb; cb.install()
from bench import place_recognition as place
from holo.capture import build_scene
from holo.spectral import SplatScene, sample_frequencies
import os
SC = os.environ.get("SCENES", "scenes") + "/"
names = ["wilsons-creek", "redrock", "research-library", "oak", "cannon", "brookline-station"]
rng = np.random.default_rng(0)
sigma = 0.025
freqs = sample_frequencies(8192, 3, 1 / sigma, rng)
grid = place.translation_grid(48, 0.25)
fps, ufps = {}, {}
for n in names:
    scene, _, _ = build_scene(SC + n + ".spz", verbose=False)
    fps[n] = place.fingerprint(scene, freqs, sigma)
    uni = SplatScene(rng.uniform(0, 1, scene.mu.shape).astype(np.float32), scene.cov.copy(), scene.amp.copy())
    ufps[n] = place.fingerprint(uni, freqs, sigma)
out = {}
for n in names:
    for w in (0.0, 1.0):
        out[f"{n} vs own-uniform w={w}"] = place.correlate(fps[n], ufps[n], freqs, grid, w)[0]
for a, b in [("wilsons-creek", "redrock"), ("wilsons-creek", "research-library"), ("cannon", "brookline-station"), ("wilsons-creek", "cannon")]:
    for w in (0.0, 1.0):
        out[f"{a} vs {b} w={w}"] = place.correlate(fps[a], fps[b], freqs, grid, w)[0]
        out[f"uniform({a}) vs uniform({b}) w={w}"] = place.correlate(ufps[a], ufps[b], freqs, grid, w)[0]
for k, v in out.items(): print(f"{v:6.3f}  {k}")
json.dump(out, open("uniform_test.json", "w"), indent=1)
