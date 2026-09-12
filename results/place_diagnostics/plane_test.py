"""One-off diagnostic for results/place_recognition.md.

Run from a checkout with bench/ importable, SCENES pointing at the gallery
.spz files, GPU via bench.cuda_backend.
Is the shared whitened phase the ground plane? Mask near-vertical frequencies
(small horizontal radius) and re-score the unrelated wide pairs and the true pairs."""

import os

import numpy as np

from bench import place_recognition as place
from holo.capture import build_scene
from holo.spectral import sample_frequencies

SC = os.environ.get("SCENES", "scenes") + "/"
rng = np.random.default_rng(0)
sigma = 0.025
freqs = sample_frequencies(8192, 3, 1 / sigma, rng)
grid = place.translation_grid(48, 0.25)
rxz = np.hypot(freqs[:, 0], freqs[:, 2])
rmax = np.abs(freqs).max()


def fp(n, frame=None):
    if frame is None:
        scene, _, _ = build_scene(SC + n + ".spz", verbose=False)
    else:
        scene, _, _ = place.build_scene_fixed(
            SC + n + ".spz", *place.crop_box(SC + frame + ".spz")
        )
    return place.fingerprint(scene, freqs, sigma)


F = {
    n: fp(n)
    for n in (
        "wilsons-creek",
        "redrock",
        "research-library",
        "oak",
        "cannon",
        "brookline-station",
    )
}
pairs = [
    ("wilsons-creek", "redrock"),
    ("wilsons-creek", "research-library"),
    ("oak", "redrock"),
    ("wilsons-creek", "cannon"),
]
true = {
    "rr-cairn(cairn cube)": (
        fp("redrock", "redrock-cairn"),
        fp("redrock-cairn", "redrock-cairn"),
    ),
    "wc-gun(gun cube)": (
        fp("wilsons-creek", "wilsons-creek-gun"),
        fp("wilsons-creek-gun", "wilsons-creek-gun"),
    ),
    "springhouse(station cube)": (
        fp("brookline-station", "brookline-station"),
        fp("springhouse-outside", "brookline-station"),
    ),
}
print(
    "mask = drop frequencies with horizontal radius below frac*max; "
    "score raw / whitened"
)
for frac in (0.0, 0.1, 0.2, 0.3, 0.5):
    keep = rxz >= frac * rmax

    def sc(a, b, w, keep=keep):
        return place.correlate(
            np.where(keep, a, 0), np.where(keep, b, 0), freqs, grid, w
        )[0]

    line = "frac=%.1f kept=%5d | " % (frac, keep.sum())
    line += " ".join(
        "%s-%s %.2f/%.2f" % (a[:3], b[:3], sc(F[a], F[b], 0), sc(F[a], F[b], 1))
        for a, b in pairs
    )
    line += " || " + " ".join(
        "%s %.2f/%.2f" % (k, sc(a, b, 0), sc(a, b, 1)) for k, (a, b) in true.items()
    )
    print(line, flush=True)
