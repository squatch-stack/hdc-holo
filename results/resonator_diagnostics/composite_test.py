"""One-off diagnostic for results/resonator_capture.md.

Two candidate objects in one physical frame: the gun where it is in
wilsons-creek, plus the cairn's splats placed elsewhere in the same cube.
A prototype of two cannon instances must prefer the gun's location over the
cairn's; a foreign prototype should not; a cairn instance should prefer the
cairn. Run from a checkout with bench/ importable, SCENES pointing at the
gallery .spz files, GPU via bench.cuda_backend.
"""

import os
from unittest.mock import patch

import numpy as np

import bench.cuda_backend as cb
from bench.place_recognition import (
    build_scene_fixed,
    correlate,
    crop_box,
    fingerprint,
    translation_grid,
)
from bench.resonator_capture import _center, prototype
from holo.capture import ALPHA_MIN, load_scene_file, weighted_quantile
from holo.spectral import SplatScene, sample_frequencies

SC = os.environ.get("SCENES", "scenes") + "/"
cb.install()
rng = np.random.default_rng(0)
sigma_units = 0.5
lo, extent = crop_box(SC + "wilsons-creek.spz")
sigma = sigma_units / extent
freqs = sample_frequencies(8192, 3, 1 / sigma, rng)
grid = translation_grid(32, 0.5) + 0.5


def in_frame(name, place_at=None):
    """A capture's splats in wilsons-creek's cube, optionally moved to a box point."""
    pos, scale, rgba, quat = load_scene_file(SC + name + ".spz")
    if place_at is not None:
        keep = rgba[:, 3] >= ALPHA_MIN
        a = rgba[keep, 3]
        c = np.array([weighted_quantile(pos[keep, i], a, 0.5) for i in range(3)])
        pos = pos - c + (lo + np.asarray(place_at) * extent)

    with patch(
        "bench.place_recognition.load_scene_file", return_value=(pos, scale, rgba, quat)
    ):
        scene, _, _ = build_scene_fixed(SC + name + ".spz", lo, extent)
    return scene


wc = in_frame("wilsons-creek")
cairn_at = np.array([0.2, 0.5, 0.2])
cairn = in_frame("redrock-cairn", place_at=cairn_at)
gun = in_frame("wilsons-creek-gun")
gun_at = np.average(gun.mu, axis=0, weights=gun.amp[:, 0])
composite = SplatScene(
    *(
        np.concatenate([getattr(s, a) for s in (wc, cairn)])
        for a in ("mu", "cov", "amp")
    )
)
S = fingerprint(composite, freqs, sigma)
print(
    "gun at %s, cairn placed at %s (box units); extent %.1f"
    % (np.round(gun_at, 3), cairn_at, extent)
)


def codeword(name):
    scene = in_frame(name, place_at=[0.5, 0.5, 0.5])
    return _center(scene, freqs, sigma)[0]


cands = {
    "cannon": codeword("cannon"),
    "rlib-cannon": codeword("research-library-cannon"),
    "cairn": codeword("redrock-cairn"),
    "saguaro": codeword("saguaro"),
    "oak": codeword("oak"),
}
probes = {
    "prototype(cannon, rlib-cannon)": prototype(
        [cands["cannon"], cands["rlib-cannon"]]
    ),
    "cannon alone": cands["cannon"],
    "rlib-cannon alone": cands["rlib-cannon"],
    "prototype(saguaro, oak) [foreign]": prototype([cands["saguaro"], cands["oak"]]),
    "cairn alone [the other object]": cands["cairn"],
    "gun itself [ceiling]": _center(gun, freqs, sigma)[0],
}
print("%-36s %6s %8s %8s  %s" % ("probe", "score", "d(gun)", "d(cairn)", "nearer"))
for label, code in probes.items():
    for w in (1.0,):
        score, t = correlate(code, S, freqs, grid, w)
        dg, dc = np.linalg.norm(t - gun_at), np.linalg.norm(t - cairn_at)
        print(
            "%-36s %6.3f %8.3f %8.3f  %s"
            % (label, score, dg, dc, "gun" if dg < dc else "cairn")
        )
