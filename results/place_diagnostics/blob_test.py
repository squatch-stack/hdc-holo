"""One-off diagnostic for results/place_recognition.md.

Run from a checkout with bench/ importable, SCENES pointing at the gallery
.spz files, GPU via bench.cuda_backend.
Confirm the central-blob mechanism: (1) delete the central blob and re-score
the wide block; (2) re-encode each wide capture in a cube scaled to the mass
core (smallest cube holding half the alpha mass, x margin) and re-score."""

import os

import numpy as np

import bench.cuda_backend as cb
from bench import place_recognition as place
from holo.capture import ALPHA_MIN, build_scene, load_scene_file, weighted_quantile
from holo.spectral import SplatScene, sample_frequencies

SC = os.environ.get("SCENES", "scenes") + "/"
cb.install()
rng = np.random.default_rng(0)
sigma = 0.025
freqs = sample_frequencies(8192, 3, 1 / sigma, rng)
grid = place.translation_grid(48, 0.25)
wide = ("wilsons-creek", "redrock", "research-library", "oak")
S = {n: build_scene(SC + n + ".spz", verbose=False)[0] for n in (*wide, "cannon")}


def fp(s, keep=None):
    if keep is None:
        return place.fingerprint(s, freqs, sigma)
    return place.fingerprint(
        SplatScene(s.mu[keep], s.cov[keep], s.amp[keep]), freqs, sigma
    )


def block(F, label):
    names = list(F)
    print(label)
    print("        " + " ".join("%7s" % n[:7] for n in names))
    for a in names:
        print(
            "%7s " % a[:7]
            + " ".join(
                "%7.3f" % place.correlate(F[a], F[b], freqs, grid, 1)[0] for b in names
            ),
            flush=True,
        )


block({n: fp(S[n]) for n in wide}, "whitened block, as encoded (reference):")
for r in (0.04, 0.08, 0.15):
    F = {}
    for n in wide:
        s = S[n]
        keep = np.linalg.norm(s.mu - 0.5, axis=1) > r
        F[n] = fp(s, keep)
        print(
            "  %-17s removed core r<%.2f: %.1f%% of splats, %.1f%% of alpha mass"
            % (
                n,
                r,
                100 * (1 - keep.mean()),
                100 * (1 - s.amp[~keep, 0].sum() / s.amp[:, 0].sum()),
            )
        )
    block(F, "whitened block with the central blob removed (r < %.2f):" % r)
print(
    "\nsubject-scaled frame: cube = margin x smallest half-mass cube, "
    "centred on the mass median"
)


def core_box(path, margin=3.0, share=0.5):
    pos, _, rgba, _ = load_scene_file(path)
    keep = rgba[:, 3] >= ALPHA_MIN
    pos, a = pos[keep], rgba[keep, 3]
    c = np.array([weighted_quantile(pos[:, i], a, 0.5) for i in range(3)])
    r = weighted_quantile(np.abs(pos - c).max(axis=1), a, share)
    return c - margin * r, float(2 * margin * r), r


F = {}
for n in (*wide, "cannon"):
    lo, ext, r = core_box(SC + n + ".spz")
    sc, _, _ = place.build_scene_fixed(SC + n + ".spz", lo, ext)
    print(
        "  %-17s half-mass radius %.2f scene units -> cube %.1f (was %s), "
        "kept %d splats"
        % (
            n,
            r,
            ext,
            {
                "wilsons-creek": 40.4,
                "redrock": "~44",
                "research-library": "?",
                "oak": "?",
                "cannon": 3.8,
            }[n],
            sc.n,
        )
    )
    F[n] = fp(sc)
block(F, "whitened block in subject-scaled frames:")
print("\ntrue pairs in the subject-scaled frame of the parent:")
for parent, crop in (
    ("wilsons-creek", "wilsons-creek-gun"),
    ("redrock", "redrock-cairn"),
):
    lo, ext, _ = core_box(SC + parent + ".spz")
    a = place.build_scene_fixed(SC + parent + ".spz", lo, ext)[0]
    b = place.build_scene_fixed(SC + crop + ".spz", lo, ext)[0]
    sa, sb = fp(a), fp(b)
    r0, o0 = place.correlate(sa, sb, freqs, grid, 0)
    r1, o1 = place.correlate(sa, sb, freqs, grid, 1)
    print(
        "  %-14s <- %-18s raw %.3f whitened %.3f offset %s"
        % (parent, crop, r0, r1, np.round(o1, 3).tolist()),
        flush=True,
    )
