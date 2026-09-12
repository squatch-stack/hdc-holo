"""One-off diagnostic for results/place_recognition.md.

Run from a checkout with bench/ importable, SCENES pointing at the gallery
.spz files, GPU via bench.cuda_backend.
Is the block the capture footprint edge, which the mass-centred crop maps to
the same normalised radius in every capture? Radial mass profiles, then a
radial taper / central restriction, then the whitened wc-rr score."""

import os

import numpy as np

import bench.cuda_backend as cb
from bench import place_recognition as place
from holo.capture import build_scene
from holo.spectral import SplatScene, sample_frequencies

SC = os.environ.get("SCENES", "scenes") + "/"
cb.install()
rng = np.random.default_rng(0)
sigma = 0.025
freqs = sample_frequencies(8192, 3, 1 / sigma, rng)
grid = place.translation_grid(48, 0.25)
S = {
    n: build_scene(SC + n + ".spz", verbose=False)[0]
    for n in ("wilsons-creek", "redrock", "research-library", "oak", "cannon")
}
print(
    "radial mass profile: share of alpha mass in shells of normalised radius "
    "from the box centre (horizontal xz radius | full 3-D radius)"
)
for n, s in S.items():
    a = s.amp[:, 0]
    c = s.mu - 0.5
    rxz = np.hypot(c[:, 0], c[:, 2])
    r3 = np.linalg.norm(c, axis=1)
    hx = np.histogram(rxz, bins=np.linspace(0, 0.75, 16), weights=a)[0] / a.sum()
    h3 = np.histogram(r3, bins=np.linspace(0, 0.9, 16), weights=a)[0] / a.sum()
    print(
        "  %-17s xz: %s\n  %-17s 3d: %s"
        % (n, " ".join("%.2f" % v for v in hx), "", " ".join("%.2f" % v for v in h3))
    )
    print(
        "  %-17s y (height) deciles of mass: %s"
        % (
            "",
            np.round(
                np.percentile(
                    np.repeat(
                        s.mu[:, 1], np.maximum(1, (a / a.max() * 20).astype(int))
                    ),
                    [5, 25, 50, 75, 95],
                ),
                2,
            ),
        )
    )


def fp(s, w):
    return place.fingerprint(SplatScene(s.mu, s.cov, s.amp * w[:, None]), freqs, sigma)


raw = {n: fp(s, np.ones(s.n, np.float32)) for n, s in S.items()}
print(
    "\nwhitened wc-rr under radial edits (score raw/whitened; "
    "self = edited vs unedited wc):"
)
for label, fn in (
    (
        "central xz radius < 0.20",
        lambda s: (np.hypot(s.mu[:, 0] - 0.5, s.mu[:, 2] - 0.5) < 0.20).astype(
            np.float32
        ),
    ),
    (
        "central xz radius < 0.35",
        lambda s: (np.hypot(s.mu[:, 0] - 0.5, s.mu[:, 2] - 0.5) < 0.35).astype(
            np.float32
        ),
    ),
    (
        "gaussian radial taper s=0.2",
        lambda s: np.exp(
            -0.5 * (np.hypot(s.mu[:, 0] - 0.5, s.mu[:, 2] - 0.5) / 0.2) ** 2
        ).astype(np.float32),
    ),
    (
        "random half of the splats",
        lambda s: (rng.uniform(size=s.n) < 0.5).astype(np.float32),
    ),
    (
        "height band: middle 50% of mass in y",
        lambda s: (
            (s.mu[:, 1] > np.percentile(s.mu[:, 1], 25))
            & (s.mu[:, 1] < np.percentile(s.mu[:, 1], 75))
        ).astype(np.float32),
    ),
    (
        "above the median height",
        lambda s: (s.mu[:, 1] > np.percentile(s.mu[:, 1], 50)).astype(np.float32),
    ),
    (
        "below the median height",
        lambda s: (s.mu[:, 1] <= np.percentile(s.mu[:, 1], 50)).astype(np.float32),
    ),
):
    a, b = (
        fp(S["wilsons-creek"], fn(S["wilsons-creek"])),
        fp(S["redrock"], fn(S["redrock"])),
    )
    print(
        "  %-38s wc-rr %.3f/%.3f   self(wc) %.3f   wc-cannon whitened %.3f"
        % (
            label,
            place.correlate(a, b, freqs, grid, 0)[0],
            place.correlate(a, b, freqs, grid, 1)[0],
            place.correlate(a, raw["wilsons-creek"], freqs, grid, 1)[0],
            place.correlate(a, fp(S["cannon"], fn(S["cannon"])), freqs, grid, 1)[0],
        ),
        flush=True,
    )
