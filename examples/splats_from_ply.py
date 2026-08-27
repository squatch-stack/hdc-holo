"""Phone capture -> holographic bundles -> queries and renders.

The minimal path through holo.capture: point it at a raw Gaussian
`.ply` (the INRIA 3DGS layout — what Scaniverse exports from a
phone; `.splat` and `.spz` load the same way), encode it into
per-cell complex64 bundles, then answer questions with inner
products: a color slice checked against the exact mixture, and an
orthographic X-ray view with no ray marching and no sort. The full
evidence driver with figures is examples/run_real_scene.py; the math and the
failure modes are docs/real-scenes.md.

    python examples/splats_from_ply.py [capture.ply]
"""

import sys
import time

import numpy as np

from holo.capture import (DIM_R, RENDER_BANDS, SIGMA_MIP, band_codebooks,
                          build_scene, decode_slice, encode_bands,
                          exact_slice, mass_mode, render_mip, render_xray,
                          slice_grid)

path = sys.argv[1] if len(sys.argv) > 1 else "data/iphone/redrock.ply"

# -- encode: splats -> band/cell bundles (this is the whole "index") --
scene, smax, box = build_scene(path)            # load, crop, normalize
books = band_codebooks(np.random.default_rng(42))
bundles, members = encode_bands(scene, smax, books)
n_cells = sum(len(b) for b in bundles.values())
mb = sum(b.nbytes for band in bundles.values() for b in band.values()) / 2**20
print(f"{scene.n:,} splats -> {n_cells} cell bundles ({mb:.0f} MB); "
      "the original splats are no longer needed.\n")

# -- query: a color slice, verified against the exact mixture --------
y = mass_mode(scene.mu[:, 1], scene.amp[:, 0], box[1])
pts, shape = slice_grid((0, box[0]), (0, box[2]), "y", y)
t0 = time.time()
holo_px = decode_slice(pts, bundles, books)     # inner products only
t1 = time.time()
truth = exact_slice(pts, scene, members)        # ground truth referee
err = (np.linalg.norm(holo_px[:, 0] - truth[:, 0])
       / np.linalg.norm(truth[:, 0]))
print(f"top-down slice at y={y:.2f}: {len(pts):,} px decoded in "
      f"{t1 - t0:.1f}s, alpha rel err {err:.0%} vs exact mixture")

# -- render: an X-ray view straight from bundles ---------------------
# Projections need their own encode at a mip level (blur = covariance
# addition) because a view only uses frequencies perpendicular to it.
mip = render_mip(scene, SIGMA_MIP)
r_books = band_codebooks(np.random.default_rng(43), RENDER_BANDS, DIM_R,
                         s_floor=SIGMA_MIP)
r_bundles, _ = encode_bands(mip, np.sqrt(smax**2 + SIGMA_MIP**2),
                            r_books, RENDER_BANDS, DIM_R)
img = render_xray(r_bundles, r_books, view=[1.0, 0.7, 0.3],
                  center=[0.5, 0.5, 0.5], half=0.5, res=160,
                  t_extent=2.0, bands=RENDER_BANDS)
print(f"x-ray view rendered: {img.shape[0]:,} px, each one inner product")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (field, sh, title) in zip(axes, [
            (truth, shape, "slice: exact ground truth"),
            (holo_px, shape, f"slice: from bundles ({err:.0%} err)"),
            (img, (160, 160), "x-ray view from bundles")]):
        rgb = np.clip(field[:, 1:4] / np.percentile(field[:, 0], 99),
                      0, 1) ** 0.7
        ax.imshow(rgb.reshape(*sh, 3), origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("out/example_splats.png", dpi=130, bbox_inches="tight")
    print("figure: out/example_splats.png")
except ImportError:
    pass                                        # figures are optional
