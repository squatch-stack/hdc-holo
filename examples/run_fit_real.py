"""Example driver: ridge-fitted vs forward-encoded real-scene bundles.

Encodes a real capture twice at IDENTICAL dimension and storage — once
by forward superposition (holo.capture.encode_bands), once by per-cell
ridge regression against the exact mixture (holo.capture.fit_cells, the
holo/fit.py idea applied cell-by-cell under a spectral prior) — then
decodes the same evidence slices from both and compares against ground
truth. Writes results/real_fit.png. See docs/real-scenes.md.

Usage:
    examples/run_fit_real.py [data/scan-tucson.spz | data/train.splat]
"""

import os
import sys
import time

import numpy as np

from holo import budget
from holo.capture import (
    band_codebooks,
    build_scene,
    decode_slice,
    encode_bands,
    exact_slice,
    fit_cells,
    mass_mode,
    slice_grid,
)

# repo root: this driver lives in examples/, its assets do not
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DEFAULT_SCENE = os.path.join(ROOT, "data", "scan-tucson.spz")


def to_rgb(field, ref):
    return np.clip(field[:, 1:4] / ref, 0.0, 1.0)


def main(path):
    t0 = time.time()
    scene, smax, box = build_scene(path)
    books = band_codebooks(np.random.default_rng(42))
    bundles, members = encode_bands(scene, smax, books)
    t1 = time.time()
    fitted = fit_cells(scene, members, books,
                       rng=np.random.default_rng(9))
    print(f"fit total {time.time() - t1:.0f}s "
          f"(forward encode was {t1 - t0:.0f}s)")

    w = scene.amp[:, 0]
    y_slice = mass_mode(scene.mu[:, 1], w, box[1])
    x_slice = mass_mode(scene.mu[:, 0], w, box[0])
    panels = []
    for title, (pts, shape) in [
            (f"top-down slice (y = {y_slice:.2f})",
             slice_grid((0, box[0]), (0, box[2]), "y", y_slice)),
            (f"side slice (x = {x_slice:.2f})",
             slice_grid((0, box[2]), (0, box[1]), "x", x_slice))]:
        truth = exact_slice(pts, scene, members)
        fwd = decode_slice(pts, bundles, books)
        fit = decode_slice(pts, fitted, books)
        errs = []
        for est in (fwd, fit):
            errs.append(float(np.linalg.norm(est[:, 0] - truth[:, 0])
                              / np.linalg.norm(truth[:, 0])))
        print(f"  {title}: forward {errs[0]:.3f} -> fitted {errs[1]:.3f} "
              f"({errs[0] / max(errs[1], 1e-9):.1f}x)")
        panels.append((title, truth, fwd, fit, shape, errs))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAGE, INK, INK2 = "#f9f9f7", "#0b0b0b", "#52514e"
    name = os.path.basename(path)
    fig, axes = plt.subplots(3, 2, figsize=(13, 12.5))
    fig.patch.set_facecolor(PAGE)
    for col, (title, truth, fwd, fit, shape, errs) in enumerate(panels):
        ref = np.percentile(truth[:, 0], 99)
        rows = [("ground truth (exact mixture)", truth, None),
                ("forward-encoded bundles", fwd, errs[0]),
                ("ridge-fitted bundles (same d, same bytes)", fit, errs[1])]
        for row, (label, field, err) in enumerate(rows):
            ax = axes[row, col]
            ax.imshow(to_rgb(field, ref).reshape(*shape, 3),
                      origin="lower", aspect="equal")
            ax.set_xticks([]), ax.set_yticks([])
            suffix = "" if err is None else f"  (rel err {100 * err:.0f}%)"
            ax.set_title(f"{label} — {title}{suffix}", fontsize=9.5,
                         color=INK)
    fig.suptitle(f"{name}: regression vs superposition at identical "
                 "dimension — per-cell fitting is sampling-limited at "
                 "real capture density", fontsize=12, color=INK)
    fig.tight_layout()
    out = os.path.join(RESULTS, "real_fit.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PAGE)
    print(f"saved {out}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with budget.heavy_run(6.0, "fit real", "--force-memory" in sys.argv):
        main(args[0] if args else DEFAULT_SCENE)
