"""Example driver: a real pretrained splat scene as holographic bundles.

The pipeline lives in `holo.capture` (loaders, mass-centered crop,
scale-banded cell encoding, cell-local decode/ground truth, X-ray
projection with mip encodes); this script points it at a capture,
renders slices and X-ray views against exact ground truth, and writes
the evidence figures to results/. See docs/real-scenes.md.

Usage:
    examples/run_real_scene.py [stats] [data/iphone/redrock.ply |
                               data/scan-tucson.spz | data/train.splat]
"""

import os
import sys
import time

import numpy as np

from holo.capture import (
    BANDS,
    DIM,
    DIM_R,
    RENDER_BANDS,
    SIGMA_MIP,
    band_codebooks,
    build_scene,
    decode_slice,
    encode_bands,
    exact_slice,
    exact_xray,
    load_scene_file,
    mass_mode,
    render_mip,
    render_xray,
    slice_grid,
)

# repo root: this driver lives in examples/, its assets do not
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DEFAULT_SCENE = os.path.join(ROOT, "data", "iphone", "redrock.ply")


def stats(path):
    pos, scale, rgba, _quat = load_scene_file(path)
    n = len(pos)
    print(f"{n:,} splats  ({os.path.basename(path)})")
    alpha = rgba[:, 3]
    print("opacity fraction >=0.05/0.1/0.25/0.5:",
          " ".join(f"{np.mean(alpha >= t):.2f}" for t in (0.05, 0.1, 0.25, 0.5)))
    for ax in range(3):
        q = np.percentile(pos[:, ax], [1, 5, 50, 95, 99])
        print(f"pos axis {ax}: p1..p99 = " +
              " ".join(f"{v:8.2f}" for v in q))
    s = scale.ravel()
    q = np.percentile(s, [1, 5, 25, 50, 75, 95, 99, 99.9])
    print("axis scales p1..p99.9:", " ".join(f"{v:.4f}" for v in q))
    print(f"scale min {s.min():.2e} max {s.max():.2e}")
    aspect = scale.max(axis=1) / np.maximum(scale.min(axis=1), 1e-9)
    print("anisotropy (max/min axis) p50/p90/p99:",
          " ".join(f"{v:.1f}" for v in np.percentile(aspect, [50, 90, 99])))


def to_rgb(field, ref):
    return np.clip(field[:, 1:4] / ref, 0.0, 1.0)


def main(path):
    t0 = time.time()
    scene, smax, box = build_scene(path)
    books = band_codebooks(np.random.default_rng(42))
    bundles, members = encode_bands(scene, smax, books)

    w = scene.amp[:, 0]
    y_slice = mass_mode(scene.mu[:, 1], w, box[1])
    x_slice = mass_mode(scene.mu[:, 0], w, box[0])
    print(f"slices at mass modes: top-down y={y_slice:.3f}, "
          f"side x={x_slice:.3f}")

    panels = []
    for title, (pts, shape) in [
            (f"top-down slice (y = {y_slice:.2f})",
             slice_grid((0, box[0]), (0, box[2]), "y", y_slice)),
            (f"side slice (x = {x_slice:.2f})",
             slice_grid((0, box[2]), (0, box[1]), "x", x_slice))]:
        t1 = time.time()
        truth = exact_slice(pts, scene, members)
        t2 = time.time()
        holo = decode_slice(pts, bundles, books)
        t3 = time.time()
        err = (np.linalg.norm(holo[:, 0] - truth[:, 0])
               / np.linalg.norm(truth[:, 0]))
        print(f"  {title}: {len(pts):,} px, GT {t2 - t1:.0f}s, "
              f"decode {t3 - t2:.0f}s, alpha rel err {err:.3f}")
        panels.append((title, truth, holo, shape, err))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAGE, INK, INK2 = "#f9f9f7", "#0b0b0b", "#52514e"
    n_cells = sum(len(b) for b in bundles.values())
    name = os.path.basename(path)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.patch.set_facecolor(PAGE)
    for col, (title, truth, holo, shape, err) in enumerate(panels):
        ref = np.percentile(truth[:, 0], 99)
        for row, (field, label) in enumerate([
                (truth, "ground truth (exact mixture)"),
                (holo, "holographic (chunked bundles)")]):
            ax = axes[row, col]
            ax.imshow(to_rgb(field, ref).reshape(*shape, 3), origin="lower",
                      aspect="equal")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{label} — {title}", fontsize=10, color=INK)
        axes[1, col].set_xlabel(f"alpha-channel rel. error {100 * err:.0f}%",
                                fontsize=9, color=INK2)
    fig.suptitle(f"{name}: {scene.n:,} real splats in {n_cells} cell "
                 f"bundles (d = {DIM:,}, {len(BANDS)} scale bands, "
                 "mixture codebooks)", fontsize=12, color=INK)
    fig.tight_layout()
    out = os.path.join(RESULTS,
                       f"real_{os.path.splitext(name)[0]}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PAGE)
    print(f"saved {out}")

    # -- X-ray projections: closed-form ray integrals from the bundles --
    mip = render_mip(scene, SIGMA_MIP)
    smax_r = np.sqrt(smax ** 2 + SIGMA_MIP ** 2)
    r_books = band_codebooks(np.random.default_rng(43), RENDER_BANDS,
                             DIM_R, s_floor=SIGMA_MIP)
    print(f"render mip: sigma_b = {SIGMA_MIP}, d = {DIM_R:,}")
    r_bundles, r_members = encode_bands(mip, smax_r, r_books,
                                        RENDER_BANDS, DIM_R)
    center, half, T, res = [0.5, 0.5, 0.5], 0.5, 2.0, 176
    views = [("view (1, 0, 0.25)", [1.0, 0.0, 0.25]),
             ("view (1, 0, 1)", [1.0, 0.0, 1.0])]
    xpanels = []
    for vtitle, view in views:
        t1 = time.time()
        sharp = exact_xray(scene, members, view, center, half, res)
        mip_gt = exact_xray(mip, r_members, view, center, half, res,
                            bands=RENDER_BANDS)
        t2 = time.time()
        holo = render_xray(r_bundles, r_books, view, center, half, res, T,
                           bands=RENDER_BANDS)
        t3 = time.time()
        err = (np.linalg.norm(holo[:, 0] - mip_gt[:, 0])
               / np.linalg.norm(mip_gt[:, 0]))
        print(f"  x-ray {vtitle}: GT {t2 - t1:.0f}s, render {t3 - t2:.0f}s, "
              f"alpha rel err vs mip {err:.3f}")
        xpanels.append((vtitle, sharp, mip_gt, holo, err))

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 15.5))
    fig.patch.set_facecolor(PAGE)
    for col, (vtitle, sharp, mip_gt, holo, err) in enumerate(xpanels):
        ref = np.percentile(mip_gt[:, 0], 99.5)
        for row, (field, label) in enumerate([
                (sharp, "analytic line integrals, full detail"),
                (mip_gt, f"analytic, mip σ_b = {SIGMA_MIP}"),
                (holo, "rendered from the mip bundles")]):
            ax = axes[row, col]
            ax.imshow(to_rgb(field, ref).reshape(res, res, 3),
                      origin="lower", aspect="equal")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{label} — {vtitle}", fontsize=10, color=INK)
        axes[2, col].set_xlabel(
            f"alpha-channel rel. error vs mip {100 * err:.0f}%",
            fontsize=9, color=INK2)
    fig.suptitle(f"{name}: orthographic X-ray views straight from the "
                 "cell bundles — no ray marching, no sort",
                 fontsize=12, color=INK)
    fig.tight_layout()
    xout = os.path.join(RESULTS,
                        f"real_{os.path.splitext(name)[0]}_xray.png")
    fig.savefig(xout, dpi=150, bbox_inches="tight", facecolor=PAGE)
    print(f"saved {xout}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    args = list(sys.argv[1:])
    do_stats = "stats" in args
    paths = [a for a in args if a != "stats"] or [DEFAULT_SCENE]
    if do_stats:
        stats(paths[0])
    else:
        main(paths[0])
