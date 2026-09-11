"""Is the residual slice error a sampling artifact rather than a coding one?

`bench/adaptive_cells.py` took Wilson's Creek from 176.7% to 72.2% by
giving every cell a member budget, and then stopped: at 32 members per
cell — a quarter-percent of the bundle's 8,192 dimensions — the error
was still 72% and still 96% concentrated in the worst 1% of pixels.
Subdivision had run out of things to fix. Cropping the same scene
reached 20.1%, so something cropping changes and subdivision does not
was carrying the rest.

The sampling grid is the suspect, and the reason is a CONSTANT of the
pipeline rather than a property of any scene. `S_LO` clamps every
splat's scale to a floor of 0.002 and `PIX` fixes the slice grid at
1/224 of the unit box, so a floor-scale splat is S_LO / PIX = **0.448
of a pixel**, always. `build_scene` divides positions AND scales by the
crop extent, so that ratio is invariant to extent — and the clamp is
not a corner case: 90.7% of Wilson's Creek's splats sit exactly on the
floor, 63.4% of the cannon's. The slice point-samples a field made of
needles thinner than its own grid, in every capture.

(An earlier draft of this file argued the ratio grew worse with crop
extent, and that cropping helped by shrinking it. That was wrong —
the same number in both units — and the run disproved it by reporting
0.45 px for all four scenes, from extent 40.4 down to 2.5. The
measurement below stands; the extent story does not, and whatever
cropping fixes is still unaccounted for.)

The experiment is four arms, because the obvious two would mislead:

  sharp / sharp       what the pipeline does today
  sharp / blurred     encode the needles, judge against the pixel
                      integral — MISMATCHED, and included precisely
                      because it looks like the fix and is not
  blurred / blurred   the matched pair: encode the field the referee
                      measures (docs/real-scenes.md)
  blurred / blurred + adaptive cells, to see whether the two
                      independent fixes compose or overlap

Measured (2026-09-11, budget 128): the matched pair beats sharp/sharp
on all four scenes (1.3-2.1x), the MISMATCHED arm is worse than doing
nothing on three of the four, and the two fixes compose — Wilson's
Creek 176.7% -> 27.4%, redrock 25.3% -> 4.9%.

Read that last number carefully. The blurred arms are scored against a
DIFFERENT, easier referee, and blurring concentrates each splat's
spectrum for the codebook (docs/real-scenes.md says so directly), so
27.4% and 176.7% are not the same measurement and do not belong in one
column. The defensible claim is only this: for a rasterized view the
pixel-integrating referee is the honest one, because it is what a
renderer shows — and under the honest referee this pipeline scores far
better than its published sharp/sharp numbers.

Usage:
    bench/footprint_test.py scene.spz [scene2.spz ...] [--budget 128]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.adaptive_cells import (
    assign_adaptive,
    assign_uniform,
    decode,
    encode,
    exact,
)


def _score(holo, truth):
    resid = holo[:, 0].astype(np.float64) - truth[:, 0]
    nt = float(np.linalg.norm(truth[:, 0]))
    sq = np.sort(resid ** 2)[::-1]
    tot = float(sq.sum()) or 1.0
    return {
        "err": float(np.linalg.norm(resid) / nt),
        "worst1pct": float(sq[:max(1, len(sq) // 100)].sum()) / tot,
    }


def _build(scene, smax, books, bands, dim, assign):
    from holo.capture import band_of

    bidx = band_of(smax, bands)
    per_band = {}
    for b, (name, _cap, cell) in enumerate(bands):
        idx = np.where(bidx == b)[0]
        per_band[name] = assign(scene.mu, idx, cell) if len(idx) else {}
    return encode(scene, per_band, books, dim)


def run_scene(path, budget, max_level):
    from holo.capture import (
        ALPHA_MIN,
        BANDS,
        DIM,
        PIX,
        band_codebooks,
        build_scene,
        footprint_blur,
        load_scene_file,
        mass_mode,
        slice_grid,
        weighted_quantile,
    )

    t0 = time.time()
    scene, smax, box = build_scene(path, verbose=False)

    # the crop extent, for the extent-dependence the test turns on
    pos, _s, rgba, _q = load_scene_file(path)
    keep = rgba[:, 3] >= ALPHA_MIN
    pos, a = pos[keep], rgba[keep][:, 3]
    ctr = np.array([weighted_quantile(pos[:, i], a, 0.5) for i in range(3)])
    extent = float(2 * 1.2 * weighted_quantile(
        np.abs(pos - ctr).max(axis=1), a, 0.75))

    books = band_codebooks(np.random.default_rng(42))
    y = mass_mode(scene.mu[:, 1], scene.amp[:, 0], box[1])
    pts, _shape = slice_grid((0, box[0]), (0, box[2]), "y", y)

    # A pixel-wide footprint is a Gaussian of equal variance; blurring
    # ADDS that variance, so the per-splat band key grows with it. Reuse
    # the mip's own rule rather than re-deriving it.
    sigma_fp = PIX / np.sqrt(12.0)
    blurred = footprint_blur(scene, PIX)
    smax_b = np.sqrt(smax ** 2 + sigma_fp ** 2)
    px_wide = float(np.median(smax)) / PIX

    row = {"scene": os.path.splitext(os.path.basename(path))[0],
           "splats": int(scene.n), "extent": extent,
           "median_splat_px": px_wide, "pixels": len(pts), "arms": {}}
    print(f"\n=== {row['scene']}: {scene.n:,} splats, extent {extent:.1f}, "
          f"median splat {px_wide:.2f} px wide", flush=True)

    def arm(name, enc_scene, enc_smax, gt_scene, assign):
        bundles, members = _build(enc_scene, enc_smax, books, BANDS, DIM,
                                  assign)
        holo = decode(pts, bundles, books, BANDS)
        # the referee sees its own field through the SAME cell
        # membership, so locality is identical across arms
        truth = exact(pts, gt_scene, members, BANDS)
        s = _score(holo, truth)
        s["cells"] = sum(len(c) for c in bundles.values())
        row["arms"][name] = s
        print(f"  {name:26s} {100 * s['err']:7.1f}%  worst1% "
              f"{100 * s['worst1pct']:3.0f}%  {s['cells']:,} cells",
              flush=True)

    arm("sharp / sharp", scene, smax, scene, assign_uniform)
    arm("sharp / blurred", scene, smax, blurred, assign_uniform)
    arm("blurred / blurred", blurred, smax_b, blurred, assign_uniform)
    arm("blurred + adaptive", blurred, smax_b, blurred,
        lambda mu, idx, cell: assign_adaptive(mu, idx, cell, budget,
                                              max_level))
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--max-level", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--numpy", action="store_true")
    args = ap.parse_args()

    if not args.numpy:
        import bench.cuda_backend as cb
        cb.install()

    rows = []
    for p in args.scenes:
        rows.append(run_scene(p, args.budget, args.max_level))
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(rows, fh, indent=2)

    print(f"\n{'scene':26s} {'extent':>7s} {'px':>6s} {'sharp':>8s} "
          f"{'mismatch':>9s} {'matched':>8s} {'+adaptive':>10s} {'gain':>7s}")
    for r in sorted(rows, key=lambda r: -r["extent"]):
        a = r["arms"]
        gain = a["sharp / sharp"]["err"] / max(a["blurred / blurred"]["err"],
                                               1e-9)
        print(f"{r['scene']:26s} {r['extent']:7.1f} "
              f"{r['median_splat_px']:6.2f} "
              f"{100 * a['sharp / sharp']['err']:7.1f}% "
              f"{100 * a['sharp / blurred']['err']:8.1f}% "
              f"{100 * a['blurred / blurred']['err']:7.1f}% "
              f"{100 * a['blurred + adaptive']['err']:9.1f}% "
              f"{gain:6.1f}x")
    if args.out:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
