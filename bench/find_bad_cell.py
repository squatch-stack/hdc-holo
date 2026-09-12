"""Find the cell that carries a scene's error, and say what is odd about it.

`results/gpu_sweep.md` established that the slice error is not one
quantity: on some captures it measures reconstruction fidelity, and on
others it is a spike detector reporting one pathological cell while the
rest of the image is faithful. Wilson's Creek is the extreme case —
176.7% error with 91% of the squared residual inside the worst 1% of
pixels.

That is a diagnosis of the METRIC. This script goes after the CAUSE: it
localises the residual, maps the worst pixels back to the cells whose
reach covers them, and then compares those cells with the population
they came from — member count, scale spread, bundle norm, how many
splats the band put in them. If blown cells look like ordinary cells on
every axis, the fault is in the decode; if they are outliers on one,
that is the lead.

The population comparison is the point. A cell holding 4,000 splats is
only suspicious if the median cell holds 40, and no amount of staring
at the bad cell alone establishes that.

Usage:
    bench/find_bad_cell.py scene.spz [--band fine] [--top 12] [--numpy]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _percentile_rank(value, population):
    """Where `value` sits in `population`, as a percentile."""
    pop = np.asarray(population, dtype=np.float64)
    if pop.size == 0:
        return float("nan")
    return float(100.0 * np.mean(pop <= value))


def _score_cells(pts, resid2, total, bundles, members, smax, bands,
                 cell_mask):
    """Credit every cell with the squared error inside its reach.

    A pixel can lie inside several cells' reach, so the residual is
    SHARED rather than assigned: each covering cell is credited with the
    pixel's squared error. Totals therefore over-count.

    Raw share is the WRONG ranking, which a first run proved by putting
    a 1-member `mid` cell on top with 99.6% of the error — its reach
    covered 13,989 of 50,176 pixels, so it contained most of the error
    the way a bucket contains rain. Band reach spans two orders of
    magnitude here (xfine 0.012, mid 0.06), so sorting by share sorts
    by band.

    `enrichment` is the honest statistic: the cell's share of the error
    divided by its share of the image. 1.0 means the error inside this
    cell is exactly what its size would predict; a blown cell
    concentrates error in its own footprint and scores far above. Every
    per-cell statistic is paired with its percentile inside the same
    band, since "4,000 members" means nothing until you know the median
    is 40.
    """
    scored = []
    for name, cap, cell in bands:
        reach = 3.0 * cap
        band_cells = bundles[name]
        if not band_cells:
            continue
        counts = np.array([len(members[name][k]) for k in band_cells])
        norms = np.array([float(np.linalg.norm(b))
                          for b in band_cells.values()])
        for k, n_members, norm in zip(band_cells, counts, norms):
            m = cell_mask(pts, k, cell, reach)
            if not m.any():
                continue
            share = float(resid2[m].sum())
            if share <= 0:
                continue
            sc = smax[members[name][k]]
            px_frac = float(m.sum()) / len(pts)
            scored.append({
                "enrichment": (share / total) / px_frac if px_frac else 0.0,
                "px_frac": px_frac,
                "band": name, "cell": [int(v) for v in k],
                "share": share / total,
                "members": int(n_members),
                "members_pct": _percentile_rank(n_members, counts),
                "bundle_norm": float(norm),
                "norm_pct": _percentile_rank(norm, norms),
                "smax_median": float(np.median(sc)),
                "smax_max": float(sc.max()),
                "smax_ratio": float(sc.max() / max(sc.min(), 1e-12)),
                "px_in_reach": int(m.sum()),
                "band_cells": len(band_cells),
                "band_members_median": float(np.median(counts)),
                "band_norm_median": float(np.median(norms)),
            })
    scored.sort(key=lambda r: -r["enrichment"])
    return scored


def _encode(path, footprint, budget, max_level):
    """Load, optionally blur, assign cells, encode, lay out the slice.

    Everything routes through `bench/adaptive_cells`, including the
    baseline: `assign_uniform` at level 0 IS the SDK's fixed lattice,
    and its keys carry the level so one mask function serves both
    lattices. That is what lets this tool localise the residual of an
    adaptively-celled encode — the two fixes could previously only be
    measured together as a single number, never attributed.

    Under `--footprint` the band caps travel with the scales via the
    sweep's own helper rather than a restatement of the transform; they
    MUST move, or `encode_bands` refuses the widened splats.
    """
    import numpy as np

    from bench import adaptive_cells as ac
    from holo.capture import (
        BANDS,
        band_codebooks,
        build_scene,
        mass_mode,
        slice_grid,
    )

    scene, smax, box = build_scene(path)
    bands = BANDS
    if footprint:
        from bench.sweep_scenes import _matched_referee
        scene, smax, bands = _matched_referee(scene, smax, BANDS)

    if budget:
        def assign(mu, idx, cell):
            return ac.assign_adaptive(mu, idx, cell, budget, max_level)
    else:
        assign = ac.assign_uniform

    from holo.capture import band_of
    bidx = band_of(smax, bands)
    per_band = {}
    for b, (name, _cap, cell) in enumerate(bands):
        idx = np.where(bidx == b)[0]
        per_band[name] = assign(scene.mu, idx, cell) if len(idx) else {}

    books = band_codebooks(np.random.default_rng(42))
    bundles, members = ac.encode(scene, per_band, books, 8192)
    y = mass_mode(scene.mu[:, 1], scene.amp[:, 0], box[1])
    pts, shape = slice_grid((0, box[0]), (0, box[2]), "y", y)
    return scene, smax, bands, books, bundles, members, pts, shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--top", type=int, default=12,
                    help="how many worst cells to report")
    ap.add_argument("--out", default="", help="write JSON here")
    ap.add_argument("--figure", default="", help="write a PNG here")
    ap.add_argument("--budget", type=int, default=0,
                    help="adaptive cells: split any cell over this many "
                         "members (0 = the SDK's fixed lattice)")
    ap.add_argument("--max-level", type=int, default=8)
    ap.add_argument("--footprint", action="store_true",
                    help="run against the matched (pixel-integrated) "
                         "referee instead of point samples")
    ap.add_argument("--numpy", action="store_true")
    args = ap.parse_args()

    if not args.numpy:
        import bench.cuda_backend as cb
        cb.install()

    from bench import adaptive_cells as ac

    scene, smax, bands, books, bundles, members, pts, shape = _encode(
        args.scene, args.footprint, args.budget, args.max_level)
    truth = ac.exact(pts, scene, members, bands)
    holo = ac.decode(pts, bundles, books, bands)
    resid2 = (holo[:, 0].astype(np.float64) - truth[:, 0]) ** 2
    total = float(resid2.sum())
    err = float(np.sqrt(total) / np.linalg.norm(truth[:, 0]))
    ref = "matched (pixel-integrated)" if args.footprint else "sharp"
    print(f"{os.path.basename(args.scene)}: {scene.n:,} splats, "
          f"{ref} referee, top-down rel err {100 * err:.1f}%")

    scored = _score_cells(pts, resid2, total, bundles, members, smax,
                          bands, ac.mask_for)
    top = scored[:args.top]

    def table(title, rows):
        print(f"\n{title}")
        print(f"{'band':7s} {'cell':>15s} {'enrich':>8s} {'share':>8s} "
              f"{'%img':>7s} {'members':>9s} {'(pct)':>7s} "
              f"{'|bundle|':>11s} {'(pct)':>7s}")
        for r in rows:
            print(f"{r['band']:7s} {tuple(r['cell'])!s:>15s} "
                  f"{r['enrichment']:8.1f} {100 * r['share']:7.1f}% "
                  f"{100 * r['px_frac']:6.2f}% {r['members']:9,} "
                  f"{r['members_pct']:6.1f}% {r['bundle_norm']:11.4g} "
                  f"{r['norm_pct']:6.1f}%")

    # A cell can be enriched and still irrelevant if it holds almost no
    # error at all, so the lead table requires both.
    material = [r for r in scored if r["share"] >= 0.005]
    table("most CONCENTRATED (enrichment, >=0.5% of total error)",
          material[:args.top])
    table("largest RAW share (dominated by band reach — see docstring)",
          sorted(scored, key=lambda r: -r["share"])[:5])
    top = material[:args.top] or top

    if top:
        w = top[0]
        print(f"\nworst cell holds {100 * w['share']:.1f}% of the squared "
              f"error from {w['px_in_reach']:,} of {len(pts):,} pixels")
        print(f"  members     {w['members']:,} vs band median "
              f"{w['band_members_median']:,.0f}  (p{w['members_pct']:.1f})")
        print(f"  |bundle|    {w['bundle_norm']:.1f} vs band median "
              f"{w['band_norm_median']:.1f}  (p{w['norm_pct']:.1f})")
        print(f"  scale span  {w['smax_median']:.5f} median, "
              f"{w['smax_max']:.5f} max, {w['smax_ratio']:.1f}x within cell")
        head = [r for r in scored if r["share"] >= 0.01]
        print(f"  cells holding >=1% of the error: {len(head)} of "
              f"{len(scored):,}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"scene": os.path.basename(args.scene), "referee": ref,
                       "splats": int(scene.n), "err_top_down": err,
                       "pixels": len(pts), "cells_scored": len(scored),
                       "top": top}, fh, indent=2)
        print(f"\nwrote {args.out}")

    if args.figure and top:
        _figure(args.figure, args.scene, pts, shape, truth, holo, resid2, top)
    return 0


def _figure(path, scene_path, pts, shape, truth, holo, resid2, top):
    """Truth, hologram, and where the residual actually is."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PAGE, INK, INK2 = "#f9f9f7", "#0b0b0b", "#52514e"
    ref = np.percentile(truth[:, 0], 99)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.patch.set_facecolor(PAGE)
    for ax, field, title in [
            (axes[0], np.clip(truth[:, 1:4] / ref, 0, 1),
             "ground truth (exact mixture)"),
            (axes[1], np.clip(holo[:, 1:4] / ref, 0, 1),
             "holographic (chunked bundles)")]:
        ax.imshow(field.reshape(*shape, 3), origin="lower", aspect="equal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title, fontsize=10, color=INK)
    # log residual: the artifact is orders above the rest, so a linear
    # scale shows one white dot on black and hides the baseline
    r = np.log10(resid2.reshape(shape) + 1e-12)
    im = axes[2].imshow(r, origin="lower", aspect="equal", cmap="magma")
    axes[2].set_xticks([]), axes[2].set_yticks([])
    axes[2].set_title("squared residual (log10)", fontsize=10, color=INK)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    axes[1].set_xlabel(
        f"worst cell carries {100 * top[0]['share']:.0f}% of the squared error",
        fontsize=9, color=INK2)
    fig.suptitle(f"{os.path.basename(scene_path)}: where the error lives",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
