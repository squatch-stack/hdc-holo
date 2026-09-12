"""Adaptive cell subdivision: split a cell when it outgrows its bundle.

`bench/find_bad_cell.py` located Wilson's Creek's 176.7% slice error in
thirteen `xfine` cells holding up to 21,064 splats against a band median
of 23. Those bundles carry `d = 8,192` dimensions, so the worst cells
hold more splats than the bundle has dimensions and crosstalk swamps
signal. Cropping tighter removes them — and the error collapses to
20.1% — but it does so by throwing away scene, which is a cure worse
than the disease on a subject like the oak.

The grid is one real problem, though it turned out not to be the whole
one. `encode_bands` puts every splat of a band into a FIXED lattice
(`xfine` at 1/32 of the unit box), and `build_scene` normalizes every
capture into that same box, so cell occupancy is whatever the subject's
density happens to hand it. A uniform lattice cannot be both fine
enough for the dense knot and cheap enough everywhere else.

This prototype compares three encoders on the same scene, same
codebooks, same decode path:

  baseline   the SDK's fixed lattice
  uniform    the same lattice, refined globally (cell / 2^L everywhere)
  adaptive   refine ONLY cells over a member budget, recursively

The interesting comparison is adaptive against uniform at matched
error: if capacity is the mechanism, both fix the blowup, and adaptive
should get there with far fewer cells — which is the whole point,
because cells are the storage.

Measured (2026-09-10, Wilson's Creek). Adaptive does that job: every
over-capacity cell gone at 1.1x the storage, where the cheapest uniform
refinement reaching zero over-capacity cells costs 3.3x and a further
level is refused outright at 16.5x. Be precise about what that buys,
because an earlier draft of this line was not: uniform L1 at 3.3x
scores 126.4% against adaptive's 129.1%, so refinement is not BETTER
here, it is three times dearer for three points. The case for adaptive
is storage at equal quality, and nothing more.

Sweeping the budget then shows capacity is only PART of the
fault — 176.7% at no budget, 129.1% at d/4, 92.1% at d/16, 78.4% at
d/64, and still 72.2% at d/256, with 96% of the error inside the worst
1% of pixels throughout. Subdivision asymptotes around 72% and the
spike survives it. `bench/footprint_test.py` chases the rest, which is
a mismatch between what the encoder bundles and what the referee
measures; the two fixes are independent and compose.

SCOPE, deliberately: the top-down slice only. That is where the
capacity failure shows, and the X-ray path has its own mip encode and
its own resolution story (`results/gpu_sweep.md`, finding three). The
algorithms now live in `holo.capture`; this module retains the
measurement driver and compatibility adapters.

Usage:
    bench/adaptive_cells.py scene.spz [--budget 2048] [--max-level 4]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from holo.capture import (
    _cell_uv_mask as uv_mask_for,
)
from holo.capture import (
    _encode_cells,
    assign_adaptive,
)
from holo.capture import (
    _numpy_exact_slice as exact,
)
from holo.capture import (
    _numpy_exact_xray as _exact_xray,
)
from holo.capture import (
    cell_mask as mask_for,
)
from holo.capture import (
    decode_slice as decode,
)
from holo.capture import (
    render_xray as _render_xray,
)

__all__ = [
    "assign_adaptive",
    "assign_uniform",
    "decode",
    "encode",
    "exact",
    "exact_xray",
    "mask_for",
    "render_xray",
    "uv_mask_for",
]


# ---------------------------------------------------------------------------
# cell assignment
# ---------------------------------------------------------------------------

def assign_uniform(mu, idx, cell, level=0):
    """Every splat into one lattice at `cell / 2**level`.

    Keys carry their level so a decoder can recover each cell's size
    without a side table — the same reason a quadtree stores depth.
    """
    size = cell / (1 << level)
    per_cell = {}
    keys = (mu[idx] // size).astype(int)
    for i, k in zip(idx, map(tuple, keys)):
        per_cell.setdefault((level, *k), []).append(i)
    return per_cell


def encode(scene, per_band, books, dim):
    bundles, members = {}, {}
    for name, per_cell in per_band.items():
        bundles[name], members[name] = _encode_cells(
            scene, per_cell, books[name][0])
    return bundles, members


def run_one(label, case, bands, assign, max_storage_gb=12.0):
    from holo.capture import band_of

    scene, smax, books, dim, pts, truth_of = case
    t0 = time.time()
    bidx = band_of(smax, bands)
    per_band, over_budget, biggest = {}, 0, 0
    for b, (name, _cap, cell) in enumerate(bands):
        idx = np.where(bidx == b)[0]
        per_band[name] = assign(scene.mu, idx, cell) if len(idx) else {}
        for ids in per_band[name].values():
            biggest = max(biggest, len(ids))
            if len(ids) > dim:
                over_budget += 1
    cells = sum(len(c) for c in per_band.values())
    storage_mb = cells * scene.channels * dim * 8 / (1 << 20)
    # Assignment is cheap (index lists); ENCODING is what costs, at
    # channels*dim*8 bytes per cell. Predict before paying: a first run
    # of this script reached 31 GB RSS on the uniform arms and had to be
    # killed, because global refinement multiplies cells by 8 per level
    # and the bundle count is the storage. Refusing the arm and
    # reporting its projected cost is the more useful result anyway —
    # "uniform refinement does not scale" is the finding, not an
    # inconvenience on the way to one.
    if storage_mb / 1024 > max_storage_gb:
        return {"label": label, "err": None, "cells": cells,
                "biggest_cell": biggest, "cells_over_dim": over_budget,
                "storage_mb": storage_mb,
                "skipped": f"projected {storage_mb / 1024:.0f} GB of bundles"}
    bundles, members = encode(scene, per_band, books, dim)
    t_enc = time.time() - t0

    truth = truth_of(members, bands)
    t1 = time.time()
    holo = decode(pts, bundles, books, bands)
    t_dec = time.time() - t1

    resid = holo[:, 0] - truth[:, 0]
    err = float(np.linalg.norm(resid) / np.linalg.norm(truth[:, 0]))
    sq = np.sort(resid.astype(np.float64) ** 2)[::-1]
    tot = float(sq.sum()) or 1.0
    share1 = float(sq[:max(1, len(sq) // 100)].sum()) / tot
    return {
        "label": label, "err": err, "err_share_1pct": share1,
        "cells": cells, "biggest_cell": biggest,
        "cells_over_dim": over_budget,
        "storage_mb": cells * scene.channels * dim * 8 / (1 << 20),
        "t_encode": round(t_enc, 1), "t_decode": round(t_dec, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--budget", type=int, default=2048,
                    help="members above which an adaptive cell splits")
    ap.add_argument("--max-level", type=int, default=4)
    ap.add_argument("--uniform-levels", default="1,2,3")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-storage-gb", type=float, default=12.0,
                    help="refuse an arm whose bundles would exceed this")
    ap.add_argument("--numpy", action="store_true")
    args = ap.parse_args()

    if not args.numpy:
        import bench.cuda_backend as cb
        cb.install()

    from holo.capture import (
        BANDS,
        DIM,
        band_codebooks,
        build_scene,
        mass_mode,
        slice_grid,
    )

    scene, smax, box = build_scene(args.scene)
    books = band_codebooks(np.random.default_rng(42))
    y = mass_mode(scene.mu[:, 1], scene.amp[:, 0], box[1])
    pts, _shape = slice_grid((0, box[0]), (0, box[2]), "y", y)
    print(f"{os.path.basename(args.scene)}: {scene.n:,} splats, "
          f"{len(pts):,} slice pixels, d={DIM:,}, budget={args.budget:,}")

    # Ground truth depends only on which splats a cell holds and how far
    # its reach extends, so it is recomputed per encoder rather than
    # shared — a refined lattice has a SMALLER reach footprint per cell
    # and a different cutoff, and reusing one truth across encoders
    # would quietly compare against the wrong referee.
    def truth_of(members, bands):
        return exact(pts, scene, members, bands)

    case = (scene, smax, books, DIM, pts, truth_of)
    plan = [("baseline (fixed 1/32)", assign_uniform)]
    for lv in [int(v) for v in args.uniform_levels.split(",") if v.strip()]:
        plan.append((f"uniform L{lv} (cell/{1 << lv})",
                     lambda mu, idx, cell, lv=lv:
                         assign_uniform(mu, idx, cell, lv)))
    plan.append((f"adaptive (budget {args.budget:,})",
                 lambda mu, idx, cell:
                     assign_adaptive(mu, idx, cell, args.budget,
                                     args.max_level)))
    runs = []
    for label, assign in plan:
        r = run_one(label, case, BANDS, assign, args.max_storage_gb)
        runs.append(r)
        if r.get("skipped"):
            print(f"  {label}: SKIPPED — {r['skipped']}", flush=True)
        else:
            print(f"  {label}: {100 * r['err']:.1f}%, {r['cells']:,} cells, "
                  f"{r['t_encode']:.0f}s encode", flush=True)

    base = runs[0]
    print(f"\n{'encoder':26s} {'err':>8s} {'worst1%':>8s} {'cells':>9s} "
          f"{'biggest':>9s} {'over d':>7s} {'MB':>8s} {'vs base':>8s}")
    for r in runs:
        err = "    n/a" if r["err"] is None else f"{100 * r['err']:6.1f}%"
        w1 = "    n/a" if r["err"] is None \
            else f"{100 * r['err_share_1pct']:6.0f}%"
        print(f"{r['label']:26s} {err:>8s} {w1:>8s} {r['cells']:9,} "
              f"{r['biggest_cell']:9,} {r['cells_over_dim']:7d} "
              f"{r['storage_mb']:8.0f} "
              f"{r['cells'] / base['cells']:7.1f}x")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"scene": os.path.basename(args.scene),
                       "splats": int(scene.n), "dim": DIM,
                       "budget": args.budget, "runs": runs}, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# X-ray over variable-size cells
# ---------------------------------------------------------------------------

def render_xray(bundles, books, cam, t_extent, bands, chunk=2048):
    """Compatibility adapter for the benchmark's packed camera."""
    return _render_xray(bundles, books, *cam, t_extent, bands, chunk)


def exact_xray(scene, members, cam, bands, chunk=1024):
    """Compatibility adapter for the benchmark's packed camera."""
    return _exact_xray(scene, members, *cam, bands, chunk)
