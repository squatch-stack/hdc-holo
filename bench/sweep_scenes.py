"""Encode a whole set of real captures and record what each one costs.

`examples/run_real_scene.py` measures one scene and prints prose. This
runs the SAME pipeline over many, on the CUDA backend, and writes one
JSON row per scene so the set can be compared — which is the only way
to test the claim the first three scenes suggested: that reconstruction
error tracks how GAUSSIAN the subject is (smooth mass encodes well,
thin high-frequency structure does not) rather than how good the scan
is.

Per scene it records what that claim needs to be checkable, not just
the error: the crop extent (the oak's error looked like an encoding
failure and was a VOLUME failure — a 12.4-unit cube against the cairn's
2.8, the same cell budget spread over ninety times the space), the
per-band split (the first cannon run put 288,640 of 311,310 splats in
the finest band and none in the coarsest, so three of four bands did
nothing), cells, timings and peak VRAM.

Stage calls and constants are `run_real_scene.py`'s, so a row here is
comparable to a number printed there; `--figures` writes the same
evidence PNGs.

Usage:
    bench/sweep_scenes.py out.json scene1.spz scene2.ply ...
    bench/sweep_scenes.py out.json --figures --dir results/ scenes/*.spz
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _describe_crop(path, row, crop_quantile=0.75, crop_margin=1.2):
    """Record how much WORLD the cells were spread over.

    `build_scene` normalizes every capture into the same unit box, so
    after it runs, a 3-metre subject and a 300-metre one look
    identical to every downstream number. The extent is the missing
    axis: the oak's X-ray error read as an encoding failure and was a
    volume failure. Recovered here from the same weighted quantiles
    build_scene itself used, so the two cannot drift apart.
    """
    from holo.capture import ALPHA_MIN, load_scene_file, weighted_quantile

    pos, _s, rgba, _q = load_scene_file(path)
    keep = rgba[:, 3] >= ALPHA_MIN
    pos, a = pos[keep], rgba[keep][:, 3]
    center = np.array([weighted_quantile(pos[:, i], a, 0.5) for i in range(3)])
    radius = weighted_quantile(np.abs(pos - center).max(axis=1), a,
                               crop_quantile)
    row["splats_loaded"] = len(rgba)
    row["crop_extent"] = float(2 * crop_margin * radius)


def _slices(scene, box, bundles, members, books, row, bands, ac=None):
    """The two axis-aligned slices, scored against the exact mixture."""
    from holo.capture import (
        decode_slice,
        exact_slice,
        mass_mode,
        slice_grid,
    )

    w = scene.amp[:, 0]
    y_slice = mass_mode(scene.mu[:, 1], w, box[1])
    x_slice = mass_mode(scene.mu[:, 0], w, box[0])

    panels, errs = [], {}
    for key, (pts, shape) in [
            ("top_down", slice_grid((0, box[0]), (0, box[2]), "y", y_slice)),
            ("side", slice_grid((0, box[2]), (0, box[1]), "x", x_slice))]:
        t1 = time.time()
        # `ac` carries the variable-cell kernels when the encode used
        # adaptive cells: its keys are (level, i, j, k) and the SDK's
        # `cell_mask` takes a 3-tuple, so mixing them raises a broadcast
        # error rather than quietly mis-masking — which is how this got
        # caught.
        if ac is None:
            truth = exact_slice(pts, scene, members, bands)
            t2 = time.time()
            holo = decode_slice(pts, bundles, books, bands)
        else:
            truth = ac.exact(pts, scene, members, bands)
            t2 = time.time()
            holo = ac.decode(pts, bundles, books, bands)
        t3 = time.time()
        # The error is RELATIVE, so it says as much about the
        # denominator as the reconstruction: a slice plane that lands in
        # a thin part of the scene has little truth to divide by and
        # reports >100% while the hologram is doing nothing unusual.
        # Record the norms and the occupancy so a big number can be read
        # rather than guessed at.
        nt = float(np.linalg.norm(truth[:, 0]))
        resid = holo[:, 0] - truth[:, 0]
        err = float(np.linalg.norm(resid) / nt)
        errs[key] = err
        row[f"norm_{key}"] = round(nt, 3)
        row[f"fill_{key}"] = round(
            float(np.mean(truth[:, 0] > 0.01 * truth[:, 0].max())), 4)
        # WHERE the error lives, not just how much. A relative L2 is a
        # single global number, and one cell whose bundle has blown up
        # can dominate it while every other pixel is faithful — which is
        # exactly what the >100% scenes turn out to be. The share of
        # squared error carried by the worst 1% and 0.1% of pixels
        # separates "reconstruction is poor" from "reconstruction is
        # good except for a handful of pathological cells".
        sq = np.sort(resid.astype(np.float64) ** 2)[::-1]
        tot = float(sq.sum()) or 1.0
        for frac, tag in ((0.01, "1pct"), (0.001, "01pct")):
            k = max(1, int(len(sq) * frac))
            row[f"err_share_{tag}_{key}"] = round(float(sq[:k].sum()) / tot, 4)
        row[f"peak_ratio_{key}"] = round(
            float(holo[:, 0].max() / max(truth[:, 0].max(), 1e-9)), 2)
        row[f"t_gt_{key}"] = round(t2 - t1, 1)
        row[f"t_decode_{key}"] = round(t3 - t2, 1)
        row[f"px_{key}"] = len(pts)
        print(f"  {key}: {len(pts):,} px, GT {t2 - t1:.0f}s, "
              f"decode {t3 - t2:.0f}s, alpha rel err {err:.3f}")
        panels.append((key, truth, holo, shape, err))
    return panels, errs


def _xrays(scene, smax, members, row, bands, ac=None, budget=0,
           max_level=8):
    """The two orthographic X-ray views, scored against the mip.

    Unaffected by the slice referee, and deliberately so: this arm
    already encodes `render_mip` and scores against that same mip, so it
    was a matched pair before the flag existed. Under --footprint the
    scene arrives pre-blurred by sigma_fp and the mip blur becomes
    sqrt(sigma_fp^2 + SIGMA_MIP^2) — 0.00810 against 0.008, a 1.3%
    change — so these numbers stay comparable to the published sweep
    rather than quietly becoming a different measurement.

    Under --budget the mip encode uses adaptive cells through the
    level-aware kernels in `bench/adaptive_cells`, whose footprint mask
    is the SDK's own disc bound with the size read from the key — so at
    level 0 it is the SDK path exactly, and the published X-ray numbers
    still reproduce.
    """
    from holo.capture import (
        DIM_R,
        RENDER_BANDS,
        SIGMA_MIP,
        band_codebooks,
        encode_bands,
        exact_xray,
        render_mip,
        render_xray,
    )

    errs = {}
    mip = render_mip(scene, SIGMA_MIP)
    smax_r = np.sqrt(smax ** 2 + SIGMA_MIP ** 2)
    r_books = band_codebooks(np.random.default_rng(43), RENDER_BANDS,
                             DIM_R, s_floor=SIGMA_MIP)
    if ac is None:
        r_bundles, r_members = encode_bands(mip, smax_r, r_books,
                                            RENDER_BANDS, DIM_R)
    else:
        # Adaptive cells on the mip encode too. Measured before this
        # existed: on wilsons-creek the r-fine band had 4 cells over
        # DIM_R, the largest holding 61,704 against a median of 163 —
        # the same capacity fault the slices had, untreated.
        from holo.capture import DIM
        from holo.capture import band_of as _band_of
        # The budget is a FRACTION of the bundle dimension, not a count:
        # capacity is members-per-d. A constant carried over from the
        # d=8,192 slice encode is d/64 there and d/256 at DIM_R=32,768,
        # which made 4x the cells at 4x the width — and with the slice
        # bundles still resident, the oak reached 46 GB RSS and the
        # kernel killed it. Same members-per-d here, and the storage is
        # predicted and refused rather than discovered.
        r_budget = max(1, budget * DIM_R // DIM)
        bidx_r = _band_of(smax_r, RENDER_BANDS)
        per_band = {}
        for b, (name, _cap, cell) in enumerate(RENDER_BANDS):
            idx = np.where(bidx_r == b)[0]
            per_band[name] = (ac.assign_adaptive(mip.mu, idx, cell,
                                                 r_budget, max_level)
                              if len(idx) else {})
        n_cells = sum(len(c) for c in per_band.values())
        gb = n_cells * mip.channels * DIM_R * 8 / 1e9
        row["xray_budget"] = r_budget
        row["xray_cells"] = n_cells
        if gb > 12.0:
            print(f"  x-ray: SKIPPED — {n_cells:,} mip cells would be "
                  f"{gb:.0f} GB of bundles", flush=True)
            row["xray_skipped"] = f"{gb:.0f} GB projected"
            return [], {}
        r_bundles, r_members = ac.encode(mip, per_band, r_books, DIM_R)
    center_p, half, T, res = [0.5, 0.5, 0.5], 0.5, 2.0, 176
    xpanels = []
    for key, view in [("xray_a", [1.0, 0.0, 0.25]),
                      ("xray_b", [1.0, 0.0, 1.0])]:
        t1 = time.time()
        sharp = (exact_xray(scene, members, view, center_p, half, res,
                            bands=bands) if members is not None
                 else None)
        cam = (view, center_p, half, res)
        if ac is None:
            mip_gt = exact_xray(mip, r_members, view, center_p, half, res,
                                bands=RENDER_BANDS)
        else:
            mip_gt = ac.exact_xray(mip, r_members, cam, RENDER_BANDS)
        t2 = time.time()
        if ac is None:
            holo = render_xray(r_bundles, r_books, view, center_p, half,
                               res, T, bands=RENDER_BANDS)
        else:
            holo = ac.render_xray(r_bundles, r_books, cam, T, RENDER_BANDS)
        t3 = time.time()
        err = float(np.linalg.norm(holo[:, 0] - mip_gt[:, 0])
                    / np.linalg.norm(mip_gt[:, 0]))
        errs[key] = err
        row[f"t_gt_{key}"] = round(t2 - t1, 1)
        row[f"t_render_{key}"] = round(t3 - t2, 1)
        print(f"  {key}: GT {t2 - t1:.0f}s, render {t3 - t2:.0f}s, "
              f"alpha rel err vs mip {err:.3f}")
        xpanels.append((key, sharp, mip_gt, holo, err))
    return xpanels, errs


def _matched_referee(scene, smax, bands):
    """Blur the scene by one slice pixel and widen the bands to match.

    The slices point-sample a field whose splats are mostly thinner than
    a pixel — S_LO / PIX = 0.448, and the clamp puts most of a real
    capture exactly on that floor — so the sharp referee asks what the
    field is at infinitely small points while a renderer asks what it
    averages over a pixel (`footprint_blur`, docs/real-scenes.md). The
    matched pair encodes the field the referee measures.

    The band caps MUST travel with the scales. `band_of` says so in its
    own docstring: widening scales without widening bands puts splats
    past the last cap and `encode_bands` refuses them. Transforming both
    by the same sqrt(x^2 + sigma^2) keeps every splat in the band it
    was already in — the map is strictly increasing, so `searchsorted`
    returns identical indices — which is what makes this a clean
    one-variable change. Only the referee moves.
    """
    from holo.capture import PIX, footprint_blur

    sigma = PIX / np.sqrt(12.0)
    blurred = footprint_blur(scene, PIX)
    smax_b = np.sqrt(smax ** 2 + sigma ** 2)
    bands_b = [(name, float(np.sqrt(cap ** 2 + sigma ** 2)), cell)
               for name, cap, cell in bands]
    return blurred, smax_b, bands_b


def run_one(path, figures=None, crop_quantile=0.75, crop_margin=1.2,
            matched=False, budget=0, max_level=8):
    from holo.capture import (
        BANDS,
        DIM,
        DIM_R,
        band_codebooks,
        band_of,
        build_scene,
        encode_bands,
    )

    t0 = time.time()
    row = {"scene": os.path.splitext(os.path.basename(path))[0],
           "file": os.path.basename(path),
           "crop_quantile": crop_quantile, "crop_margin": crop_margin}

    scene, smax, box = build_scene(path, crop_quantile=crop_quantile,
                                   crop_margin=crop_margin)
    # The X-ray arm is ALREADY a matched pair — it encodes `render_mip`
    # and scores against that same mip — so the flag changes the SLICE
    # referee and leaves the X-ray comparable to the published sweep.
    bands = BANDS
    if matched:
        scene, smax, bands = _matched_referee(scene, smax, BANDS)
    row["referee"] = "matched (pixel-integrated)" if matched else "sharp"
    row["splats_encoded"] = int(scene.n)
    row["box"] = [float(b) for b in box]
    _describe_crop(path, row, crop_quantile, crop_margin)

    bidx = band_of(smax, bands)
    row["band_split"] = {name: int(np.sum(bidx == b))
                         for b, (name, _c, _z) in enumerate(bands)}

    books = band_codebooks(np.random.default_rng(42))
    t_enc = time.time()
    ac = None
    if budget:
        # Adaptive cells for the slice encode; `_xrays` applies the same
        # budget to the mip encode through the level-aware kernels.
        from bench import adaptive_cells as ac
        bidx_a = band_of(smax, bands)
        per_band = {}
        for b, (name, _cap, cell) in enumerate(bands):
            idx = np.where(bidx_a == b)[0]
            per_band[name] = (ac.assign_adaptive(scene.mu, idx, cell,
                                                 budget, max_level)
                              if len(idx) else {})
        bundles, members = ac.encode(scene, per_band, books, DIM)
    else:
        bundles, members = encode_bands(scene, smax, books, bands)
    row["budget"] = budget
    row["t_encode"] = round(time.time() - t_enc, 1)
    row["cells"] = int(sum(len(b) for b in bundles.values()))
    row["cells_per_band"] = {k: len(v) for k, v in bundles.items()}

    panels, errs = _slices(scene, box, bundles, members, books, row,
                           bands, ac)
    if ac is not None:
        # The X-ray arm encodes its own mip bundles at DIM_R; holding
        # the slice bundles alongside them is what doubled the peak.
        import gc
        del bundles
        gc.collect()
        members = None
    xpanels, xerrs = _xrays(scene, smax, members, row, bands, ac, budget,
                            max_level)
    errs.update(xerrs)

    for k in ("xray_a", "xray_b"):
        errs.setdefault(k, float("nan"))
    row["err"] = {k: round(v, 4) for k, v in errs.items()}
    row["t_total"] = round(time.time() - t0, 1)
    row["dim"] = DIM
    row["dim_render"] = DIM_R
    try:
        import cupy
        # total_bytes(), not used_bytes(): the pool holds freed blocks, so
        # used_bytes() reads ~0 at the end of a run and the first sweep
        # recorded 0.0 GB for every scene. total_bytes() is the high-water
        # mark of what was actually taken from the device.
        row["peak_vram_gb"] = round(
            cupy.get_default_memory_pool().total_bytes() / 1e9, 2)
        row["backend"] = "cupy-cuda"
    except ImportError:
        row["backend"] = "numpy"

    if figures and xpanels:
        _figures(figures, row["scene"], scene, panels, xpanels, row["cells"])
    return row


def _figures(outdir, name, scene, panels, xpanels, n_cells):
    import matplotlib

    from holo.capture import BANDS, DIM, SIGMA_MIP
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAGE, INK, INK2 = "#f9f9f7", "#0b0b0b", "#52514e"

    def to_rgb(field, ref):
        return np.clip(field[:, 1:4] / ref, 0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.patch.set_facecolor(PAGE)
    for col, (title, truth, holo, shape, err) in enumerate(panels):
        ref = np.percentile(truth[:, 0], 99)
        for r, (field, label) in enumerate([
                (truth, "ground truth (exact mixture)"),
                (holo, "holographic (chunked bundles)")]):
            ax = axes[r, col]
            ax.imshow(to_rgb(field, ref).reshape(*shape, 3), origin="lower",
                      aspect="equal")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{label} — {title}", fontsize=10, color=INK)
        axes[1, col].set_xlabel(f"alpha-channel rel. error {100 * err:.0f}%",
                                fontsize=9, color=INK2)
    fig.suptitle(f"{name}: {scene.n:,} real splats in {n_cells} cell "
                 f"bundles (d = {DIM:,}, {len(BANDS)} scale bands)",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"real_{name}.png"), dpi=150,
                bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)

    res = int(np.sqrt(len(xpanels[0][1])))
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 15.5))
    fig.patch.set_facecolor(PAGE)
    for col, (vtitle, sharp, mip_gt, holo, err) in enumerate(xpanels):
        ref = np.percentile(mip_gt[:, 0], 99.5)
        # Under adaptive cells there is no SDK-lattice membership to
        # build the full-detail panel from, so it is absent rather than
        # faked; the mip row stands in and the label says which.
        detail = (sharp, "analytic line integrals, full detail") \
            if sharp is not None \
            else (mip_gt, "full detail unavailable (adaptive cells)")
        for r, (field, label) in enumerate([
                detail,
                (mip_gt, f"analytic, mip σ_b = {SIGMA_MIP}"),
                (holo, "rendered from the mip bundles")]):
            ax = axes[r, col]
            ax.imshow(to_rgb(field, ref).reshape(res, res, 3),
                      origin="lower", aspect="equal")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{label} — {vtitle}", fontsize=10, color=INK)
        axes[2, col].set_xlabel(
            f"alpha-channel rel. error vs mip {100 * err:.0f}%",
            fontsize=9, color=INK2)
    fig.suptitle(f"{name}: orthographic X-ray views straight from the "
                 "cell bundles", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"real_{name}_xray.png"), dpi=150,
                bbox_inches="tight", facecolor=PAGE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--dir", default="results")
    ap.add_argument("--numpy", action="store_true",
                    help="skip the CUDA backend (reference timings)")
    ap.add_argument("--budget", type=int, default=0,
                    help="adaptive cells on the slice encode: split any "
                         "cell over this many members (0 = fixed lattice)")
    ap.add_argument("--max-level", type=int, default=8)
    ap.add_argument("--footprint", action="store_true",
                    help="matched referee: encode and score the "
                         "pixel-integrated field instead of point samples")
    ap.add_argument("--crop-quantile", type=float, default=0.75,
                    help="build_scene crop quantile; lower crops tighter")
    ap.add_argument("--crop-margin", type=float, default=1.2)
    ap.add_argument("--tag", default="",
                    help="label carried into every row, for sweeps that "
                         "run the same scene at several crops")
    args = ap.parse_args()

    if not args.numpy:
        import bench.cuda_backend as cb
        cb.install()

    figdir = None
    if args.figures:
        figdir = args.dir
        os.makedirs(figdir, exist_ok=True)

    rows = []
    for i, path in enumerate(args.scenes, 1):
        print(f"\n=== [{i}/{len(args.scenes)}] {os.path.basename(path)}",
              flush=True)
        try:
            row = run_one(path, figures=figdir,
                          crop_quantile=args.crop_quantile,
                          crop_margin=args.crop_margin,
                          matched=args.footprint, budget=args.budget,
                          max_level=args.max_level)
        except Exception as exc:
            # One unloadable capture must not cost the other ten their
            # run; the sweep is long and unattended.
            traceback.print_exc()
            row = {"scene": os.path.splitext(os.path.basename(path))[0],
                   "file": os.path.basename(path),
                   "error": f"{type(exc).__name__}: {exc}"}
        row["tag"] = args.tag
        rows.append(row)
        print(f"  -> {row.get('t_total', '?')}s", flush=True)
        with open(args.out, "w") as fh:                # checkpoint each
            json.dump(rows, fh, indent=2)

    ok = [r for r in rows if "err" in r]
    print(f"\n{len(ok)}/{len(rows)} scenes measured -> {args.out}")
    if ok:
        print(f"\n{'scene':28s} {'splats':>10s} {'extent':>8s} "
              f"{'cells':>7s} {'top':>7s} {'side':>7s} {'xray':>7s} "
              f"{'time':>7s}")
        for r in sorted(ok, key=lambda r: r["err"]["top_down"]):
            e = r["err"]
            print(f"{r['scene']:28s} {r['splats_encoded']:10,} "
                  f"{r['crop_extent']:8.1f} {r['cells']:7,} "
                  f"{100 * e['top_down']:6.1f}% {100 * e['side']:6.1f}% "
                  f"{100 * e['xray_a']:6.1f}% {r['t_total']:6.0f}s")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
