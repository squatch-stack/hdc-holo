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


def _describe_crop(path, row):
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
    radius = weighted_quantile(np.abs(pos - center).max(axis=1), a, 0.75)
    row["splats_loaded"] = len(rgba)
    row["crop_extent"] = float(2 * 1.2 * radius)


def _slices(scene, box, bundles, members, books, row):
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
        truth = exact_slice(pts, scene, members)
        t2 = time.time()
        holo = decode_slice(pts, bundles, books)
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


def _xrays(scene, smax, members, row):
    """The two orthographic X-ray views, scored against the mip."""
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
    r_bundles, r_members = encode_bands(mip, smax_r, r_books,
                                        RENDER_BANDS, DIM_R)
    center_p, half, T, res = [0.5, 0.5, 0.5], 0.5, 2.0, 176
    xpanels = []
    for key, view in [("xray_a", [1.0, 0.0, 0.25]),
                      ("xray_b", [1.0, 0.0, 1.0])]:
        t1 = time.time()
        sharp = exact_xray(scene, members, view, center_p, half, res)
        mip_gt = exact_xray(mip, r_members, view, center_p, half, res,
                            bands=RENDER_BANDS)
        t2 = time.time()
        holo = render_xray(r_bundles, r_books, view, center_p, half, res, T,
                           bands=RENDER_BANDS)
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


def run_one(path, figures=None):
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
           "file": os.path.basename(path)}

    scene, smax, box = build_scene(path)
    row["splats_encoded"] = int(scene.n)
    row["box"] = [float(b) for b in box]
    _describe_crop(path, row)

    bidx = band_of(smax, BANDS)
    row["band_split"] = {name: int(np.sum(bidx == b))
                         for b, (name, _c, _z) in enumerate(BANDS)}

    books = band_codebooks(np.random.default_rng(42))
    t_enc = time.time()
    bundles, members = encode_bands(scene, smax, books)
    row["t_encode"] = round(time.time() - t_enc, 1)
    row["cells"] = int(sum(len(b) for b in bundles.values()))
    row["cells_per_band"] = {k: len(v) for k, v in bundles.items()}

    panels, errs = _slices(scene, box, bundles, members, books, row)
    xpanels, xerrs = _xrays(scene, smax, members, row)
    errs.update(xerrs)

    row["err"] = {k: round(v, 4) for k, v in errs.items()}
    row["t_total"] = round(time.time() - t0, 1)
    row["dim"] = DIM
    row["dim_render"] = DIM_R
    try:
        import cupy
        row["peak_vram_gb"] = round(
            cupy.get_default_memory_pool().used_bytes() / 1e9, 2)
        row["backend"] = "cupy-cuda"
    except ImportError:
        row["backend"] = "numpy"

    if figures:
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
        for r, (field, label) in enumerate([
                (sharp, "analytic line integrals, full detail"),
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
            row = run_one(path, figures=figdir)
        except Exception as exc:
            # One unloadable capture must not cost the other ten their
            # run; the sweep is long and unattended.
            traceback.print_exc()
            row = {"scene": os.path.splitext(os.path.basename(path))[0],
                   "file": os.path.basename(path),
                   "error": f"{type(exc).__name__}: {exc}"}
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
