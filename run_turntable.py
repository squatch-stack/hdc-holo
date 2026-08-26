#!/usr/bin/env python3
"""Turntable of a REAL capture, every frame straight from holographic
bundles: encode the mip once (blur = covariance addition), then each
view folds a projection-slice factor into the cell bundles and reads
out — no geometry, no rasterizer, no per-frame re-encode.

    run_turntable.py [data/scan-tucson.spz | data/train.splat]
                     [--frames N] [--res R]

Outputs results/real_turntable-<name>.gif and a contact-sheet PNG.
Channels are premultiplied (alpha, aR, aG, aB), so the RGB channels
composite correctly on black as-is; exposure is normalized once,
globally, from a high percentile across all frames so the orbit does
not flicker.
"""

import argparse
import os
import time

import numpy as np

from holo.capture import (DIM_R, RENDER_BANDS, SIGMA_MIP, band_codebooks,
                          build_scene, encode_bands, render_mip, render_xray)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default="data/scan-tucson.spz")
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--res", type=int, default=200)
    ap.add_argument("--elev", type=float, default=1.2,
                    help="y (up) component of the orbit direction")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="tone-map exponent: X-ray line integrals through "
                         "the ground plane dominate linear exposure, so "
                         "compress to keep the rest of the scene visible")
    args = ap.parse_args()
    name = os.path.splitext(os.path.basename(args.path))[0]

    t0 = time.time()
    scene, smax, _ = build_scene(args.path)
    mip = render_mip(scene, SIGMA_MIP)
    smax_r = np.sqrt(smax ** 2 + SIGMA_MIP ** 2)
    r_books = band_codebooks(np.random.default_rng(43), RENDER_BANDS, DIM_R)
    bundles, _ = encode_bands(mip, smax_r, r_books, RENDER_BANDS, DIM_R)
    n_cells = sum(len(b) for b in bundles.values())
    nbytes = sum(b.nbytes for band in bundles.values()
                 for b in band.values())
    print(f"encoded {name}: {n_cells} cell bundles, "
          f"{nbytes / 2**20:.0f} MB of complex64, {time.time()-t0:.0f}s")

    center, half, T = [0.5, 0.5, 0.5], 0.5, 2.0
    raw = []
    t1 = time.time()
    for i in range(args.frames):
        az = 2 * np.pi * i / args.frames
        view = [np.cos(az), args.elev, np.sin(az)]
        out = render_xray(bundles, r_books, view, center, half,
                          args.res, T, bands=RENDER_BANDS)
        raw.append(out.reshape(args.res, args.res, 4))
        if i % 6 == 0:
            print(f"  frame {i:>3}/{args.frames} "
                  f"({(time.time()-t1)/(i+1):.1f} s/frame)", flush=True)
    per = (time.time() - t1) / args.frames

    ref = np.percentile(np.stack([f[:, :, 1:4] for f in raw]), 99.0)
    frames = [np.clip(f[:, :, 1:4] / ref, 0, 1) ** args.gamma for f in raw]

    from PIL import Image
    os.makedirs("results", exist_ok=True)
    imgs = [Image.fromarray((f * 255).astype(np.uint8)[::-1])
            for f in frames]
    gif = f"results/real_turntable-{name}.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=110, loop=0)
    sheet = np.concatenate(
        [frames[i] for i in range(0, args.frames,
                                  max(args.frames // 6, 1))][:6], axis=1)
    png = f"results/real_turntable-{name}.png"
    Image.fromarray((sheet * 255).astype(np.uint8)[::-1]).save(png)
    print(f"{args.frames} frames at {args.res}x{args.res}: "
          f"{per:.1f} s/frame; saved {gif}, {png}")


if __name__ == "__main__":
    main()
