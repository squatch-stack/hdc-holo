"""Measure the storage codecs on real-capture bundles (SDK.md 0.2 log).

Capture bundles are the wide-dynamic-range case the HG codec targets
(|S| spans ~1000x within a band): round-trip every saguaro fine-band
cell bundle through HM/HG at 8 and 4 bits, decode the evidence slices
from the round-tripped bundles, and compare against both the
uncompressed decode (drift: fidelity TO THE BUNDLE) and the exact
mixture (fidelity to ground truth). The two metrics diverge on forward
bundles because small components are mostly crosstalk noise — see the
"accidental shrinkage denoiser" entry in SDK.md's 0.2 log.

Usage: examples/run_codec_capture.py [data/scan-tucson.spz] [--shrink]

--shrink adds the deliberate denoiser (holo/denoise.py, issue #1) to the
table: soft and hard shrinkage at magnitude percentiles, and the
shrink-then-persist pairing. That is what the "accidental denoiser"
above should have been all along.
"""

import os
import sys
import time

import numpy as np

from holo.capture import (
    band_codebooks,
    build_scene,
    decode_slice,
    encode_bands,
    exact_slice,
    mass_mode,
    slice_grid,
)
from holo.denoise import percentile_threshold, shrink
from holo.phase import pack_complex, pack_polar, unpack

# repo root: this driver lives in examples/, its assets do not
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCENE = os.path.join(ROOT, "data", "scan-tucson.spz")


def roundtrip(bundles, codec, bits, gamma=0.5):
    out = {}
    for band, cells in bundles.items():
        out[band] = {}
        for k, b in cells.items():
            rb = np.empty_like(b)
            for c in range(b.shape[0]):
                buf = (pack_complex(b[c], bits=bits) if codec == "HM"
                       else pack_polar(b[c], bits=bits, gamma=gamma))
                rb[c] = unpack(buf)
            out[band][k] = rb
    return out


def denoise_all(bundles, pct, mode):
    """Shrink every cell bundle at its own per-channel percentile."""
    return {band: {k: shrink(b, percentile_threshold(b, pct), mode)
                   for k, b in cells.items()}
            for band, cells in bundles.items()}


def main(path, with_shrink=False):
    t0 = time.time()
    scene, smax, box = build_scene(path, verbose=False)
    books = band_codebooks(np.random.default_rng(42))
    bundles, members = encode_bands(scene, smax, books, verbose=False)

    mags = np.concatenate([np.abs(b).ravel()[::7]
                           for b in bundles["fine"].values()])
    q = np.percentile(mags[mags > 0], [50, 99.9])
    print(f"fine-band |S| p99.9/p50 dynamic range: {q[1] / q[0]:.0f}x")

    w = scene.amp[:, 0]
    slices = [("top-down", slice_grid((0, box[0]), (0, box[2]), "y",
                                      mass_mode(scene.mu[:, 1], w, box[1]))),
              ("side", slice_grid((0, box[2]), (0, box[1]), "x",
                                  mass_mode(scene.mu[:, 0], w, box[0])))]
    truth = {n: exact_slice(pts, scene, members) for n, (pts, _) in slices}
    base = {n: decode_slice(pts, bundles, books) for n, (pts, _) in slices}

    d = next(iter(bundles["fine"].values())).shape[1]
    raw = 4 * d * 8
    print(f"{'codec':>9} {'bytes/cell':>11} {'of c64':>7} "
          f"{'top-down vs GT':>15} {'side vs GT':>11} {'drift':>8}")
    for codec, bits in [(None, None), ("HM", 8), ("HG", 8),
                        ("HM", 4), ("HG", 4)]:
        rt = bundles if codec is None else roundtrip(bundles, codec, bits)
        nbytes = raw if codec is None else 4 * (8 + 2 * bits * d // 8)
        errs, drift = [], 0.0
        for n, (pts, _) in slices:
            est = decode_slice(pts, rt, books)
            errs.append(float(np.linalg.norm(est[:, 0] - truth[n][:, 0])
                              / np.linalg.norm(truth[n][:, 0])))
            drift = max(drift, float(
                np.linalg.norm(est[:, 0] - base[n][:, 0])
                / np.linalg.norm(base[n][:, 0])))
        label = "complex64" if codec is None else f"{codec}-{bits}"
        dcol = "—" if codec is None else f"{drift:.4f}"
        print(f"{label:>9} {nbytes:>11,} {nbytes / raw:>6.2f}x "
              f"{errs[0]:>15.3f} {errs[1]:>11.3f} {dcol:>8}")
    if with_shrink:
        print()
        print(f"{'denoiser':>14} {'bytes/cell':>11} {'of c64':>7} "
              f"{'top-down vs GT':>15} {'side vs GT':>11}")
        for pct in (25, 40, 60):
            for mode in ("soft", "hard"):
                sh = denoise_all(bundles, pct, mode)
                errs = [float(np.linalg.norm(
                    decode_slice(pts, sh, books)[:, 0] - truth[n][:, 0])
                    / np.linalg.norm(truth[n][:, 0]))
                    for n, (pts, _) in slices]
                print(f"{mode + ' p' + str(pct):>14} {raw:>11,} {1.0:>6.2f}x "
                      f"{errs[0]:>15.3f} {errs[1]:>11.3f}")
        # the pairing that matters: denoise, THEN persist faithfully
        sh = denoise_all(bundles, 25, "soft")
        for codec, bits in (("HG", 8), ("HM", 4)):
            rt = roundtrip(sh, codec, bits)
            nbytes = 4 * (8 + 2 * bits * d // 8)
            errs = [float(np.linalg.norm(
                decode_slice(pts, rt, books)[:, 0] - truth[n][:, 0])
                / np.linalg.norm(truth[n][:, 0]))
                for n, (pts, _) in slices]
            print(f"{'soft p25 -> ' + codec + '-' + str(bits):>14} "
                  f"{nbytes:>11,} {nbytes / raw:>6.2f}x "
                  f"{errs[0]:>15.3f} {errs[1]:>11.3f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else DEFAULT_SCENE, "--shrink" in sys.argv)
