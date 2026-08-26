"""Measure the storage codecs on real-capture bundles (SDK.md 0.2 log).

Capture bundles are the wide-dynamic-range case the HG codec targets
(|S| spans ~1000x within a band): round-trip every saguaro fine-band
cell bundle through HM/HG at 8 and 4 bits, decode the evidence slices
from the round-tripped bundles, and compare against both the
uncompressed decode (drift: fidelity TO THE BUNDLE) and the exact
mixture (fidelity to ground truth). The two metrics diverge on forward
bundles because small components are mostly crosstalk noise — see the
"accidental shrinkage denoiser" entry in SDK.md's 0.2 log.

Usage: run_codec_capture.py [data/scan-tucson.spz]
"""

import os
import sys
import time

import numpy as np

from holo.capture import (band_codebooks, build_scene, decode_slice,
                          encode_bands, exact_slice, mass_mode, slice_grid)
from holo.phase import pack_complex, pack_polar, unpack

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENE = os.path.join(HERE, "data", "scan-tucson.spz")


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


def main(path):
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
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENE)
