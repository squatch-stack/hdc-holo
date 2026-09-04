#!/usr/bin/env python3
"""The public scene benchmark: encode -> slice -> score, one command.

    python bench/scene_bench.py [scene.ply|.spz|.splat] [--pix N] [--json out.json]

Runs the fixed real-capture pipeline on one scene and prints what the
HDC/VSA community's spatial benchmarks do not have: a superposed
representation exercised at capture scale, scored against an exact
analytic referee rather than against images, with a determinism
checksum a second machine can compare to ~1e-8.

Stages, each timed:
  1. load + normalize     (build_scene: alpha floor, mass-centered crop)
  2. banded encode        (scale bands x spatial cells, mixture codebooks)
  3. two axis slices      exact-mixture ground truth vs decoded-from-
                          bundles, relative alpha error
  4. checksums            float64 magnitude sums per band - deterministic
                          on one machine, cross-machine agreement ~1e-8
                          (byte-identity is NOT expected across BLAS
                          implementations; the sums are the contract)

The referee is the expensive part by design - it is the check, not the
product. --pix trades referee resolution for time (default 112 gives a
112x112 grid per slice; the evidence figures in docs/real-scenes.md use
224).

A released capture to run this against: the Brookline springhouse scan
(data/brookline-station.ply here; published with the studio's gallery).
Numbers this reproduces are registered in claims/registry.jsonl
(capture.err_brookline and friends).
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holo.capture import (  # noqa: E402
    band_codebooks,
    build_scene,
    decode_slice,
    encode_bands,
    exact_slice,
    mass_mode,
    slice_grid,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default = next(
        (p for p in (
            os.path.join(ROOT, "data", "brookline-station.ply"),
            os.path.join(ROOT, "data", "iphone", "redrock.ply"),
        ) if os.path.exists(p)), None)
    ap.add_argument("scene", nargs="?", default=default)
    ap.add_argument("--pix", type=int, default=112,
                    help="referee grid edge per slice (default 112)")
    ap.add_argument("--json", help="write the result table to this path")
    args = ap.parse_args()
    if not args.scene:
        ap.error("no scene given and no default capture present in data/")

    report = {"scene": os.path.basename(args.scene), "pix": args.pix}

    t0 = time.time()
    scene, smax, box = build_scene(args.scene)
    report["splats"] = int(scene.n)
    report["load_s"] = round(time.time() - t0, 2)

    t1 = time.time()
    books = band_codebooks(np.random.default_rng(42))
    bundles, members = encode_bands(scene, smax, books)
    report["encode_s"] = round(time.time() - t1, 2)
    report["cells"] = int(sum(len(b) for b in bundles.values()))

    # Determinism contract: float64 sums of bundle magnitudes, per band.
    # Bit-identical on one machine; agreeing to ~1e-8 across backends.
    report["checksums"] = {
        band: float(np.sum(np.abs(np.asarray(list(cells.values())))))
        for band, cells in bundles.items() if cells
    }

    w = scene.amp[:, 0]
    y_mode = mass_mode(scene.mu[:, 1], w, box[1])
    x_mode = mass_mode(scene.mu[:, 0], w, box[0])

    # Same slices as the evidence figures, at the requested referee size.
    global_pix = 1.0 / args.pix
    slices = []
    for name, (pts, shape) in [
        ("top-down", slice_grid((0, box[0]), (0, box[2]), "y", y_mode,
                                pix=global_pix)),
        ("side", slice_grid((0, box[2]), (0, box[1]), "x", x_mode,
                            pix=global_pix)),
    ]:
        ta = time.time()
        truth = exact_slice(pts, scene, members)
        tb = time.time()
        holo = decode_slice(pts, bundles, books)
        tc = time.time()
        err = float(np.linalg.norm(holo[:, 0] - truth[:, 0])
                    / np.linalg.norm(truth[:, 0]))
        slices.append({
            "slice": name, "pixels": int(len(pts)),
            "referee_s": round(tb - ta, 2), "decode_s": round(tc - tb, 2),
            "rel_err": round(err, 4),
        })
        print(f"{name:9s} {len(pts):7,d} px  referee {tb - ta:6.1f}s  "
              f"decode {tc - tb:6.1f}s  alpha rel err {err:.3f}")
    report["slices"] = slices

    digest = hashlib.sha256(
        json.dumps(report["checksums"], sort_keys=True).encode()).hexdigest()[:16]
    print(f"\n{report['splats']:,} splats -> {report['cells']} cell bundles "
          f"in {report['encode_s']}s;  checksum digest {digest}")
    report["digest"] = digest

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
