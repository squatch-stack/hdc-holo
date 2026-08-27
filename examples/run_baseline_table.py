#!/usr/bin/env python3
"""Fidelity per byte: holographic bundles against per-splat codecs.

    examples/run_baseline_table.py [capture] [--points N] [--out FILE]

The comparison the 3DGS compression literature expects, run with ONE
referee so the families are commensurable: every row is scored by
evaluating the field it reconstructs at the same query points and
comparing against the exact Gaussian mixture of the source capture.
Per-splat formats (PLY / SPZ / SOG) lose to quantization; holographic
bundles lose to crosstalk; the referee does not care which.

Read the result honestly — see docs/baselines.md. A bundle is not a
compression format, and this table is where that stops being an
opinion: it is a queryable, mergeable, renderable-without-geometry
field whose size buys something the codecs do not sell. The one axis
where the shapes genuinely differ is density: bundle bytes are fixed
per cell no matter how many splats land in it, while every per-splat
format grows linearly.
"""

import argparse
import os
import sys
import tempfile

import numpy as np

from holo.capture import (
    DIM,
    band_codebooks,
    build_scene,
    decode_slice,
    encode_bands,
    load_ply_sh,
    load_scene_file,
    save_ply,
    save_spz,
)
from holo.phase import pack_complex, unpack
from holo.sog import save_sog
from holo.spectral import eval_scene_exact


def field_error(scene_path, pts, truth, tmp):
    """Rebuild a capture from `scene_path` and score its field."""
    scene, _, _ = build_scene(scene_path, verbose=False)
    got = eval_scene_exact(scene, pts)[:, 0]
    return np.linalg.norm(got - truth) / np.linalg.norm(truth)


def bundle_bytes(bundles, bits=None):
    """complex64 by default, or the packed size at `bits` per part."""
    n = sum(b.size for band in bundles.values() for b in band.values())
    if bits is None:
        return n * 8
    one = next(iter(next(iter(bundles.values())).values()))
    return int(len(pack_complex(one.ravel(), bits=bits)) / one.size * n)


def per_splat_rows(splats, src, src_mb, ref, tmp):
    """PLY / SPZ / SOG, each written from the same splats and scored by
    reconstructing its field. SOG is read back with the spec decoder
    the tests use, then round-tripped through our lossless PLY writer
    so every row reaches the referee the same way."""
    pos, scale, rgba, quat, sh = splats
    pts, truth = ref
    rows = [("PLY (SH-0, our writer)", src_mb,
             field_error(src, pts, truth, tmp), "lossless")]

    p = os.path.join(tmp, "ours.spz")
    save_spz(p, pos, scale, rgba, quat)
    rows.append(("SPZ v3", os.path.getsize(p) / 2**20,
                 field_error(p, pts, truth, tmp), "quantized"))

    p = os.path.join(tmp, "ours.sog")
    save_sog(p, pos, scale, rgba, quat, sh=sh)
    sog_mb = os.path.getsize(p) / 2**20
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tests"))
    from test_sog import _read_sog  # spec decoder
    dpos, dscale, drgba, dquat, _ = _read_sog(p)
    q = os.path.join(tmp, "sog_rt.ply")
    save_ply(q, dpos * [1, -1, -1], dscale, drgba, dquat)
    rows.append(("SOG (SH palette)", sog_mb,
                 field_error(q, pts, truth, tmp), "quantized + palette"))
    return rows


def density_table(splats, books, rng, tmp):
    """How each family scales with density at a fixed scene extent:
    bundle bytes follow occupied cells, per-splat bytes follow content."""
    pos, scale, rgba, quat = splats[:4]
    print("\n  density scaling (same crop, more splats in it):")
    out = ("| splats | SPZ MB | bundle MB (d=8,192) | ratio |\n"
           "|---:|---:|---:|---:|\n")
    for frac in (0.1, 0.25, 0.5, 1.0):
        sub = rng.choice(len(pos), max(int(frac * len(pos)), 2),
                         replace=False)
        sp = os.path.join(tmp, f"d{frac}.spz")
        save_spz(sp, pos[sub], scale[sub], rgba[sub], quat[sub])
        pp = os.path.join(tmp, f"d{frac}.ply")
        save_ply(pp, pos[sub], scale[sub], rgba[sub], quat[sub])
        sc, sm, _ = build_scene(pp, verbose=False)
        bb, _ = encode_bands(sc, sm, books, verbose=False)
        smb, bmb = os.path.getsize(sp) / 2**20, bundle_bytes(bb) / 2**20
        out += (f"| {sc.n:,} | {smb:.2f} | {bmb:.0f} | "
                f"{bmb / max(smb, 1e-9):.0f}x |\n")
        print(f"    {sc.n:>7,} splats: SPZ {smb:5.2f} MB, "
              f"bundles {bmb:6.0f} MB")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", nargs="?",
                    default=os.path.join("data", "iphone", "redrock.ply"))
    ap.add_argument("--points", type=int, default=6_000)
    ap.add_argument("--splats", type=int, default=60_000,
                    help="subsample the capture to this many splats: the "
                         "exact-mixture referee is O(splats x points), and "
                         "every row is scored on the SAME subsample, so "
                         "bytes and error stay commensurable")
    ap.add_argument("--out", default=os.path.join("results",
                                                  "baseline_table.md"))
    args = ap.parse_args()

    src = os.path.abspath(args.capture)
    rng = np.random.default_rng(0)
    pos_a, scale_a, rgba_a, quat_a = load_scene_file(src)
    sh_a = load_ply_sh(src)
    keep = rng.choice(len(pos_a), min(args.splats, len(pos_a)),
                      replace=False)
    pos, scale, rgba, quat = (pos_a[keep], scale_a[keep], rgba_a[keep],
                              quat_a[keep])
    sh = None if sh_a is None else sh_a[keep]
    tmp = tempfile.mkdtemp(prefix="baseline-")
    src = os.path.join(tmp, "subsample.ply")          # the common source
    save_ply(src, pos, scale, rgba, quat)
    scene, smax, _box = build_scene(src, verbose=False)
    # query where the scene actually is: splat centres, jittered off
    # them so this is not a peak-sampling artefact
    pick = scene.mu[rng.choice(scene.n, args.points, replace=False)]
    pts = (pick + rng.normal(0, 0.01, pick.shape)).astype(np.float32)
    truth = eval_scene_exact(scene, pts)[:, 0]
    src_mb = os.path.getsize(src) / 2**20
    print(f"{scene.n:,} splats after crop; {args.points:,} query points; "
          f"referee = exact mixture of the source")

    rows = per_splat_rows((pos, scale, rgba, quat, sh), src, src_mb,
                          (pts, truth), tmp)

    # -- holographic bundles: encode, decode, score -----------------
    for dim in (2048, DIM):
        books = band_codebooks(np.random.default_rng(42), dim=dim)
        bundles, _ = encode_bands(scene, smax, books, dim=dim, verbose=False)
        got = decode_slice(pts, bundles, books)[:, 0]
        err = np.linalg.norm(got - truth) / np.linalg.norm(truth)
        mb = bundle_bytes(bundles) / 2**20
        rows.append((f"holographic bundles (d={dim:,})", mb, err,
                     "crosstalk"))
        if dim == DIM:
            for bits in (8, 4):
                # round-trip every bundle through the codec and re-score:
                # a byte count next to an error implies they belong to
                # the same artifact, so they must
                coded = {b: {k: unpack(pack_complex(v.ravel(), bits=bits))
                             .reshape(v.shape)
                             for k, v in band.items()}
                         for b, band in bundles.items()}
                got_c = decode_slice(pts, coded, books)[:, 0]
                err_c = (np.linalg.norm(got_c - truth)
                         / np.linalg.norm(truth))
                mb_c = bundle_bytes(bundles, bits=bits) / 2**20
                rows.append((f"  same, HM-{bits} codec", mb_c, err_c,
                             "crosstalk + quantization"))

    dens = density_table((pos, scale, rgba, quat), books, rng, tmp)

    hdr = ("| representation | MB | B/splat | field err | loss source |\n"
           "|---|---:|---:|---:|---|\n")
    body = ""
    for name, mb, err, kind in rows:
        body += (f"| {name} | {mb:.1f} | {mb * 2**20 / scene.n:.0f} | "
                 f"{err:.1%} | {kind} |\n")
    table = hdr + body
    print("\n" + table)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("<!-- generated by examples/run_baseline_table.py -->\n\n"
                + table + "\n" + dens)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
