"""SOG export — Spatially Ordered Gaussians, the web-delivery format.

SOG (PlayCanvas, format version 2) stores a splat scene as a handful
of lossless WebP images inside a zip: 16-bit log-space positions split
across two images, codebook-indexed scales and DC color, smallest-three
quaternions, and — the reason to prefer it over SPZ — a PALETTE of
higher-order spherical harmonics, the view-dependent term SPZ v2 drops
entirely (73% of a raw 3DGS PLY's bytes; `docs/real-scenes.md`).

Two ideas carry the compression, and both are ours in spirit:

  spatial ordering   splats are Morton-sorted before being written into
                     the image grid, so neighbouring pixels hold nearby
                     splats and the images become smooth — the same
                     locality medicine as `holo/spatial.py`'s cells.
                     WebP then has something to compress; unsorted, the
                     images are noise and the format buys little.
  codebooks          scales, DC color, and the SH palette are vector-
                     quantized, exactly the rate-distortion trade
                     `holo/phase.py` makes for bundles — one layer down
                     the stack, on per-splat attributes instead.

Writer only: the ecosystem's readers (Spark, SuperSplat, PlayCanvas
engine, splat-transform) consume what this produces, and
`tests/test_sog.py` decodes it back with the spec's own arithmetic.
"""

import io
import json
import zipfile

import numpy as np

from .capture import SH_C0, _to_y_up

SOG_VERSION = 2
_SH_COEFFS = {1: 3, 2: 8, 3: 15}      # AC coefficients per band count


def _spread3(v):
    """Interleave a 16-bit integer with two zero bits per bit."""
    v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v


def morton_order(q):
    """Sort order for 16-bit quantized positions (N, 3): Z-order, so
    adjacent pixels hold spatially adjacent splats."""
    key = (_spread3(q[:, 0]) | (_spread3(q[:, 1]) << np.uint64(1))
           | (_spread3(q[:, 2]) << np.uint64(2)))
    return np.argsort(key, kind="stable")


def _codebook(values, size=256):
    """256 representative floats (quantiles — robust to the heavy
    tails real captures have) plus the index of each value."""
    qs = np.linspace(0, 100, size)
    book = np.percentile(values, qs).astype(np.float64)
    book = np.maximum.accumulate(book)                 # non-decreasing
    idx = np.searchsorted(book, values)
    idx = np.clip(idx, 0, size - 1)
    lower = np.clip(idx - 1, 0, size - 1)
    take_lower = np.abs(values - book[lower]) < np.abs(values - book[idx])
    return book, np.where(take_lower, lower, idx).astype(np.uint8)


def _kmeans(X, k, seed=0, iters=6, sample=50_000, chunk=8192):
    """Lloyd's algorithm on a subsample, then assign everything."""
    rng = np.random.default_rng(seed)
    n = len(X)
    sub = X[rng.choice(n, min(n, sample), replace=False)]
    C = sub[rng.choice(len(sub), min(k, len(sub)), replace=False)].copy()
    if len(C) < k:                                      # tiny inputs
        C = np.concatenate([C, np.zeros((k - len(C), X.shape[1]),
                                        X.dtype)])

    def assign(A):
        out = np.empty(len(A), np.int32)
        cn = (C ** 2).sum(1)
        for i in range(0, len(A), chunk):
            blk = A[i:i + chunk]
            d = cn[None, :] - 2.0 * (blk @ C.T)
            out[i:i + chunk] = np.argmin(d, axis=1)
        return out

    for _ in range(iters):
        lab = assign(sub)
        for j in np.unique(lab):
            C[j] = sub[lab == j].mean(0)
    return C, assign(X)


def _grid(count, align=64):
    """Image dimensions with width*height >= count (spec: trailing
    pixels are ignored); width aligned so rows stay tidy."""
    w = int(np.ceil(np.sqrt(count)))
    w = int(np.ceil(w / align) * align)
    return w, int(np.ceil(count / w))


def _image(channels, w, h):
    """Stack uint8 channel arrays into a (h, w, C) image, zero-padded."""
    c = len(channels)
    img = np.zeros((h * w, c), np.uint8)
    n = len(channels[0])
    for i, ch in enumerate(channels):
        img[:n, i] = ch
    return img.reshape(h, w, c)


def save_sog(path, pos, scale, rgba, quat, sh=None, sh_clusters=1024,
             seed=0, generator="holo"):
    """Write a bundled `.sog` (zip of meta.json + lossless WebPs).

    Takes the loader-level representation `load_scene_file` returns —
    y-up world, linear color and alpha in [0, 1], wxyz quaternions —
    and writes the ecosystem's y-down convention, as `save_ply` does.

    sh: optional (N, 3, K) higher-order SH in the FILE's frame (what
    `load_ply_sh` returns), K in {3, 8, 15} for 1..3 bands. Omit it
    and the export is DC-only, like SPZ v2.

    `sh_clusters` is the SH palette size. The format allows up to
    65536, but READERS vary: Spark 2.1.0 renders 256 and 1024 palettes
    and shows nothing at 2048 (measured on Red Rock — the file decodes
    correctly either way under the spec's own arithmetic, so this is a
    reader ceiling, not a writer bug). 1024 is the tested default;
    raise it only against a reader you have verified.

    Lossy, by design and by measurement (`docs/real-scenes.md`):
    positions to 16 bits in a symmetric-log space, scales and DC color
    through 256-entry codebooks, quaternions smallest-three at 8 bits
    per component, and higher-order SH through a `sh_clusters`-entry
    palette. Requires Pillow (`pip install -e '.[viz]'`).
    """
    from PIL import Image

    pos = np.asarray(pos, np.float64)
    quat = np.asarray(quat, np.float64)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True),
                             1e-9)
    pos, quat = _to_y_up(pos, quat)                  # back to on-disk RDF
    rgba = np.asarray(rgba, np.float32)
    scale = np.maximum(np.asarray(scale, np.float64), 1e-12)
    n = len(pos)

    # -- positions: symmetric log, 16-bit, Morton-ordered ------------
    lg = np.sign(pos) * np.log1p(np.abs(pos))
    mins, maxs = lg.min(0), lg.max(0)
    span = np.maximum(maxs - mins, 1e-12)
    q16 = np.clip(np.round((lg - mins) / span * 65535), 0, 65535) \
        .astype(np.uint32)
    order = morton_order(q16)
    q16, pos, scale, rgba, quat = (q16[order], pos[order], scale[order],
                                   rgba[order], quat[order])

    w, h = _grid(n)
    files, meta_extra = {}, {}
    files["means_l.webp"] = _image([(q16[:, i] & 0xFF).astype(np.uint8)
                                    for i in range(3)], w, h)
    files["means_u.webp"] = _image([(q16[:, i] >> 8).astype(np.uint8)
                                    for i in range(3)], w, h)

    # -- scales: one 256-entry codebook in log space -----------------
    s_book, s_idx = _codebook(np.log(scale).ravel())
    files["scales.webp"] = _image([s_idx.reshape(n, 3)[:, i]
                                   for i in range(3)], w, h)

    # -- DC color + opacity ------------------------------------------
    dc = (np.clip(rgba[:, :3], 0, 1) - 0.5) / SH_C0
    c_book, c_idx = _codebook(dc.ravel().astype(np.float64))
    alpha = np.round(np.clip(rgba[:, 3], 0, 1) * 255).astype(np.uint8)
    files["sh0.webp"] = _image([c_idx.reshape(n, 3)[:, 0],
                                c_idx.reshape(n, 3)[:, 1],
                                c_idx.reshape(n, 3)[:, 2], alpha], w, h)

    # -- quaternions: smallest three ---------------------------------
    drop = np.argmax(np.abs(quat), axis=1)
    sign = np.where(quat[np.arange(n), drop] < 0, -1.0, 1.0)
    q = quat * sign[:, None]
    mask = np.ones((n, 4), bool)
    mask[np.arange(n), drop] = False
    keep = q[mask].reshape(n, 3)
    comp = np.round(np.clip(keep * np.sqrt(2) / 2 + 0.5, 0, 1) * 255) \
        .astype(np.uint8)
    files["quats.webp"] = _image([comp[:, 0], comp[:, 1], comp[:, 2],
                                  (252 + drop).astype(np.uint8)], w, h)

    # -- higher-order SH: a palette of per-splat AC vectors ----------
    if sh is not None:
        sh = np.asarray(sh, np.float64)[order]        # (N, 3, K)
        k_coef = sh.shape[2]
        bands = {v: b for b, v in _SH_COEFFS.items()}[k_coef]
        k = int(min(sh_clusters, n))
        C, labels = _kmeans(sh.reshape(n, -1), k, seed=seed)
        sh_book, sh_idx = _codebook(C.ravel())
        cw = 64 * k_coef
        ch = int(np.ceil(k / 64))
        cent = np.zeros((ch, cw, 3), np.uint8)
        idx3 = sh_idx.reshape(k, 3, k_coef)
        for e in range(k):
            u0, v = (e % 64) * k_coef, e // 64
            cent[v, u0:u0 + k_coef, :] = idx3[e].T     # (K, 3) RGB
        files["shN_centroids.webp"] = cent
        files["shN_labels.webp"] = _image(
            [(labels & 0xFF).astype(np.uint8),
             ((labels >> 8) & 0xFF).astype(np.uint8)], w, h)
        meta_extra["shN"] = {"count": k, "bands": bands,
                             "codebook": [float(v) for v in sh_book],
                             "files": ["shN_centroids.webp",
                                       "shN_labels.webp"]}

    meta = {
        "version": SOG_VERSION,
        "asset": {"generator": generator},
        "count": int(n),
        "means": {"mins": [float(v) for v in mins],
                  "maxs": [float(v) for v in maxs],
                  "files": ["means_l.webp", "means_u.webp"]},
        "scales": {"codebook": [float(v) for v in s_book],
                   "files": ["scales.webp"]},
        "quats": {"files": ["quats.webp"]},
        "sh0": {"codebook": [float(v) for v in c_book],
                "files": ["sh0.webp"]},
        **meta_extra,
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("meta.json", json.dumps(meta))
        for name, arr in files.items():
            if arr.shape[2] == 2:                      # labels: R, G
                arr = np.concatenate(
                    [arr, np.zeros_like(arr[:, :, :1])], axis=2)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, "WEBP", lossless=True,
                                      quality=100, method=4)
            z.writestr(name, buf.getvalue())
    return meta


__all__ = ["SOG_VERSION", "morton_order", "save_sog"]
