"""Real-capture pipeline: pretrained Gaussian-splat scenes as bundles.

Loaders for the two common interchange formats, a mass-centered crop
(real captures put most splats in a far background shell), scale
clamping (a resolution floor + a reach cap), scale-banded spatial
chunking with per-band mixture codebooks, cell-local decode/ground
truth for slices, and closed-form X-ray projection straight from the
cell bundles (with the dedicated mip encode projections need).
`docs/real-scenes.md` carries the narrative; `examples/run_real_scene.py` is the
example driver that produced the evidence figures.

Formats (both byte-verified against the reference implementations):
  .splat (antimatter15) — 32 B/splat: float32[3] pos, float32[3] linear
    scale, u8[4] RGBA (alpha = opacity), u8[4] quaternion ((v-128)/128,
    w first).
  .spz v2/v3 (Niantic, gzip legacy layout from nianticlabs/spz) —
    16 B header {magic, version, numPoints, shDegree, fracBits,
    flags}, then per-attribute sections: positions 24-bit signed
    fixed point, alpha u8 (sigmoid), color u8[3] (SH DC:
    (v/255-0.5)/0.15), scale u8[3] (log: v/16-10), rotation, then SH
    ((v-128)/128, coefficient-major with color varying fastest —
    `parse_spz_sh`). v3 changed rotations only: smallest-three, 4 B
    (2-bit index of the omitted largest component in the top bits,
    then three sign+9-bit magnitudes) instead of v2's 3 B xyz. The
    byte count 16 + N*(16 + rot + shDim*3) must match exactly.
    v4 replaced the container with per-attribute ZSTD streams behind
    a 32 B plaintext header: `spz_header` reads it, `parse_spz`
    refuses it with instructions.

Hard-won codebook rules (violations render as structured artifacts, not
subtle noise — see docs/spectral.md): every band's mixture must span
down to the GLOBAL scale floor (bands are assigned by max axis scale,
but needle splats keep thin axes at the floor), and the finest
component sits at beta = 1 (sigma_rho = 1/floor) so floor-scale
directions sample as unit phasors.
"""

import gzip
import struct
import time

import numpy as np

from . import accel as _accel
from .spectral import SplatScene, decode_weights, sample_frequencies, spectral_bundle

# axis scales are clamped to [S_LO, S_HI] of the normalized scene extent
# — a resolution floor (min) and a cap bounding every cell's query reach
# (max); real 3DGS scale distributions span 5 decades
S_LO, S_HI = 0.002, 0.05
ALPHA_MIN = 0.1
# bands by max axis scale: (name, upper cap, cell size); reach = 3 * cap.
# The xfine/fine split at 0.004 exists because reach FOLLOWS the cap:
# floor-scale splats (most of a real capture) don't need 0.024 of reach,
# and halving it cuts each query's in-reach crosstalk volume ~3x —
# measured 33-44% slice-error reduction on the saguaro capture for 1.5x
# storage. Raising d instead does NOT buy this (2-4% for 1.5x memory):
# dense-scene residual error is coherent, not Monte-Carlo.
BANDS = [
    ("xfine",  0.004, 1 / 32),
    ("fine",   0.008, 1 / 32),
    ("mid",    0.02,  1 / 8),
    ("coarse", S_HI,  1 / 4),
]
DIM = 8192
PIX = 1.0 / 224             # slice pixel size in normalized units
SH_C0 = 0.28209479177

# X-ray renders use a dedicated mip encode: only frequencies nearly
# perpendicular to the view direction carry projection signal, so the
# scene is blurred by SIGMA_MIP (concentrating its spectrum) and encoded
# at DIM_R so the perpendicular slice holds enough effective dimensions
SIGMA_MIP = 0.008
DIM_R = 32768
RENDER_BANDS = [
    ("r-fine",   0.02,  1 / 8),
    ("r-coarse", 0.055, 1 / 4),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _to_y_up(pos, quat):
    """Normalize a right-down-front (COLMAP/3DGS) scene to the y-up
    right-up-back world every other loader already produces: rotate
    180 deg about x. Positions (x, y, z) -> (x, -y, -z); each splat
    quaternion is premultiplied by that rotation, r = (0, 1, 0, 0):
    (w, x, y, z) -> (-x, w, -z, y). Scales are rotation-invariant.

    Raw 3DGS `.ply` (INRIA layout, what training pipelines and
    Scaniverse's raw export emit) and antimatter15 `.splat` arrive
    y-DOWN and render upside down without this; `.spz` is specified
    y-up (the official PLY->SPZ conversion applies the same flip) and
    ARKit LiDAR clouds are gravity-aligned y-up — verified empirically
    on all five in-house captures (see SDK.md's 0.2 log).
    """
    pos = pos.copy()
    pos[:, 1] *= -1.0
    pos[:, 2] *= -1.0
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    quat = np.stack([-x, w, -z, y], axis=1)
    return pos, quat


def load_splat(path):
    raw = np.fromfile(path, dtype=np.uint8).reshape(-1, 32)
    floats = raw[:, :24].copy().view(np.float32).reshape(-1, 6)
    pos = floats[:, :3].astype(np.float64)
    scale = floats[:, 3:6].astype(np.float64)
    rgba = raw[:, 24:28].astype(np.float32) / 255.0
    quat = (raw[:, 28:32].astype(np.float64) - 128.0) / 128.0
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    pos, quat = _to_y_up(pos, quat)
    return pos, scale, rgba, quat


SPZ_MAGIC = 0x5053474E
SPZ_SH_DIM = {0: 0, 1: 3, 2: 8, 3: 15}       # coefficients per degree


def load_spz(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    return parse_spz(buf)


def spz_header(buf):
    """Identify any SPZ file from its first bytes.

    Returns a dict with at least `version`. Versions 1-3 share the
    16-byte legacy header inside a gzip stream; version 4 replaced it
    with a 32-byte plaintext NGSP header ahead of per-attribute ZSTD
    streams — deliberately readable without decompressing anything,
    which is why this works on a raw v4 file while `parse_spz` cannot
    yet decode one (see `_SPZ_V4_HELP`).
    """
    if buf[:2] == b"\x1f\x8b":                      # gzip: legacy
        # decompress only what the header needs, so a short read of a
        # big file still identifies it
        import zlib
        buf = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(buf, 64)
    magic, ver = struct.unpack_from("<II", buf, 0)
    if magic != SPZ_MAGIC:
        raise ValueError("not an SPZ file")
    if ver >= 4:
        (_, _, n, sh_deg, frac_bits, flags, n_streams,
         toc) = struct.unpack_from("<IIIBBBBI", buf, 0)
        return {"version": ver, "count": n, "sh_degree": sh_deg,
                "fractional_bits": frac_bits, "flags": flags,
                "streams": n_streams, "toc_offset": toc,
                "container": "zstd"}
    _, _, n, sh_deg, frac_bits, flg, _ = struct.unpack_from(
        "<IIIBBBB", buf, 0)
    return {"version": ver, "count": n, "sh_degree": sh_deg,
            "fractional_bits": frac_bits, "flags": flg,
            "container": "gzip"}


_SPZ_V4_HELP = (
    "SPZ v{ver} stores per-attribute ZSTD streams, not one gzip blob. "
    "Python has no stdlib zstd before 3.14 and this SDK takes no new "
    "runtime dependency without review, so v4 decoding is not built in. "
    "Options: convert with the reference tool (`splat-transform in.spz "
    "out.ply`), or open an issue to add an optional zstd extra. "
    "`spz_header()` reads v4 metadata without decompressing."
)


def _spz_quat_v3(buf, o, n):
    """Smallest-three rotations (v3+): a 2-bit index of the OMITTED
    (largest) component in the top bits, then three 9-bit magnitudes
    each with a sign bit, most-significant component first. Mirrors
    nianticlabs/spz `unpackQuaternionSmallestThree`.
    """
    r = np.frombuffer(buf, np.uint8, n * 4, o).reshape(n, 4) \
        .astype(np.uint32)
    comp = r[:, 0] | (r[:, 1] << 8) | (r[:, 2] << 16) | (r[:, 3] << 24)
    i_largest = (comp >> 30).astype(np.int64)
    mask = (1 << 9) - 1
    quat = np.zeros((n, 4), np.float64)          # xyzw, per the reference
    rows = np.arange(n)
    for i in range(3, -1, -1):                   # low bits are index 3
        active = i != i_largest
        mag = (comp & mask).astype(np.float64)
        neg = ((comp >> 9) & 1).astype(bool)
        val = np.sqrt(0.5) * mag / mask
        val = np.where(neg, -val, val)
        quat[rows[active], i] = val[active]
        comp = np.where(active, comp >> 10, comp)
    ss = (quat ** 2).sum(axis=1)
    quat[rows, i_largest] = np.sqrt(np.clip(1.0 - ss, 0.0, None))
    return np.concatenate([quat[:, 3:4], quat[:, :3]], axis=1)  # -> wxyz


def _spz_sections(buf):
    """Header, per-point byte offsets, and sizes for a legacy SPZ."""
    magic, ver, n, sh_deg, frac_bits, _flags, _ = struct.unpack_from(
        "<IIIBBBB", buf, 0)
    if magic != SPZ_MAGIC:
        raise ValueError("not an SPZ file")
    if ver >= 4:
        raise NotImplementedError(_SPZ_V4_HELP.format(ver=ver))
    if ver not in (2, 3):
        raise ValueError(f"unsupported legacy SPZ version {ver}")
    sh_dim = SPZ_SH_DIM[sh_deg]
    rot = 3 if ver == 2 else 4                   # v3 packs 4 bytes
    expect = 16 + n * (9 + 1 + 3 + 3 + rot + sh_dim * 3)
    if len(buf) != expect:
        raise ValueError(f"size mismatch: {len(buf)} vs {expect}")
    o = {"pos": 16}
    o["alpha"] = o["pos"] + n * 9
    o["color"] = o["alpha"] + n
    o["scale"] = o["color"] + n * 3
    o["rot"] = o["scale"] + n * 3
    o["sh"] = o["rot"] + n * rot
    return ver, n, sh_deg, sh_dim, frac_bits, o


def parse_spz(buf):
    """Parse decompressed legacy SPZ bytes (v2 or v3).

    v3 differs from v2 in exactly one place — rotations became
    smallest-three (2-bit index + three 10-bit signed components,
    4 bytes) instead of 8-bit xyz (3 bytes) — which also removes v2's
    ill-conditioning near 180-degree rotations, since the component
    reconstructed from the others is always the largest.
    """
    ver, n, _sh_deg, _sh_dim, frac_bits, o = _spz_sections(buf)
    p24 = np.frombuffer(buf, np.uint8, n * 9, o["pos"]).reshape(n, 3, 3) \
        .astype(np.int32)
    fixed = p24[..., 0] | (p24[..., 1] << 8) | (p24[..., 2] << 16)
    fixed -= (fixed & 0x800000) << 1              # sign extension
    pos = fixed.astype(np.float64) / (1 << frac_bits)
    alpha = np.frombuffer(buf, np.uint8, n, o["alpha"]) \
        .astype(np.float32) / 255.0
    col = np.frombuffer(buf, np.uint8, n * 3, o["color"]).reshape(n, 3) \
        .astype(np.float32)
    color = np.clip(0.5 + SH_C0 * (col / 255.0 - 0.5) / 0.15, 0, 1)
    slog = np.frombuffer(buf, np.uint8, n * 3, o["scale"]).reshape(n, 3) \
        .astype(np.float64)
    scale = np.exp(slog / 16.0 - 10.0)
    if ver == 2:
        r3 = np.frombuffer(buf, np.uint8, n * 3, o["rot"]).reshape(n, 3) \
            .astype(np.float64)
        xyz = (r3 - 127.5) / 127.5
        w = np.sqrt(np.clip(1.0 - (xyz**2).sum(axis=1), 0.0, None))
        quat = np.concatenate([w[:, None], xyz], axis=1)   # -> (w,x,y,z)
    else:
        quat = _spz_quat_v3(buf, o["rot"], n)
    rgba = np.concatenate([color, alpha[:, None]], axis=1)
    return pos, scale, rgba, quat


def parse_spz_sh(buf):
    """Higher-order SH from legacy SPZ bytes as (N, 3, K), or None.

    Every SPZ version carries SH when the header's `shDegree` says so
    — the DC-only files this pipeline writes are a choice of OUR
    writer, not a limit of the format. Bytes are coefficient-major
    with the color channel varying fastest; `unquantizeSH` is
    (x - 128) / 128.
    """
    _ver, n, _sh_deg, sh_dim, _frac, o = _spz_sections(buf)
    if sh_dim == 0:
        return None
    raw = np.frombuffer(buf, np.uint8, n * sh_dim * 3, o["sh"]) \
        .reshape(n, sh_dim, 3).astype(np.float32)
    return np.transpose((raw - 128.0) / 128.0, (0, 2, 1))   # -> (N,3,K)


def load_spz_sh(path):
    """`parse_spz_sh` for a file on disk."""
    with gzip.open(path, "rb") as f:
        return parse_spz_sh(f.read())


def load_ply(path, sigma_scale=0.75):
    """Point-cloud PLY -> points as isotropic splats (iPhone LiDAR
    captures: EigenCapture's binary xyz scans, the SceneDepthPointCloud
    sample's ascii xyz+rgb clouds). Not a Gaussian-splat PLY parser —
    plain clouds only; each point becomes an isotropic splat with sigma
    = sigma_scale * (median nearest-neighbor distance, estimated on a
    subsample), full opacity, identity rotation, and the point's color
    (white when the cloud is colorless)."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply", "not a PLY file"
        fmt = None
        n = 0
        props = []
        while True:
            line = f.readline().strip()
            if line == b"end_header":
                break
            parts = line.split()
            if parts[0] == b"format":
                fmt = parts[1].decode()
            elif parts[0] == b"element":
                in_vertex = parts[1] == b"vertex"
                if in_vertex:
                    n = int(parts[2])
            elif parts[0] == b"property" and in_vertex \
                    and parts[1] != b"list":
                props.append((parts[2].decode(), parts[1].decode()))
        names = [p[0] for p in props]
        assert names[:3] == ["x", "y", "z"], f"unsupported layout {names}"
        if {"f_dc_0", "opacity", "scale_0", "rot_0"} <= set(names):
            # full 3DGS Gaussian PLY (INRIA layout; Scaniverse raw
            # exports use it): SH DC color, sigmoid opacity, log scales,
            # wxyz quaternion. Ignores normals and higher SH bands.
            assert fmt == "binary_little_endian", fmt
            dt = np.dtype([(nm, "<f4") for nm, _ in props])
            rec = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
            pos = np.stack([rec["x"], rec["y"], rec["z"]], 1) \
                .astype(np.float64)
            color = np.clip(0.5 + SH_C0 * np.stack(
                [rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"]], 1), 0, 1) \
                .astype(np.float32)
            alpha = (1.0 / (1.0 + np.exp(-rec["opacity"])))[:, None] \
                .astype(np.float32)
            scale = np.exp(np.stack(
                [rec["scale_0"], rec["scale_1"], rec["scale_2"]], 1)) \
                .astype(np.float64)
            quat = np.stack([rec["rot_0"], rec["rot_1"], rec["rot_2"],
                             rec["rot_3"]], 1).astype(np.float64)
            quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True),
                               1e-9)
            pos, quat = _to_y_up(pos, quat)
            return pos, scale, np.concatenate([color, alpha], 1), quat
        has_rgb = names[3:6] == ["red", "green", "blue"]
        if fmt == "binary_little_endian":
            typemap = {"float": "<f4", "uchar": "u1", "double": "<f8"}
            dt = np.dtype([(nm, typemap[t]) for nm, t in props])
            rec = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
            pos = np.stack([rec["x"], rec["y"], rec["z"]], 1) \
                .astype(np.float64)
            rgb = (np.stack([rec["red"], rec["green"], rec["blue"]], 1)
                   .astype(np.float32) / 255.0) if has_rgb else None
        elif fmt == "ascii":
            data = np.loadtxt(f, dtype=np.float64, max_rows=n,
                              usecols=range(len(props)))
            pos = data[:, :3]
            rgb = (data[:, 3:6] / 255.0).astype(np.float32) \
                if has_rgb else None
        else:
            raise AssertionError(f"unsupported PLY format {fmt}")

    # sigma from the cloud's own sampling density (subsampled brute NN)
    rng = np.random.default_rng(0)
    sub = pos[rng.choice(len(pos), min(2000, len(pos)), replace=False)]
    d2 = ((sub[None, :, :] - sub[:, None, :]) ** 2).sum(-1)
    d2[np.diag_indices_from(d2)] = np.inf
    sigma = sigma_scale * float(np.median(np.sqrt(d2.min(1))))
    scale = np.full((len(pos), 3), sigma)
    rgba = np.concatenate(
        [rgb if rgb is not None else np.ones((len(pos), 3), np.float32),
         np.ones((len(pos), 1), np.float32)], axis=1)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (len(pos), 1))
    return pos, scale, rgba, quat


def load_scene_file(path):
    p = str(path)
    if p.endswith(".spz"):
        return load_spz(p)
    if p.endswith(".ply"):
        return load_ply(p)
    return load_splat(p)


def save_ply(path, pos, scale, rgba, quat):
    """Write splats as a raw 3DGS Gaussian PLY (INRIA layout, SH deg 0)
    — the bridge OUT of this pipeline into the display ecosystem
    (splat-transform, SuperSplat, Spark, engine plugins).

    Takes the loader-level representation `load_scene_file` returns —
    y-up world, linear color and alpha in [0, 1], wxyz quaternions —
    and writes the ecosystem's conventions: COLMAP-style y-DOWN axes
    (the inverse of `_to_y_up`, which is its own inverse; viewers
    apply their own flip, so a y-up file would render upside down),
    SH DC color, logit opacity, log scales. Round trip through
    `load_ply` reproduces the inputs to float32 rounding (alpha is
    clamped to (1e-6, 1-1e-6) so the logit stays finite; quaternion
    sign may flip — same rotation).
    """
    pos = np.asarray(pos, np.float64)
    quat = np.asarray(quat, np.float64)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True),
                             1e-9)
    pos, quat = _to_y_up(pos, quat)            # back to on-disk RDF
    color = np.asarray(rgba, np.float32)[:, :3]
    alpha = np.clip(np.asarray(rgba, np.float32)[:, 3], 1e-6, 1 - 1e-6)
    s = np.maximum(np.asarray(scale, np.float64), 1e-12)
    fields = ["x", "y", "z", "nx", "ny", "nz",
              "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]
    rec = np.zeros(len(pos), dtype=np.dtype([(f, "<f4") for f in fields]))
    for i, ax in enumerate("xyz"):
        rec[ax] = pos[:, i]
        rec[f"f_dc_{i}"] = (color[:, i] - 0.5) / SH_C0
        rec[f"scale_{i}"] = np.log(s[:, i])
    rec["opacity"] = np.log(alpha / (1.0 - alpha))
    for i in range(4):
        rec[f"rot_{i}"] = quat[:, i]
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n"
                + f"element vertex {len(pos)}\n".encode()
                + b"".join(f"property float {fl}\n".encode()
                           for fl in fields)
                + b"end_header\n" + rec.tobytes())


def load_ply_sh(path):
    """Higher-order SH from a 3DGS PLY as (N, 3, K), or None.

    The view-dependent term `load_ply` drops: 73% of a raw capture's
    bytes for ~10% of its color energy (`docs/real-scenes.md`), which
    only SOG export currently carries. INRIA writes `f_rest_i` in
    channel-major order (i = channel * K + coefficient). Returned in
    the FILE's frame — no y-up flip, since rotating a scene rotates
    the SH basis too, and the export writes y-down anyway.
    """
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply", "not a PLY file"
        props, n, fmt = [], 0, None
        while True:
            line = f.readline().strip()
            if line == b"end_header":
                break
            parts = line.split()
            if parts[0] == b"format":
                fmt = parts[1].decode()
            elif parts[0] == b"element":
                in_vertex = parts[1] == b"vertex"
                if in_vertex:
                    n = int(parts[2])
            elif parts[0] == b"property" and in_vertex \
                    and parts[1] != b"list":
                props.append(parts[2].decode())
        rest = [p for p in props if p.startswith("f_rest_")]
        if not rest or fmt != "binary_little_endian":
            return None
        dt = np.dtype([(nm, "<f4") for nm in props])
        rec = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    k = len(rest) // 3
    if k not in (3, 8, 15):
        return None
    return np.stack([np.stack([rec[f"f_rest_{c * k + i}"]
                               for i in range(k)], 1)
                     for c in range(3)], 1).astype(np.float32)


def save_spz(path, pos, scale, rgba, quat, frac_bits=12, version=3):
    """Write splats as legacy SPZ (v3 by default, v2 on request) — the
    compressed delivery format (~20 B/splat vs ~68 B in an SH-0 PLY;
    the exact inverse of `parse_spz`, so round trips are
    quantization-only).

    Quantization grid (measured on Red Rock in SDK.md's log):
    positions to 24-bit fixed point (2^-frac_bits units), scales to a
    log-u8 grid (~6% relative), alpha and color to u8.

    Rotations are where the versions differ, and why v3 is the
    default: v2 stores 8-bit x/y/z and recovers w, so its angular
    error grows as ~grid/w and blows up near 180-degree rotations;
    v3 stores the SMALLEST three components at 9-bit magnitude plus a
    sign and recovers the largest, which is never ill-conditioned.
    Pass version=2 only for readers that predate v3.

    SPZ is already y-up — positions pass through unflipped. This
    writer emits DC color only (shDegree 0); the FORMAT carries
    higher-order SH at every version, and `parse_spz_sh` reads it.
    """
    pos = np.asarray(pos, np.float64)
    n = len(pos)
    fixed = np.round(pos * (1 << frac_bits)).astype(np.int64)
    assert np.abs(fixed).max() < (1 << 23), "position exceeds 24-bit range"
    p24 = np.zeros((n, 3, 3), np.uint8)
    for b in range(3):
        p24[..., b] = (fixed >> (8 * b)) & 0xFF
    rgba = np.asarray(rgba, np.float32)
    a_u8 = np.round(np.clip(rgba[:, 3], 0, 1) * 255).astype(np.uint8)
    col = np.round(np.clip(
        ((rgba[:, :3] - 0.5) * 0.15 / SH_C0 + 0.5) * 255, 0, 255)) \
        .astype(np.uint8)
    s_u8 = np.round(np.clip(
        (np.log(np.maximum(np.asarray(scale, np.float64), 1e-12)) + 10.0)
        * 16.0, 0, 255)).astype(np.uint8)
    if version not in (2, 3):
        raise ValueError(f"unsupported SPZ write version {version}")
    q = np.asarray(quat, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    if version == 2:
        q = q * np.where(q[:, :1] < 0, -1.0, 1.0)  # w >= 0 (recovered)
        r_u8 = np.round(np.clip(q[:, 1:] * 127.5 + 127.5, 0, 255)) \
            .astype(np.uint8)
    else:
        r_u8 = _pack_quat_v3(q)
    head = struct.pack("<IIIBBBB", SPZ_MAGIC, version, n, 0, frac_bits,
                       0, 0)
    with gzip.open(path, "wb") as f:
        f.write(head + p24.tobytes() + a_u8.tobytes() + col.tobytes()
                + s_u8.tobytes() + r_u8.tobytes())


def _pack_quat_v3(q):
    """Inverse of `_spz_quat_v3`: drop the largest component, store the
    other three as 9-bit magnitude + sign, index in the top 2 bits."""
    q = np.asarray(q, np.float64)
    q = np.concatenate([q[:, 1:], q[:, :1]], axis=1)      # wxyz -> xyzw
    n = len(q)
    i_largest = np.argmax(np.abs(q), axis=1)
    rows = np.arange(n)
    q = q * np.where(q[rows, i_largest] < 0, -1.0, 1.0)[:, None]
    mask = (1 << 9) - 1
    comp = (i_largest.astype(np.uint32) << np.uint32(30))
    for i in range(4):
        sel = i != i_largest
        val = np.clip(q[:, i] / np.sqrt(0.5), -1.0, 1.0)
        mag = np.round(np.abs(val) * mask).astype(np.uint32)
        piece = mag | ((val < 0).astype(np.uint32) << np.uint32(9))
        # component i sits in the slot counted from the low end,
        # skipping the omitted one — the order `_spz_quat_v3` reads
        below = (np.arange(4)[None, :] > i) & (
            np.arange(4)[None, :] != i_largest[:, None])
        shift = (below.sum(axis=1) * 10).astype(np.uint32)
        comp = np.where(sel, comp | (piece << shift), comp)
    out = np.zeros((n, 4), np.uint8)
    for b in range(4):
        out[:, b] = ((comp >> np.uint32(8 * b)) & np.uint32(0xFF)) \
            .astype(np.uint8)
    return out


def quat_to_rot(q):
    """(N, 4) quaternions (w, x, y, z) -> (N, 3, 3) rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - w*z), 2 * (x*z + w*y)], -1),
        np.stack([2 * (x*y + w*z), 1 - 2 * (x**2 + z**2), 2 * (y*z - w*x)], -1),
        np.stack([2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x**2 + y**2)], -1),
    ], axis=1)


# ---------------------------------------------------------------------------
# Crop, normalize, clamp
# ---------------------------------------------------------------------------

def weighted_quantile(v, w, q):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(np.interp(q * cw[-1], cw, v[o]))


def build_scene(path, alpha_min=ALPHA_MIN, s_lo=S_LO, s_hi=S_HI,
                crop_quantile=0.75, crop_margin=1.2, verbose=True):
    """Load, filter, crop to a mass-centered cube (real captures carry
    background floaters out to +-100x the subject), normalize to the
    unit box, clamp scales; returns scene, per-splat max axis scale, box.

    Channels are premultiplied color: (alpha, alpha*R, alpha*G, alpha*B).
    """
    pos, scale, rgba, quat = load_scene_file(path)
    keep = rgba[:, 3] >= alpha_min
    pos, scale, rgba, quat = pos[keep], scale[keep], rgba[keep], quat[keep]
    a = rgba[:, 3]
    center = np.array([weighted_quantile(pos[:, i], a, 0.5)
                       for i in range(3)])
    radius = weighted_quantile(np.abs(pos - center).max(axis=1), a,
                               crop_quantile)
    lo, hi = center - crop_margin * radius, center + crop_margin * radius
    inside = np.all((pos >= lo) & (pos <= hi), axis=1)
    pos, scale, rgba, quat = (pos[inside], scale[inside], rgba[inside],
                              quat[inside])
    extent = float((hi - lo).max())
    if verbose:
        print(f"mass-centered crop: cube of {extent:.1f} scene units "
              f"at {center.round(1)}")
    pos = ((pos - lo) / extent).astype(np.float32)
    scale = np.clip(scale / extent, s_lo, s_hi)
    rots = quat_to_rot(quat)
    cov = np.einsum("nij,nj,nkj->nik", rots, scale**2, rots) \
        .astype(np.float32)
    alpha = rgba[:, 3:4]
    amp = np.concatenate([alpha, alpha * rgba[:, :3]], axis=1) \
        .astype(np.float32)                      # premultiplied RGBA
    smax = scale.max(axis=1)
    box = ((hi - lo) / extent).astype(np.float32)
    if verbose:
        print(f"kept {len(pos):,} splats (alpha>={alpha_min}, "
              f"mass-centered crop); normalized box {box.round(2)}")
    return SplatScene(mu=pos, cov=cov, amp=amp), smax, box


def render_mip(scene, sigma_b):
    """Analytic mip: blur every splat by sigma_b (covariances add under
    Gaussian convolution), rescaling peak amplitude to preserve mass."""
    cov = scene.cov + (sigma_b ** 2) * np.eye(3, dtype=np.float32)
    ratio = np.sqrt(np.linalg.det(scene.cov.astype(np.float64))
                    / np.linalg.det(cov.astype(np.float64)))
    return SplatScene(mu=scene.mu, cov=cov.astype(np.float32),
                      amp=scene.amp * ratio[:, None].astype(np.float32))


# ---------------------------------------------------------------------------
# Bands, cells, codebooks
# ---------------------------------------------------------------------------

def band_codebooks(rng, bands=None, dim=DIM, s_floor=S_LO):
    """Per band: a mixture codebook spanning [s_floor, band cap] — NOT
    just the band's own scale range. Bands are assigned by MAX axis
    scale, so a mid-band needle splat can still have thin axes at the
    global floor; if the band's codebook does not sample out to
    1/s_floor, the importance weights along those thin directions are
    heavy-tailed and that cell decodes as plane-wave herringbone.
    Finest component at beta = 1 so floor-scale directions sample as
    unit phasors."""
    books = {}
    for name, cap, _cell in (bands or BANDS):
        n_comp = 3 + max(2, int(round(np.log2(cap / s_floor))))
        rho = list(1.0 / np.geomspace(s_floor, cap, n_comp))
        freqs = sample_frequencies(dim, 3, rho, rng)
        books[name] = (freqs, rho, decode_weights(freqs, rho))
    return books


def band_of(smax, bands=None):
    """Index into the band list for each splat by its max axis scale."""
    caps = np.array([cap for _, cap, _ in (bands or BANDS)])
    return np.searchsorted(caps, smax, side="left")


def encode_bands(scene, smax, books, bands=None, dim=DIM, verbose=True):
    bands = bands or BANDS
    bundles, members = {}, {}
    bidx = band_of(smax, bands)
    for b, (name, cap, cell) in enumerate(bands):
        idx = np.where(bidx == b)[0]
        freqs = books[name][0]
        per_cell = {}
        for i in idx:
            k = tuple((scene.mu[i] // cell).astype(int))
            per_cell.setdefault(k, []).append(i)
        bundles[name], members[name] = {}, {}
        t0 = time.time()
        for k, ids in per_cell.items():
            ids = np.array(ids)
            sub = SplatScene(mu=scene.mu[ids], cov=scene.cov[ids],
                             amp=scene.amp[ids])
            bundles[name][k] = spectral_bundle(sub, freqs, chunk=2048)
            members[name][k] = ids
        if verbose:
            nbytes = len(per_cell) * scene.channels * dim * 8
            print(f"  {name}: {len(idx):,} splats -> {len(per_cell)} cells "
                  f"({nbytes / (1 << 20):.0f} MB of complex64) "
                  f"in {time.time() - t0:.0f}s")
    return bundles, members


def cell_mask(points, cell, cell_size, reach):
    lo = np.array(cell, dtype=np.float32) * cell_size
    nearest = np.clip(points, lo, lo + cell_size)
    return ((points - nearest) ** 2).sum(axis=1) <= reach * reach


# ---------------------------------------------------------------------------
# Per-cell ridge fitting (the holo/fit.py idea, applied cell-by-cell)
# ---------------------------------------------------------------------------

def _exact_subset(scene, ids, pts, chunk=2048):
    """Exact mixture of just the splats `ids`, evaluated at pts."""
    out = np.zeros((len(pts), scene.channels), dtype=np.float32)
    inv_cov = np.linalg.inv(scene.cov[ids].astype(np.float64)) \
        .astype(np.float32)
    for lo in range(0, len(ids), chunk):
        sub = slice(lo, lo + chunk)
        diff = pts[None, :, :] - scene.mu[ids[sub]][:, None, :]
        quad = np.einsum("npi,nij,npj->np", diff, inv_cov[sub], diff)
        out += np.exp(-0.5 * quad).T @ scene.amp[ids[sub]]
    return out


def fit_cells(scene, members, books, bands=None, lam=1e-3,
              prior_frac=0.35, samples_per_splat=16, min_samples=1000,
              max_samples=3000, rng=None, verbose=True):
    """Ridge-fit every cell bundle to its members' exact field — the
    regression view of holo/fit.py, cell by cell. The forward bundle is
    a Monte-Carlo estimate with sqrt(1/d) crosstalk baked in; regression
    finds the OPTIMAL vector in the same feature basis, so the fitted
    bundles drop straight into decode_slice / render_xray with lower
    noise at identical dimension and storage.

    Per cell: sample points half near the member splats (signal), half
    uniform over the cell + reach box (zeros — teaches the fit not to
    hallucinate); targets are the cell-local exact mixture; solve the
    dual ridge on the backend (accel.ridge_cell_fit) under a spectral
    prior exp(-1/2 (prior_frac*cap)^2 |w|^2) — without it, minimum-norm
    regression memorizes samples as bumps at the codebook's finest
    scale — and divide out the band's importance weights so the result
    stays in encode_bands' bundle convention.
    Returns {band: {cell: (C, d) complex64}}.

    MEASURED LIMIT (docs/real-scenes.md): fitting ties forward encoding
    on sparse cells (few splats each: 0.037 vs 0.036 rel err on a toy)
    but LOSES to it at real capture density — hundreds of floor-scale
    splats per cell need target coverage at their own kernel width,
    i.e. tens of thousands of samples per cell, beyond the dual solve's
    reach (saguaro: forward 0.52/0.38 vs fitted 0.72/0.53, with
    dropout speckle where sampling starved). Open direction: replace
    point sampling with the analytic L2 projection — the region Gram
    G_jk = integral over the cell+reach box of e^{i(w_j - w_k) . p} dp
    is a separable product of sincs, so the closed-form optimum needs
    no samples at all.
    """
    rng = rng or np.random.default_rng(0)
    fitted = {}
    for name, cap, cell in (bands or BANDS):
        freqs, _, weights = books[name]
        reach = 3.0 * cap
        prior = np.exp(-0.5 * (prior_frac * cap) ** 2
                       * (freqs ** 2).sum(axis=1)).astype(np.float32)
        fitted[name] = {}
        t0 = time.time()
        for k, ids in members[name].items():
            n_pts = int(np.clip(samples_per_splat * len(ids) + min_samples,
                                min_samples, max_samples))
            near_ids = ids[rng.integers(0, len(ids), n_pts // 2)]
            spread = np.sqrt(scene.cov[near_ids, 0, 0]
                             + scene.cov[near_ids, 1, 1]
                             + scene.cov[near_ids, 2, 2])[:, None]
            near = scene.mu[near_ids] \
                + (1.5 * spread * rng.standard_normal((n_pts // 2, 3))) \
                .astype(np.float32)
            lo = np.array(k, dtype=np.float32) * cell - reach
            hi = lo + cell + 2 * reach
            far = rng.uniform(lo, hi, (n_pts - n_pts // 2, 3)) \
                .astype(np.float32)
            pts = np.clip(np.concatenate([near, far]), lo, hi) \
                .astype(np.float32)
            y = _exact_subset(scene, ids, pts)
            weighted = _accel.ridge_cell_fit(freqs, pts, y, lam,
                                             prior=prior)
            fitted[name][k] = (weighted / weights[None, :]) \
                .astype(np.complex64)
        if verbose and members[name]:
            print(f"  fit {name}: {len(members[name])} cells "
                  f"in {time.time() - t0:.0f}s")
    return fitted


# ---------------------------------------------------------------------------
# Cell-local decode and exact ground truth
# ---------------------------------------------------------------------------

def decode_slice(points, bundles, books, bands=None, chunk=4096):
    n_ch = next(b.shape[0] for cells in bundles.values()
                for b in cells.values())
    out = np.zeros((len(points), n_ch), dtype=np.float32)
    for name, cap, cell in (bands or BANDS):
        freqs, _, weights = books[name]
        reach = 3.0 * cap
        if _accel.active():
            pairs = [(cell_mask(points, k, cell, reach),
                      b * weights[None, :])
                     for k, b in bundles[name].items()]
            pairs = [(m, b) for m, b in pairs if m.any()]
            if pairs:
                out += _accel.cell_decode(freqs, points, pairs)
            continue
        cells = {k: (b * weights[None, :]).T.astype(np.complex64)
                 for k, b in bundles[name].items()}
        for plo in range(0, len(points), chunk):
            pts = points[plo:plo + chunk]
            E = np.exp(1j * (pts @ freqs.T)).astype(np.complex64)
            for k, wb in cells.items():
                m = cell_mask(pts, k, cell, reach)
                if m.any():
                    out[plo:plo + chunk][m] += (E[m] @ wb).real
    return out


def exact_slice(points, scene, members, bands=None, chunk=2048):
    """Ground truth with the same cell-locality cutoff as the hologram."""
    out = np.zeros((len(points), scene.channels), dtype=np.float32)
    inv_cov = np.linalg.inv(scene.cov.astype(np.float64)).astype(np.float32)
    for name, cap, cell in (bands or BANDS):
        reach = 3.0 * cap
        for k, ids in members[name].items():
            m = cell_mask(points, k, cell, reach)
            if not m.any():
                continue
            pts = points[m]
            acc = np.zeros((len(pts), scene.channels), dtype=np.float32)
            for slo in range(0, len(ids), chunk):
                sub = ids[slo:slo + chunk]
                diff = pts[None, :, :] - scene.mu[sub][:, None, :]
                quad = np.einsum("npi,nij,npj->np", diff, inv_cov[sub], diff)
                acc += np.exp(-0.5 * quad).T @ scene.amp[sub]
            out[m] += acc
    return out


# ---------------------------------------------------------------------------
# X-ray projections (see holo/render.py for the single-bundle z-up form)
# ---------------------------------------------------------------------------

def camera_basis_yup(view):
    """Orthonormal (v, u1, u2) with u2 as close to +y as possible, so
    y-up captures render upright (holo/render.py assumes z-up)."""
    v = np.asarray(view, dtype=np.float64)
    v = v / np.linalg.norm(v)
    up = np.array([0.0, 1.0, 0.0])
    if abs(v @ up) > 0.98:
        up = np.array([0.0, 0.0, 1.0])
    u1 = np.cross(up, v)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    return (v.astype(np.float32), u1.astype(np.float32),
            u2.astype(np.float32))


def _pixel_grid(center, v, u1, u2, half, res, t_extent):
    xs = np.linspace(-half, half, res, dtype=np.float32)
    px, py = np.meshgrid(xs, xs)
    uv = np.stack([px.ravel(), py.ravel()], axis=1)
    origins = (np.asarray(center, dtype=np.float32)
               + uv[:, :1] * u1 + uv[:, 1:] * u2
               - (t_extent / 2) * v)
    return uv, origins


def _cell_uv_mask(uv, cell, cell_size, reach, center, u1, u2):
    """Pixels whose ray passes within reach of the cell: distance on the
    image plane from the cell's projected footprint."""
    c3 = (np.asarray(cell, dtype=np.float32) + 0.5) * cell_size \
        - np.asarray(center, dtype=np.float32)
    cuv = np.array([c3 @ u1, c3 @ u2], dtype=np.float32)
    r = np.sqrt(3.0) / 2.0 * cell_size + reach
    return ((uv - cuv) ** 2).sum(axis=1) <= r * r


def render_xray(bundles, books, view, center, half, res, t_extent,
                bands=None, chunk=2048):
    """Orthographic X-ray straight from the cell bundles: fold the
    projection-slice factor (holo/render.py) AND the mixture importance
    weights into each band's bundles; mask cells by their projected
    footprint so ray crosstalk stays local, as in decode_slice."""
    v, u1, u2 = camera_basis_yup(view)
    uv, origins = _pixel_grid(center, v, u1, u2, half, res, t_extent)
    n_ch = next(b.shape[0] for cells in bundles.values()
                for b in cells.values())
    out = np.zeros((len(origins), n_ch), dtype=np.float32)
    for name, cap, cell in (bands or BANDS):
        freqs, _, weights = books[name]
        reach = 3.0 * cap
        a = (freqs @ v).astype(np.float64)
        T = float(t_extent)
        F = np.where(np.abs(a) * T < 1e-6, T,
                     (np.exp(1j * a * T) - 1.0)
                     / (1j * np.where(a == 0, 1, a)))
        wf = (weights * F).astype(np.complex64)
        if _accel.active():
            pairs = [(_cell_uv_mask(uv, k, cell, reach, center, u1, u2),
                      b * wf[None, :]) for k, b in bundles[name].items()]
            pairs = [(m, b) for m, b in pairs if m.any()]
            if pairs:
                out += _accel.cell_decode(freqs, origins, pairs)
            continue
        cells = {k: (b * wf[None, :]).T.astype(np.complex64)
                 for k, b in bundles[name].items()}
        masks = {k: _cell_uv_mask(uv, k, cell, reach, center, u1, u2)
                 for k in cells}
        for plo in range(0, len(origins), chunk):
            pts = origins[plo:plo + chunk]
            E = np.exp(1j * (pts @ freqs.T)).astype(np.complex64)
            for k, wb in cells.items():
                m = masks[k][plo:plo + chunk]
                if m.any():
                    out[plo:plo + chunk][m] += (E[m] @ wb).real
    return out


def exact_xray(scene, members, view, center, half, res, bands=None,
               chunk=1024):
    """Analytic full-line integral of every anisotropic splat:
    integral = alpha sqrt(2 pi / q) exp(-1/2 (d^T S^-1 d - s^2/q)),
    q = v^T S^-1 v, s = d^T S^-1 v — with the same cell footprint masks."""
    v, u1, u2 = camera_basis_yup(view)
    uv, _ = _pixel_grid(center, v, u1, u2, half, res, 0.0)
    plane = (np.asarray(center, dtype=np.float32)
             + uv[:, :1] * u1 + uv[:, 1:] * u2)
    out = np.zeros((len(plane), scene.channels), dtype=np.float32)
    inv_cov = np.linalg.inv(scene.cov.astype(np.float64)).astype(np.float32)
    for name, cap, cell in (bands or BANDS):
        reach = 3.0 * cap
        for k, ids in members[name].items():
            m = _cell_uv_mask(uv, k, cell, reach, center, u1, u2)
            if not m.any():
                continue
            pts = plane[m]
            acc = np.zeros((len(pts), scene.channels), dtype=np.float32)
            for slo in range(0, len(ids), chunk):
                sub = ids[slo:slo + chunk]
                ic = inv_cov[sub]
                delta = pts[None, :, :] - scene.mu[sub][:, None, :]
                icv = ic @ v                                   # (n, 3)
                q = (icv @ v)[:, None]                         # (n, 1)
                s = np.einsum("npi,ni->np", delta, icv)
                quad = np.einsum("npi,nij,npj->np", delta, ic, delta)
                line = np.sqrt(2 * np.pi / q) \
                    * np.exp(-0.5 * (quad - s * s / q))
                acc += line.T @ scene.amp[sub]
            out[m] += acc
    return out


# ---------------------------------------------------------------------------
# Slice placement helpers
# ---------------------------------------------------------------------------

def mass_mode(values, weights, extent):
    """Center of the heaviest histogram bin — where the scene's mass is."""
    hist, edges = np.histogram(values, bins=48, range=(0, extent),
                               weights=weights)
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def slice_grid(u_range, v_range, plane, w_value, pix=PIX):
    """Points on an axis-aligned plane. 'y': u=x, v=z; 'x': u=z, v=y."""
    nu = max(int(round((u_range[1] - u_range[0]) / pix)), 8)
    nv = max(int(round((v_range[1] - v_range[0]) / pix)), 8)
    us = np.linspace(*u_range, nu, dtype=np.float32)
    vs = np.linspace(*v_range, nv, dtype=np.float32)
    U, V = np.meshgrid(us, vs)
    W = np.full_like(U, w_value)
    axes = {"y": (U, W, V), "x": (W, V, U)}[plane]
    pts = np.stack([a.ravel() for a in axes], axis=1)
    return pts, (nv, nu)
