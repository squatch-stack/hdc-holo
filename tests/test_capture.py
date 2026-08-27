"""holo/capture.py — loaders, crop/clamp, banded cells, slice/X-ray."""

import gzip
import struct

import numpy as np
import pytest

from holo.capture import (
    BANDS,
    S_HI,
    S_LO,
    band_codebooks,
    band_of,
    build_scene,
    decode_slice,
    encode_bands,
    exact_slice,
    exact_xray,
    fit_cells,
    load_splat,
    parse_spz,
    render_mip,
    render_xray,
)
from holo.spectral import SplatScene


def _write_splat(path, pos, scale, rgba_u8, quat):
    """Author antimatter15 .splat bytes: f32[3] pos, f32[3] scale,
    u8[4] RGBA, u8[4] quaternion ((v-128)/128, w first)."""
    n = len(pos)
    raw = np.zeros((n, 32), dtype=np.uint8)
    raw[:, :12] = np.asarray(pos, np.float32).view(np.uint8).reshape(n, 12)
    raw[:, 12:24] = np.asarray(scale, np.float32).view(np.uint8) \
        .reshape(n, 12)
    raw[:, 24:28] = rgba_u8
    raw[:, 28:32] = np.clip(np.asarray(quat) * 128.0 + 128.0,
                            0, 255).astype(np.uint8)
    raw.tofile(path)


def test_splat_loader_roundtrip(tmp_path):
    pos = np.array([[0.5, -1.25, 3.0], [10.0, 0.0, -2.5]], np.float32)
    scale = np.array([[0.1, 0.2, 0.3], [0.05, 0.05, 0.4]], np.float32)
    rgba = np.array([[200, 100, 50, 255], [10, 20, 30, 128]], np.uint8)
    quat = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    path = tmp_path / "two.splat"
    _write_splat(path, pos, scale, rgba, quat)
    lpos, lscale, lrgba, lquat = load_splat(str(path))
    # .splat is y-down (COLMAP world); the loader normalizes to y-up
    # by a 180-deg rotation about x — see capture._to_y_up
    assert np.allclose(lpos, pos * [1, -1, -1])   # f32 fields are exact
    assert np.allclose(lscale, scale)
    assert np.allclose(lrgba, rgba / 255.0)
    # quaternions survive u8 quantization to ~1/128 per component:
    # loaded rotation must equal flip . authored (proper rotation, so
    # compare rotation matrices, not raw components)
    from holo.capture import quat_to_rot
    F = np.diag([1.0, -1.0, -1.0])
    assert np.allclose(quat_to_rot(lquat), F @ quat_to_rot(quat),
                       atol=0.05)


def _spz_v2_bytes(pos, scale, alpha, color_u8, quat_xyz, frac_bits=12):
    """Author SPZ v2 (uncompressed layout; caller gzips): the format
    holo/capture.py parses, built independently from the spec."""
    n = len(pos)
    head = struct.pack("<IIIBBBB", 0x5053474E, 2, n, 0, frac_bits, 0, 0)
    fixed = np.round(np.asarray(pos) * (1 << frac_bits)).astype(np.int64)
    p24 = np.zeros((n, 3, 3), np.uint8)
    for b in range(3):
        p24[..., b] = (fixed >> (8 * b)) & 0xFF
    a_u8 = np.round(np.asarray(alpha) * 255).astype(np.uint8)
    s_u8 = np.round((np.log(np.asarray(scale)) + 10.0) * 16.0) \
        .astype(np.uint8)
    r_u8 = np.clip(np.asarray(quat_xyz) * 127.5 + 127.5, 0, 255) \
        .astype(np.uint8)
    return (head + p24.tobytes() + a_u8.tobytes() + color_u8.tobytes()
            + s_u8.tobytes() + r_u8.tobytes())


def test_spz_v2_parser_roundtrip():
    pos = np.array([[1.5, -0.25, 0.0], [-3.0, 2.0, 4.5]])
    scale = np.array([[0.05, 0.1, 0.2], [0.3, 0.02, 0.02]])
    alpha = np.array([1.0, 0.5])
    color = np.array([[128, 200, 60], [10, 128, 250]], np.uint8)
    quat_xyz = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])  # w recovered
    buf = _spz_v2_bytes(pos, scale, alpha, color, quat_xyz)
    lpos, lscale, lrgba, lquat = parse_spz(buf)
    assert np.allclose(lpos, pos, atol=1.5 / (1 << 12))   # 24-bit fixed
    assert np.allclose(lscale, scale, rtol=0.04)          # log-u8 grid
    assert np.allclose(lrgba[:, 3], alpha, atol=1 / 255)
    assert np.allclose(lquat[0], [1, 0, 0, 0], atol=0.01)
    assert np.allclose(lquat[1], [0.8, 0.6, 0, 0], atol=0.01)


def test_spz_gzip_and_size_check(tmp_path):
    buf = _spz_v2_bytes(np.zeros((3, 3)), np.full((3, 3), 0.1),
                        np.ones(3), np.full((3, 3), 128, np.uint8),
                        np.zeros((3, 3)))
    path = tmp_path / "tiny.spz"
    with gzip.open(path, "wb") as f:
        f.write(buf)
    from holo.capture import load_spz
    lpos, _, _, _ = load_spz(str(path))
    assert len(lpos) == 3


def test_ply_loader_both_variants(tmp_path):
    from holo.capture import load_ply
    rng = np.random.default_rng(9)
    pts = rng.uniform(0, 1, (300, 3))
    # binary little-endian, xyz only (the EigenCapture scan layout)
    b = tmp_path / "scan.ply"
    with open(b, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n"
                b"element vertex 300\nproperty float x\nproperty float y\n"
                b"property float z\nend_header\n")
        f.write(pts.astype("<f4").tobytes())
    pos, scale, rgba, quat = load_ply(str(b))
    assert np.allclose(pos, pts, atol=1e-6)
    assert np.all(rgba == 1.0)                 # colorless -> white, alpha 1
    assert scale.std() == 0 and scale[0, 0] > 0
    # ascii with rgb (the SceneDepthPointCloud sample layout)
    a = tmp_path / "cloud.ply"
    rgb = rng.integers(0, 256, (300, 3))
    with open(a, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex 300\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\n"
                "property uchar blue\nelement face 0\n"
                "property list uchar int vertex_indices\nend_header\n")
        for p, c in zip(pts, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
    pos, scale, rgba, _quat = load_ply(str(a))
    assert np.allclose(pos, pts, atol=1e-5)
    assert np.allclose(rgba[:, :3], rgb / 255.0, atol=1e-6)


def test_ascii_ply_colour_is_clamped_to_the_declared_uchar_range(tmp_path):
    from holo.capture import load_ply
    # `property uchar` is a declaration, not a guarantee. The binary
    # branch cannot violate it (u1 dtype); the ascii branch parses
    # floats and would carry anything the writer emitted. A real iPhone
    # LiDAR export in data/ does exactly this — channels from -44 to
    # 298 over 1.8% of its points — and unclamped those become negative
    # premultiplied amplitudes downstream in build_scene, i.e. light
    # with negative energy.
    pts = np.zeros((3, 3))
    rgb = [(-44, 128, 298), (0, 255, 255), (300, -1, 128)]
    a = tmp_path / "outofrange.ply"
    with open(a, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex 3\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\n"
                "property uchar blue\nend_header\n")
        for p, c in zip(pts, rgb):
            f.write("%g %g %g %d %d %d\n" % (p[0], p[1], p[2], *c))
    _pos, _scale, rgba, _quat = load_ply(str(a))
    assert rgba[:, :3].min() >= 0.0
    assert rgba[:, :3].max() <= 1.0
    # in-range values must pass through untouched, not be rescaled
    assert np.isclose(rgba[1, 0], 0.0)
    assert np.isclose(rgba[1, 1], 1.0)
    assert np.isclose(rgba[0, 1], 128 / 255.0)


def test_ply_loader_gaussian_3dgs_layout(tmp_path):
    from holo.capture import SH_C0, load_ply
    rng = np.random.default_rng(21)
    n = 50
    fields = (["x", "y", "z", "nx", "ny", "nz",
               "f_dc_0", "f_dc_1", "f_dc_2"]
              + [f"f_rest_{i}" for i in range(9)]
              + ["opacity", "scale_0", "scale_1", "scale_2",
                 "rot_0", "rot_1", "rot_2", "rot_3"])
    rec = np.zeros(n, dtype=np.dtype([(f, "<f4") for f in fields]))
    pos = rng.uniform(-2, 2, (n, 3)).astype(np.float32)
    for i, ax in enumerate("xyz"):
        rec[ax] = pos[:, i]
    rec["f_dc_0"] = 1.0                       # -> 0.5 + SH_C0, clipped
    rec["opacity"] = 0.0                      # sigmoid -> 0.5
    for i in range(3):
        rec[f"scale_{i}"] = np.log(0.05)      # log-stored -> 0.05
    rec["rot_0"] = 2.0                        # w-first, unnormalized
    p = tmp_path / "gauss.ply"
    with open(p, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n"
                + f"element vertex {n}\n".encode()
                + b"".join(f"property float {f_}\n".encode()
                           for f_ in fields)
                + b"end_header\n" + rec.tobytes())
    lpos, scale, rgba, quat = load_ply(str(p))
    # 3DGS PLY is y-down; loader rotates 180 deg about x to y-up
    assert np.allclose(lpos, pos * [1, -1, -1], atol=1e-6)
    assert np.allclose(rgba[:, 0], min(0.5 + SH_C0, 1.0), atol=1e-6)
    assert np.allclose(rgba[:, 3], 0.5, atol=1e-6)      # sigmoid(0)
    assert np.allclose(scale, 0.05, atol=1e-6)          # exp(log 0.05)
    # identity, premultiplied by the x-180 rotation (0, 1, 0, 0)
    assert np.allclose(quat, [[0, 1, 0, 0]] * n, atol=1e-6)


def test_y_up_normalization_preserves_covariance(tmp_path):
    """The loader's y-up flip must transform each splat's WHOLE
    Gaussian congruently: sigma' = F sigma F for F = diag(1,-1,-1).
    Authored via the Gaussian-PLY path (f32 quats, no quantization)."""
    from holo.capture import load_ply, quat_to_rot
    rng = np.random.default_rng(3)
    n = 40
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]
    rec = np.zeros(n, dtype=np.dtype([(f, "<f4") for f in fields]))
    pos = rng.uniform(-2, 2, (n, 3))
    scales = rng.uniform(0.01, 0.3, (n, 3))
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for i, ax in enumerate("xyz"):
        rec[ax] = pos[:, i]
        rec[f"scale_{i}"] = np.log(scales[:, i])
    for i in range(4):
        rec[f"rot_{i}"] = q[:, i]
    p = tmp_path / "aniso.ply"
    with open(p, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n"
                + f"element vertex {n}\n".encode()
                + b"".join(f"property float {f_}\n".encode()
                           for f_ in fields)
                + b"end_header\n" + rec.tobytes())
    lpos, lscale, _, lquat = load_ply(str(p))
    F = np.diag([1.0, -1.0, -1.0])
    R, Rl = quat_to_rot(q), quat_to_rot(lquat)
    S2 = np.einsum("ni,ij->nij", scales ** 2, np.eye(3))
    cov = np.einsum("nab,nbc,ndc->nad", R, S2, R)
    lcov = np.einsum("nab,nbc,ndc->nad", Rl, S2, Rl)
    assert np.allclose(lpos, pos * [1, -1, -1], atol=1e-6)
    assert np.allclose(lscale, scales, atol=1e-6)
    assert np.allclose(lcov, F @ cov @ F, atol=1e-5)


def test_build_scene_crops_floaters_and_clamps_scales(tmp_path):
    rng = np.random.default_rng(0)
    n_core, n_far = 60, 10
    pos = np.concatenate([rng.normal(0, 1.0, (n_core, 3)),
                          rng.normal(0, 80.0, (n_far, 3)) + 200])
    scale = np.concatenate([np.full((n_core, 3), 0.02),
                            np.full((n_far, 3), 5.0)])
    rgba = np.full((n_core + n_far, 4), 255, np.uint8)
    quat = np.tile([1.0, 0, 0, 0], (n_core + n_far, 1))
    path = tmp_path / "scene.splat"
    _write_splat(path, pos, scale, rgba, quat)
    scene, smax, _box = build_scene(str(path), verbose=False)
    # the mass-centered cube keeps the core and drops the 200-away shell
    assert n_core * 0.7 <= scene.n <= n_core + 1
    assert np.all(scene.mu >= -1e-6) and np.all(scene.mu <= 1 + 1e-6)
    assert np.all(smax >= S_LO - 1e-9) and np.all(smax <= S_HI + 1e-9)


def _toy_scene(rng, n=40):
    mu = rng.uniform(0.2, 0.8, (n, 3)).astype(np.float32)
    s = rng.uniform(0.004, 0.03, (n, 1)) * np.ones((1, 3))
    cov = np.einsum("ni,ij->nij", (s**2)[:, 0:1].repeat(3, 1),
                    np.eye(3)).astype(np.float32)
    cov = np.stack([np.eye(3) * (si[0] ** 2) for si in s]) \
        .astype(np.float32)
    amp = rng.uniform(0.5, 1.0, (n, 1)).astype(np.float32)
    return SplatScene(mu=mu, cov=cov, amp=amp), s.max(axis=1) \
        .astype(np.float32)


def test_banded_cell_encode_decodes_the_mixture():
    rng = np.random.default_rng(1)
    scene, smax = _toy_scene(rng)
    books = band_codebooks(np.random.default_rng(2))
    bundles, members = encode_bands(scene, smax, books, verbose=False)
    # probe near centers, where the field carries signal: a uniform grid
    # is mostly zeros and the relative error degenerates to noise/0
    pts = (scene.mu[rng.integers(0, scene.n, 300)]
           + 0.005 * rng.standard_normal((300, 3))).astype(np.float32)
    truth = exact_slice(pts, scene, members)
    holo = decode_slice(pts, bundles, books)
    # signal ~ amp >= 0.5, crosstalk ~ sqrt(local/2d) ~ few %: 0.15 is
    # several sigma of margin on both backends
    rel = (np.linalg.norm(holo[:, 0] - truth[:, 0])
           / np.linalg.norm(truth[:, 0]))
    assert rel < 0.15


def test_band_assignment_respects_caps():
    smax = np.array([0.003, 0.01, 0.03], np.float32)
    idx = band_of(smax)
    caps = [cap for _, cap, _ in BANDS]
    for s, b in zip(smax, idx):
        assert s <= caps[b]
        assert b == 0 or s > caps[b - 1]


def test_fit_cells_decodes_the_mixture():
    rng = np.random.default_rng(11)
    n = 12
    mu = rng.uniform(0.25, 0.75, (n, 3)).astype(np.float32)
    s = rng.uniform(0.006, 0.02, n)
    cov = np.stack([np.eye(3) * (si ** 2) for si in s]).astype(np.float32)
    scene = SplatScene(mu=mu, cov=cov,
                       amp=np.ones((n, 1), dtype=np.float32))
    smax = s.astype(np.float32)
    # small d keeps the CPU-only CI runner fast; the fit's sampling
    # floor, not d, dominates its error here
    books = band_codebooks(np.random.default_rng(12), dim=1024)
    _, members = encode_bands(scene, smax, books, dim=1024, verbose=False)
    fitted = fit_cells(scene, members, books, min_samples=600,
                       max_samples=900, rng=np.random.default_rng(13),
                       verbose=False)
    pts = (scene.mu[rng.integers(0, n, 300)]
           + 0.004 * rng.standard_normal((300, 3))).astype(np.float32)
    truth = exact_slice(pts, scene, members)
    est = decode_slice(pts, fitted, books)
    rel = (np.linalg.norm(est[:, 0] - truth[:, 0])
           / np.linalg.norm(truth[:, 0]))
    assert rel < 0.2   # sampling-limited fit floor ~0.05 at these sizes


def test_xray_render_matches_analytic_projection():
    rng = np.random.default_rng(3)
    # a few fat splats: fat spectra keep the projection slice populated
    scene = SplatScene(
        mu=rng.uniform(0.35, 0.65, (6, 3)).astype(np.float32),
        cov=np.stack([np.eye(3) * 0.04**2] * 6).astype(np.float32),
        amp=np.ones((6, 1), np.float32))
    mip = render_mip(scene, 0.01)
    smax = np.full(6, np.sqrt(0.04**2 + 0.01**2), np.float32)
    bands = [("r", 0.055, 0.25)]
    books = band_codebooks(np.random.default_rng(4), bands,
                           dim=1 << 14, s_floor=0.01)
    bundles, members = encode_bands(mip, smax, books, bands,
                                    dim=1 << 14, verbose=False)
    view, center = [1.0, 0.0, 0.3], [0.5, 0.5, 0.5]
    truth = exact_xray(mip, members, view, center, 0.4, 48, bands=bands)
    holo = render_xray(bundles, books, view, center, 0.4, 48, 2.0,
                       bands=bands)
    # projections use only the ~perpendicular spectrum slice, so the
    # budget is far looser than point decodes (docs/render.md)
    rel = (np.linalg.norm(holo[:, 0] - truth[:, 0])
           / np.linalg.norm(truth[:, 0]))
    assert rel < 0.35


def test_save_ply_roundtrip(tmp_path):
    """save_ply -> load_ply reproduces the loader-level representation
    to float32 rounding: the bridge out of the pipeline is lossless."""
    from holo.capture import load_ply, quat_to_rot, save_ply
    rng = np.random.default_rng(11)
    n = 60
    pos = rng.uniform(-3, 3, (n, 3))
    scale = rng.uniform(0.01, 0.4, (n, 3))
    rgba = np.concatenate([rng.uniform(0, 1, (n, 3)),
                           rng.uniform(0.05, 0.95, (n, 1))], axis=1)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p = tmp_path / "out.ply"
    save_ply(str(p), pos, scale, rgba, q)
    lpos, lscale, lrgba, lquat = load_ply(str(p))
    assert np.allclose(lpos, pos, atol=1e-5)
    assert np.allclose(lscale, scale, atol=1e-6)
    assert np.allclose(lrgba, rgba, atol=1e-5)
    # double flip may negate the quaternion — same rotation
    assert np.allclose(quat_to_rot(lquat), quat_to_rot(q), atol=1e-5)


def test_save_ply_writes_ecosystem_convention(tmp_path):
    """On disk the file is COLMAP y-down (what every external viewer
    expects); the y-up flip lives only inside our loaders."""
    from holo.capture import save_ply
    pos = np.array([[1.0, 2.0, 3.0]])
    save_ply(str(tmp_path / "c.ply"), pos, [[0.1, 0.1, 0.1]],
             [[1, 1, 1, 0.5]], [[1.0, 0, 0, 0]])
    raw = open(tmp_path / "c.ply", "rb").read()
    body = raw.split(b"end_header\n", 1)[1]
    x, y, z = np.frombuffer(body, "<f4", 3)
    assert (x, y, z) == (1.0, -2.0, -3.0)


def test_save_ply_alpha_edges_and_build_scene(tmp_path):
    """alpha 0/1 stay finite through the logit, and the written file
    feeds straight back into build_scene."""
    from holo.capture import save_ply
    rng = np.random.default_rng(12)
    n = 40
    pos = rng.normal(0, 0.5, (n, 3))
    rgba = np.concatenate([rng.uniform(0, 1, (n, 3)),
                           np.ones((n, 1))], axis=1)   # alpha exactly 1
    rgba[0, 3] = 0.0                                   # and exactly 0
    p = tmp_path / "s.ply"
    save_ply(str(p), pos, np.full((n, 3), 0.05), rgba,
             np.tile([1.0, 0, 0, 0], (n, 1)))
    assert np.isfinite(np.frombuffer(
        open(p, "rb").read().split(b"end_header\n", 1)[1], "<f4")).all()
    scene, _smax, _box = build_scene(str(p), verbose=False)
    # the alpha-0 splat is filtered, alpha-1 splats survive the logit
    # round trip at ~1 (the crop may drop a few more — not under test)
    assert scene.n >= n * 0.75
    assert np.allclose(scene.amp[:, 0], 1.0, atol=1e-4)


def test_save_spz_roundtrip(tmp_path):
    """save_spz -> load_spz reproduces splats on the format's
    quantization grid: 2^-12 positions, ~6% log-u8 scales, u8 color."""
    from holo.capture import load_spz, save_spz
    rng = np.random.default_rng(13)
    n = 80
    pos = rng.uniform(-4, 4, (n, 3))
    scale = rng.uniform(0.01, 0.4, (n, 3))
    rgba = np.concatenate([rng.uniform(0.1, 0.9, (n, 3)),
                           rng.uniform(0.1, 1.0, (n, 1))], axis=1)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p = tmp_path / "out.spz"
    save_spz(str(p), pos, scale, rgba, q)
    lpos, lscale, lrgba, lquat = load_spz(str(p))
    assert np.allclose(lpos, pos, atol=1.5 / (1 << 12))
    assert np.allclose(lscale, scale, rtol=0.04)
    assert np.allclose(lrgba, rgba, atol=1.5 / 255 / (0.15 / 0.2821))
    # rotation error as the ANGLE of the relative rotation. The format
    # stores xyz and reconstructs w = sqrt(1-|xyz|^2), which amplifies
    # the u8 grid near w=0 (rotations within ~1 deg of 180) — a real
    # SPZ v2 property, so assert the well-conditioned region and only
    # sanity-bound the tail.
    dot = np.abs(np.sum(lquat * q, axis=1))
    ang = 2 * np.arccos(np.clip(dot, -1, 1))
    # error grows as ~grid/w (dw = |xyz| dr / w): well-conditioned
    # for |w| > 0.3, degrading toward w = 0
    ok = np.abs(q[:, 0]) > 0.3
    assert np.all(ang[ok] < 0.03)              # < ~1.7 deg
    assert np.all(ang < 0.2)                   # w~0 tail still bounded


def _spz_v3_bytes(pos, scale, alpha, color_u8, quat_xyzw, sh=None,
                  frac_bits=12):
    """Author SPZ v3 bytes independently, from nianticlabs/spz's
    unpackQuaternionSmallestThree: 2-bit index of the OMITTED largest
    component in the top bits, then three (sign, 9-bit magnitude)
    fields, component 3 in the lowest bits."""
    n = len(pos)
    sh_deg = 0 if sh is None else {3: 1, 8: 2, 15: 3}[sh.shape[2]]
    head = struct.pack("<IIIBBBB", 0x5053474E, 3, n, sh_deg, frac_bits,
                       0, 0)
    fixed = np.round(np.asarray(pos) * (1 << frac_bits)).astype(np.int64)
    p24 = np.zeros((n, 3, 3), np.uint8)
    for b in range(3):
        p24[..., b] = (fixed >> (8 * b)) & 0xFF
    a_u8 = np.round(np.asarray(alpha) * 255).astype(np.uint8)
    s_u8 = np.round((np.log(np.asarray(scale)) + 10.0) * 16.0) \
        .astype(np.uint8)
    q = np.asarray(quat_xyzw, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    rot = np.zeros((n, 4), np.uint8)
    for k in range(n):
        big = np.argmax(np.abs(q[k]))
        v = q[k] * (-1.0 if q[k][big] < 0 else 1.0)
        word = big << 30
        rest = [i for i in range(4) if i != big]      # ascending
        for slot, i in enumerate(rest):               # slot 0 = highest
            m = round(abs(v[i]) / np.sqrt(0.5) * 511)
            piece = m | ((1 << 9) if v[i] < 0 else 0)
            word |= piece << (10 * (2 - slot))
        rot[k] = [(word >> (8 * b)) & 0xFF for b in range(4)]
    body = (p24.tobytes() + a_u8.tobytes() + color_u8.tobytes()
            + s_u8.tobytes() + rot.tobytes())
    if sh is not None:                       # (N,3,K) -> coeff-major
        q8 = np.round(np.transpose(sh, (0, 2, 1)) * 128.0 + 128.0)
        body += np.clip(q8, 0, 255).astype(np.uint8).tobytes()
    return head + body


def test_spz_v3_parser_matches_reference_packing():
    from holo.capture import parse_spz
    rng = np.random.default_rng(31)
    n = 200
    pos = rng.uniform(-4, 4, (n, 3))
    scale = np.exp(rng.uniform(np.log(0.01), np.log(0.4), (n, 3)))
    alpha = rng.uniform(0.1, 1.0, n)
    color = rng.integers(0, 256, (n, 3)).astype(np.uint8)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)     # xyzw
    buf = _spz_v3_bytes(pos, scale, alpha, color, q)
    lpos, lscale, lrgba, lquat = parse_spz(buf)
    assert np.allclose(lpos, pos, atol=1.5 / (1 << 12))
    assert np.allclose(lscale, scale, rtol=0.04)
    assert np.allclose(lrgba[:, 3], alpha, atol=1 / 255)
    # our quaternions are wxyz; the file's are xyzw
    ref = np.concatenate([q[:, 3:4], q[:, :3]], axis=1)
    dot = np.abs((lquat * ref).sum(1))
    assert np.degrees(2 * np.arccos(np.clip(dot, -1, 1))).max() < 0.5


def test_spz_v3_rotations_beat_v2_near_180_degrees():
    """Why v3 is the default writer version: v2 stores xyz and
    recovers w, so error explodes as w -> 0; v3 drops the LARGEST
    component instead and never hits that."""
    import os
    import tempfile

    from holo.capture import load_spz, save_spz
    rng = np.random.default_rng(32)
    n = 400
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    ang = np.pi - rng.uniform(0, 0.05, n)          # near 180 degrees
    q = np.concatenate([np.cos(ang / 2)[:, None],
                        axis * np.sin(ang / 2)[:, None]], axis=1)
    pos = rng.uniform(-1, 1, (n, 3))
    scale = np.full((n, 3), 0.05)
    rgba = np.concatenate([np.full((n, 3), 0.5), np.ones((n, 1))], 1)
    errs = {}
    for ver in (2, 3):
        p = os.path.join(tempfile.mkdtemp(), f"v{ver}.spz")
        save_spz(p, pos, scale, rgba, q, version=ver)
        _, _, _, lq = load_spz(p)
        dot = np.abs((lq * q).sum(1))
        errs[ver] = np.degrees(2 * np.arccos(np.clip(dot, -1, 1))).max()
    assert errs[3] < 1.0                    # well conditioned
    assert errs[3] < errs[2] / 5            # and far better than v2


def test_spz_carries_sh_at_every_version():
    """The DC-only files this pipeline writes are OUR writer's choice,
    not a format limit: shDegree > 0 rides in v2 and v3 alike."""
    from holo.capture import parse_spz_sh
    rng = np.random.default_rng(33)
    n, k = 50, 15
    sh = rng.uniform(-0.9, 0.9, (n, 3, k))
    buf = _spz_v3_bytes(rng.uniform(-1, 1, (n, 3)),
                        np.full((n, 3), 0.05), np.ones(n),
                        np.full((n, 3), 128, np.uint8),
                        np.tile([0.0, 0, 0, 1.0], (n, 1)), sh=sh)
    got = parse_spz_sh(buf)
    assert got.shape == (n, 3, k)
    assert np.allclose(got, sh, atol=1.5 / 128)      # u8 quantization
    assert parse_spz_sh(_spz_v3_bytes(
        rng.uniform(-1, 1, (n, 3)), np.full((n, 3), 0.05), np.ones(n),
        np.full((n, 3), 128, np.uint8),
        np.tile([0.0, 0, 0, 1.0], (n, 1)))) is None   # shDegree 0


def test_spz_v4_is_identified_with_an_actionable_error():
    """v4 moved to per-attribute ZSTD streams; we can still read its
    plaintext header and must say why decoding is unavailable."""
    from holo.capture import parse_spz, spz_header
    head = struct.pack("<IIIBBBBI", 0x5053474E, 4, 1234, 3, 12, 0, 6, 64)
    buf = head + b"\x00" * 12 + b"payload"
    info = spz_header(buf)
    assert info["version"] == 4 and info["count"] == 1234
    assert info["sh_degree"] == 3 and info["streams"] == 6
    assert info["container"] == "zstd"
    with pytest.raises(NotImplementedError, match="zstd"):
        parse_spz(buf)


def _sh_basis(d):
    """Real SH basis, degrees 1-3, in the 3DGS coefficient order."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    return np.stack([
        -0.4886025119029199 * y, 0.4886025119029199 * z,
        -0.4886025119029199 * x,
        1.0925484305920792 * x * y, -1.0925484305920792 * y * z,
        0.31539156525252005 * (2 * zz - xx - yy),
        -1.0925484305920792 * x * z, 0.5462742152960396 * (xx - yy),
        -0.5900435899266435 * y * (3 * xx - yy),
        2.890611442640554 * x * y * z,
        -0.4570457994644658 * y * (4 * zz - xx - yy),
        0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
        -0.4570457994644658 * x * (4 * zz - xx - yy),
        1.445305721320277 * z * (xx - yy),
        -0.5900435899266435 * x * (xx - 3 * yy)], 1)


def test_sh_flip_matches_a_real_rotation_of_the_basis():
    """The loaders rotate scenes 180 degrees about x; the SH basis has
    to turn with them, or view-dependent color comes out mirrored.
    Pins the sign table against the basis functions themselves: the
    color seen looking along d before the flip must equal the color
    seen along the flipped d after it."""
    from holo.capture import sh_flip_x180
    rng = np.random.default_rng(41)
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    coef = rng.normal(size=(1, 3, 15))                 # one splat, RGB
    before = np.einsum("ck,nk->nc", coef[0], _sh_basis(d))
    after = np.einsum("ck,nk->nc", sh_flip_x180(coef)[0],
                      _sh_basis(d * [1, -1, -1]))
    assert np.allclose(before, after, atol=1e-5)
    assert np.allclose(sh_flip_x180(sh_flip_x180(coef)), coef)


def _sh_monomials(d):
    """The angular parts of the same basis, degrees 1-3, in 3DGS order,
    with every normalisation constant stripped out."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    return np.stack([
        y, z, x,
        x * y, y * z, 2 * zz - xx - yy, x * z, xx - yy,
        y * (3 * xx - yy), x * y * z, y * (4 * zz - xx - yy),
        z * (2 * zz - 3 * xx - 3 * yy), x * (4 * zz - xx - yy),
        z * (xx - yy), x * (xx - 3 * yy)], 1)


def test_sh_flip_signs_follow_from_monomial_parity():
    """A second derivation of _SH_FLIP_X180, from a different starting
    point than the one that produced it.

    The test above pins the sign table against `_sh_basis` — but the
    table was itself derived numerically from that basis, so the pair
    is one implementation checking itself: a convention error shared by
    both would pass. This derives the signs algebraically instead. A
    180-degree turn about x sends (x, y, z) -> (x, -y, -z), so each
    basis function maps to a scalar multiple of itself, and that scalar
    is fixed by how many flipped factors the MONOMIAL carries — xy and
    xz carry one, yz carries two, the even terms carry none.

    Normalisation constants cannot participate: they are nonzero
    scalars and cancel in the ratio, whichever sign they have. (In
    `_sh_basis` several are negative, which is exactly why the ratio,
    rather than the value, is the thing to test.) So this route touches
    neither the constants nor the convention that produced the table.
    """
    from holo.capture import _SH_FLIP_X180
    rng = np.random.default_rng(7)
    d = rng.normal(size=(4000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    before = _sh_monomials(d)
    after = _sh_monomials(d * [1.0, -1.0, -1.0])
    # near-zero denominators make the ratio noise, not evidence
    usable = np.abs(before) > 1e-3
    assert usable.sum(0).min() > 3000, "too few usable samples to conclude"
    signs = []
    for k in range(before.shape[1]):
        r = after[usable[:, k], k] / before[usable[:, k], k]
        assert np.allclose(np.abs(r), 1.0, atol=1e-9), \
            "coefficient %d does not map to +-itself" % k
        assert r.std() < 1e-9, \
            "coefficient %d's ratio varies with direction" % k
        signs.append(round(float(r.mean())))
    assert np.array_equal(np.array(signs, np.float32), _SH_FLIP_X180)


def test_ply_and_spz_sh_agree_on_the_y_up_convention(tmp_path):
    """load_ply_sh flips (files are y-down); load_spz_sh does not
    (SPZ is y-up). Both must hand back the SAME world."""
    from holo.capture import load_ply_sh, parse_spz_sh, sh_flip_x180
    rng = np.random.default_rng(42)
    n, k = 30, 15
    sh_world = rng.uniform(-0.8, 0.8, (n, 3, k))       # y-up truth
    # author a PLY holding the y-DOWN version of that world
    on_disk = sh_flip_x180(sh_world)
    fields = (["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
              + [f"f_rest_{i}" for i in range(45)]
              + ["opacity", "scale_0", "scale_1", "scale_2",
                 "rot_0", "rot_1", "rot_2", "rot_3"])
    rec = np.zeros(n, dtype=np.dtype([(f, "<f4") for f in fields]))
    for c in range(3):
        for i in range(k):
            rec[f"f_rest_{c * k + i}"] = on_disk[:, c, i]
    rec["rot_0"] = 1.0
    p = tmp_path / "sh.ply"
    with open(p, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n"
                + f"element vertex {n}\n".encode()
                + b"".join(f"property float {f_}\n".encode()
                           for f_ in fields)
                + b"end_header\n" + rec.tobytes())
    assert np.allclose(load_ply_sh(str(p)), sh_world, atol=1e-5)
    # the SPZ path stores the same world directly (already y-up)
    buf = _spz_v3_bytes(np.zeros((n, 3)), np.full((n, 3), 0.05),
                        np.ones(n), np.full((n, 3), 128, np.uint8),
                        np.tile([0.0, 0, 0, 1.0], (n, 1)),
                        sh=sh_world)
    assert np.allclose(parse_spz_sh(buf), sh_world, atol=1.5 / 128)


def test_encode_bands_refuses_splats_larger_than_every_band():
    """Splats above the last band cap match NO band. Left alone they
    are dropped silently — the scene loses content while every
    downstream number (slice error, byte count, render) still looks
    healthy — so the encoder refuses instead."""
    from holo.capture import band_codebooks, encode_bands
    rng = np.random.default_rng(51)
    n = 40
    mu = rng.uniform(0.2, 0.8, (n, 3)).astype(np.float32)
    s = np.full(n, 0.003)
    s[:5] = 0.09                                  # above the coarse cap
    cov = np.einsum("n,ij->nij", s ** 2, np.eye(3)).astype(np.float32)
    scene = SplatScene(mu, cov, np.ones((n, 1), np.float32))
    books = band_codebooks(rng, dim=256)
    with pytest.raises(ValueError, match="exceed the largest band cap"):
        encode_bands(scene, s, books, dim=256, verbose=False)
    # every splat inside the bands still encodes, none lost
    s_ok = np.full(n, 0.003)
    _bundles, members = encode_bands(scene, s_ok, books, dim=256,
                                     verbose=False)
    encoded = sum(len(ids) for band in members.values()
                  for ids in band.values())
    assert encoded == n


def test_band_of_covers_every_clamped_scale():
    """build_scene clamps to S_HI and the coarse cap IS S_HI, so a
    clamped scene can never reach the out-of-range index."""
    from holo.capture import BANDS, S_HI, S_LO, band_of
    probe = np.array([S_LO, 0.004, 0.0041, 0.02, S_HI])
    assert np.all(band_of(probe) < len(BANDS))
    assert band_of(np.array([S_HI * 1.001]))[0] == len(BANDS)


def test_footprint_blur_matches_brute_force_pixel_averaging():
    """The closed form (covariances add) must equal what a rasterizer
    would get by averaging point samples across the pixel — verified
    against a 3-D supersample of the pixel volume."""
    from holo.capture import footprint_blur
    from holo.spectral import eval_scene_exact
    rng = np.random.default_rng(61)
    n = 12
    mu = rng.uniform(0.3, 0.7, (n, 3)).astype(np.float32)
    ax = rng.uniform(0.004, 0.02, (n, 3))
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    from holo.capture import quat_to_rot
    R = quat_to_rot(q)
    cov = np.einsum("nab,nb,ncb->nac", R, ax ** 2, R).astype(np.float32)
    scene = SplatScene(mu, cov, rng.uniform(0.5, 1, (n, 1)).astype(np.float32))

    pix = 0.01
    pts = rng.uniform(0.35, 0.65, (40, 3)).astype(np.float32)
    closed = eval_scene_exact(footprint_blur(scene, pix), pts)

    # brute force: average point samples over the pixel's own volume,
    # weighted by the same box -> the supersample IS the integral
    g = (np.arange(9) + 0.5) / 9 - 0.5                  # 9^3 samples
    off = np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3) * pix
    acc = np.zeros((len(pts), scene.channels), np.float64)
    for o in off:
        acc += eval_scene_exact(scene, (pts + o).astype(np.float32))
    brute = acc / len(off)
    rel = np.linalg.norm(closed[:, 0] - brute[:, 0]) / np.linalg.norm(brute[:, 0])
    assert rel < 0.02, rel     # box vs equal-variance Gaussian, 9^3 grid


def test_footprint_makes_sub_pixel_needles_visible():
    """The point/footprint gap this evaluator exists to close: a splat
    thinner than a pixel is nearly invisible to point sampling."""
    from holo.capture import footprint_blur
    from holo.spectral import eval_scene_exact
    pix = 0.01
    mu = np.array([[0.5, 0.5, 0.5]], np.float32)
    needle = np.diag([1e-4, 0.02, 0.02]).astype(np.float32) ** 2
    scene = SplatScene(mu, needle[None], np.ones((1, 1), np.float32))
    # sample a row of pixel centres crossing the needle's thin axis
    xs = 0.5 + (np.arange(-4, 5) + 0.5) * pix
    pts = np.stack([xs, np.full(9, 0.5), np.full(9, 0.5)], 1).astype(np.float32)
    point = eval_scene_exact(scene, pts)[:, 0]
    fp = eval_scene_exact(footprint_blur(scene, pix), pts)[:, 0]
    assert point.max() < 1e-3          # point samples miss it entirely
    assert fp.max() > 10 * point.max()  # the pixel integral finds it
