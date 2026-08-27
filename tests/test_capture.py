"""holo/capture.py — loaders, crop/clamp, banded cells, slice/X-ray."""

import gzip
import struct

import numpy as np

from holo.capture import (BANDS, S_HI, S_LO, band_codebooks, band_of,
                          build_scene, decode_slice, encode_bands,
                          exact_slice, exact_xray, fit_cells, load_splat,
                          parse_spz, render_mip, render_xray)
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
    pos, scale, rgba, quat = load_ply(str(a))
    assert np.allclose(pos, pts, atol=1e-5)
    assert np.allclose(rgba[:, :3], rgb / 255.0, atol=1e-6)


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
    scene, smax, box = build_scene(str(path), verbose=False)
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
