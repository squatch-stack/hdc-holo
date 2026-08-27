"""holo/sog.py — SOG export, decoded back with the spec's arithmetic.

The reference decoder below is transcribed from the published SOG v2
spec (PlayCanvas), deliberately NOT from holo/sog.py's internals: it
is the independent referee, the same way tests/test_capture.py authors
.splat and .spz bytes from their specs.
"""

import json
import zipfile

import numpy as np
import pytest

from holo.capture import SH_C0
from holo.sog import save_sog

pytest.importorskip("PIL", reason="SOG export needs Pillow ([viz])")


def _read_sog(path):
    """Decode a bundled .sog per the spec. Returns pos, scale, rgba,
    quat (wxyz), sh or None — in the file's own (y-down) frame."""
    from PIL import Image
    out = {}
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json"))
        for name in z.namelist():
            if name.endswith(".webp"):
                with z.open(name) as f:
                    out[name] = np.asarray(Image.open(f).convert("RGBA"))
    n = meta["count"]
    flat = {k: v.reshape(-1, 4)[:n] for k, v in out.items()}

    lo, hi = flat["means_l.webp"], flat["means_u.webp"]
    q = (hi[:, :3].astype(np.uint32) << 8) | lo[:, :3].astype(np.uint32)
    mins = np.array(meta["means"]["mins"])
    maxs = np.array(meta["means"]["maxs"])
    lg = mins + (maxs - mins) * (q / 65535.0)
    pos = np.sign(lg) * (np.exp(np.abs(lg)) - 1.0)

    s_book = np.array(meta["scales"]["codebook"])
    scale = np.exp(s_book[flat["scales.webp"][:, :3]])

    c_book = np.array(meta["sh0"]["codebook"])
    sh0 = flat["sh0.webp"]
    rgb = 0.5 + c_book[sh0[:, :3]] * SH_C0
    alpha = sh0[:, 3:4] / 255.0
    rgba = np.concatenate([rgb, alpha], 1)

    qw = flat["quats.webp"]
    comp = (qw[:, :3] / 255.0 - 0.5) * 2.0 / np.sqrt(2)
    mode = qw[:, 3].astype(int) - 252
    d = np.sqrt(np.maximum(0.0, 1.0 - (comp ** 2).sum(1)))
    quat = np.zeros((n, 4))
    for m in range(4):
        sel = mode == m
        if not sel.any():
            continue
        keep_cols = [c for c in range(4) if c != m]
        quat[np.ix_(sel, keep_cols)] = comp[sel]
        quat[sel, m] = d[sel]

    sh = None
    if "shN" in meta:
        bands = meta["shN"]["bands"]
        k = {1: 3, 2: 8, 3: 15}[bands]
        book = np.array(meta["shN"]["codebook"])
        lab = flat["shN_labels.webp"]
        idx = lab[:, 0].astype(np.int32) + (lab[:, 1].astype(np.int32) << 8)
        cent = out["shN_centroids.webp"]
        sh = np.zeros((n, 3, k))
        for c in range(k):
            u = (idx % 64) * k + c
            v = idx // 64
            sh[:, :, c] = book[cent[v, u, :3]]
    return pos, scale, rgba, quat, sh


def _scene(rng, n, k=None):
    pos = rng.uniform(-3, 3, (n, 3))
    scale = np.exp(rng.uniform(np.log(0.005), np.log(0.5), (n, 3)))
    rgba = np.concatenate([rng.uniform(0.05, 0.95, (n, 3)),
                           rng.uniform(0.1, 1.0, (n, 1))], 1)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sh = rng.normal(0, 0.05, (n, 3, k)) if k else None
    return pos, scale, rgba, q, sh


def test_sog_roundtrip_through_the_spec_decoder(tmp_path):
    rng = np.random.default_rng(5)
    pos, scale, rgba, q, _ = _scene(rng, 500)
    p = tmp_path / "s.sog"
    save_sog(str(p), pos, scale, rgba, q)
    dpos, dscale, drgba, dquat, dsh = _read_sog(str(p))
    assert dsh is None                                  # DC-only export

    # SOG is written y-down (like save_ply); undo for comparison, and
    # Morton ordering permutes splats — match by nearest position
    dpos_up = dpos * [1, -1, -1]
    order = np.argmin(((dpos_up[:, None, :] - pos[None, :, :]) ** 2)
                      .sum(-1), axis=1)
    assert len(np.unique(order)) == len(pos)            # a permutation

    # 16-bit log-space positions over a 6-unit span: sub-millimeter
    assert np.abs(dpos_up - pos[order]).max() < 5e-4
    # 256-entry log codebook over a 100x scale range
    assert np.all(np.abs(dscale - scale[order]) / scale[order] < 0.06)
    assert np.abs(drgba[:, :3] - rgba[order][:, :3]).max() < 0.03
    assert np.abs(drgba[:, 3] - rgba[order][:, 3]).max() < 1.5 / 255
    # smallest-three at 8 bits, and the file frame is the y-up flip:
    # compare rotation MATRICES, decoded == F . original (as in
    # tests/test_capture.py's loader checks)
    from holo.capture import quat_to_rot
    F = np.diag([1.0, -1.0, -1.0])
    assert np.abs(quat_to_rot(dquat)
                  - F @ quat_to_rot(q[order])).max() < 0.02


def test_sog_carries_higher_order_sh(tmp_path):
    """The reason to prefer SOG over SPZ v2: the view-dependent term
    survives, through a palette."""
    rng = np.random.default_rng(6)
    n = 400
    pos, scale, rgba, q, sh = _scene(rng, n, k=15)
    p = tmp_path / "sh.sog"
    meta = save_sog(str(p), pos, scale, rgba, q, sh=sh, sh_clusters=64)
    assert meta["shN"]["bands"] == 3 and meta["shN"]["count"] == 64
    _, _, _, _, dsh = _read_sog(str(p))
    assert dsh is not None and dsh.shape == (n, 3, 15)
    # A 64-entry palette over 400 i.i.d.-random splats is deliberately
    # coarse (real captures cluster; random noise does not), so the
    # claim under test is that the view-dependent term SURVIVES at all
    # — SPZ v2 drops it to exactly zero.
    assert np.linalg.norm(dsh) > 0.4 * np.linalg.norm(sh)


def test_morton_order_groups_neighbours():
    """Spatial ordering is what makes the images compressible."""
    from holo.sog import morton_order
    rng = np.random.default_rng(7)
    q = rng.integers(0, 65536, (4000, 3)).astype(np.uint32)
    order = morton_order(q)
    p = q[order].astype(np.float64)
    step = np.abs(np.diff(p, axis=0)).sum(1).mean()
    base = np.abs(np.diff(q.astype(np.float64), axis=0)).sum(1).mean()
    assert step < 0.5 * base            # neighbours really are nearer


def test_sog_is_a_zip_of_webp_and_meta(tmp_path):
    rng = np.random.default_rng(8)
    pos, scale, rgba, q, _ = _scene(rng, 300)
    p = tmp_path / "z.sog"
    save_sog(str(p), pos, scale, rgba, q)
    with zipfile.ZipFile(p) as z:
        names = set(z.namelist())
        meta = json.loads(z.read("meta.json"))
        assert z.read("means_l.webp")[:4] == b"RIFF"
    assert {"meta.json", "means_l.webp", "means_u.webp", "scales.webp",
            "sh0.webp", "quats.webp"} <= names
    assert meta["version"] == 2 and meta["count"] == 300
    assert len(meta["scales"]["codebook"]) == 256
