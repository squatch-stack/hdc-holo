"""Backend dispatch: the universal readout kernel, both paths."""

import numpy as np
import pytest

from holo import accel
from holo.capture import cell_mask, decode_slice


def _reference(points, W, S):
    """The complex-arithmetic definition the kernel must reproduce."""
    E = np.exp(1j * (points @ W.T)).astype(np.complex64)
    out = np.real(E @ np.conj(np.atleast_2d(S)).T) / W.shape[0]
    return out if np.ndim(S) == 2 else out[:, 0]


def test_readout_matches_complex_definition():
    rng = np.random.default_rng(41)
    W = rng.normal(0, 20, size=(2048, 3)).astype(np.float32)
    S = (rng.normal(size=(2048,)) + 1j * rng.normal(size=(2048,))) \
        .astype(np.complex64)
    P = rng.uniform(0, 1, size=(500, 3)).astype(np.float32)
    got = accel.readout(P, W, S)
    assert got.shape == (500,)
    assert np.allclose(got, _reference(P, W, S), atol=2e-5)


def test_readout_multichannel_shape_and_values():
    rng = np.random.default_rng(43)
    W = rng.normal(0, 20, size=(1024, 2)).astype(np.float32)
    S = (rng.normal(size=(3, 1024)) + 1j * rng.normal(size=(3, 1024))) \
        .astype(np.complex64)
    P = rng.uniform(0, 1, size=(300, 2)).astype(np.float32)
    got = accel.readout(P, W, S)
    assert got.shape == (300, 3)
    assert np.allclose(got, _reference(P, W, S), atol=2e-5)


def test_readout_chunking_is_invisible():
    rng = np.random.default_rng(47)
    W = rng.normal(0, 20, size=(512, 2)).astype(np.float32)
    S = (rng.normal(size=(512,)) + 1j * rng.normal(size=(512,))) \
        .astype(np.complex64)
    P = rng.uniform(0, 1, size=(1000, 2)).astype(np.float32)
    assert np.allclose(accel.readout(P, W, S, chunk=64),
                       accel.readout(P, W, S, chunk=100000), atol=2e-5)


@pytest.mark.skipif(not accel.active(), reason="MLX backend not present")
def test_mlx_and_numpy_paths_agree():
    rng = np.random.default_rng(53)
    W = rng.normal(0, 30, size=(4096, 3)).astype(np.float32)
    S = (rng.normal(size=(3, 4096)) + 1j * rng.normal(size=(3, 4096))) \
        .astype(np.complex64)
    P = rng.uniform(0, 1, size=(2000, 3)).astype(np.float32)
    gpu = accel.readout(P, W, S)
    saved = accel._HAVE_MLX
    try:
        accel._HAVE_MLX = False        # force the NumPy fallback
        cpu = accel.readout(P, W, S)
    finally:
        accel._HAVE_MLX = saved
    assert np.allclose(gpu, cpu, atol=5e-5)


def test_importing_the_toolkit_does_not_initialise_metal():
    """`from holo import HoloMap` must not touch the GPU.

    holo/accel.py used to import mlx.core at module load, and most of the
    package imports accel — so asking for a hypervector map selected the
    Metal backend and initialised MLX. That is exposure the caller never
    asked for, and it is not hypothetical: on a shared machine a Metal
    fault raised by an unrelated process killed a run here, with this
    process as the innocent victim.

    Checked in a subprocess because the backend is probed once and
    cached, so an in-process check would see whatever an earlier test
    already triggered.
    """
    import subprocess
    import sys

    code = ("import sys\n"
            "from holo import FHRR, HoloMap\n"
            "assert 'mlx.core' not in sys.modules, 'importing holo pulled in MLX'\n"
            "from holo import accel\n"
            "accel.backend_name()\n"
            "print('probed-on-ask', 'mlx.core' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=False)
    assert out.returncode == 0, out.stderr
    # and the probe still happens when something actually asks
    assert out.stdout.strip().startswith("probed-on-ask")


def test_mixed_level_pairs_decode_on_both_backends(monkeypatch):
    rng = np.random.default_rng(84)
    freqs = rng.normal(size=(128, 3)).astype(np.float32)
    points = rng.uniform(-0.5, 1.5, (80, 3)).astype(np.float32)
    keys = [(0, 0, 0), (1, 1, 0, 0), (2, 2, 1, 0)]
    bundles = {"mix": {
        k: (rng.normal(size=(2, 128)) + 1j * rng.normal(size=(2, 128)))
        .astype(np.complex64) for k in keys}}
    weights = np.full(128, 1/128, dtype=np.float32)
    books = {"mix": (freqs, None, weights)}
    bands = [("mix", 0.1, 1.0)]
    pairs = [(cell_mask(points, k, 1.0, 0.3), b * weights[None, :])
             for k, b in bundles["mix"].items()]
    expected = np.zeros((80, 2), dtype=np.float32)
    for mask, weighted in pairs:
        E = np.exp(1j * (points[mask] @ freqs.T)).astype(np.complex64)
        expected[mask] += (E @ weighted.T).real
    # Exercise real acceleration when available; CPU-only CI still checks
    # both the NumPy decoder and dispatch's mask/bundle contract below.
    if accel.active():
        np.testing.assert_allclose(
            accel.cell_decode(freqs, points, pairs), expected, atol=2e-5)
    monkeypatch.setattr(accel, "active", lambda: False)
    np.testing.assert_allclose(
        decode_slice(points, bundles, books, bands), expected, atol=2e-5)
    calls = []

    def dispatch(w, pts, cells):
        calls.append(cells)
        out = np.zeros_like(expected)
        for mask, weighted in cells:
            E = np.exp(1j * (pts[mask] @ w.T)).astype(np.complex64)
            out[mask] += (E @ weighted.T).real
        return out

    monkeypatch.setattr(accel, "active", lambda: True)
    monkeypatch.setattr(accel, "cell_decode", dispatch)
    np.testing.assert_allclose(
        decode_slice(points, bundles, books, bands), expected, atol=2e-5)
    assert len(calls) == 1
    assert len(calls[0]) == len(keys)
