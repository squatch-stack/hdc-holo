"""Backend dispatch: the universal readout kernel, both paths."""

import numpy as np
import pytest

from holo import accel


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
