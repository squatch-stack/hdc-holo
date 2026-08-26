"""holo/spectral.py — spectral encoder, mixture codebooks, translation."""

import numpy as np
import pytest

from holo.spectral import (SplatScene, decode_field, decode_field_phasor,
                           phasor_bundle, random_scene, sample_frequencies,
                           spectral_bundle, translate_bundle)
from holo.spectral import eval_scene_exact


def _single_splat(sigma=0.03):
    return SplatScene(
        mu=np.array([[0.5, 0.5]], dtype=np.float32),
        cov=np.array([np.eye(2) * sigma**2], dtype=np.float32),
        amp=np.array([[1.0]], dtype=np.float32))


def test_spectral_decode_matches_analytic_gaussian():
    rng = np.random.default_rng(0)
    scene = _single_splat()
    rho = 1.3 / 0.03
    freqs = sample_frequencies(1 << 14, 2, rho, rng)
    bundle = spectral_bundle(scene, freqs)
    pts = rng.uniform(0.35, 0.65, size=(400, 2)).astype(np.float32)
    approx = decode_field(bundle, freqs, rho, pts)
    exact = eval_scene_exact(scene, pts)
    # MC error ~ chi/sqrt(d) ~ 0.01 at d=16384; 0.06 is >4 sigma of margin
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 0.06


def test_phasor_bundle_is_the_shared_kernel_special_case():
    rng = np.random.default_rng(1)
    scene = _single_splat()
    freqs = sample_frequencies(1 << 14, 2, 1.0 / 0.03, rng)
    bundle = phasor_bundle(scene, freqs)
    pts = rng.uniform(0.35, 0.65, size=(400, 2)).astype(np.float32)
    approx = decode_field_phasor(bundle, freqs, pts)
    exact = eval_scene_exact(scene, pts)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 0.06


def test_mixture_codebook_decodes_unbiased():
    rng = np.random.default_rng(2)
    scene = _single_splat()
    rho = [0.4 / 0.03, 1.3 / 0.03, 4.0 / 0.03]   # 3-component mixture
    freqs = sample_frequencies(1 << 14, 2, rho, rng)
    bundle = spectral_bundle(scene, freqs)
    pts = rng.uniform(0.35, 0.65, size=(400, 2)).astype(np.float32)
    approx = decode_field(bundle, freqs, rho, pts)
    exact = eval_scene_exact(scene, pts)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 0.08


def test_translate_bundle_shifts_the_decoded_field():
    rng = np.random.default_rng(3)
    scene = random_scene(20, dim=2, rng=rng, scale_range=(0.03, 0.05))
    rho = 1.3 / 0.03
    freqs = sample_frequencies(1 << 14, 2, rho, rng)
    bundle = spectral_bundle(scene, freqs)
    t = np.array([0.12, -0.07], dtype=np.float32)
    shifted = translate_bundle(bundle, freqs, t)
    pts = rng.uniform(0.25, 0.75, size=(400, 2)).astype(np.float32)
    # decoding the bound-translated bundle at p equals the original
    # mixture at p - t: the shift theorem is exact on the bundle
    approx = decode_field(shifted, freqs, rho, pts)
    exact = eval_scene_exact(scene, pts - t)
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 0.15   # same MC noise budget as the untranslated decode


def test_error_shrinks_with_dimension():
    rng = np.random.default_rng(4)
    scene = random_scene(30, dim=2, rng=rng, scale_range=(0.03, 0.05))
    rho = 1.3 / 0.03
    rels = []
    for d in (1 << 10, 1 << 14):
        book_rng = np.random.default_rng(7)
        freqs = sample_frequencies(d, 2, rho, book_rng)
        bundle = spectral_bundle(scene, freqs)
        pts = np.random.default_rng(8).uniform(
            0.2, 0.8, size=(400, 2)).astype(np.float32)
        exact = eval_scene_exact(scene, pts)
        approx = decode_field(bundle, freqs, rho, pts)
        rels.append(np.linalg.norm(approx - exact) / np.linalg.norm(exact))
    assert rels[1] < rels[0]           # ~1/sqrt(d)
    assert rels[1] < 0.15


def test_gpu_and_numpy_paths_agree():
    from holo import accel
    if not accel.active():
        pytest.skip("MLX backend not active")
    rng = np.random.default_rng(5)
    scene = random_scene(10, dim=3, rng=rng, scale_range=(0.03, 0.05))
    freqs = sample_frequencies(1 << 12, 3, 1.3 / 0.03, rng)
    fast = accel.spectral_bundle(scene, freqs)
    # reference: the pure-NumPy formulation, computed inline
    pairs = [(i, j) for i in range(3) for j in range(i, 3)]
    wq = np.stack([freqs[:, i] * freqs[:, j] * (1.0 if i == j else 2.0)
                   for i, j in pairs], axis=1)
    cq = np.stack([scene.cov[:, i, j] for i, j in pairs], axis=1)
    norm = ((2 * np.pi) ** 1.5
            * np.sqrt(np.linalg.det(scene.cov.astype(np.float64))))
    env = np.exp(-0.5 * (cq @ wq.T))
    ref = (scene.amp * norm[:, None]).T.astype(np.complex64) @ \
        ((env * np.exp(-1j * (scene.mu @ freqs.T)))).astype(np.complex64)
    # identical math, different silicon: float32 rounding only
    assert np.max(np.abs(fast - ref)) / np.max(np.abs(ref)) < 1e-5
