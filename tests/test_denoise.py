"""Shrinkage denoising (holo/denoise.py).

The interesting test is not that the arithmetic is right — it is that
the estimator actually recovers a planted signal from planted noise,
and that soft beats hard while doing it. Those are the two claims the
module makes, so those are what fail here if it regresses.
"""

import numpy as np
import pytest

from holo.denoise import percentile_threshold, shrink


def _planted(seed=0, d=4096, k=64):
    """A sparse strong signal in weak noise — magnitudes well separated."""
    rng = np.random.default_rng(seed)
    clean = np.zeros(d, np.complex64)
    idx = rng.choice(d, k, replace=False)
    clean[idx] = rng.normal(size=k) + 1j * rng.normal(size=k)
    noise = (rng.normal(scale=0.05, size=d)
             + 1j * rng.normal(scale=0.05, size=d)).astype(np.complex64)
    return clean, (clean + noise).astype(np.complex64)


def _overlapping(seed=1, d=4096):
    """The regime a forward-encoded capture bundle is actually in: a
    heavy-tailed, CONTINUOUS magnitude distribution that reaches down
    into the noise scale, because a mixture codebook weights frequencies
    unequally. There is no clean gap to threshold at."""
    rng = np.random.default_rng(seed)
    mag = np.exp(rng.normal(-1.5, 1.2, size=d))
    clean = (mag * np.exp(1j * rng.uniform(0, 2 * np.pi, d))).astype(np.complex64)
    noise = (rng.normal(scale=0.15, size=d)
             + 1j * rng.normal(scale=0.15, size=d)).astype(np.complex64)
    return clean, (clean + noise).astype(np.complex64)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_shrinkage_recovers_a_planted_signal():
    clean, noisy = _planted()
    before = _rel(noisy, clean)
    out = shrink(noisy, 0.15, mode="soft")
    assert _rel(out, clean) < before, "shrinkage did not denoise"
    assert _rel(out, clean) < 0.5 * before


def test_soft_beats_hard_where_signal_and_noise_overlap():
    """Soft's advantage is REGIME-SPECIFIC, and this pins the regime
    rather than the slogan. Where magnitudes overlap and the threshold
    is low — which is where the real capture optimum sits, p25 — soft
    wins because most components near the threshold are mixtures of
    signal and crosstalk, and zeroing them outright discards the signal
    with the noise."""
    clean, noisy = _overlapping()
    t = percentile_threshold(noisy, 25)
    soft = _rel(shrink(noisy, t, mode="soft"), clean)
    hard = _rel(shrink(noisy, t, mode="hard"), clean)
    assert soft < hard, "soft %.4f should beat hard %.4f here" % (soft, hard)


def test_hard_beats_soft_when_the_signal_is_well_separated():
    """The other half of the same fact, kept as a test so nobody
    'simplifies' the module into always recommending soft: with a sparse
    strong signal in weak noise there is a real gap to cut at, and soft's
    shrinkage of the surviving components is pure bias."""
    clean, noisy = _planted()
    soft = _rel(shrink(noisy, 0.15, mode="soft"), clean)
    hard = _rel(shrink(noisy, 0.15, mode="hard"), clean)
    assert hard < soft


def test_hard_zeroes_below_and_keeps_above():
    v = np.array([0.1, 0.5, 1.0], np.complex64)
    out = shrink(v, 0.4, mode="hard")
    assert out[0] == 0
    assert np.allclose(out[1:], v[1:])


def test_soft_shrinks_magnitude_and_preserves_phase():
    ang = np.array([0.3, -2.0, 1.7])
    v = (2.0 * np.exp(1j * ang)).astype(np.complex64)
    out = shrink(v, 0.5, mode="soft")
    assert np.allclose(np.abs(out), 1.5, atol=1e-6)      # 2.0 - 0.5
    assert np.allclose(np.angle(out), ang, atol=1e-6)    # phase untouched


def test_soft_does_not_produce_nan_at_exact_zeros():
    # the guarded divide exists for this: a zero component has no phase
    v = np.array([0.0, 1.0], np.complex64)
    out = shrink(v, 0.5, mode="soft")
    assert np.all(np.isfinite(out))
    assert out[0] == 0


def test_percentile_threshold_is_per_channel():
    # a premultiplied alpha channel and a colour channel do not share a
    # magnitude scale, so one global threshold would gut the quieter one
    v = np.stack([np.full(100, 1.0), np.full(100, 100.0)]).astype(np.complex64)
    t = percentile_threshold(v, 50)
    assert t.shape == (2, 1)
    assert np.allclose(t.ravel(), [1.0, 100.0])
    out = shrink(v, t, mode="hard")
    assert np.count_nonzero(out) > 0          # neither channel is wiped


def test_percentile_threshold_handles_one_dimensional_input():
    v = np.arange(1, 101).astype(np.complex64)
    assert np.isclose(percentile_threshold(v, 50), 50.5)


def test_output_is_complex64():
    v = (np.arange(8) + 1j * np.arange(8)).astype(np.complex128)
    for mode in ("soft", "hard"):
        assert shrink(v, 1.0, mode=mode).dtype == np.complex64


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match=r"soft.*hard"):
        shrink(np.ones(4, np.complex64), 0.1, mode="threshold")
