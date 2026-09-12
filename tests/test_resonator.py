"""Synthetic recovery, honest failures and the two encoder phase signs."""

import numpy as np
import pytest

from holo.attribute_field import AttributeSplatField
from holo.fhrr import FHRR
from holo.resonator import deflate, factorize_all, grid_codebook, resonator
from holo.spectral import SplatScene, spectral_bundle, translate_bundle

# 2026-09-12: OPENBLAS_NUM_THREADS=1 python -m bench.resonator_sweep
# --dims 4096 --grids 16 --objects 3 --trials 50 --no-plot
# Exact object recovery 127/150; 42/50 scenes recovered completely.
MEASURED_RATE = 127 / 150
N_TRIALS = 50


def _trials(space, n_objects, seed=0):
    field = AttributeSplatField(space, .04)
    values = np.linspace(0, 1, 16)
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        field.attrs.get(label)
    books = [field.attrs.matrix(), *(grid_codebook(field.W[:, k], values)
                                    for k in range(2))]
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, space.dim, 16, n_objects]))
    for _ in range(N_TRIALS):
        flat = rng.choice(26 * 16 * 16, n_objects, replace=False)
        truth = {tuple(map(int, np.unravel_index(i, (26, 16, 16)))) for i in flat}
        field.S.fill(0)
        field.splats.clear()
        for label, x, y in sorted(truth):
            field.add_splat([values[x], values[y]], field.attrs.labels[label])
        yield truth, factorize_all(field.S, books, n_objects)


def _found(results):
    return {tuple(r.indices) for r in results if r.converged}


def test_single_object_is_recovered_exactly(space):
    for truth, results in _trials(space, 1):
        assert _found(results) == truth


@pytest.mark.parametrize("n_objects", [2, 3, 4])
def test_multi_object_recovery_meets_the_measured_rate(space, n_objects):
    # Independent stream for regression; 50 scenes are the independent units.
    recovered = sum(len(truth & _found(results))
                    for truth, results in _trials(space, n_objects, seed=17))
    rate = recovered / (N_TRIALS * n_objects)
    # Same sweep with --objects 2 3 4: rates .90, .8467, .63.
    measured = {2: .90, 3: MEASURED_RATE, 4: .63}[n_objects]
    margin = 4 * np.sqrt(measured * (1 - measured) / N_TRIALS)
    assert rate >= measured - margin


def test_converges_within_forty_iterations(space):
    for _, results in _trials(space, 1):
        result = results[0]
        assert result.converged
        assert 3 <= result.n_iters <= 40
        assert len(result.trace) == result.n_iters
        assert all(len(row) == 3 for row in result.trace)


def test_sign_convention_round_trips_a_spectral_translation(space):
    rng = np.random.default_rng(21)
    freqs = rng.normal(0, 25, (space.dim, 2)).astype(np.float32)
    values = np.linspace(-.5, .5, 16)
    indices = [3, 11]
    translation = values[indices]
    scene = SplatScene(np.zeros((1, 2), dtype=np.float32),
                       np.array([np.eye(2) * .0001], dtype=np.float32),
                       np.ones((1, 4), dtype=np.float32))
    bundle = spectral_bundle(scene, freqs)
    shifted = translate_bundle(bundle, freqs, translation)
    books = [grid_codebook(freqs[:, k], values, sign=-1) for k in range(2)]
    result = resonator(shifted[0], books)
    assert result.converged
    assert result.indices == indices
    wrong = resonator(shifted[0], [m.conj() for m in books])
    assert wrong.indices == [15 - i for i in indices]


def test_capacity_cliff_is_reported_not_hidden():
    trials = list(_trials(FHRR(1024, seed=0), 8))
    recovered = sum(len(truth & _found(results)) for truth, results in trials)
    assert recovered / (N_TRIALS * 8) < .5
    assert any(not r.converged for _, results in trials for r in results)


def test_deflation_fits_real_amplitude_without_mutating_input(space):
    books = [space.random(5), space.random(4)]
    chosen = FHRR.bind(books[0][2], books[1][1])
    signal = 2.5 * chosen
    original = signal.copy()
    result = resonator(signal, books, init=[books[0][2], books[1][1]])
    np.testing.assert_allclose(deflate(signal, result, books), 0, atol=2e-6)
    np.testing.assert_array_equal(signal, original)


def test_projection_update_and_hysteresis(space):
    books = [space.random(4), space.random(3)]
    initial = [FHRR.normalize(space.random()), FHRR.normalize(space.random())]
    signal = FHRR.bind(books[0][1], books[1][2])
    result = resonator(signal, books, iters=1, init=initial, hysteresis=.25)
    for j, m in enumerate(books):
        q = signal * FHRR.normalize(initial[1 - j]).conj()
        projected = FHRR.normalize(m.T @ (m.conj() @ q))
        expected = FHRR.normalize(.75 * projected + .25 * FHRR.normalize(initial[j]))
        np.testing.assert_allclose(result.estimates[j], expected, atol=2e-5)
    assert not result.converged


def test_zero_signal_and_exhausted_budget_are_honest(space):
    books = [space.random(4)]
    result = resonator(space.zeros(), books)
    assert not result.converged
    assert result.n_iters == 40
    assert result.scores == [0.]
    assert not resonator(books[0][0], books, iters=0).converged
    assert len(factorize_all(space.zeros(), books, 4)) == 1


def test_seeded_initialization_is_repeatable(space):
    books = [space.random(4)]
    a = resonator(books[0][1], books, rng=np.random.default_rng(9))
    b = resonator(books[0][1], books, rng=np.random.default_rng(9))
    assert a.indices == b.indices == [1]
    assert a.trace == b.trace


@pytest.mark.parametrize("books", [[], [np.ones((2, 3))]])
def test_invalid_codebooks_raise(space, books):
    with pytest.raises(ValueError):
        resonator(space.zeros(), books)
