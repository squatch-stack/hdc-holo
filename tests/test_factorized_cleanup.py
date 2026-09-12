"""Seeded CPU checks of cleanup, candidate coverage and allocation guards."""

import numpy as np
import pytest

from bench import factorized_cleanup as fc


def fixture_books(n=4):
    rng = np.random.default_rng(fc.SEED)
    return fc.axis_codebooks(rng.uniform(-np.pi, np.pi, (1024, 3)), n)


def test_product_row_is_the_elementwise_product_of_factor_rows():
    books = fixture_books()
    product = fc.product_codebook(books)
    for row, ix in enumerate(np.ndindex(4, 4, 4)):
        np.testing.assert_allclose(product[row], books[0][ix[0]] * books[1][ix[1]]
                                   * books[2][ix[2]], rtol=1e-6, atol=1e-6)


def test_resonator_equals_brute_force_at_load_one():
    ops = fc.workload(np, 1024, 4, 1, 8, np.random.default_rng(fc.SEED))
    brute, resonator = ops['brute'](), ops['resonator']()
    np.testing.assert_array_equal(resonator['argmax'], brute['argmax'])
    assert fc.recover(resonator, brute['truth_index']) == 1


def test_coarse_fine_equals_brute_force_when_top_r_covers_everything():
    books = fixture_books(5)
    query, truth = fc._queries(books, 16, 5, np.random.default_rng(7))
    memory = fc.product_codebook(books)
    brute = fc.cleanup_brute((query.real, query.imag), (memory.real, memory.imag))
    fine = fc.cleanup_coarse_fine(query, books, 2, 8)
    np.testing.assert_array_equal(fine['argmax'], brute['argmax'])
    assert all(t in choices for t, choices in zip(truth, fine['candidates']))


def test_budget_refusal_before_materialising(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('allocation/backend selection before budget check')

    monkeypatch.setattr(fc, 'product_codebook', forbidden)
    monkeypatch.setattr(fc, 'pick_backend', forbidden)
    with pytest.raises(MemoryError, match='exceeds'):
        fc.run_matrix('numpy', 1024, 8, [1], 4, 1, 1e-6)


def test_workload_keys_and_determinism():
    a = fc.workload(np, 1024, 4, 2, 3, np.random.default_rng(11))
    b = fc.workload(np, 1024, 4, 2, 3, np.random.default_rng(11))
    assert set(a) == {'brute', 'coarse_fine', 'resonator'}
    for key in a:
        x, y = a[key](), b[key]()
        np.testing.assert_array_equal(x['argmax'], y['argmax'])
        np.testing.assert_array_equal(x['truth_index'], y['truth_index'])
        np.testing.assert_allclose(x['scores'], y['scores'], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(x['checksum'], y['checksum'], rtol=2e-6)


def test_recovery_falls_with_load_for_every_cleanup():
    # Deliberately span the noise regime; low loads alone can all be perfect.
    values = {key: [] for key in ('brute', 'coarse_fine', 'resonator')}
    for load in (1, 32, 256):
        ops = fc.workload(np, 1024, 4, load, 12, np.random.default_rng(fc.SEED),
                          coarse=2, top_r=4)
        for key, fn in ops.items():
            result = fn()
            values[key].append(fc.recover(result, result['truth_index']))
    for rates in values.values():
        assert rates[0] - rates[-1] >= .25
        # One-query tolerance for this small seeded fixture.
        assert np.all(np.diff(rates) <= 1 / 12 + 1e-8)


def test_candidate_miss_and_spurious_diagnostics():
    report = fc.run_matrix('numpy', 1024, 4, [256], 4, 1, 1,
                           coarse=2, top_r=1, iters=8)
    methods = report['rows'][0]['methods']
    fine, resonator = methods['coarse_fine'], methods['resonator']
    assert fine['recovery'] <= fine['true_cell_selected_rate']
    assert resonator['spurious_rate'] <= resonator['converged_rate']
    assert resonator['mean_iterations'] <= 8
    assert methods['resonator']['bytes_held'] < methods['brute']['bytes_held']


@pytest.mark.parametrize('kwargs', [{'loads': []}, {'top_r': 0}, {'reps': 0},
                                   {'max_gb': float('nan')}, {'backend': 'mlx'}])
def test_invalid_run_parameters(kwargs):
    params = {'backend': 'numpy', 'd': 1024, 'n': 4, 'loads': [1],
              'Q': 2, 'reps': 1, 'max_gb': 1}
    params.update(kwargs)
    with pytest.raises(ValueError):
        fc.run_matrix(**params)


def test_krop_control_matches_dense_rotation_dictionary():
    angles = np.linspace(0, 2 * np.pi, 12)[1:-1]
    memory = np.stack([fc.krop_row(angles, i) for i in range(1024)])
    query = np.random.default_rng(17).standard_normal((3, 1024))
    expected = query @ memory.T
    result = fc.cleanup_krop(query, angles)
    np.testing.assert_array_equal(result['argmax'], expected.argmax(axis=1))
    np.testing.assert_allclose(result['scores'], expected.max(axis=1),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result['checksum'], np.abs(expected).sum(),
                               rtol=1e-12, atol=1e-12)


def test_measure_diagnostics_describe_the_last_timed_result():
    calls = []

    def changing():
        value = len(calls)
        calls.append(value)
        return {'argmax': np.array([value]), 'truth_index': np.array([2]),
                'checksum': float(value), 'scores': np.array([value])}

    row = fc._measure(changing, 2, 128, 'numpy')
    assert len(calls) == 3  # One warmup, two timed calls, no extra accuracy call.
    assert row['argmax'] == [2]
    assert row['checksum'] == 2
    assert row['recovery'] == 1
