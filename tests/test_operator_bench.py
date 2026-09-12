"""Small deterministic operator checks keep timing claims tied to known math."""

import numpy as np
import pytest

from bench.operator_bench import (
    _verify,
    main,
    op_bind,
    op_bind_hrr,
    op_bundle,
    op_cleanup,
    op_cleanup_hrr,
    op_similarity,
    predict_bytes,
    run_matrix,
)
from holo.fhrr import FHRR, ItemMemory


def test_cleanup_kernel_matches_itemmemory_scores():
    space = FHRR(32, seed=4)
    memory = ItemMemory(space)
    for label in range(13):
        memory.get(label)
    m = memory.matrix()
    queries = space.random(7)
    expected = np.stack([memory.scores(q) for q in queries]) * space.dim
    for chunk in (1, 5, 100):
        got = op_cleanup(queries.real, queries.imag, m.real, m.imag, chunk)
        np.testing.assert_array_equal(got['argmax'], expected.argmax(axis=1))
        np.testing.assert_allclose(got['scores'], expected.max(axis=1), atol=2e-6)
        np.testing.assert_allclose(got['checksum'], expected.sum(dtype=np.float64),
                                   atol=1e-5)


@pytest.mark.parametrize('d', [7, 8])
def test_hrr_bind_matches_circular_convolution(d):
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((2, 3, d)).astype(np.float32)
    expected = np.stack([sum(a[:, j] * b[:, (k - j) % d] for j in range(d))
                         for k in range(d)], axis=1)
    np.testing.assert_allclose(op_bind_hrr(a, b), expected, atol=2e-6)


def test_refuses_when_predicted_bytes_exceed_budget():
    predicted = predict_bytes(32, 10, 8, np.float32)
    with pytest.raises(MemoryError, match='predicted'):
        run_matrix('numpy', 32, 10, 8, 1, (predicted - 1) / 1e9)
    assert predict_bytes(32768, 100000, 4096, np.float32) > 26e9


def test_checksums_are_deterministic_at_fixed_seed():
    a = run_matrix('numpy', 32, 13, 7, 1, 1)
    b = run_matrix('numpy', 32, 13, 7, 1, 1)
    assert _verify(a, b) == dict.fromkeys(a['operators'], 0.0)
    assert a['accuracy'] == b['accuracy']
    for key in ('cleanup', 'cleanup_hrr'):
        assert a['operators'][key]['argmax_histogram'] == (
            b['operators'][key]['argmax_histogram'])
        assert sum(a['operators'][key]['argmax_histogram']) == 7


def test_equal_bytes_rows_exist():
    rows = run_matrix('numpy', 32, 13, 7, 1, 1)['accuracy']
    assert set(rows) == {'FHRR@d', 'FHRR@d/2', 'HRR@d'}
    # The same-d control necessarily uses twice the bytes of the matched pair.
    assert rows['FHRR@d/2']['bytes_per_codeword'] == (
        rows['HRR@d']['bytes_per_codeword'])
    assert rows['FHRR@d']['bytes_per_codeword'] == (
        2 * rows['HRR@d']['bytes_per_codeword'])
    assert rows['FHRR@d/2']['N'] == rows['HRR@d']['N'] == 8


def test_bind_and_spatial_kernels():
    rng = np.random.default_rng(9)
    a, b = rng.standard_normal((2, 5, 16, 2)).astype(np.float32)
    ar, ai = op_bind(a[..., 0], a[..., 1], b[..., 0], b[..., 1])
    np.testing.assert_allclose(ar + 1j * ai,
                               (a[..., 0] + 1j * a[..., 1])
                               * (b[..., 0] + 1j * b[..., 1]), atol=1e-6)
    mu, cov, amp, W, wq, points = (rng.random(shape, dtype=np.float32)
                                  for shape in [(5, 3), (5, 6), (5, 4),
                                                (16, 3), (16, 6), (7, 3)])
    s = amp.T @ (np.exp(-0.5 * cov @ wq.T) * np.exp(-1j * (mu @ W.T)))
    sr, si = op_bundle(mu, cov, amp, W, wq, chunk=2)
    np.testing.assert_allclose(sr + 1j * si, s, atol=1e-6)
    expected = np.real(np.exp(1j * (points @ W.T)) @ s.T)
    np.testing.assert_allclose(op_similarity(points, W, sr, si), expected,
                               atol=4e-6)


def test_cleanup_ties_and_hrr_scores():
    q = np.array([[1, 0], [0, 1]], dtype=np.float32)
    m = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    result = op_cleanup(q, q * 0, m, m * 0, 1)
    np.testing.assert_array_equal(result['argmax'], [0, 2])
    hrr = op_cleanup_hrr(q, m)
    assert hrr['checksum'] == float((q @ m.T).sum())
    np.testing.assert_array_equal(hrr['argmax'], result['argmax'])


def test_tf32_refused_before_backend_import(monkeypatch):
    monkeypatch.setenv('CUPY_TF32', '1')
    with pytest.raises(ValueError, match='TF32'):
        run_matrix('cupy', 32, 13, 7, 1, 1)


def test_cli_verification_and_bad_checksum(tmp_path):
    out = tmp_path / 'report.json'
    report = main(['--d', '8', '--K', '3', '--Q', '2', '--reps', '1',
                   '--verify', '--out', str(out)])
    assert out.exists()
    assert report['verification'][0]['K'] == 1000
    a = {'operators': {'cleanup': {'checksum': 1.0}}}
    b = {'operators': {'cleanup': {'checksum': 1.01}}}
    with pytest.raises(ValueError, match='verification failed'):
        _verify(a, b)
