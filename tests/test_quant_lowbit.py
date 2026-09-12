"""Low-bit rate and distortion checks independent of any proposed packer."""

import numpy as np
import pytest

from bench.quant_lowbit import HEADER_BYTES, quant_polar, synthetic_cells
from holo.phase import pack, pack_polar, unpack


def rayleigh_bundle(n=65536):
    rng = np.random.default_rng(17)
    return (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)


@pytest.mark.parametrize("d", [1, 3, 7, 8, 11, 33])
@pytest.mark.parametrize("mbits,pbits", [(0, 1), (1, 1), (1, 3), (2, 6), (8, 8)])
def test_lowbit_sizes_are_ceil_of_bits(d, mbits, pbits):
    q, size = quant_polar(np.ones(d), mbits, pbits)
    assert size == (d * mbits + 7) // 8 + (d * pbits + 7) // 8 + HEADER_BYTES
    assert q.shape == (d,) and q.dtype == np.complex64


def test_lowbit_error_is_monotone_in_bits():
    v = rayleigh_bundle()
    errors = [np.linalg.norm(quant_polar(v, b, b)[0] - v)
              for b in (1, 2, 3, 4, 6, 8)]
    assert np.all(np.diff(errors) < 0)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_lowbit_phase_similarity_meets_the_sinc_bound(bits):
    rng = np.random.default_rng(8)
    v = np.exp(1j * rng.uniform(-np.pi, np.pi, 65536)).astype(np.complex64)
    q, _ = quant_polar(v, 0, bits)
    q = q / np.abs(q)  # the scalar gain is not the phase code under test
    assert np.mean((q * v.conj()).real) >= np.sinc(1 / 2 ** bits) - 0.03


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 6, 8, 16])
@pytest.mark.parametrize("scale_rule", ["max", "p99.9"])
def test_mbits_zero_reproduces_the_hp_codec(bits, scale_rule):
    v = rayleigh_bundle(513)
    v[:3] = [0, -1, 1]
    q, _ = quant_polar(v, 0, bits, scale_rule=scale_rule)
    hp = unpack(pack(v, bits=bits))
    gain = np.vdot(hp, v).real / v.size
    np.testing.assert_allclose(q, hp * gain, rtol=2e-6, atol=1e-6)


def test_p999_scale_survives_a_single_outlier():
    rng = np.random.default_rng(22)
    v = np.exp(1j * rng.uniform(-np.pi, np.pi, 8192)).astype(np.complex64)
    v[0] *= 100
    clipped, _ = quant_polar(v, 2, 4, scale_rule="p99.9")
    maximum, _ = quant_polar(v, 2, 4, scale_rule="max")
    assert np.all(np.abs(clipped[1:]) > 0.9)
    assert np.all(maximum[1:] == 0)


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_symmetric_quantisation_matches_hg(bits):
    v = rayleigh_bundle(1024)
    np.testing.assert_allclose(quant_polar(v, bits, bits)[0],
                               unpack(pack_polar(v, bits)), rtol=2e-6, atol=1e-6)


def test_zero_magnitude_stays_zero():
    q, _ = quant_polar(np.zeros(11), 2, 3, scale_rule="p99.9")
    np.testing.assert_array_equal(q, np.zeros(11, np.complex64))


@pytest.mark.parametrize("kwargs", [{"mbits": -1}, {"pbits": 0},
                                   {"mbits": 1.5}, {"gamma": 0},
                                   {"scale_rule": "bad"}])
def test_invalid_quantiser_settings_fail(kwargs):
    settings = {"mbits": 2, "pbits": 2}
    settings.update(kwargs)
    with pytest.raises(ValueError):
        quant_polar(np.ones(8), **settings)


def test_synthetic_fixture_is_seeded_and_anisotropic():
    from holo.capture import BANDS, S_LO

    cells, half = synthetic_cells(ncells=2, npoints=8, nsplats=12)
    again, _ = synthetic_cells(ncells=2, npoints=8, nsplats=12)
    for (scene, pts), (other, repeated) in zip(cells, again):
        np.testing.assert_array_equal(scene.cov, other.cov)
        np.testing.assert_array_equal(pts, repeated)
        scales = np.sqrt(np.linalg.eigvalsh(scene.cov))
        assert scales.min() >= S_LO * (1 - 1e-6)
        assert scales.max() <= BANDS[0][1] * (1 + 1e-6)
        assert np.all(scales[:, 0] <= S_LO * 1.1 * (1 + 1e-6))
        assert np.all(np.abs(pts) <= half)


def test_d2_records_complete_grid_and_raw_drift(monkeypatch):
    from bench import precision_battery as pb
    from bench import quant_lowbit as ql
    from holo import spectral

    cells = synthetic_cells(npoints=8, nsplats=3)
    monkeypatch.setattr(ql, "synthetic_cells", lambda: cells)
    monkeypatch.setattr(ql, "LADDER", [(8, 2, 2)])
    result = pb.d2_lowbit(synthetic=True)
    rows = [r for r in result.values() if isinstance(r, dict)]
    assert len(rows) == 12
    assert {r["payload_budget"] for r in rows} == {8192, 16384, 32768}
    assert all(np.isfinite(r["median_drift"]) for r in rows)
    # An identity quantiser must yield zero drift, including batched readout.
    monkeypatch.setattr(ql, "quant_polar", lambda v, *_a, **_kw: (v, 0))
    monkeypatch.setattr(spectral, "spectral_bundle",
                        lambda _local, freqs: np.ones((1, len(freqs)), np.complex64))
    result = pb.d2_lowbit(synthetic=True)
    rows = [r for r in result.values() if isinstance(r, dict)]
    assert all(r["median_drift"] < 1e-6 for r in rows if not r["shrink"])


def test_synthetic_cli_records_to_a_relative_output(tmp_path, monkeypatch):
    import json

    from bench import precision_battery as pb
    from bench.quant_lowbit import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pb, "d2_lowbit", lambda synthetic: {"synthetic": synthetic})
    main(["--synthetic", "--output", "d2.jsonl"])
    row = json.loads((tmp_path / "d2.jsonl").read_text())
    assert row["id"] == "D2" and row["result"] == {"synthetic": True}


@pytest.mark.parametrize("d,block", [(1, 32), (31, 16), (64, 32), (129, 64)])
@pytest.mark.parametrize("mbits,pbits", [(1, 1), (2, 2), (1, 3), (4, 4)])
@pytest.mark.parametrize("scale_code", ["e8m0", "u8"])
def test_block_sizes_count_the_scales(d, block, mbits, pbits, scale_code):
    from bench.quant_lowbit import quant_block

    q, size = quant_block(rayleigh_bundle(d), mbits, pbits, block, scale_code)
    assert size == ((d * mbits + 7) // 8 + (d * pbits + 7) // 8
                    + (d + block - 1) // block + HEADER_BYTES)
    assert q.shape == (d,) and q.dtype == np.complex64


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("gamma", [0.5, 1.0, 0.7])
def test_block_equals_vector_scale_when_block_is_d(bits, gamma):
    from bench.quant_lowbit import quant_block

    for v in (rayleigh_bundle(513), np.zeros(17, np.complex64)):
        actual, _ = quant_block(v, bits, bits, len(v), "u8", gamma)
        expected, _ = quant_polar(v, bits, bits, gamma, "max")
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("scale_code", ["e8m0", "u8"])
def test_block_scale_isolates_an_outlier(scale_code):
    from bench.quant_lowbit import quant_block

    rng = np.random.default_rng(22)
    v = np.exp(1j * rng.uniform(-np.pi, np.pi, 8192)).astype(np.complex64)
    v[0] *= 100
    q, _ = quant_block(v, 2, 4, 32, scale_code)
    assert np.count_nonzero(q[:32]) >= 1
    assert np.all(np.mean(q[32:].reshape(-1, 32) != 0, axis=1) > 0.9)
    assert np.all(quant_polar(v, 2, 4)[0][1:] == 0)


@pytest.mark.parametrize("scale_code", ["e8m0", "u8"])
def test_block_error_is_monotone_in_bits(scale_code):
    from bench.quant_lowbit import quant_block

    v = rayleigh_bundle()
    errors = [np.linalg.norm(quant_block(v, b, b, 32, scale_code)[0] - v)
              for b in (1, 2, 3, 4, 6, 8)]
    assert np.all(np.diff(errors) < 0)


def test_e8m0_scale_is_a_power_of_two_and_covers_the_block_max():
    from bench.quant_lowbit import _block_scales, quant_block

    maxima = np.array([0, 2.0 ** -140, 0.5, 1, np.nextafter(
        np.float32(1), np.float32(2)), 3, 8, 100, 2.0 ** 127], np.float32)
    scales = _block_scales(np.repeat(maxima, 3), 3, "e8m0")
    expected = np.array([2.0 ** -127, 2.0 ** -127, 0.5, 1, 2, 4, 8,
                         128, 2.0 ** 127], np.float32)
    np.testing.assert_array_equal(scales, expected)
    assert np.all(scales >= maxima)
    np.testing.assert_array_equal(np.log2(scales), np.round(np.log2(scales)))
    # Observe the scales through reconstruction, not only the helper.
    v = np.repeat(np.array([0, 1, 3, 100], np.complex64), 3)
    q, _ = quant_block(v, 1, 4, 3)
    np.testing.assert_allclose(np.abs(q), np.repeat([0, 1, 4, 128], 3),
                               rtol=2e-6, atol=1e-6)
    with pytest.raises(ValueError, match="E8M0 range"):
        quant_block(np.array([np.finfo(np.float32).max]), 2, 2)


def test_mbits_zero_is_refused_for_blocks():
    from bench.quant_lowbit import quant_block

    with pytest.raises(ValueError, match="mbits"):
        quant_block(np.ones(32), 0, 2)


@pytest.mark.parametrize("scale_code", ["e8m0", "u8"])
def test_block_zero_and_rounded_zero_scales(scale_code):
    from bench.quant_lowbit import quant_block

    v = np.zeros(65, np.complex64)
    np.testing.assert_array_equal(quant_block(v, 2, 2, 32, scale_code)[0], v)
    v[0] = 1000
    v[32:] = 0.01
    q, _ = quant_block(v, 2, 2, 32, scale_code)
    assert np.all(np.isfinite(q))
    if scale_code == "u8":
        assert np.all(q[32:] == 0)


@pytest.mark.parametrize("kwargs", [{"block": 0}, {"block": 1.5},
                                   {"block": True}, {"scale_code": "bad"},
                                   {"scale_bits": 4}, {"gamma": 0},
                                   {"pbits": 0}])
def test_invalid_block_settings_fail(kwargs):
    from bench.quant_lowbit import quant_block

    settings = {"mbits": 2, "pbits": 2}
    settings.update(kwargs)
    with pytest.raises(ValueError):
        quant_block(np.ones(8), **settings)


@pytest.mark.parametrize("v", [[], [[1]], [np.nan], [np.inf]])
def test_invalid_block_vectors_fail(v):
    from bench.quant_lowbit import quant_block

    with pytest.raises(ValueError):
        quant_block(v, 2, 2)


def test_block_ladder_fits_payload_budgets():
    from bench.quant_lowbit import LADDER_BLOCK, quant_block

    for factor in (0.5, 1, 2):
        for d, mbits, pbits, block in LADDER_BLOCK:
            dimension = int(d * factor) // block * block
            _, size = quant_block(np.ones(dimension), mbits, pbits, block)
            assert dimension % block == 0
            assert size - HEADER_BYTES <= int(16384 * factor)
    for block in (16, 64):
        d = next(d for d, m, p, b in LADDER_BLOCK if b == block)
        assert d == (16384 // (block // 2 + 1)) * block
        assert (d + block) // block * (block // 2 + 1) > 16384


def test_d3_records_complete_grid(monkeypatch):
    from bench import precision_battery as pb
    from bench import quant_lowbit as ql
    from holo import spectral

    scenes, half = synthetic_cells(npoints=2, nsplats=1)
    cells = ([(scene, scene.mu.copy()) for scene, _ in scenes], half)
    monkeypatch.setattr(ql, "synthetic_cells", lambda: cells)
    monkeypatch.setattr(ql, "LADDER_BLOCK", [(16, 2, 2, 4)])
    result = pb.d3_blockscale(synthetic=True)
    rows = [r for r in result.values() if isinstance(r, dict)]
    assert len(rows) == 12  # Two scale codes and two references per budget.
    expected = set()
    for factor in (0.5, 1, 2):
        budget = int(16384 * factor)
        for code in ("e8m0", "u8"):
            expected.add("B=%d/d=%d/m=2/p=2/block=4/%s" %
                         (budget, int(16 * factor), code))
        for d, bits in ((8192, 8), (16384, 4)):
            expected.add("B=%d/d=%d/m=%d/p=%d/block=0/max" %
                         (budget, int(d * factor), bits, bits))
    assert {k for k in result if k.startswith("B=")} == expected
    assert all(np.isfinite(r[field]) for r in rows for field in
               ("median_rel_err", "median_drift", "raw_median_rel_err"))
    monkeypatch.setattr(ql, "quant_block", lambda v, *_a, **_kw: (v, 0))
    monkeypatch.setattr(ql, "quant_polar", lambda v, *_a, **_kw: (v, 0))
    monkeypatch.setattr(spectral, "spectral_bundle",
                        lambda _local, freqs: np.ones((1, len(freqs)), np.complex64))
    result = pb.d3_blockscale(synthetic=True)
    assert all(r["median_drift"] < 1e-6 for r in result.values()
               if isinstance(r, dict))


def test_d3_synthetic_cli_records(tmp_path, monkeypatch):
    import json

    from bench import precision_battery as pb
    from bench.quant_lowbit import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pb, "d3_blockscale", lambda synthetic: {"synthetic": synthetic})
    main(["--experiment", "D3", "--synthetic", "--output", "d3.jsonl"])
    row = json.loads((tmp_path / "d3.jsonl").read_text())
    assert row["id"] == "D3" and row["result"] == {"synthetic": True}
