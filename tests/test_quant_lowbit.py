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
