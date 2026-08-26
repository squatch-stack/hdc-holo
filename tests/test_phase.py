"""Phase-only and quantized storage."""

import pytest
import numpy as np

from holo import FHRR, HoloMap
from holo import phase


def test_phase_quantization_roundtrip(space):
    v = space.random()
    assert space.sim(phase.from_phases(phase.to_phases(v)), v) == \
        pytest.approx(1.0, abs=1e-3)
    assert space.sim(phase.dequantize(phase.quantize(v, 8), 8), v) > 0.999
    assert space.sim(phase.dequantize(phase.quantize(v, 4), 4), v) > 0.99


def test_phase_projected_bundle_still_retrieves(space):
    m = HoloMap(space)
    pairs = {f"k{i}": f"v{i % 40}" for i in range(200)}
    for k, v in pairs.items():
        m.put(k, v)
    tiny = phase.dequantize(phase.quantize(m.M, 8), 8)  # 8x smaller storage
    for k, v in pairs.items():
        v_hat = FHRR.unbind(tiny, m.keys.get(k))
        assert m.values.cleanup(v_hat)[0] == v


def test_pack_unpack_roundtrip_and_sizes(space):
    v = space.random()
    for bits, payload in [(8, space.dim), (4, space.dim // 2),
                          (16, 2 * space.dim)]:
        buf = phase.pack(v, bits)
        assert len(buf) == 8 + payload       # tagged: 8-byte header
        assert space.sim(phase.unpack(buf), v) > (0.99 if bits == 4
                                                  else 0.999)


def test_pack_rejects_foreign_bytes(space):
    with pytest.raises(ValueError, match="format tag"):
        phase.unpack(b"not a phase blob")
    with pytest.raises(ValueError, match="format tag"):
        phase.unpack(phase.quantize(space.random(), 8).tobytes())


def test_codec_curve_shape_and_sanity():
    rows = phase.codec_curve(dims=(512,), bits_list=(4, 8),
                             n_splats=10, n_pairs=30, n_probe=200)
    assert len(rows) == 7    # raw + 2 HP + 2 HM + 2 HG
    by_bits = {r["bits"]: r for r in rows}
    assert by_bits[4]["nbytes"] < by_bits[8]["nbytes"] < by_bits[64]["nbytes"]
    for r in rows:
        assert np.isfinite(r["field_rmse"]) and 0 <= r["map_acc"] <= 1


def test_pack_complex_roundtrip_preserves_magnitudes(space):
    rng = np.random.default_rng(71)
    bundle = (rng.normal(size=space.dim) * 3
              + 1j * rng.normal(size=space.dim)).astype(np.complex64)
    for bits, payload in [(8, 2 * space.dim), (4, space.dim),
                          (16, 4 * space.dim)]:
        buf = phase.pack_complex(bundle, bits)
        assert len(buf) == 12 + payload
        got = phase.unpack(buf)                # unpack() dispatches on magic
        rel = np.linalg.norm(got - bundle) / np.linalg.norm(bundle)
        assert rel < {4: 0.25, 8: 0.02, 16: 0.001}[bits]
    # phase-only projection of the same bundle loses magnitudes; HM keeps
    # them: HM must reconstruct strictly better at equal bytes (8-bit HP
    # vs 4-bit HM are both ~1 byte/dim)
    hp = phase.unpack(phase.pack(bundle, 8))
    hm = phase.unpack(phase.pack_complex(bundle, 4))
    err = lambda v: np.linalg.norm(v - bundle)  # noqa: E731
    assert err(hm) < err(hp)


def test_pack_polar_roundtrip_and_dispatch(space):
    rng = np.random.default_rng(73)
    # Rayleigh-ish magnitudes, the shape bundles actually have
    bundle = (rng.normal(size=space.dim)
              + 1j * rng.normal(size=space.dim)).astype(np.complex64)
    for bits, payload in [(8, 2 * space.dim), (4, space.dim),
                          (16, 4 * space.dim)]:
        buf = phase.pack_polar(bundle, bits)
        assert len(buf) == 16 + payload
        got = phase.unpack(buf)               # dispatches on 'HG' magic
        rel = np.linalg.norm(got - bundle) / np.linalg.norm(bundle)
        assert rel < {4: 0.22, 8: 0.03, 16: 0.002}[bits]
    # companding must beat linear re/im at the 4-bit end on
    # heavy-tailed magnitudes (equal bytes: both are 1 byte/dim)
    err = lambda v: np.linalg.norm(v - bundle)  # noqa: E731
    hg4 = err(phase.unpack(phase.pack_polar(bundle, 4)))
    hm4 = err(phase.unpack(phase.pack_complex(bundle, 4)))
    assert hg4 < hm4
