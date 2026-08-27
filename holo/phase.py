"""Phase-only and quantized storage: shrinking complex64 up to 8x.

An FHRR *codeword* is all phase — unit magnitude by construction — so
storing its complex64 form (8 bytes/dim) wastes half the bits on
redundant magnitudes. Store the angle instead:

    float32 phases   4 bytes/dim   lossless in practice   (2x)
    uint8 codes      1 byte/dim    256 phase levels        (8x)

Quantization to b bits adds uniform phase noise of std (2pi/2^b)/sqrt(12)
per component, which scales expected similarity by E[cos eps] ~ 1 -
sigma^2/2 — at 8 bits that is a factor 0.9997, at 4 bits still 0.994:
phase quantization is nearly free, which is why phasor VSAs suit analog
and photonic hardware where phase precision is physically limited.

A *bundle* is different: superposition gives components varying
magnitudes, and projecting them back to unit phasors ('phasor
projection') discards real information. It still mostly works — the
demo measures the retrieval cost — but magnitude matters at high load,
so quantize codewords freely and think before phase-projecting bundles.
"""

import collections

import numpy as np

from .demokit import Table, banner
from .fhrr import FHRR
from .hashmap import HoloMap


def to_phases(v):
    """complex64 -> float32 angles (2x smaller). Magnitudes discarded."""
    return np.angle(v).astype(np.float32)


def from_phases(p):
    return np.exp(1j * p).astype(np.complex64)


def quantize(v, bits=8):
    """complex -> b-bit phase codes (uint8 for b<=8, else uint16)."""
    levels = 1 << bits
    codes = np.round((np.angle(v) + np.pi) * (levels / (2 * np.pi)))
    codes = codes.astype(np.int64) % levels
    return codes.astype(np.uint8 if bits <= 8 else np.uint16)


def dequantize(codes, bits=8):
    levels = 1 << bits
    phases = codes.astype(np.float32) * (2 * np.pi / levels) - np.pi
    return from_phases(phases)


# -- tagged storage envelope (format v1) ------------------------------------
#
# quantize()/dequantize() are the raw-array layer; pack()/unpack() are what
# goes to disk or over a wire: an 8-byte self-describing header (magic "HP",
# version, bits, dim) followed by the codes — nibble-packed two-per-byte
# when bits <= 4, so a 4-bit codeword really is dim/2 bytes. Changing this
# layout bumps STORAGE_VERSION.

import struct  # noqa: E402

STORAGE_VERSION = 1
_MAGIC = b"HP"
_HEADER = struct.Struct("<2sBBI")  # magic, version, bits, dim


def pack(v, bits=8):
    """complex vector -> tagged quantized-phase bytes (8 + payload)."""
    codes = quantize(v, bits)
    dim = len(codes)
    if bits <= 4:
        if dim % 2:
            codes = np.concatenate([codes, np.zeros(1, codes.dtype)])
        payload = ((codes[0::2].astype(np.uint8) << 4)
                   | codes[1::2].astype(np.uint8)).tobytes()
    else:
        payload = codes.tobytes()
    return _HEADER.pack(_MAGIC, STORAGE_VERSION, bits, dim) + payload


def unpack(buf):
    """Tagged storage bytes -> complex64 vector. Dispatches on the
    magic: 'HP' phase-only, 'HM' magnitude-preserving (linear re/im),
    'HG' gamma-companded polar."""
    if len(buf) >= _HEADER_GAMMA.size and buf[:2] == _MAGIC_GAMMA:
        return _unpack_polar(buf)
    if len(buf) >= _HEADER_MAG.size and buf[:2] == _MAGIC_MAG:
        return _unpack_complex(buf)
    if len(buf) < _HEADER.size or buf[:2] != _MAGIC:
        raise ValueError("not a holo phase blob: missing the 'HP'/'HM' "
                         "format tag")
    _, ver, bits, dim = _HEADER.unpack_from(buf)
    if ver != STORAGE_VERSION:
        raise ValueError(f"phase blob is storage version {ver}; this build "
                         f"speaks {STORAGE_VERSION}")
    if bits <= 4:
        packed = np.frombuffer(buf, np.uint8, offset=_HEADER.size)
        codes = np.empty(len(packed) * 2, dtype=np.uint8)
        codes[0::2] = packed >> 4
        codes[1::2] = packed & 0x0F
        codes = codes[:dim]
    elif bits <= 8:
        codes = np.frombuffer(buf, np.uint8, count=dim, offset=_HEADER.size)
    else:
        codes = np.frombuffer(buf, np.uint16, count=dim, offset=_HEADER.size)
    return dequantize(codes, bits)


# -- magnitude-preserving codec ('HM') --------------------------------------
#
# The codec_curve below shows WHY this exists: phase-only codes floor
# amplitude FIELDS at ~0.24 relative RMSE regardless of bit depth,
# because a weighted mixture's signal lives in component magnitudes.
# 'HM' stores scaled signed-integer re/im at `bits` per component
# (2 * dim * bits/8 payload + 12-byte header): 16x smaller than
# complex64 at 4 bits, 8x at 8 bits, with the magnitudes intact.

_MAGIC_MAG = b"HM"
_HEADER_MAG = struct.Struct("<2sBBIf")  # magic, version, bits, dim, scale


def pack_complex(v, bits=8):
    """complex vector -> tagged magnitude-preserving bytes."""
    a = np.ascontiguousarray(v, dtype=np.complex64)
    dim = a.shape[0]
    comps = np.concatenate([a.real, a.imag])           # (2*dim,) float32
    scale = float(np.max(np.abs(comps))) or 1.0
    top = (1 << (bits - 1)) - 1
    q = np.round(np.clip(comps / scale, -1, 1) * top).astype(np.int32)
    if bits <= 4:
        u = (q + 8).astype(np.uint8)                   # [-7,7] -> [1,15]
        payload = ((u[0::2] << 4) | u[1::2]).astype(np.uint8).tobytes()
    elif bits <= 8:
        payload = q.astype(np.int8).tobytes()
    else:
        payload = q.astype(np.int16).tobytes()
    return _HEADER_MAG.pack(_MAGIC_MAG, STORAGE_VERSION, bits, dim,
                            scale) + payload


def _unpack_complex(buf):
    _, ver, bits, dim, scale = _HEADER_MAG.unpack_from(buf)
    if ver != STORAGE_VERSION:
        raise ValueError(f"magnitude blob is storage version {ver}; this "
                         f"build speaks {STORAGE_VERSION}")
    top = (1 << (bits - 1)) - 1
    if bits <= 4:
        packed = np.frombuffer(buf, np.uint8, offset=_HEADER_MAG.size)
        u = np.empty(len(packed) * 2, dtype=np.int32)
        u[0::2] = packed >> 4
        u[1::2] = packed & 0x0F
        q = u[:2 * dim] - 8
    elif bits <= 8:
        q = np.frombuffer(buf, np.int8, count=2 * dim,
                          offset=_HEADER_MAG.size).astype(np.int32)
    else:
        q = np.frombuffer(buf, np.int16, count=2 * dim,
                          offset=_HEADER_MAG.size).astype(np.int32)
    comps = q.astype(np.float32) * (scale / top)
    return (comps[:dim] + 1j * comps[dim:]).astype(np.complex64)


# -- gamma-companded polar codec ('HG') -------------------------------------
#
# HM's weakness at low bits is dynamic range: linear re/im quantization
# with max-scaling spends most levels on the tail of the magnitude
# distribution. 'HG' codes each component in POLAR form — magnitude
# through a compressive power curve (m/scale)^gamma, phase uniformly —
# `bits` each, so byte cost matches HM at equal bits. Since v0.1.0
# shipped the HM header, this is a NEW magic, not a layout change.

_MAGIC_GAMMA = b"HG"
_HEADER_GAMMA = struct.Struct("<2sBBIff")  # magic, ver, bits, dim, scale, gamma


def _pack_block(codes, bits):
    if bits <= 4:
        if len(codes) % 2:
            codes = np.concatenate([codes, np.zeros(1, codes.dtype)])
        u = codes.astype(np.uint8)
        return ((u[0::2] << 4) | u[1::2]).astype(np.uint8).tobytes()
    if bits <= 8:
        return codes.astype(np.uint8).tobytes()
    return codes.astype(np.uint16).tobytes()


def _unpack_block(buf, offset, n, bits):
    if bits <= 4:
        nbytes = (n + 1) // 2
        packed = np.frombuffer(buf, np.uint8, count=nbytes, offset=offset)
        u = np.empty(nbytes * 2, dtype=np.uint16)
        u[0::2] = packed >> 4
        u[1::2] = packed & 0x0F
        return u[:n], offset + nbytes
    if bits <= 8:
        return (np.frombuffer(buf, np.uint8, count=n, offset=offset)
                .astype(np.uint16), offset + n)
    return (np.frombuffer(buf, np.uint16, count=n, offset=offset),
            offset + 2 * n)


def pack_polar(v, bits=8, gamma=0.5):
    """complex vector -> gamma-companded polar bytes (magnitude and
    phase at `bits` each; 2 * dim * bits/8 payload + 16-byte header)."""
    a = np.ascontiguousarray(v, dtype=np.complex64)
    dim = a.shape[0]
    m, ph = np.abs(a), np.angle(a)
    scale = float(m.max()) or 1.0
    mtop = (1 << bits) - 1
    mq = np.round(np.clip(m / scale, 0, 1) ** gamma * mtop) \
        .astype(np.uint16)
    plev = 1 << bits
    pq = (np.round((ph + np.pi) * (plev / (2 * np.pi)))
          .astype(np.int64) % plev).astype(np.uint16)
    return (_HEADER_GAMMA.pack(_MAGIC_GAMMA, STORAGE_VERSION, bits, dim,
                               scale, gamma)
            + _pack_block(mq, bits) + _pack_block(pq, bits))


def _unpack_polar(buf):
    _, ver, bits, dim, scale, gamma = _HEADER_GAMMA.unpack_from(buf)
    if ver != STORAGE_VERSION:
        raise ValueError(f"polar blob is storage version {ver}; this "
                         f"build speaks {STORAGE_VERSION}")
    mtop = (1 << bits) - 1
    plev = 1 << bits
    mq, off = _unpack_block(buf, _HEADER_GAMMA.size, dim, bits)
    pq, _ = _unpack_block(buf, off, dim, bits)
    m = (mq.astype(np.float32) / mtop) ** (1.0 / gamma) * scale
    ph = pq.astype(np.float32) * (2 * np.pi / plev) - np.pi
    return (m * np.exp(1j * ph)).astype(np.complex64)


# -- rate-distortion: bytes vs task fidelity --------------------------------

#: everything a codec measurement needs that is fixed for one d — the
#: alternative was a twelve-argument function or a late-binding closure
_Cell = collections.namedtuple(
    "_Cell", "d field truth peak holomap pairs probes")


def _codec_measure(cell, S_field, M_map, nbytes, bits, kind):
    """One (d, codec, bits) cell: field RMSE against the exact mixture,
    and HoloMap retrieval accuracy, after a codec round trip."""
    from .accel import readout
    rmse = float(np.sqrt(np.mean(
        (readout(cell.probes, cell.field.W, S_field) - cell.truth) ** 2)))
    rmse /= cell.peak
    hits = sum(cell.holomap.values.cleanup(
        np.conj(cell.holomap.keys.get(k)) * M_map)[0] == v
        for k, v in cell.pairs)
    return {"d": cell.d, "bits": bits, "nbytes": nbytes, "kind": kind,
            "field_rmse": rmse, "map_acc": hits / len(cell.pairs)}


def codec_curve(dims=(512, 1024, 2048, 4096, 8192),
                bits_list=(2, 3, 4, 6, 8, 16),
                n_splats=40, n_pairs=200, n_probe=1500, seed=0):
    """The storage question made quantitative: for a fixed byte budget,
    is it better to spend on MORE DIMENSIONS or MORE PHASE PRECISION?

    For every (d, bits) cell, two task measurements after a round trip
    through the tagged codec (which includes the bundle's phase
    projection — the lossy part):
      * field: a 40-splat mixture bundle, relative RMSE vs the exact
        mixture at probe points;
      * map: a 200-pair HoloMap bundle, retrieval accuracy against a
        64-value codebook.
    Plus the raw complex64 baseline per d (bits=64). Returns dict rows:
    {d, bits, nbytes, field_rmse (rel peak), map_acc}."""
    from .fhrr import FHRR
    from .field import GaussianSplatField
    from .hashmap import HoloMap

    rng = np.random.default_rng(seed)
    mus = rng.uniform(0.1, 0.9, (n_splats, 2))
    alphas = rng.uniform(0.5, 1.0, n_splats)
    P = rng.uniform(0, 1, (n_probe, 2)).astype(np.float32)
    pairs = [(f"k{i}", f"v{i % 64}") for i in range(n_pairs)]

    rows = []
    for d in dims:
        field = GaussianSplatField(d, np.eye(2) * 0.05 ** 2, seed=seed)
        for mu, a in zip(mus, alphas):
            field.add_splat(mu, float(a))
        truth = field.exact(P)
        peak = float(truth.max())
        m = HoloMap(FHRR(d, seed=seed))
        for k, v in pairs:
            m.put(k, v)

        cell = _Cell(d, field, truth, peak, m, pairs, P)

        def measure(S_field, M_map, nbytes, bits, kind, cell=cell):
            # `cell` bound as a default, not captured: this closure is
            # called in-iteration today, but a late-binding capture
            # would silently measure the LAST d for every row if anyone
            # ever deferred the call
            rows.append(_codec_measure(cell, S_field, M_map, nbytes,
                                       bits, kind))

        measure(field.S, m.M, 8 * d, 64, "raw")       # complex64 baseline
        for bits in bits_list:
            buf_f, buf_m = pack(field.S, bits), pack(m.M, bits)
            measure(unpack(buf_f), unpack(buf_m), len(buf_f), bits, "HP")
        for bits in (b for b in bits_list if b >= 4):
            buf_f = pack_complex(field.S, bits)
            buf_m = pack_complex(m.M, bits)
            measure(unpack(buf_f), unpack(buf_m), len(buf_f), bits, "HM")
        for bits in (b for b in bits_list if b >= 4):
            buf_f = pack_polar(field.S, bits)
            buf_m = pack_polar(m.M, bits)
            measure(unpack(buf_f), unpack(buf_m), len(buf_f), bits, "HG")
    return rows


def _codec_table(rows):
    t = Table(("d", 6), ("codec", 6), ("bits", 5), ("bytes", 8, ","),
              ("field RMSE", 11, ".3f"), ("map acc", 8, ".1%"),
              indent="  ")
    t.header()
    for r in rows:
        t.row(r["d"], r["kind"], r["bits"], r["nbytes"],
              r["field_rmse"], r["map_acc"])


def _codec_symbols_finding(rows):
    """Cleanup needs DIRECTIONS, so phase-only plus few bits wins big
    at equal bytes."""
    solid = [r for r in rows if r["map_acc"] >= 0.995]
    best = min(solid, key=lambda r: r["nbytes"], default=None)
    base = min((r for r in solid if r["bits"] == 64),
               key=lambda r: r["nbytes"], default=None)
    if not (best and base):
        return
    print(f"  symbols: {best['bits']}-bit d={best['d']} reaches "
          f"~100% in {best['nbytes']:,} B — "
          f"{base['nbytes'] / best['nbytes']:.0f}x fewer bytes "
          f"than the cheapest complex64 config")


def _codec_fields_finding(rows):
    """The other half of the split: a weighted mixture's signal lives
    in component MAGNITUDES, so the phase projection sets a floor."""
    floor = min(r["field_rmse"] for r in rows if r["kind"] == "HP")
    raw = min(r["field_rmse"] for r in rows if r["kind"] == "raw")
    hm = min((r for r in rows if r["kind"] == "HM"),
             key=lambda r: r["field_rmse"])
    print(f"  fields: phase-only (HP) floors RMSE at ~{floor:.2f} at ANY "
          f"bit depth (complex64 reaches {raw:.3f}); the "
          f"magnitude-preserving HM codec breaks the floor — "
          f"{hm['bits']}-bit re/im at d={hm['d']} reaches "
          f"{hm['field_rmse']:.3f} in {hm['nbytes']:,} B "
          f"({8 * hm['d'] / hm['nbytes']:.0f}x smaller than complex64)")


def _codec_companding_finding(rows):
    def mean_rmse(kind, bits):
        sub = [r for r in rows if r["kind"] == kind and r["bits"] == bits]
        if not sub:
            return float("nan")
        return min(np.mean([r["field_rmse"] for r in sub if r["d"] >= 2048]),
                   9.9)
    print(f"  companding (HG, gamma 0.5) vs linear (HM) at the low-bit "
          f"end, field RMSE averaged over d>=2048: "
          f"4-bit {mean_rmse('HG', 4):.3f} vs {mean_rmse('HM', 4):.3f}; "
          f"8-bit {mean_rmse('HG', 8):.3f} vs {mean_rmse('HM', 8):.3f}")


def _codec_plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print()
        return
    import os
    os.makedirs("out", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    dims = sorted({r["d"] for r in rows})
    cmap = plt.get_cmap("magma")
    for i, d in enumerate(dims):
        color = cmap(0.15 + 0.7 * i / max(len(dims) - 1, 1))
        for kind, style in (("HP", ":o"), ("HM", "-s"), ("HG", "--^")):
            sub = sorted((r for r in rows
                          if r["d"] == d and r["kind"] == kind),
                         key=lambda r: r["nbytes"])
            xs = [r["nbytes"] for r in sub]
            label = f"d={d} {kind}" if i in (0, len(dims) - 1) else None
            axes[0].plot(xs, [r["field_rmse"] for r in sub], style,
                         color=color, label=label, markersize=4)
            axes[1].plot(xs, [r["map_acc"] for r in sub], style,
                         color=color, markersize=4)
        base = next(r for r in rows if r["d"] == d and r["kind"] == "raw")
        axes[0].plot([base["nbytes"]], [base["field_rmse"]], "*",
                     color=color, markersize=11)
        axes[1].plot([base["nbytes"]], [base["map_acc"]], "*",
                     color=color, markersize=11)
    for ax, title, ylabel in [
            (axes[0], "field task (lower is better)", "relative RMSE"),
            (axes[1], "map task (higher is better)", "retrieval accuracy")]:
        ax.set_xscale("log")
        ax.set_xlabel("stored bytes (tagged codec)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Symbols survive phase-only codes (HP, dotted); amplitude "
                 "fields need magnitudes (HM solid, HG companded dashed; "
                 "star = complex64)", fontsize=11)
    fig.tight_layout()
    fig.savefig("out/codec_curve.png", dpi=110)
    plt.close(fig)
    print("  saved out/codec_curve.png")
    print()


def demo_codec(dim=4096, seed=0, save_png=True):
    banner("Codec rate-distortion: bytes vs task fidelity")
    rows = codec_curve(seed=seed)
    _codec_table(rows)
    # the punchline: the two tasks SPLIT
    _codec_symbols_finding(rows)
    _codec_fields_finding(rows)
    _codec_companding_finding(rows)
    if not save_png:
        print()
        return
    _codec_plot(rows)

def demo(dim=4096, seed=0):
    banner("Phase-only & quantized storage", dim)
    space = FHRR(dim, seed=seed)
    v = space.random()
    for bits in [16, 8, 4, 2]:
        r = space.sim(dequantize(quantize(v, bits), bits), v)
        print(f"  codeword at {bits:>2}-bit phases: similarity to "
              f"original {r:.4f}  ({bits/64:.3%} of complex64 size)")

    print("  bundle storage cost (HoloMap, value alphabet 256):")
    print(f"  {'pairs N':>8} {'complex64':>10} {'phase f32':>10} "
          f"{'uint8':>7} {'uint4':>7}")
    for n_pairs in [250, 500, 1000, 2000]:
        space = FHRR(dim, seed=seed)
        m = HoloMap(space)
        rng = np.random.default_rng(seed + 1)
        pairs = [(f"key{i}", f"val{rng.integers(256)}")
                 for i in range(n_pairs)]
        for k, val in pairs:
            m.put(k, val)

        def accuracy(bundle, m=m, pairs=pairs, n_pairs=n_pairs):
            hits = 0
            for k, val in pairs:
                v_hat = FHRR.unbind(bundle, m.keys.get(k))
                hits += m.values.cleanup(v_hat)[0] == val
            return hits / n_pairs

        full = accuracy(m.M)
        phase32 = accuracy(from_phases(to_phases(m.M)))
        q8 = accuracy(dequantize(quantize(m.M, 8), 8))
        q4 = accuracy(dequantize(quantize(m.M, 4), 4))
        print(f"  {n_pairs:>8} {full:>10.1%} {phase32:>10.1%} "
              f"{q8:>7.1%} {q4:>7.1%}")
    print(f"  (bytes per bundle: {8*dim} / {4*dim} / {dim} / {dim//2} — "
          "phase projection costs accuracy only near the capacity cliff)")
    print()
