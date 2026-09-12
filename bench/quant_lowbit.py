"""Measure sub-nibble storage before implementing a byte stream.

The abstract of arXiv:2604.25939 (checked online on 2026-09-12) describes
qFHRR's 3--4-bit phase indices and integer operations through modular
arithmetic and lookup tables. That supports testing low-bit storage, but
not replacing our amplitude-bearing bundles with integer bundling.
Per-dimension histograms with eight or more bins and even 8-bit counters
cost at least 64 bits/dimension (and overflow beyond 255 contributions).
A fixed-point real/imaginary accumulator is HM with a shared scale. LUT
similarity is exact for HP codes; a speed benefit requires suitable integer
hardware. These are analytical objections, not measured speed claims.

D1 stopped at four bits because the existing packer has a nibble floor.
Quantisation distortion does not depend on packing, so the rates here are
analytic: separately rounded magnitude and phase streams plus the 17-byte
proposed HQ header <2sBBBIff (magic, version, mbits, pbits, d, scale, gamma).
Even HP uses that common hypothetical header for equal-rate comparisons;
no HQ bytes are emitted. With mbits=0 the phase codes are exactly HP's and
the header's scale slot carries the least-squares scalar gain, so the
phase-only row is comparable at equal bytes. Scale selection has no
effect on that row.

Run all 84 settings over 24 synthetic cells (1500 probes, 330 splats each):
    HDC_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m bench.quant_lowbit \
        --synthetic --output /tmp/quantphase-d2.jsonl
Omit --synthetic and supply --capture for the real-cell confirmation.
Payload budgets are 8192, 16384 and 32768 bytes; headers are additional,
matching D1's convention. The synthetic outcome is provisional.

The D2 tables, the knee they locate and the packer decision are in
results/quant_lowbit.md. In one line: the knee is 4+4 bits at every
budget on the synthetic fixture and on two captures (where 2/2 at 4d is
about 2x worse than 4/4 at 2d), nothing at one bit survives, and
phase-only is not a rate point for a spectral bundle at any bit depth,
so no HQ packer is written.

Failure modes. (1) A max-derived scale lets one outlier eat the levels at
one or two magnitude bits; p99.9 clipping recovers the bulk and the test
pins it. (2) Shrink-then-quantise is itself a ~0.11 drift, so it helps
only where quantisation noise is larger than that (two bits and below) and
hurts at four bits and above. (3) Phase-only storage: a spectral bundle's
magnitudes are the Gaussian envelope, so unit-modulus phases whiten the
spectrum and the decoded field is dominated by frequencies that carried
nothing; a scalar gain cannot repair a spectral shape. That is the
projection floor holo.phase.codec_curve reports for fields, not a bit-depth
effect. (4) The synthetic cells are uniform in occupancy and free of the
coherent-crosstalk floor of real cells (holo/capture.py), so the
dimension-over-precision trend D1 measured on captures is only weakly
present here; the knee is provisional until rerun on a capture.
"""

import argparse
import os
import struct
import time

import numpy as np

from holo.phase import check_bits, dequantize, quantize

HEADER_BYTES = struct.calcsize("<2sBBBIff")
LADDER = ((8192, 8, 8), (16384, 4, 4), (32768, 2, 2),
          (65536, 1, 1), (32768, 1, 3), (65536, 0, 2), (16384, 2, 6))


def quant_polar(v, mbits, pbits, gamma=0.5, scale_rule="max"):
    """Return complex64 reconstruction and ideal stream bytes including header.

    Percentile clipping trades the largest 0.1% for useful magnitude levels
    in the bulk, where max scaling can otherwise round everything to zero.
    Empty vectors and nonfinite data are refused rather than reporting a
    plausible rate for an undefined scale.
    """
    mbits = 0 if mbits == 0 else check_bits(mbits, "mbits")
    pbits = check_bits(pbits, "pbits")
    a = np.asarray(v, dtype=np.complex64)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError("v must be a nonempty finite vector")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if scale_rule not in ("max", "p99.9"):
        raise ValueError("scale_rule must be max or p99.9")
    nbytes = (a.size * mbits + 7) // 8 + (a.size * pbits + 7) // 8
    phasor = dequantize(quantize(a, pbits), pbits)
    if mbits == 0:
        # Phase-only keeps HP's codes exactly; the header's scale slot holds
        # the least-squares gain Re<phasor, a>/d so the row measures phase
        # distortion rather than the missing amplitude unit.
        gain = np.float32(np.vdot(phasor, a).real / a.size)
        return (phasor * gain).astype(np.complex64), nbytes + HEADER_BYTES
    mag = np.abs(a)
    scale = float(mag.max() if scale_rule == "max"
                  else np.percentile(mag, 99.9)) or 1.0
    top = (1 << mbits) - 1
    codes = np.round(np.clip(mag / scale, 0, 1) ** gamma * top)
    restored = (codes / top) ** (1.0 / gamma) * scale
    return (restored * phasor).astype(np.complex64), nbytes + HEADER_BYTES


def synthetic_cells(ncells=24, seed=7, npoints=1500, nsplats=330):
    """Match xfine occupancy with randomly oriented, anisotropic Gaussians.

    One principal scale stays within 10% of S_LO while the other two range
    up to the xfine cap. This avoids an isotropic easy case; it does not
    model the spatial correlations or outlier distribution of a capture.
    """
    from holo.capture import BANDS, S_LO
    from holo.spectral import SplatScene, random_rotations_3d

    _, cap, cell = BANDS[0]
    rng = np.random.default_rng(seed)
    cells = []
    for _ in range(ncells):
        scales = rng.uniform(S_LO, cap, (nsplats, 3))
        scales[:, 0] = rng.uniform(S_LO, 1.1 * S_LO, nsplats)
        rot = random_rotations_3d(nsplats, rng)
        cov = (rot * scales[:, None, :] ** 2) @ rot.transpose(0, 2, 1)
        local = SplatScene(
            mu=rng.uniform(-cell / 2, cell / 2, (nsplats, 3)).astype(np.float32),
            cov=cov.astype(np.float32),
            amp=rng.uniform(0.1, 1, (nsplats, 1)).astype(np.float32))
        pts = rng.uniform(-cell / 2, cell / 2, (npoints, 3)).astype(np.float32)
        cells.append((local, pts))
    return cells, cell / 2


def candidates(bundle, configs):
    """Batch the four scale/shrink variants so readout reuses query phases."""
    from holo.denoise import percentile_threshold, shrink

    shrunk = shrink(bundle, percentile_threshold(bundle, 25))
    rows, vectors = [], [bundle]
    for budget, mbits, pbits in configs:
        for rule in ("max", "p99.9"):
            for denoise, source in ((False, bundle), (True, shrunk)):
                q, nbytes = quant_polar(source, mbits, pbits, scale_rule=rule)
                rows.append({"payload_budget": budget, "d": len(bundle),
                             "mbits": mbits, "pbits": pbits,
                             "scale_rule": rule, "shrink": denoise,
                             "analytic_bytes": nbytes})
                vectors.append(q)
    return rows, np.stack(vectors)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--capture")
    ap.add_argument("--output", default="/tmp/quantphase-d2.jsonl")
    args = ap.parse_args(argv)
    from bench import precision_battery as pb

    if args.capture:
        pb.set_capture(args.capture)
    output = os.path.abspath(args.output)
    start = time.monotonic()
    result = pb.d2_lowbit(synthetic=args.synthetic)
    exp = next(e for e in pb.REGISTRY if e.id == "D2")
    pb.record(exp, result, time.monotonic() - start, pb.budget.peak_rss_gb(),
              path=output)
    pb.report(output)


if __name__ == "__main__":
    main()
