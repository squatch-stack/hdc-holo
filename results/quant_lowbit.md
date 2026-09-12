# Quantised-phase storage below the nibble floor (D2, synthetic)

Ungated results note. Tool: `bench/quant_lowbit.py`; experiment `D2` in
`bench/precision_battery.py`. Run 2026-09-12 on NumPy, 24 synthetic xfine
cells (330 splats, 1500 probes each, fixture seed 7, frequency seed 42,
gamma 0.5), 84 settings in 282 s at 1.61 GB peak RSS. Every number below
is provisional until the same command is rerun on real cells
(`--capture <scan.spz>`): the synthetic fixture has uniform occupancy and
no coherent-crosstalk floor, and it does not reproduce D1's measured
dimension-over-precision trend (see the last section).

## Question

D1 (`bench/precision_battery.py`) found that at a fixed byte budget,
spending bits on dimension beats spending them on precision — d=4096@16
bit 0.2223 → d=8192@8 bit 0.1781 → d=16384@4 bit 0.1283 on real cells —
and stopped at 4 bits because the packer nibble-packs everything below.
qFHRR (arXiv:2604.25939) stores 3–4-bit phase indices. Does the trend
continue below the nibble? The measurement needs no packer: distortion is
set by the quantiser, bytes are `ceil(d·bits/8)` per stream plus a
17-byte header, and the rows here are computed that way.

## The ladder

Median relative reconstruction error against the exact field over 24
cells. `B` is the payload budget in bytes (header extra); `m/p` are
magnitude/phase bits; columns are scale rule × shrink-then-quantise.
`0/p` is phase-only with HP's exact codes and a least-squares scalar gain
in the header.

```
     B       d   m/p   max/off    max/on  p999/off   p999/on
  8192    4096   8/8    0.1008    0.1335    0.1008    0.1338
  8192    8192   4/4    0.1201    0.1559    0.1190    0.1556
  8192   16384   2/2    0.3090    0.2659    0.3057    0.2585
  8192   32768   1/1    0.7332    0.7111    0.6995    0.6784
  8192   16384   1/3    0.6904    0.6736    0.6540    0.6336
  8192   32768   0/2     68.94     61.22     68.94     61.22
  8192    8192   2/6    0.2981    0.2447    0.2931    0.2287
 16384    8192   8/8    0.1024    0.1548    0.1046    0.1567
 16384   16384   4/4    0.0970    0.1260    0.0961    0.1258
 16384   32768   2/2    0.3064    0.2647    0.3006    0.2559
 16384   65536   1/1    0.6442    0.6264    0.6079    0.5874
 16384   32768   1/3    0.7332    0.7111    0.6991    0.6782
 16384   65536   0/2     200.8     177.9     200.8     177.9
 16384   16384   2/6    0.3236    0.2565    0.3209    0.2504
 32768   16384   8/8    0.0799    0.1253    0.0782    0.1247
 32768   32768   4/4    0.1023    0.1331    0.0992    0.1315
 32768   65536   2/2    0.2864    0.2446    0.2793    0.2299
 32768  131072   1/1    0.6544    0.6361    0.6134    0.5970
 32768   65536   1/3    0.6438    0.6264    0.6079    0.5874
 32768  131072   0/2     42.94     37.65     42.94     37.65
 32768   32768   2/6    0.3195    0.2575    0.3145    0.2456
```

Drift against the unquantised decode at the same d (shrink included):

```
     B       d   m/p   max/off    max/on  p999/off   p999/on
  8192    4096   8/8    0.0061    0.1303    0.0064    0.1309
  8192    8192   4/4    0.0621    0.1342    0.0598    0.1343
  8192   16384   2/2    0.2739    0.2351    0.2700    0.2275
  8192   32768   1/1    0.6253    0.6080    0.5952    0.5777
  8192   16384   1/3    0.6237    0.6071    0.5913    0.5721
  8192   32768   0/2     63.99     56.88     63.99     56.88
  8192    8192   2/6    0.3208    0.2569    0.3161    0.2475
 16384    8192   8/8    0.0043    0.1329    0.0053    0.1335
 16384   16384   4/4    0.0484    0.1150    0.0459    0.1159
 16384   32768   2/2    0.2614    0.2242    0.2552    0.2167
 16384   65536   1/1    0.6224    0.6058    0.5890    0.5691
 16384   32768   1/3    0.6253    0.6080    0.5951    0.5775
 16384   65536   0/2     197.0     174.5     197.0     174.5
 16384   16384   2/6    0.2917    0.2272    0.2866    0.2208
 32768   16384   8/8    0.0031    0.1144    0.0039    0.1149
 32768   32768   4/4    0.0435    0.1103    0.0423    0.1115
 32768   65536   2/2    0.2753    0.2355    0.2713    0.2227
 32768  131072   1/1    0.6499    0.6317    0.6089    0.5912
 32768   65536   1/3    0.6220    0.6057    0.5890    0.5691
 32768  131072   0/2     42.81     37.55     42.81     37.55
 32768   32768   2/6    0.2752    0.2180    0.2717    0.2086
```

Unquantised (raw) median error by dimension, for reference: d=4096
0.1008, 8192 0.1024, 16384 0.0798, 32768 0.0891, 65536 0.0328, 131072
0.0184.

## Reading

- **The knee is 4+4 bits.** At the central budget (16,384 B) the best
  point is d=16384 at 4/4 (0.0961, p99.9 scale, no shrink), ahead of
  d=8192 at 8/8 (0.1024) by 6%. Halving bits again to d=32768 at 2/2
  costs 0.2559 even with clipping and shrink — 2.7× worse. The answer to
  "does (32768, 2) beat (16384, 4) at equal bytes" is no.
- **Nothing at one bit survives.** Every 1-bit-magnitude row (1/1 and
  1/3) sits at 0.59–0.73; the 1/3 allocation is no better than 1/1, so
  the bits lost are the magnitude bits, not the phase bits. Two-bit
  magnitudes with six-bit phases (2/6) land with 2/2 at 0.25–0.32:
  below four bits the magnitude stream is the whole problem.
- **Phase-only is not a rate point for a spectral bundle.** With HP's
  exact codes and the best scalar gain, 0/2 decodes at 38–200× the field.
  A spectral bundle's magnitudes are the Gaussian envelope of the cell;
  unit modulus whitens the spectrum and the decoded field is dominated by
  frequencies that carried nothing. This is the projection floor
  `holo.phase.codec_curve` already reports for fields, at any bit depth,
  and a gain cannot repair a spectral shape.
- **Scale rule.** p99.9 clipping over max helps where levels are scarce
  (2/2: 0.3064 → 0.3006; 1/1: 0.6442 → 0.6079) and is neutral at 8 bits.
  `tests/test_quant_lowbit.py` pins the single-outlier case at two bits,
  where a max scale rounds every other component to zero.
- **Shrink.** `shrink(v, percentile_threshold(v, 25))` is itself a ~0.11
  drift. It helps only where quantisation noise exceeds that (2/2:
  0.3006 → 0.2559; 1/1: 0.6079 → 0.5874) and hurts at four bits and above
  (4/4: 0.0961 → 0.1258; 8/8: 0.1024 → 0.1548).
- **Budget dependence.** At the half budget (8,192 B) 8/8 at d=4096
  wins (0.1008 vs 0.1190 for 4/4); at the double budget (32,768 B) 8/8 at
  d=16384 wins (0.0782 vs 0.0992). Only the central budget favours
  4/4, and only by 6%.

## What this does not establish

D1 measured a 28% gain from 8 to 4 bits at equal bytes on real cells.
This fixture shows 6% at one budget and the reverse at the other two, so
it does not reproduce D1's trend; the raw errors are not even monotone in
d (0.1008 → 0.1024 → 0.0798 → 0.0891). Real cells differ in occupancy,
in outlier magnitude distribution, and in the coherent-crosstalk floor
(`holo/capture.py`) below which extra dimensions stop paying. Whether the
knee is at 4 bits or whether 2-bit magnitudes become competitive where
crosstalk, not quantisation, sets the floor, is what the capture rerun
decides.

## Recommendation

Do not write an HQ packer on this evidence. No symmetric or asymmetric
sub-nibble setting beat the existing 4-bit rate point, one-bit magnitudes
never came within 5× of it, and the tool that would justify a bit-streamed
`HQ` magic (`<2sBBBIff` header) is a measured win below four bits that the
synthetic fixture does not show. Keep the analytic quantiser, rerun the
same 84 settings on real cells, and decide then. If the capture rerun
shows (32768, 2/2) within a few percent of (16384, 4/4), the packer costs
one function and a golden-bytes test; if it does not, D1's
"monotonically" gets a registered upper limit at four bits.
