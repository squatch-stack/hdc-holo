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

## Real cells (2026-09-12, two captures)

Same command with `--capture`, run on the 5090 host in NumPy (D2 is CPU
work): `cell_scenes("xfine", 24)` on `cannon.spz` (167 s, 2.82 GB peak
RSS) and on `wilsons-creek.spz` (368 s, 2.78 GB). The captures are the
gallery exports measured in `results/gpu_sweep.md`.

Cannon — median relative reconstruction error:

```
     B       d   m/p   max/off    max/on  p999/off   p999/on
  8192    4096   8/8    0.2026    0.2778    0.2026    0.2778
  8192    8192   4/4    0.1727    0.2744    0.1733    0.2754
  8192    8192   2/6    0.3500    0.2601    0.3430    0.2531
  8192   16384   2/2    0.2890    0.2888    0.2852    0.2886
  8192   16384   1/3    0.8078    0.7206    0.8027    0.7159
  8192   32768   1/1    1.1243    1.0291    1.1106    1.0236
  8192   32768   0/2      80.6      65.7      80.6      65.7
 16384    8192   8/8    0.1556    0.2732    0.1556    0.2733
 16384   16384   4/4    0.1175    0.2587    0.1173    0.2589
 16384   16384   2/6    0.3221    0.2466    0.3132    0.2407
 16384   32768   2/2    0.2778    0.2819    0.2688    0.2796
 16384   32768   1/3    0.8235    0.7268    0.8108    0.7125
 16384   65536   1/1    1.1105    1.0214    1.0922    1.0127
 16384   65536   0/2     240.5     195.0     240.5     195.0
 32768   16384   8/8    0.1055    0.2580    0.1053    0.2580
 32768   32768   4/4    0.0940    0.2525    0.0939    0.2529
 32768   32768   2/6    0.3014    0.2406    0.2902    0.2369
 32768   65536   2/2    0.2620    0.2794    0.2530    0.2788
 32768   65536   1/3    0.8132    0.7163    0.8042    0.6999
 32768  131072   1/1    1.1009    1.0167    1.0896    1.0066
 32768  131072   0/2      50.4      41.1      50.4      41.1
```

Unquantised by dimension: 4096 0.2025, 8192 0.1554, 16384 0.1053, 32768 0.0865, 65536 0.0537, 131072 0.0354.

Wilson's Creek — median relative reconstruction error:

```
     B       d   m/p   max/off    max/on  p999/off   p999/on
  8192    4096   8/8    0.2464    0.3026    0.2460    0.3026
  8192    8192   4/4    0.2067    0.3006    0.2056    0.2996
  8192    8192   2/6    0.3637    0.2810    0.3575    0.2819
  8192   16384   2/2    0.2757    0.2999    0.2700    0.3017
  8192   16384   1/3    0.7608    0.6552    0.7504    0.6465
  8192   32768   1/1    1.0675    0.9730    1.0616    0.9677
  8192   32768   0/2      83.4      64.6      83.4      64.6
 16384    8192   8/8    0.1882    0.2973    0.1882    0.2974
 16384   16384   4/4    0.1329    0.2756    0.1329    0.2760
 16384   16384   2/6    0.3086    0.2552    0.3027    0.2559
 16384   32768   2/2    0.2477    0.3023    0.2432    0.3025
 16384   32768   1/3    0.7305    0.6399    0.7225    0.6275
 16384   65536   1/1    1.0713    0.9700    1.0628    0.9636
 16384   65536   0/2     246.1     191.2     246.1     191.2
 32768   16384   8/8    0.1235    0.2748    0.1235    0.2748
 32768   32768   4/4    0.1019    0.2790    0.1017    0.2792
 32768   32768   2/6    0.2725    0.2556    0.2686    0.2558
 32768   65536   2/2    0.2283    0.2973    0.2243    0.2976
 32768   65536   1/3    0.7433    0.6411    0.7385    0.6294
 32768  131072   1/1    1.0649    0.9723    1.0525    0.9608
 32768  131072   0/2      52.6      40.1      52.6      40.1
```

Unquantised by dimension: 4096 0.2464, 8192 0.1879, 16384 0.1234, 32768 0.0924, 65536 0.0641, 131072 0.0440.

### What the real cells change

- **D1's trend is back.** On captures the unquantised error falls
  monotonically with d (cannon 0.2025 → 0.0354 from 4096 to 131072), and
  at every budget 4/4 at 2d beats 8/8 at d: 0.1556 → 0.1175 on cannon
  and 0.1882 → 0.1329 on Wilson's Creek at 16 KB (−24% and −29%; D1
  measured −28%). The synthetic fixture's 6%-and-reversed result was the
  fixture, as the note above suspected.
- **The knee is still four bits, and it is sharper.** 2/2 at 4d costs
  0.2688 (cannon) and 0.2432 (Wilson's Creek) at 16 KB against 0.1175 and
  0.1329 for 4/4 — 2.3× and 1.8× worse — with no rescue from clipping or
  shrink. One-bit magnitudes exceed 1.0 (the reconstruction is worse than
  reporting zero). The asymmetric 2/6 does not help either: the magnitude
  stream is the whole problem below four bits, on real cells as on
  synthetic ones.
- **Shrink hurts everywhere on real cells** (4/4: 0.1175 → 0.2587;
  8/8: 0.1556 → 0.2732; even 2/2 gains nothing). `percentile_threshold(v,
  25)` is a ~0.25 drift on capture bundles, where the synthetic fixture
  put it at 0.11. The synthetic 2-bit gain from shrink does not transfer.
- **p99.9 versus max is a wash on real cells** (fourth digit). Capture
  bundles do not carry the single dominant magnitude that the synthetic
  outlier test guards against; the guard stays because it is cheap and
  the case is real, but it buys nothing here.
- **Phase-only: 40–246× the field**, as on the fixture. Not a rate point.

### Decision

The knee is at four bits per field on both captures, by a factor of
about two, so no `HQ` packer is written and D1's "monotonically" gets its
upper limit: dimension beats precision down to four bits and not below.
The remaining sub-nibble question is a different one — whether 2-bit
magnitudes with a *per-block* scale (not a per-vector one) close the gap
— and that is a new experiment, not this ladder.

## D3 (synthetic)

Squatch Stack's block-scale experiment adds independent magnitude scales
and keeps D2's seeded 24-cell fixture (330 splats and 1500 probes per cell),
frequency seed 42, gamma 0.5, exact-field error, and drift against the raw
decode at the same dimension. All variants at a dimension share the
codebook, bundle, and batched readout. The two D2 max-scale references
(4/4 and 8/8, no shrink) are recomputed inside this run. There are 42
settings: 36 block-scale settings and six vector references.

The motivation comes from the [OCP MX v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
[BATQuant, arXiv:2603.16590](https://arxiv.org/abs/2603.16590), and
[DuQuant++, arXiv:2604.17789](https://arxiv.org/abs/2604.17789), read online
on 2026-09-12. BATQuant describes block-wise affine shaping and clipping;
DuQuant++ aligns outlier-aware rotations with microscaling groups. This
experiment applies neither transformation. Our unsigned, companded
magnitude codes are our definitions, not MXFP4 E2M1 or a reproduction of
these papers. E8M0 supplies a covering power-of-two scale with exponent
-127 through 127 (zero blocks use -127; larger required scales raise).
The u8 alternative quantises each block max to a linear 8-bit fraction of
the vector max held as float32. A rounded-down scale saturates magnitude
codes at their maximum; a scale rounded to zero restores a zero block.
Only 8-bit scales are defined. Rates are analytic; no packer is emitted.
The common hypothetical header remains 17 bytes for comparability.

The payload includes both bit streams and all scale bytes; the header
is extra. The four requested block-32 dimensions are retained. The
block-16 and block-64 suggestions need arithmetic corrections: with 2/2
bits, a full block costs `block/2 + 1` bytes. Consequently, at 16,384
bytes, `d = block * floor(16384 / (block/2 + 1))` is 29,120 or 31,744.
No whole-block dimension lands on exactly 16,384 bytes because neither
9 nor 33 divides it. The suggested 29,184 exceeds the budget (16,416
bytes), while 31,488 leaves 148 bytes unused. The corrected points are
the largest whole-block dimensions that fit. The other budgets multiply
d by 0.5 or 2, retaining whole blocks; none was dropped.

```
  B       d    m/p  block  payload  unused
16384   15872  4/4     32    16368      16
16384   30720  2/2     32    16320      64
16384   30720  1/3     32    16320      64
16384   58240  1/1     32    16380       4
16384   29120  2/2     16    16380       4
16384   31744  2/2     64    16368      16
16384   16384  4/4      0    16384       0
16384    8192  8/8      0    16384       0
```

Reproduce the full synthetic measurement (JSONL appends a record per run):

```sh
HDC_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m bench.quant_lowbit \
  --experiment D3 --synthetic --output /tmp/blockscale-d3.jsonl
```

Capture confirmation, with the path supplied by the maintainer:

```sh
HDC_BACKEND=numpy OPENBLAS_NUM_THREADS=1 .venv/bin/python -m bench.quant_lowbit \
  --experiment D3 --capture "$SCENES/cannon.spz" \
  --output /tmp/blockscale-d3-capture.jsonl
```

Run on 2026-09-12 with NumPy 1.26.4: **483.2 s, 1.71 GB peak RSS**.
All three budgets completed in about eight minutes on CPU.

Median relative reconstruction error against the exact field over 24 cells.
`block=0` and `max/ref` identify the same-run D2 vector references; a dash
means the scale code does not apply. Headers are additional to `B`.

Error:

```
     B       d   m/p  block    e8m0      u8  max/ref
  8192    4096   8/8      0       -       -  0.1008
  8192    7936   4/4     32  0.0981  0.0965       -
  8192    8192   4/4      0       -       -  0.1201
  8192   14560   2/2     16  0.1632  0.1530       -
  8192   15360   1/3     32  0.5947  0.3263       -
  8192   15360   2/2     32  0.1983  0.1833       -
  8192   15872   2/2     64  0.1913  0.1809       -
  8192   29120   1/1     32  0.6387  0.3774       -
 16384    8192   8/8      0       -       -  0.1024
 16384   15872   4/4     32  0.0685  0.0667       -
 16384   16384   4/4      0       -       -  0.0970
 16384   29120   2/2     16  0.1491  0.1468       -
 16384   30720   1/3     32  0.6368  0.3298       -
 16384   30720   2/2     32  0.1832  0.1746       -
 16384   31744   2/2     64  0.2163  0.2003       -
 16384   58240   1/1     32  0.5743  0.3457       -
 32768   16384   8/8      0       -       -  0.0799
 32768   31744   4/4     32  0.0684  0.0651       -
 32768   32768   4/4      0       -       -  0.1023
 32768   58240   2/2     16  0.1256  0.1205       -
 32768   61440   1/3     32  0.5223  0.2764       -
 32768   61440   2/2     32  0.1464  0.1383       -
 32768   63488   2/2     64  0.1986  0.1861       -
 32768  116480   1/1     32  0.5807  0.3527       -
```

Drift against the unquantised decode at the same d:

```
     B       d   m/p  block    e8m0      u8  max/ref
  8192    4096   8/8      0       -       -  0.0061
  8192    7936   4/4     32  0.0463  0.0415       -
  8192    8192   4/4      0       -       -  0.0621
  8192   14560   2/2     16  0.1275  0.1215       -
  8192   15360   1/3     32  0.5053  0.2686       -
  8192   15360   2/2     32  0.1527  0.1443       -
  8192   15872   2/2     64  0.1954  0.1807       -
  8192   29120   1/1     32  0.5601  0.3275       -
 16384    8192   8/8      0       -       -  0.0043
 16384   15872   4/4     32  0.0377  0.0357       -
 16384   16384   4/4      0       -       -  0.0484
 16384   29120   2/2     16  0.1168  0.1135       -
 16384   30720   1/3     32  0.5435  0.2700       -
 16384   30720   2/2     32  0.1471  0.1378       -
 16384   31744   2/2     64  0.1924  0.1805       -
 16384   58240   1/1     32  0.5687  0.3433       -
 32768   16384   8/8      0       -       -  0.0031
 32768   31744   4/4     32  0.0327  0.0302       -
 32768   32768   4/4      0       -       -  0.0435
 32768   58240   2/2     16  0.1208  0.1163       -
 32768   61440   1/3     32  0.5302  0.2823       -
 32768   61440   2/2     32  0.1508  0.1393       -
 32768   63488   2/2     64  0.1891  0.1779       -
 32768  116480   1/1     32  0.5640  0.3420       -
```

Raw median relative error by dimension (shared across settings):

```
     d  raw error
  4096     0.1007
  7936     0.0855
  8192     0.1024
 14560     0.0774
 15360     0.0878
 15872     0.0562
 16384     0.0798
 29120     0.0720
 30720     0.0771
 31744     0.0570
 32768     0.0891
 58240     0.0287
 61440     0.0292
 63488     0.0352
116480     0.0248
```

### Reading D3

- **No: (30,720, 2/2, 32) does not beat (16,384, 4/4).** At the
  central budget the vector reference has error 0.09698; block u8 gives
  0.17461 (absolute increase 0.07763, 1.80x / 80.0% worse) and E8M0 gives
  0.18316 (increase 0.08619, 1.89x / 88.9% worse). The corresponding
  drifts are 0.0484 versus 0.1378 and 0.1471. Even the best two-bit
  configuration, block 16 with u8, loses to the vector 4/4 reference at
  all three budgets (errors 0.1530, 0.1468, 0.1205 versus 0.1201,
  0.0970, 0.1023).
- **u8 wins over E8M0** in both error and drift for every matched row.
  At 16 KB, 2/2 with block 32 improves from 0.1832 to 0.1746 (4.7%).
  The difference is much larger with one-bit magnitudes: 1/3 improves
  from 0.6368 to 0.3298 and 1/1 from 0.5743 to 0.3457. A power-of-two
  covering scale leaves more unused element range than a linear scale.
- **Smaller blocks help the two-bit stream.** At 16 KB, u8 error for
  blocks 16/32/64 is 0.1468/0.1746/0.2003; drift is
  0.1135/0.1378/0.1805. E8M0 has the same ordering. Drift increases with
  block size at every budget for both scale codes. Half-budget exact
  error has a small 32/64 reversal: these rows also change d and hence
  the random codebook and raw error, so it is not a controlled
  fixed-d estimate of the block-size effect.
- **The one-bit rows remain far from competitive.** Across budgets,
  1/3 errors are 0.2764–0.3298 with u8 and 0.5223–0.6368 with E8M0;
  1/1 errors are 0.3457–0.3774 and 0.5743–0.6387. Three phase bits
  improve on one at equal budgets with u8, but cannot repair the
  magnitude loss enough to beat four-bit storage.
- **Block-scaled 4/4 is the best measured row at each budget**, with u8
  errors 0.0965/0.0667/0.0651. Some advantage over the vector reference
  comes from different raw errors at the slightly different dimensions;
  the raw table above prevents attributing the entire improvement to
  quantisation. Its drift also beats the vector 4/4 drift at each
  budget. This is evidence for exploring scales at four bits, not for
  lowering the magnitude precision to two bits.

This is **provisional until the maintainer reruns D3 on captures** using
`--experiment D3 --capture ...`. The synthetic fixture does not reproduce
real occupancy correlations or the coherent-crosstalk floor. On this
fixture the answer below four bits is negative: the knee stays at four
bits and D1's limit stands. A negative capture rerun would confirm that
limit despite block scaling; these CPU results alone do not establish it
for captures. No packer or claims-surface change is justified here.

Validation: the quantizer file has 131 passing cases in 0.67 s (slowest
case 0.39 s). The full suite reports `3 failed, 419 passed, 9 skipped in
21.74s`; all three failures arise solely from `tests.count` (registry
321, derived 333). Ruff is clean and `holo-quality check` reports
`lint debt: 50 (baseline 50)`. `holo-facts check --strict` reports
`1 FAIL, 25 WARN`, with `tests.count` the sole FAIL. Updating that
registry and its cites is outside this lane and left to the maintainer.
The original D2 functions, existing tests, and prior results are intact.
