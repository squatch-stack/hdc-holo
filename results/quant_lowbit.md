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

