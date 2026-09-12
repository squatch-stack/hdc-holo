# Place recognition on the twelve gallery captures

Ungated results note. Tool: `bench/place_recognition.py` (PRs #101, #104
and the whitening lane). All runs on the RTX 5090 through
`bench/cuda_backend`, d=8192, σ_rec=0.025 of the box, 16 yaws, 48³
translation grid with local refinement (`--limit 0.25`), phase-surrogate
null. Scenes are the gallery exports measured in `results/gpu_sweep.md`;
labels below abbreviate `research-library`→`rlib`, `brookline-station`→
`brook`, `springhouse-outside`→`spring`, `wilsons-creek`→`wc`,
`redrock`→`rr`. Known partners: brook↔spring (the one true re-capture),
rr-cairn⊂rr and wc-gun⊂wc (exact crops of their parents).

## Raw cross-power correlation, per-capture frames (`--whiten 0`)

Phase correlation (row = reference, column = query; max over 16 yaws
and the translation search, so the matrix is not symmetric):

```
        brook-  brook cannon    oak rr-cai     rr rlib-c   rlib saguar spring wc-gun     wc
brook-2  1.000  0.304  0.305  0.297  0.305  0.379  0.377  0.413  0.563  0.313  0.452  0.353
brook    0.320  1.000  0.204  0.215  0.275  0.266  0.295  0.297  0.403  0.286  0.276  0.251
cannon   0.289  0.225  1.000  0.284  0.272  0.232  0.241  0.242  0.242  0.219  0.174  0.212
oak      0.314  0.247  0.346  1.000  0.268  0.751  0.223  0.700  0.403  0.257  0.189  0.746
rr-cairn 0.332  0.281  0.284  0.254  1.000  0.252  0.371  0.255  0.358  0.267  0.361  0.213
rr       0.397  0.292  0.287  0.756  0.267  1.000  0.257  0.933  0.500  0.272  0.238  0.956
rlib-can 0.330  0.311  0.245  0.220  0.405  0.214  1.000  0.242  0.354  0.230  0.569  0.196
rlib     0.420  0.338  0.292  0.692  0.283  0.933  0.278  1.000  0.582  0.294  0.262  0.903
saguaro  0.566  0.400  0.274  0.411  0.378  0.507  0.367  0.578  1.000  0.314  0.422  0.445
spring   0.313  0.286  0.230  0.212  0.274  0.242  0.250  0.252  0.320  1.000  0.185  0.222
wc-gun   0.430  0.301  0.170  0.169  0.347  0.230  0.593  0.264  0.432  0.196  1.000  0.230
wc       0.363  0.284  0.269  0.750  0.237  0.956  0.237  0.905  0.452  0.256  0.225  1.000
```

Radial-power control (exactly pose-invariant, arrangement-free):

```
        brook-  brook cannon    oak rr-cai     rr rlib-c   rlib saguar spring wc-gun     wc
brook-2  1.000  0.942  0.841  0.792  0.916  0.797  0.871  0.773  0.841  0.957  0.924  0.767
brook    0.942  1.000  0.906  0.909  0.924  0.903  0.883  0.882  0.885  0.937  0.865  0.856
cannon   0.841  0.906  1.000  0.882  0.914  0.846  0.703  0.802  0.691  0.865  0.727  0.799
oak      0.792  0.909  0.882  1.000  0.784  0.951  0.734  0.931  0.796  0.789  0.695  0.897
rr-cairn 0.916  0.924  0.914  0.784  1.000  0.785  0.838  0.718  0.770  0.920  0.850  0.725
rr       0.797  0.903  0.846  0.951  0.785  1.000  0.797  0.979  0.842  0.763  0.727  0.964
rlib-can 0.871  0.883  0.703  0.734  0.838  0.797  1.000  0.778  0.954  0.801  0.945  0.765
rlib     0.773  0.882  0.802  0.931  0.718  0.979  0.778  1.000  0.846  0.730  0.721  0.970
saguaro  0.841  0.885  0.691  0.796  0.770  0.842  0.954  0.846  1.000  0.746  0.898  0.816
spring   0.957  0.937  0.865  0.789  0.920  0.763  0.801  0.730  0.746  1.000  0.842  0.731
wc-gun   0.924  0.865  0.727  0.695  0.850  0.727  0.945  0.721  0.898  0.842  1.000  0.728
wc       0.767  0.856  0.799  0.897  0.725  0.964  0.765  0.970  0.816  0.731  0.728  1.000
```

Phase-surrogate null over 24 draws: mean 0.147, σ 0.056,
p95 0.245, max 0.265.

| query | rank-1 | best positive | best negative | separation |
|---|---|---|---|---|
| 1 brook | no | 0.286 | 0.400 | -2.0σ |
| 4 rr-cairn | no | 0.267 | 0.405 | -2.5σ |
| 5 rr | no | 0.252 | 0.956 | -12.6σ |
| 9 spring | no | 0.286 | 0.314 | -0.5σ |
| 10 wc-gun | no | 0.225 | 0.569 | -6.1σ |
| 11 wc | no | 0.230 | 0.956 | -12.9σ |

**0 of 6.** The failure is not noise: the four wide outdoor parents
(oak, rr, rlib, wc — each a 480,000-splat subsampled export in a 30–40
unit mass-centred cube) score 0.70–0.96 against *each other*, at offsets
within a grid step of zero, twelve σ above the null. The true pairs sit
at 0.23–0.29. A capture blurred at σ_rec = box/40 is a bump in the middle
of its cube, and the raw cross-power spectrum is dominated by that
bump's lowest frequencies: the peak measures the envelope, not the
arrangement. The radial control says the same thing from the other
side — it is 0.70–0.98 everywhere, because every blurred capture has
nearly the same power profile.

## Shared physical frame (`--frame`)

| pair | score | offset (box units) | null mean ± σ (max) | radial |
|---|---|---|---|---|
| wilsons-creek ↔ gun, parent's cube | 0.993 | [-0.0, -0.001, 0.001] | 0.070 ± 0.007 (max 0.083) | 0.972 |
| wilsons-creek ↔ gun, gun's cube | 0.258 | [0.005, 0.25, -0.021] | 0.277 ± 0.038 (max 0.345) | 0.913 |
| redrock ↔ cairn, parent's cube | 0.990 | [0.001, -0.001, 0.001] | 0.081 ± 0.007 (max 0.095) | 0.975 |
| redrock ↔ cairn, cairn's cube | 0.904 | [0.004, 0.012, 0.001] | 0.178 ± 0.023 (max 0.226) | 0.983 |
| brookline-station ↔ springhouse-outside, station's cube | 0.184 | [0.172, -0.25, 0.241] | 0.113 ± 0.019 (max 0.136) | 0.909 |

- Crop in the parent's cube: 0.99 at zero offset for both crops. This is
  the linearity prediction (`B_parent = B_crop + B_rest`) and it is close
  to trivial — after the 480k subsample the gun and cairn regions hold
  most of their parents' mass, so `B_rest` is small.
- Parent restricted to the crop's cube: 0.904 for the cairn, 0.258 for
  the gun against a null of 0.277 ± 0.038. The gun's cube is 4 of the
  parent's 40 units on a side, so the subsampled parent has roughly a
  thousandth of its splats there against the crop's 428,000 — the
  restricted parent is too sparse to be the same field. The cairn's
  parent is denser in that region and passes.
- The one true re-capture (station interior ↔ springhouse yard, encoded
  in the station's cube) scores 0.184 against a null of 0.113 ± 0.019 —
  3.7 σ, with the recovered offset pinned to the search limit on one
  axis. Not a match by this tool's own bar.

## Whitened correlation (`--whiten 1`, PHAT)

Each component of conj(A)·B divided by its magnitude, so only phase votes
and the score is the fraction of live components aligned at the peak.
Phase-surrogate null over 24 draws: mean 0.0387, σ 0.0020,
max 0.0421 — whitening makes the null tight, as it should.

```
        brook-  brook cannon    oak rr-cai     rr rlib-c   rlib saguar spring wc-gun     wc
brook-2  1.000  0.087  0.060  0.131  0.090  0.162  0.068  0.162  0.152  0.075  0.093  0.169
brook    0.076  1.000  0.071  0.099  0.072  0.109  0.064  0.120  0.091  0.077  0.064  0.117
cannon   0.069  0.065  1.000  0.114  0.062  0.098  0.058  0.092  0.088  0.099  0.067  0.108
oak      0.139  0.100  0.112  1.000  0.085  0.505  0.076  0.327  0.383  0.101  0.098  0.596
rr-cairn 0.091  0.086  0.068  0.075  1.000  0.094  0.079  0.108  0.104  0.092  0.086  0.084
rr       0.178  0.127  0.104  0.512  0.119  1.000  0.098  0.547  0.488  0.082  0.140  0.823
rlib-can 0.069  0.071  0.060  0.077  0.081  0.087  1.000  0.113  0.089  0.066  0.085  0.089
rlib     0.172  0.120  0.091  0.323  0.112  0.549  0.113  1.000  0.258  0.089  0.122  0.473
saguaro  0.158  0.086  0.101  0.380  0.095  0.479  0.089  0.264  1.000  0.113  0.149  0.501
spring   0.069  0.082  0.096  0.087  0.086  0.084  0.067  0.083  0.110  1.000  0.065  0.079
wc-gun   0.090  0.075  0.062  0.095  0.086  0.131  0.091  0.122  0.138  0.076  1.000  0.122
wc       0.179  0.135  0.110  0.596  0.108  0.823  0.098  0.487  0.504  0.087  0.126  1.000
```

| query | rank-1 | best positive | best negative | separation |
|---|---|---|---|---|
| 1 brook | no | 0.082 | 0.135 | -26.5σ |
| 4 rr-cairn | yes | 0.119 | 0.112 | +3.5σ |
| 5 rr | no | 0.094 | 0.823 | -368.0σ |
| 9 spring | no | 0.077 | 0.113 | -18.2σ |
| 10 wc-gun | no | 0.126 | 0.149 | -11.7σ |
| 11 wc | no | 0.122 | 0.823 | -353.8σ |

**1 of 6**, and the one hit (rr-cairn, +3.5σ) is inside the 12-scene
noise of the raw run. Whitening did not remove the wide-capture block:
it *sharpened* it. wc↔rr is 0.823 — 368σ above a null whose maximum is
0.042 — with oak, rr, rlib and wc at 0.33–0.82 among themselves while
the true pairs sit at 0.08–0.13. Half whitening (`--whiten 0.5`) sits
between the two: 0/6, null 0.070 ± 0.013, wc↔rr 0.951.

Shared-frame pairs, whitened:

| pair | score | offset (box units) | null mean ± σ (max) |
|---|---|---|---|
| wilsons-creek ↔ gun, gun's cube | 0.291 | [0.001, 0.003, 0.0] | 0.040 ± 0.002 (max 0.042) |
| redrock ↔ cairn, cairn's cube | 0.745 | [-0.0, 0.004, -0.0] | 0.039 ± 0.002 (max 0.041) |
| station ↔ springhouse, station's cube | 0.082 | [-0.227, -0.148, 0.007] | 0.040 ± 0.003 (max 0.045) |
| station ↔ springhouse, station's cube, 64³ grid, limit 0.5 | 0.119 | [0.381, -0.474, -0.157] | 0.043 ± 0.001 (max 0.045) |

The cairn crop still localises in its own cube (0.745, offset zero);
the gun crop does not (0.291 — 125σ above the whitened null, but a
tenth of what the cairn gets, for the sparsity reason given above);
the springhouse re-capture is 0.08–0.12 with the offset wandering to
the search limit whichever limit is set. Not found.

## What the wide-capture block is not

wc↔rr at 0.823 whitened means 82% of 8,192 frequency components have
the same phase at one translation. Two different places cannot do that
by arrangement, so something the fingerprints share is being scored.
Six diagnostics on the box (`results/place_diagnostics/*_test.py`, same
codebook, same search):

| hypothesis | test | result |
|---|---|---|
| the crop cube itself | each wide capture vs a uniform random fill of its own cube carrying its own covariances and masses | raw 0.11–0.28, whitened 0.04–0.12: a capture is not its box. Uniform fills *do* score 0.91–0.96 raw against each other (the box explains the raw block's floor) but 0.05–0.10 whitened. |
| a shared ground plane | drop frequencies with horizontal radius below 0.1–0.5 of the maximum | wc↔rr whitened 0.83 → 0.82 → 0.77 → 0.60 at 8%, of components kept; the block survives the mask. |
| a centred, symmetric envelope (real spectrum) | fraction of live components with \|phase\| < π/4 | 0.25 for every capture and every uniform fill: phases are uniform, not real. |
| low frequencies only | drop \|w\| below 0.2 / 0.4 / 0.6 of the maximum | wc↔rr whitened 0.78 / 0.58 / 0.39 with 84% / 33% / 5% of components kept; the block lives at high frequency too. |
| a few enormous splats | remove the top 1% of splats by α·√det Σ (11–16% of mass) | full-vs-rest 0.98–1.00 in every capture; the top 1% of wc against the top 1% of rr scores 0.06; the remaining 99% score 0.82. It is the bulk. |
| shared splats | exact position overlap between exports | 0.0000 for every pair; distinct md5, sizes, bounding boxes. |

The mechanism is unidentified. What is known: it is carried by the bulk
of the splats, at all frequencies, in the phase and not the envelope,
between the four captures that are 388k–1.16M splats in a 30–40 unit
cube, and not between those and the object scans (cannon, the station,
the crops) or uniform fills. The next question for this tool is that
one, and it is a question about what a d=8192 spectral bundle of a wide
outdoor capture actually encodes at σ_rec = box/40 — not about the
scorer.

## Conclusion

- **Negative for whole-scene place recognition on this corpus**, raw
  and whitened: 0/6 and 1/6 rank-1 against a bar of 6/6 at ≥ 3σ. The
  one true re-capture is not found under any setting tried.
- **Positive for sub-map localisation in a shared frame**: a crop
  encoded in its parent's cube is found at its true offset (0.99, both
  crops), and the parent restricted to the crop's cube is found when
  the restricted region is dense enough (cairn 0.90 raw / 0.75
  whitened; gun 0.26 / 0.29, too sparse after the 480k subsample).
- The phase surrogate is the right null (tight, and the scorer's own
  control); position scrambling is not, on captures.
- Whitening is the right scorer for arrangement and the wrong one for
  the radial question; both stay in the tool.
- The wide-capture block is the open finding. Until it is explained, no
  score between two wide captures from this tool should be read as a
  match, and the descriptor should not be promoted.

Figures: `out/place/similarity-corpus.png` (raw) and
`out/place/similarity-corpus-whitened.png`.
