# Ridge-fitting holograms from data

*[← docs index](README.md) · learning & imaging*

**What.** The readout `f(p) = Re<e^{i W p}, conj(S)>/D` is LINEAR in S:
a hologram is the weight vector of a random-Fourier-features regression
model, and the codeword map is the feature map. So S can be FIT to raw
samples `{(p_i, y_i)}` of any target — no splats, no mixture model —
by exact ridge regression. This is the holographic analog of 3DGS's
training loop, except the problem is convex with a closed form.

In real coordinates the features are `[cos(Wp), sin(Wp), 1]` and the
solver picks whichever Gram matrix is smaller:

    primal:  (A^T A + lam n I) theta = A^T y        (features <= samples)
    dual:    theta = A^T (A A^T + lam n I)^{-1} y   (otherwise; same optimum)

Multi-channel targets (RGB) are extra right-hand sides on ONE
factorization. `FrequencyBands` concatenates W blocks at several kernel
scales for broadband targets like photographs.

**Results.** On a known 200-splat mixture, the fitted S beats the
forward-built bundle ~70x on held-out RMSE (0.005 vs 0.37): regression
finds the OPTIMAL vector in the same basis and corrects the coherent
crosstalk that bundling just accepts. Fitting from noisy samples
denoises (0.024 under 0.1-sigma noise) — bundling can't, it never sees
data. Grace Hopper's portrait fits recognizably at D=2048 (equal bytes
to the 128x128 image), improving monotonically with D.

**Where the win stops transferring** (0.2 finding; `fit_cells` in
`holo/capture.py`, `ridge_cell_fit(prior=)` in `holo/accel.py`). The
~70x result above is a sparse-scene, matched-basis outcome; per-cell
fitting of REAL captures behaves differently. (1) Under mixture
codebooks a spectral prior is NECESSARY: naive minimum-norm ridge is
12x worse than forward encoding — energy spreads into the codebook's
finest frequencies and samples are memorized as kernel-width bumps;
with the prior `exp(-1/2 (0.35 cap)^2 |w|^2)` the fit ties forward on
sparse scenes (0.037 vs 0.036). (2) At dense-capture scale the fit is
SAMPLING-LIMITED and loses (saguaro: fitted 0.72/0.53 vs forward
0.52/0.38, with dropout speckle where sampling starved): hundreds of
floor-scale splats per cell need target coverage at their own kernel
width — tens of thousands of samples per cell, beyond the dual solve.
**The analytic L2 projection, measured** (`examples/run_analytic_projection.py`,
issue #2). The sampling limit above is a property of *fitting to
samples*; projecting the exact mixture onto the codebook needs none.
`spectral_bundle` already computes the projection's right-hand side —
it IS the mixture's Fourier transform — so the whole difference from
forward encoding is replacing the diagonal importance weighting with a
Gram solve. Median relative error over the six most populated xfine
cells of the saguaro capture, d=2048:

| objective | interior splats only | whole cell |
|---|---|---|
| forward encoding | 0.1464 | 0.1879 |
| box + TSVD | **0.0273** (5.4x better) | 0.2052 (*worse*) |
| Gaussian window, s = h/2 | 0.0609 | **0.0739** (2.5x better) |

Read those two columns together, because the story is entirely in the
difference. **The box is the better objective and the window is the
usable one.** The box's right-hand side is the whole-space transform,
which is only correct for splats well inside the cell; the exact
box-restricted transform of an anisotropic Gaussian needs the complex
error function, does not separate for non-diagonal covariance, and is
not available in numpy. Real cells hold splats at their boundaries, so
that approximation costs more than the method wins. The window has no
such problem — a Gaussian window times a Gaussian splat is another
Gaussian, so its right-hand side is exact at any splat position, which
is why it barely degrades between the columns (0.0609 -> 0.0739) where
the box collapses (0.0273 -> 0.2052).

**One eigendecomposition serves a whole band**, because both Grams
factorise as `G_c = D G0 D^H` with `D` a unitary diagonal of
cell-centre phases — cell position enters only as a phase, so the
expensive step is per band, not per cell. Measured at production
d=8192: 63 s per band, amortised over thousands of cells.

**Truncation is mandatory, and this is where a smaller study misleads.**
At d=2048 the window is stable at full rank while the box detonates if
under-truncated (0.0273 at keep=1536, 8.6 at 2048), which reads like
the window needing no tuning. It is an artifact of the dimension. At
the production d=8192 the window Gram's condition number is 1.6e20 to
3.2e20 across the four bands — past what float64 can invert — and
solving at full rank returns garbage of order 1e5 rather than a
degraded answer. Keeping **25%** of the spectrum is safe on every band
and sits near the ~3,300 space-bandwidth degrees of freedom a cell of
this size actually supports.

**But 25% was chosen for safety, not accuracy, and it costs a lot.**
That figure came from a stability argument — full rank detonates, 25%
does not — and nothing in between was ever measured. Sweeping it on
saguaro, one eigendecomposition shared across every truncation:

| keep | top-down | side | \|solved\| / \|forward\| |
|---|---|---|---|
| 0.10 | +24.0% | +21.8% | 1.1 |
| 0.25 | +38.0% | +35.9% | 1.1 |
| 0.40 | +54.1% | +46.1% | 1.3 |
| **0.55** | **+59.3%** | **+58.8%** | 5.3 |
| 0.70 | **-1417.0%** | -838.2% | 97.4 |
| 0.85 | -5256.2% | -3188.9% | 148,911 |
| 1.00 | -71836511.5% | -73842301.8% | 5.2e11 |

On saguaro the shipped setting appears to give up more than twenty
points, and the optimum appears to sit immediately next to a precipice:
0.55 looks best and 0.70 is 37x worse than not projecting at all.

**That reading was wrong, and Red Rock shows why.** The `.spz` saguaro
capture is a lossy export; `redrock.ply` is the lossless one this repo
recommends. Run there, keep=0.55 again looks best by slice error
(+28.6% / +39.8% against 0.25's +17.8% / +27.8%) — while the divergence
guard reports every cell of the `fine` band at **1030x** the forward
bundle norm. Both are true, and the second one is what matters:

| keep | fine-band norm ratio | cells over the 20x limit |
|---|---|---|
| 0.25 | median 1.21, p99 1.77 | 0.0% |
| 0.40 | median 10, p99 23.9 | 4.9% |
| 0.55 | median 1030, p99 2430 | 100% |

The slice error cannot see it because of how the bands divide the scene:

| band | splats | cells |
|---|---:|---:|
| xfine | 542,122 | 1,615 |
| fine | 3,854 | 1,241 |
| mid | 641 | 130 |
| coarse | 21 | 14 |

**The `fine` band holds 0.7% of the splats.** Corrupting it by three
orders of magnitude barely moves a slice error dominated by xfine's
542k, so the metric rewards a setting that has destroyed a band. Those
bundles are still garbage for anything that reads them — a render, a
`what_is_at` query, a stored scene — and only this particular
measurement is blind to it.

So **keep=0.25 stays**: it is the last truncation with every cell clean
on the lossless capture. And the wider lesson is about the referee, not
the knob — slice error against the exact mixture is not a sufficient
acceptance test for a per-cell solve, because a band can be destroyed
without it moving. The norm ratio sees what the slice error misses, and
belongs alongside it whenever a truncation changes.

**So it is a gate, not a warning.** `run_projection_pipeline.py` refuses
a band whose solved bundles exceed the limit rather than printing and
carrying on, because printing is not a defence against a metric that
says the run was the best of four. The refusal names both ways out — a
smaller `keep`, or `--allow-divergence` for a sweep deliberately
exploring past the cliff, which is how the cliff was found in the first
place. The ratio is recorded either way, so a refused run still says how
far past it went.

**And the referee reports per band now, not just in aggregate.** The
norm ratio is a proxy: it reads the bundles, not the field they decode
to. So the pipeline also scores every band against *its own*
exact-mixture ground truth, which is what turns "1030x the forward
bundle norm" into a statement about the decoded field, in the quantity
every other result here is quoted in. It is close to free —
`decode_slice` and `exact_slice` both already took a band list, and the
bands partition the splats, so four single-band passes touch what one
full pass does. The aggregate stays the headline number; it is simply
not the acceptance test on its own.

Nor is the edge in the same place on every capture. On the LiDAR
`lidar-dense` capture the top-down slice peaks near 0.40 (+26.9%) and
has already turned over by 0.55 (+18.7%), while its side slice is still
climbing at 0.55 (+10.9%) — the two slices of one capture do not agree
on a best truncation, let alone two captures.

**`keep` is not a regularisation level, and that is most of why the
edge moves.** Truncation here keeps the largest `keep*d` eigenvalues — a
RANK fraction, which says nothing about how small the smallest survivor
is, and the operator divides by that survivor. Each band draws its
frequencies at its own scale cap, so its Gram decays at its own rate and
one `keep` lands somewhere different in every band. The relative
eigenvalue each setting actually cuts at, at the shipped d=8192
(`run_projection_pipeline.py --spectrum`, which needs no capture because
the Gram is scene-independent):

| band | keep=0.25 | keep=0.40 | keep=0.55 | keep=0.70 |
|---|---:|---:|---:|---:|
| xfine | 2.56e-04 | 5.77e-06 | 5.09e-08 | 1.20e-10 |
| **fine** | 1.77e-05 | 2.57e-08 | **2.47e-12** | 2.19e-17 |
| mid | 4.67e-03 | 3.03e-03 | 5.13e-04 | 8.66e-06 |
| coarse | 4.01e-03 | 3.96e-03 | 1.79e-03 | 6.82e-05 |

At keep=0.55 the `fine` band is cut at 2.47e-12 where `coarse` is cut at
1.79e-03 — the same nominal setting spanning a factor of 7e8 in what it
actually regularises, with the operator dividing by those numbers.
`fine` is the band a shared rank fraction regularises least, and `fine`
is the band that ran at 1030x on Red Rock. That is a mechanism, not a
coincidence.

Two things follow. The spread widens with `d`: a rehearsal at d=2048 put
`fine` at keep=0.55 at 4.7e-6, six orders of magnitude short of the
2.47e-12 above — the same trap that made an early per-cell study
conclude no truncation was needed. And the classical form of this solve
does not use a rank fraction at all. The Fourier-extension literature
truncates at a THRESHOLD on the singular values, with accuracy going as
the square root of it ([related-work.md](related-work.md)); a threshold
adapts to each band's spectrum by construction, where a rank fraction
cannot.

Read the other way round, one threshold does that adaptation by itself.
The rank fraction each implies at d=8192:

| band | eps=1e-3 | eps=1e-4 | eps=1e-5 | eps=1e-6 |
|---|---:|---:|---:|---:|
| xfine | 0.186 | 0.290 | 0.380 | 0.459 |
| **fine** | 0.103 | **0.195** | 0.266 | 0.324 |
| mid | 0.509 | 0.625 | 0.696 | 0.747 |
| coarse | 0.593 | 0.689 | 0.744 | 0.783 |

Against the shipped uniform keep=0.25, `eps=1e-4` truncates `fine`
*harder* — 0.195, and `fine` is the band that needs it — while letting
the well-conditioned `mid` and `coarse` keep about two and a half times
more of their spectrum than a flat quarter allows. That is the shape a
fix would have. The spectra themselves are committed
(`out/gram_spectrum_d8192.npz`), so the next threshold question is a
lookup rather than another eleven minutes of eigendecomposition.

## What the threshold actually scores, and what it says about the gate

Red Rock, one process, four settings sharing one eigendecomposition per
band (2,927 s, 7.19 GB peak), scored per band. `keep=0.25` reproduces
the published +17.8% / +27.8% exactly:

| setting | aggregate | xfine | fine | mid | coarse |
|---|---|---|---|---|---|
| forward | 0.1945 / 0.2160 | 0.1944 / 0.2160 | 0.4350 / 0.5580 | 0.9277 / 0.5301 | 1.4294 / 1.0726 |
| keep=0.25 | **+17.8% / +27.8%** | +17.7% / +27.8% | **-24.6% / -33.1%** | +52.8% / +7.7% | +29.8% / +14.8% |
| eps=1e-3 | +8.7% / +23.7% | +8.7% / +23.7% | **-1.0% / +10.6%** | +48.9% / +31.2% | +21.8% / +19.9% |
| eps=1e-4 | +13.5% / +24.8% | +13.5% / +24.8% | -2.1% / -5.8% | +40.0% / +33.1% | +4.5% / +7.2% |
| eps=1e-5 | +14.6% / +25.7% | +14.6% / +25.8% | -36.0% / -44.9% | +31.0% / +24.8% | -15.6% / -20.0% |

**The aggregate IS xfine.** The two columns agree to the digit at every
setting, because xfine holds 542,122 of 546,638 splats. Every
projection number this repo has published is an xfine number wearing a
scene's name.

**The shipped setting damages a band.** At keep=0.25 — the setting Red
Rock validated, with 0.0% of cells over the divergence limit — `fine`
decodes 24.6% / 33.1% WORSE than not projecting at all. The aggregate
improves regardless.

**And the gate could not see that, because its limit is far too loose.**
All sixteen (band, setting) norm ratios in this run are at or below
4.34, against a `DIVERGENCE_RATIO` of 20, so the gate passed every one.
The ratio is not a bad signal though — it is a badly calibrated one.
Inside each sparse band it tracks the per-band error monotonically:

| band | norm ratio -> per-band change, top-down |
|---|---|
| fine | 1.04 -> -1.0%, 1.10 -> -2.1%, 1.32 -> -24.6%, 1.39 -> -36.0% |
| mid | 1.14 -> +52.8%, 1.18 -> +48.9%, 1.57 -> +40.0%, 2.23 -> +31.0% |
| coarse | 1.21 -> +29.8%, 1.28 -> +21.8%, 2.00 -> +4.5%, 4.34 -> -15.6% |

20 was calibrated against catastrophe — saguaro's 5.3-versus-97 gap —
and catastrophe is not the failure that happens at a working setting.
**A limit near 1.3 would have refused `fine` at keep=0.25.** It is not
being changed here: one capture is what made 20 wrong, and one capture
is not enough to make 1.3 right. Confirming the relationship on a
second capture is cheap and is the next thing.

**The reparameterisation did what it was designed to do.** `eps=1e-3` is
the only setting where no band is meaningfully degraded — its worst is
`fine` at -1.0% — and it holds every band's ratio at or below 1.28,
because it truncates `fine` hardest (846 of 8,192) exactly where a flat
quarter truncates it least. It pays for that in aggregate: +8.7%
against keep=0.25's +17.8% on the top-down slice.

So the choice is not between a good default and a bad one. It is
between **+17.8% with one band a quarter worse** and **+8.7% with no
band worse** — measured on a metric that, until this run, only ever
reported the first number.

**The divergence is silent in the decode and loud in the norm.** The
last column above is the median solved-bundle norm over the forward
one: everything that works sits at or below 5.3 and everything broken
at or above 97, an 18x gap. `run_projection_pipeline.py` checks it after
every band and says so, because a bundle that decodes to garbage without
raising is the failure this technique most needs caught.

That combination — a large gain, a sharp cliff, a scene-dependent edge,
and a catastrophic rather than graceful failure — is why the projection
stays a measured spike and does not become an SDK default. Automatic
truncation selection is the path, and the norm ratio is the signal it
would use; neither is done.

## Through the whole pipeline

The per-cell numbers above are a different quantity from the slice
error the rest of this repo reports. Encoding EVERY cell analytically
and decoding the same evidence slices against the same exact-mixture
referee (`examples/run_projection_pipeline.py`):

| capture | forward encoding | analytic projection | change |
|---|---|---|---|
| saguaro | 0.3501 / 0.2132 | 0.2170 / 0.1367 | **+38.0% / +35.9%** |
| train (dense) | 0.9591 / 0.4948 | **0.3765 / 0.1716** | **+60.7% / +65.3%** |

The gain is **largest on the dense capture** — the case where doubling
`d` bought 2-4% for +600 MB and where orthogonal coupling bought 1.9%
([spatial.md](spatial.md)). Train's top-down slice goes from an error
as large as its signal (0.96) to 0.38.

**What it costs, and how much of that was avoidable.** Profiling put
98% of the fixed cost in one place: the per-band `eigh` was 106 s of a
108.6 s fixed cost, while per-cell solves were 27 ms and irrelevant.
Two changes remove most of it. Solving cells in batches of 256 turns
hundreds of BLAS-2 matvecs into one BLAS-3 matmul — **bit-identical
arithmetic**, 7.1x faster on that step. And Tikhonov regularisation
needs no eigendecomposition at all, so an explicit inverse replaces
`eigh` at ~6x less cost. Together: 770 s -> ~250 s on train, against
forward encoding's 51-100 s. Decode and storage are unchanged either
way — the output is an ordinary bundle, so codecs, replication and
rendering are untouched.

**Tikhonov beats truncation, and its knob is scene-dependent by five
orders of magnitude.** Replacing the truncated pseudo-inverse with
`(G + lambda I)^-1` is both cheaper and more accurate, but only at the
right lambda, and the right lambda is not portable:

| lambda (x max Gram entry) | saguaro | train (dense) |
|---|---|---|
| 1e-6 | **0.1227 / 0.0803** (+65.0% / +62.3%) | 1.0351 / 0.3727 (**-7.9%** / +24.7%) |
| 1e-3 | 0.1377 / 0.0963 (+60.7% / +54.8%) | 0.3792 / 0.1608 (+60.5% / +67.5%) |
| 1e-1 | — | **0.2986 / 0.1388** (+68.9% / +71.9%) |

The sparse capture wants 1e-6 and the dense one wants 1e-1 or more —
the train sweep had not turned at the largest value tried. **Using the
sparse setting on the dense capture is a 7.9% REGRESSION against
forward encoding**, which is the failure mode to guard: this is a knob
that must be set per scene, not a constant to inherit. The plausible
mechanism is that a dense cell puts more energy at the frequencies
where the Gram is near-singular, so it needs heavier damping; that is
consistent with the direction but has not been tested directly.

At its best setting each capture beats the truncated solve
(saguaro 0.1227 against 0.2170; train 0.2986 against 0.3765) at a
quarter of the encode cost.

**Shrinkage still adds on top, contradicting the obvious prediction.**
A solved bundle is already L2-optimal on its window and already
regularised, so shrinkage ought to move it away from the optimum. It
does not: on saguaro, `shrink` at the 10th percentile improves the
solved bundle a further **+6.4% / +3.6%** (0.2031 / 0.1317 against
0.2170 / 0.1367). The optimum is for the *windowed per-cell* objective,
which is not the quantity being scored, and truncation means the solve
is not even optimal for that. The gain is smaller where the solve is
stronger (+1.4% on train under Tikhonov), which fits.

Window WIDTH is a real knob and not a forgiving one: s = h/2 gives
0.0739, s = h gives 0.1587, and wider is worse still, because a wide
window weights territory the cell's own splats do not describe.

*Positioning.* This is the classical Fourier extension problem (see
[related-work.md](related-work.md)) with the smooth-window variant
being the partition-of-unity method familiar from meshfree
approximation. Our frequencies are RANDOM (mixture-drawn) rather than a
lattice, so the plunge structure smears — but the Adcock-Huybrechs
frames theory covers arbitrary frames, and the stability guarantees
survive. Not promoted to the SDK surface: this is a measured spike, and
`SDK.md`'s charter wants a deterministic test and a documented failure
mode before an API exists.

**Failure modes.** The finest band must stay >= the sample spacing or
the model memorizes training points and rings between them (train PSNR
47dB, test 17dB — the classic signature); lam is a real knob in the
interpolation regime (features > samples). Platform: NumPy is pinned
<2.0 because macOS Accelerate float32 GEMV corrupts with heap-dependent
NaNs — the OpenBLAS wheels are clean (`holo/fit.py` NOTE).

**API.**
```python
from holo import FrequencyBands, HoloRegressor
bands = FrequencyBands(dims=[1024, 2048, 4096, 9216],
                       sigmas=[0.12, 0.04, 0.016, 0.008], ndim=2, seed=0)
reg = HoloRegressor(bands).fit(points, values, lam=1e-2)  # values (n,) or (n,c)
pred = reg.eval(anywhere)     # the fitted S is a first-class hologram
```

**Evidence.** A photograph fit as one complex vector:

![Grace Hopper portrait fit by ridge regression at increasing D](../out/fit_photo.png)

The per-cell real-scene comparison (the honest negative at density):

![forward vs ridge-fitted bundles on the saguaro capture](../results/real_fit.png)

`tests/test_fit.py` (recovers-from-samples, beats-bundle, denoises,
multichannel); `holo-demos fit color`.
