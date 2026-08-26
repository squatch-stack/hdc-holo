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
Open direction: the analytic L2 projection — the box-region Gram
`G_jk = integral e^{i(w_j-w_k)p} dp` is a separable product of sincs,
so the optimum has a closed form needing zero samples. This is the
classical Fourier extension problem (see
[related-work.md](related-work.md)): expect plunge-region
ill-conditioning and regularize from day one. One difference from the
classical setting: our frequencies are RANDOM (mixture-drawn), not a
lattice — the plunge structure smears, but the Adcock-Huybrechs frames
theory covers arbitrary frames, so the stability guarantees survive.

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
