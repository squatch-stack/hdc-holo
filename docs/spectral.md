# Spectral encoder and mixture-of-Gaussians codebooks

*[← docs index](README.md) · fields & scenes*

*(implementation: `holo/spectral.py`, exported via `holo.encode`;
example drivers `run_prototype.py`, `run_mog.py`; `hdc_splat.py` is a
compatibility shim)*

**What.** Drops the shared-covariance restriction of the FPE field
([fields.md](fields.md)): every splat keeps its OWN anisotropic Sigma
inside one bundle. The trick is to store the splat's Fourier SPECTRUM
sampled at a shared random codebook `w_j ~ rho`:

    g_hat(w) = a (2 pi)^{D/2} |Sigma|^{1/2} exp(-1/2 w^T Sigma w) e^{-i w.mu}
    s_k[j]   = g_hat_k(w_j)                       (complex64, length d)

A scene is the bundle `S = sum_k s_k`; decoding at p is Monte-Carlo
Fourier inversion with importance weights `1/rho(w_j)` — still one
inner product per point (`accel.decode` on the GPU). The classical FPE
case falls out when every splat shares Sigma0: draw `w ~ N(0,
Sigma0^-1)` and the envelope is constant, vectors are unit phasors,
no weights needed. Whole-scene translation is one elementwise multiply
on the bundle (shift theorem).

**The variance problem and the MoG fix.** A single-scale codebook pays
an importance-sampling penalty when splat scales vary: frequencies in
rho's tail get huge weights. Measured penalty 16-33x over matched-scale
encoding; drawing the codebook from a MIXTURE of Gaussians spanning the
splat-scale range cuts it to 2.4-3.2x (3-10x noise reduction).
Complementary to bands ([spatial.md](spatial.md)): bands quantize
covariance into discrete classes, the mixture keeps it continuous, and
the two compose.

**Budget.** Capacity curves fit `d^-0.50` exactly — the Monte Carlo law
again, with the importance-weight variance as a multiplier.

**Failure modes.** The codebook's FINEST component must reach the
smallest axis scale present (beta = 1 at the scale floor, splats
clamped to it): a wider finest component leaves per-sample weights
heavy-tailed, and single tail frequencies paint visible plane-wave
stripes/herringbone across renders.
`results/failure_herringbone.png` is the preserved exhibit — the first
real-scene run before the rule was learned: the rectangular herringbone
patches ARE cell bundles whose band codebook stopped short of the
global scale floor while needle splats kept thin axes at it.

**Evidence.** Crosstalk follows `sqrt(N/2d)` — measured curves fit
`d^-0.50` exactly:

![capacity curves: crosstalk of superposed splats follows sqrt(N/2d)](../results/capacity_curve.png)

The mixture codebook collapses the importance-sampling penalty:

![single-sigma vs mixture codebook penalty, and who pays it by splat scale](../results/mog_penalty.png)

The preserved herringbone failure-mode exhibit (band codebook stopped
short of the global scale floor):

![herringbone failure mode: per-cell plane-wave patches](../results/failure_herringbone.png)

Also `results/recon_2d.png`, `results/translation.png` (shift-theorem
equivariance); `run_prototype.py`, `run_mog.py`.
