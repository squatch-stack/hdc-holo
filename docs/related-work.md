# Related work (arXiv sweep, 2026-08-26)

*[← docs index](README.md) · positioning*

A positioning survey run against every novelty-sensitive claim in this
repo. Method: keyword sweeps over arXiv via web search plus abstract
reads of the nearest neighbors. Caveat that must travel with this page:
**absence from a search is weak evidence of absence** — re-sweep before
any external claim of novelty (papers also appear daily; see the
awesome-gaussians tracker).

## Theory anchors (we build directly on these)

- **Vector Function Architectures** — Frady, Kleyko, Kymn, Olshausen,
  Sommer, *Computing on Functions Using Randomized Vector
  Representations* (arXiv:2109.03429). The FPE/Bochner kernel bridge our
  field encoders instantiate.
- Kleyko et al., *HDC/VSA surveys parts I & II* (arXiv:2111.06077,
  arXiv:2112.15424); *Improved Cleanup and Decoding of Fractional Power
  Encodings* (arXiv:2412.00488); Torchhd (arXiv:2205.09208).

## Nearest neighbors (cite these; our positioning in italics)

- **VSA-OGM** — *Brain-Inspired Probabilistic Occupancy Grid Mapping
  with VSAs* (arXiv:2408.09066; npj Unconventional Computing 2026).
  SSP-encoded occupancy fields with sparse local updates; 45x latency /
  400x memory vs dense baselines. *Closest published relative of our
  attribute/chunked fields — but scalar occupancy only: no per-splat
  covariance, color, rendering, learning, replication, or deletion.*
  Related: hyperdimensional OGMs for RL exploration (arXiv:2502.09393).
- **HyperSpace** (arXiv:2604.15113) — modular open-source framework for
  VSA pipelines (HRR/FHRR operators incl. regression; finds
  similarity/cleanup dominate spatial workloads). *The nearest "VSA
  SDK" effort; no scene, splat, or replication content.*
- **GVKF: Gaussian Voxel Kernel Functions** (arXiv:2411.01853, NeurIPS
  2024) — 3DGS made a continuous opacity field through kernel
  regression. *Independently validates our "splatting is a kernel
  mixture" bridge from the graphics side — but keeps per-Gaussian
  parametric storage; no random-feature/holographic superposition.*
- **CryoSplat** (arXiv:2508.04929) and **R2-Gaussian**
  (arXiv:2405.20693) — Gaussian splatting under the Fourier slice
  theorem (cryo-EM) and closed-form X-ray/tomographic splatting (CT),
  on the heritage of frequency-domain volume rendering. *Prior art for
  closed-form Gaussian projections; our render.py differs in WHAT is
  projected — a random-frequency bundle, where a whole view folds into
  one vector — not per-Gaussian rasterization.*
- **EKS** (arXiv:2508.02831) — anisotropic-Gaussian-kernel spatial
  encoding for NeRF editing. *Kernel-space scene features, adjacent to
  our encode strand; not superposed/holographic.*
- **3DGS compression** — survey arXiv:2502.19457 (plus VQ, texture-
  coding, feed-forward lines). *A dense field with no fixed-size
  superposition/random-feature approach found in it — our
  fidelity-per-byte curves should benchmark against these baselines.*

## Apparently still open (as of this sweep — reverify before claiming)

No hits found for: splat scenes stored AS holographic bundles with
algebraic queries (what_is_at / where_is by unbinding); ridge-fitted
random-feature holograms positioned as a scene representation (RFF
regression itself is classical — Rahimi & Recht — the scene framing is
what seems absent); CRDT-replicated scene fields, observed-remove
deletion, or collaborative splat painting over sync; per-splat
anisotropic covariance via importance-sampled mixture spectral
codebooks; rendering by folding projection-slice factors into a
random-feature bundle.

## 0.2 delta (2026-08-26, second sweep)

Prior art located for the analytic per-cell L2 projection direction
(the closed-form box-region Gram of complex exponentials): this is the
classical **Fourier extension** problem — approximating a function on a
subdomain with exponentials from a larger box — and its pathologies are
fully mapped. Adcock's *On the numerical stability of Fourier
extensions* (arXiv:1206.4111) proves the frame system is severely
ill-conditioned (the sinc/prolate Gram's eigenvalues collapse through
Slepian's "plunge region") yet REGULARIZED least squares is numerically
stable to ~sqrt(machine-eps) accuracy; Adcock–Huybrechs *Frames and
Numerical Approximation II* develops the general frames+regularization
theory, and the AZ algorithm (arXiv:1912.03648) gives fast solvers for
exactly these systems. Implications for `fit_cells`: (a) the empirical
spectral prior that rescued per-cell ridge is a known-in-theory
regularizer of a frame system, (b) the zero-sample closed-form Gram
will hit the plunge-region conditioning and needs truncated-SVD or
Tikhonov treatment from the start — the literature says this is
workable, not fatal.

Re-run this sweep before any writeup or release; log deltas here with
the date. Track daily 3DGS postings via the awesome-gaussians list
(github.com/longxiang-ai/awesome-gaussians).
