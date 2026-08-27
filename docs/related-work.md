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

## Dispatch delta (2026-08-26, near-enough dispatch)

Positioning for `holo/dispatch.py` (rules as similarity). Adjacent
lines that exist: the HDC/VSA surveys (part II, arXiv:2112.15424)
name **similarity-based reasoning** — search, classification,
analogy — as a core capability of the framework, and HDC text/intent
classification is a standard application; *Probabilistic Abduction
for Visual Abstract Reasoning via Learning Rules in Vector-symbolic
Architectures* (arXiv:2401.16024) learns and executes RULES inside a
VSA (visual domain); similarity-preserving hypervector encodings of
sequences (arXiv:2201.11691, arXiv:2112.15475) are the literature's
answer to exactly the order-blindness gap our trigram conditions
carry. On the engineering side, embedding-based "semantic routers"
(learned sentence embeddings + nearest-centroid intent routing) are
established practice. *What appears open as of this sweep: the
engineered synthesis — a rule ENGINE contract with hash-derived
determinism (no learned model), an explicit sqrt(N/2d) capacity
budget, whole-rulebook-as-one-vector and banded/clustered variants
with measured compute trades, abstention as policy, and
CRDT-mergeable rule tables.* Re-verify before claiming externally.

## 0.3 delta (2026-08-27, publication-prep sweep)

Run before the publication pass, against every novelty-sensitive claim
we hold. Four deltas, two of which move our own framing.

**Quantized-phase FHRR now has a paper, and it corroborates our codec
split.** qFHRR — *Rethinking Fourier Holographic Reduced
Representations through Quantized Phase and Integer Arithmetic*
(Snyder, Poursiami, Parsa, arXiv:2604.25939) — stores each dimension
as a discrete phase index in K bins, making binding modular addition,
unbinding modular subtraction, and similarity a cosine lookup, at
**3-4 bits per dimension** against 64-bit complex, with FPE's spatial
similarity structure preserved. That representation IS our HP codec
(`holo/phase.py`) arrived at from the hardware side rather than the
storage side, so it must be cited, and it removes any claim that
phase-only quantized FHRR is itself novel.
*What it does not touch — and in fact confirms:* their bundling is
**not closed** under quantized phase, so they map to Cartesian, sum,
and project back onto the unit circle. That projection discards
magnitude, which is exactly the failure our codec measurements found
from the other direction (phase-only floors amplitude fields at ~0.24
rel RMSE at any bit depth, which is why `HM`/`HG` exist). Two
independent routes to the same boundary. Our contribution is the
measured rate-distortion curve across it, not the representation.

**Our scale floor is an ad-hoc version of a published filter.**
Mip-Splatting (arXiv:2311.16493, CVPR 2024) constrains Gaussian size
by the maximal sampling frequency of the input views — a principled
`S_LO` — and replaces 3DGS's 2D dilation with a 2D Mip filter that
simulates a box filter. Analytic-Splatting (arXiv:2403.11056, ECCV
2024) integrates the Gaussian analytically over the 2D pixel window
via a logistic-CDF approximation, framing the problem exactly as our
own measurements did: 3DGS "treats each pixel as an isolated single
point rather than an area." Our `footprint_blur` is the 3-D,
pre-projection form of the same convolution. *Position:* adopt and
cite; the anti-aliasing lane is solved and is not ours to contest.
The open piece our measurements added is narrow and worth keeping —
that clamping a thin axis while holding PEAK amplitude changes
radiometry (11x scene-mass inflation on Red Rock), which the
mass-preserving alternative would fix.

**Motion-as-phase-ramp has prior art; our earlier framing overstated
it.** The shift property of fractional binding is standard in the
Spatial Semantic Pointer literature (Komer & Eliasmith, already cited)
— convolving a spatial representation with an SSP translates it — and
*Simulating and Predicting Dynamical Systems With Spatial Semantic
Pointers* (Voelker et al., Neural Computation 33(8), 2021) already
simulates arbitrary trajectories for single AND multiple objects in
this algebra. `translate_bundle` is that shift theorem, not a new
idea. *What remains open* is narrower and should be claimed that way:
the combination at capture scale — per-object sub-bundles over real
650k-splat scenes, a whole animation bound into ONE vector by time
codewords, `where_is(object, t)` as an unbinding, and CRDT-mergeable
animated scenes — plus the cost model (phase ramp is O(d) regardless
of splat count; 350x cheaper than re-encode at 20k splats).

**CRDT-replicated vector state has a nearest neighbour now.**
*Conflict-Free Replicated Data Types for Neural Network Model
Merging* (arXiv:2605.19373, 2026) wraps merge strategies in an OR-Set
contribution layer plus a deterministic merge over a canonically
ordered set. Different object (network weights, not hypervector
bundles) but the same shape as `holo/orset.py`, and it sharpens a
contrast worth making in any writeup: they report that **all 26**
merge strategies they tested fail commutativity, associativity or
idempotency, whereas superposition gives us the first two for free and
fails only idempotency — which is precisely the gap the G-Counter and
observed-remove recipes close.

**Still not found:** splat scenes stored as holographic bundles with
algebraic queries; rendering by folding projection-slice factors into
a random-feature bundle; per-splat anisotropic covariance via
importance-sampled mixture spectral codebooks; a rule engine with a
capacity contract and abstention as policy. Those remain the claims to
lead with.

Re-run this sweep before any writeup or release; log deltas here with
the date. Track daily 3DGS postings via the awesome-gaussians list
(github.com/longxiang-ai/awesome-gaussians).
