# Holographic Scene Representation: Gaussian Splats as Hypervectors

*Draft v0 — prose against [PAPER.md](../PAPER.md)'s three claims.
Markdown by choice, not by default: no LaTeX toolchain here would let
me verify a `.tex` compiles, and drafting in a gated surface means the
claims checker drift-tests every number below against
`claims/registry.jsonl`. Conversion is mechanical once a toolchain
exists.*

**Status.** Sections 1–5 are drafted prose. Sections 6–9 are drafted at
outline-plus-argument density and need a second pass. Nothing here is
citation-formatted yet; references are named inline and collected in
[PAPER.md §9](../PAPER.md).

---

## Abstract

A 3D Gaussian splatting scene is a list of primitives: fast to
rasterize, awkward to query, merge, or carry. We show that such a scene
can instead be *superposed* into a single fixed-size complex vector,
and that three capabilities then follow from one algebra rather than
from three mechanisms. Encoding a point as `e^{iWp}` with frequency
rows drawn from `N(0, Σ⁻¹)` makes the inner product of two encodings
equal a Gaussian kernel, so a mixture of anisotropic Gaussians is a
random-feature bundle: evaluating the field anywhere is an inner
product, symbolic attributes attached by binding are recovered by
unbinding, an entire orthographic view folds into another vector of the
same kind, and independently-edited replicas merge by addition. A
single capacity law, `σ ~ √(N·R / 2d)`, predicts the noise of every one
of those readouts and states where each stops working. We demonstrate
the representation on real captures of up to 682,000 splats, evaluated
against exact analytic ground truth rather than against images, and we
report the cost plainly: a bundle is roughly two orders of magnitude
larger than a modern splat codec at the same scene. The contribution is
not compression. It is that query, view synthesis, and coordination-free
merge become the *same operation* on the same object, with a measurable
budget attached.

## 1. Introduction

3D Gaussian splatting represents a scene as an explicit collection of
anisotropic Gaussian primitives, and that explicitness is the source of
both its speed and its awkwardness. Rasterizing a list of primitives is
fast. Asking a question of it — *what is at this point?*, *where are the
objects labelled "chair"?* — means iterating the list. Merging two
independently-edited copies means reconciling two lists. Carrying a
scene means carrying every primitive.

Vector-symbolic architectures (VSAs) make composition algebraic:
structures are built by binding and bundling high-dimensional vectors,
and questions are answered by inverse operations rather than by
traversal. They are usually applied to symbols. The bridge to geometry
is fractional power encoding: representing a position `p` as the
hypervector `e^{iWp}` with frequency rows `w ~ N(0, Σ⁻¹)` makes the
inner product of two such encodings equal the Gaussian kernel
`exp(−½(p−q)ᵀΣ⁻¹(p−q))` — Bochner's theorem, in the form Rahimi and
Recht made practical as random Fourier features and Frady et al.
developed as Vector Function Architectures.

The consequence we build on is small to state and large in effect: **a
Gaussian splat mixture is already a random-feature bundle.** Summing the
encodings of N splats, weighted by their amplitudes, produces one vector
whose inner product with the encoding of any query point returns the
mixture's value there. The scene stops being a list.

This paper reports what that buys, what it costs, and where it fails.

**Three claims.**

1. **A splat scene is one vector, and queries are algebra.** With
   importance-sampled mixture spectral codebooks, every splat keeps its
   own anisotropic covariance inside a shared basis, and role-filler
   payloads attached by binding answer attribute queries by unbinding.
2. **A view is also just a bundle.** By the projection-slice theorem,
   folding one per-frequency factor into the conjugate bundle turns an
   entire orthographic render into another bundle of the same kind:
   every pixel is one inner product, with no ray marching, no depth
   sort, and no geometry present at render time.
3. **Superposition is almost a CRDT, and the "almost" is exactly one
   axiom.** Bundling is commutative and associative but not idempotent;
   two standard recipes close that single gap and make holographic
   scene state mergeable without coordination.

**Two things we state up front rather than in a discussion section.**

*The boundary.* This is an emission — tomographic — representation.
Alpha compositing is non-linear and therefore outside superposition's
reach. We render X-ray projections and evaluate them against analytic
line integrals; we do not perform novel-view synthesis, and we do not
compete with rasterizers on photorealism.

*The cost.* On the same capture, a holographic bundle is approximately
400× larger and 50× less accurate at reproducing the field than a
current splat codec (§7). If the task is to store a scene and rasterize
it later, the correct advice is to use the codec. The claims above are
about capabilities the codecs do not provide, and the paper is written
so that a reader can weigh that trade with real numbers.

## 2. Background

**FHRR.** A hypervector is `d` unit-magnitude complex phasors. Binding
is elementwise complex multiplication (phases add) and is exactly
invertible by the conjugate; bundling is addition; a fixed random
permutation tags order or role. Two independent random hypervectors have
similarity `0 ± 1/√(2d)`, which is the noise floor everything else
trades against.

**Capacity as a contract.** Bundling N items puts crosstalk of standard
deviation `σ ~ √(N·R / 2d)` under every readout, where R is the
component power of what was bundled. Capacity is therefore a
signal-to-noise budget, never a table size, and structures fail *soft* —
noise, then errors — rather than by allocation failure. We treat this
law as an API: every constructor documents its budget, and every
experiment reports measurement against prediction.

**Fractional power encoding.** Raising a hypervector to a real power
multiplies its phases, extending binding from a discrete operator to a
continuous one and giving the `e^{iWp}` encoding above. Komer and
Eliasmith's Spatial Semantic Pointers develop this for continuous space,
including the shift property we use in §5.

**3D Gaussian splatting.** A scene is a set of anisotropic Gaussians
with position, covariance, opacity, and view-dependent colour, rendered
by projecting and alpha-compositing them front-to-back.

## 3. Claim 1 — A splat scene is one vector

### 3.1 Encoding

The naive route — one shared covariance for all splats — fails
immediately on real data, where axis scales span five decades. We
instead store each splat's *Fourier spectrum* sampled at a shared random
codebook, which keeps per-splat anisotropic covariance inside one
bundle at the cost of an importance-sampling variance penalty.

Drawing the codebook from a **mixture of Gaussians** spanning the
scene's scale range reduces that penalty from 16–33× to 2.4–3.2×. Two
rules were learned the expensive way, and both are visible as structured
artifacts rather than as noise when violated: the finest mixture
component must sit at the global scale floor, and *every* band's
codebook must reach that floor — because bands are assigned by maximum
axis scale, while a mid-band needle splat still has thin axes at the
floor. Violating the second rule paints herringbone
(`results/failure_herringbone.png`).

Real captures then require two more mechanisms. **Scale bands** group
splats by maximum axis scale, each band with its own codebook; **spatial
cells** chunk each band, so a query's crosstalk follows *local* density
rather than N. Query reach follows the band's scale cap, which is why
splitting the finest band cut slice error by 33–44%.

### 3.2 Queries

Evaluating the encoded field at a point is one inner product. Attaching
role-filler records to splats by binding, and recovering them by
unbinding, gives `what_is_at(p)` and `where_is(label)` directly — the
second returning a *field* over space rather than a list of matches.
Because the codewords are hash-derived from `(dim, seed)` rather than
from sequential RNG state, any process reconstructs them independently,
which is what makes §5 possible.

### 3.3 Evidence

Capacity curves fit `d^-0.50` exactly, matching the law. Four real
captures — two phone-scanned outdoor scenes, an indoor LiDAR cloud, and
a Tanks & Temples scene, from 244k to 682k splats after cropping — pass
through one fixed pipeline, and slices decode at 19%/22% relative error
against the *exact Gaussian mixture* evaluated cell-locally. Kernels
agree across three compute backends to 2.5e-8.

*Positioning.* VSA-OGM is the closest published relative: SSP-encoded
occupancy fields with sparse local updates, but scalar occupancy only —
no per-splat covariance, colour, rendering, learning, or replication.
GVKF independently validates the "splatting is a kernel mixture" bridge
from the graphics side while keeping per-Gaussian parametric storage.

## 4. Claim 2 — A view is also just a bundle

### 4.1 Folding a view into the vector

The bundle is the scene's Fourier transform sampled at random
frequencies. By the projection-slice theorem, the integral of the field
along a ray has a closed form in that domain: the ray integral depends
only on the spectrum in the plane perpendicular to the view direction.
Folding the corresponding per-frequency factor into the conjugate bundle
therefore turns an entire orthographic view into another bundle, and
every pixel becomes a single inner product against it.

There is no ray marching, no depth sort, and — the part that surprises
people — no geometry present at render time. The renderer's input is a
vector.

Because blur is covariance addition, mip levels are free: convolving the
scene with an isotropic Gaussian is adding `σ²I` to every covariance.
This matters, because a projection uses only the spectrum slice
perpendicular to the view, so renders need their own dimension budget;
we encode a dedicated mip for rendering.

### 4.2 Evidence

On a synthetic 240-splat trefoil knot with analytic line integrals as
ground truth, views rendered from a single vector reach 5–7% RMSE. On
real captures, a full turntable orbit is rendered entirely from 135 cell
bundles with no geometry at render time. Colour rides as amplitude on
one shared frequency basis, so RGB renders reuse the same phasors.

### 4.3 The boundary, stated as a result

Emission and tomography only. Occlusion requires ordering and
compositing, which are non-linear operations that superposition does not
admit. We consider this worth stating precisely rather than hedging: the
representation renders what integrates, and a scene where visibility
matters needs a compositing pass outside the algebra.

*Positioning.* CryoSplat and R2-Gaussian both exploit the Fourier slice
theorem for closed-form Gaussian projection — per primitive. Folding a
whole *view* into one random-feature vector appears to be new.

## 5. Claim 3 — Superposition is almost a CRDT

### 5.1 The one missing axiom

A conflict-free replicated data type requires a merge that is
commutative, associative, and idempotent. Bundling — vector addition —
is trivially commutative and associative. It is not idempotent: adding
the same contribution twice double-counts it, so redelivered updates
corrupt state.

That single gap has a standard fix. Each writer accumulates only its own
bundle, and the merged hologram is the sum over writers, so redelivery
is absorbed by the underlying delta protocol's version vectors. Merges
are then bit-identical on every replica that has seen the same updates.

Deletion is subtler. Subtracting a contribution has PN-counter
semantics, and two peers concurrently retracting the same item
over-cancel into a *negative phantom*. Changing the type of the
operation fixes it: removals become tombstones in a grow-only set, the
subtraction is derived once by every reader, concurrent duplicate
removal is idempotent, and a concurrent re-add wins because it carries a
fresh identifier the remover never observed.

Determinism is what makes all of this coordination-free. Codewords and
frequency matrices are hash-derived from `(dim, seed)`; a codebook built
from sequential RNG state would assign vectors by creation order and
replicas would silently disagree.

### 5.2 Evidence

Two peers paint halves of a scene offline and delta-sync into one, with
matching state and render digests; attributed scenes merge with labels
riding a grow-only registry, so a peer answers queries for labels it
never used locally. A two-process demonstration exchanges only
length-prefixed deltas over TCP and includes the adversarial case: both
painters concurrently undo the same stroke, which is tombstoned twice
and subtracted once.

*Positioning, and a contrast.* Recent work wrapping neural-network model
merging in CRDT semantics reports that all 26 merge strategies tested —
weight averaging, SLERP, TIES, DARE, Fisher, evolutionary — fail
commutativity, associativity, or idempotency. Superposition satisfies
the first two by construction and fails only the third. The recipes
above close that one gap, which is a cheaper starting position than the
merging literature usually enjoys.

## 6. The law, and where it bends

*(Draft density: argument complete, prose needs a pass.)*

The capacity law is the paper's spine, and this section is where it is
tested rather than asserted. Every demonstration prints the measured
curve against prediction. Four honest deviations:

- **Dense scenes correlate their noise.** Nearby splats share
  frequencies, inflating σ roughly 1.5–3× over the i.i.d. law. The error
  is *coherent*, not Monte-Carlo, which we established by the negative
  result that doubling `d` bought 2–4% for +600 MB. More dimensions
  cannot wash out correlated error.
- **Per-cell ridge fitting is sampling-limited.** Fitting a hologram to
  samples beats forward encoding by ~70× held-out on synthetic mixtures,
  but at real capture density the fit *loses* (0.72/0.53 vs 0.52/0.38),
  because hundreds of floor-scale splats per cell need coverage at their
  own kernel width. A spectral prior was required even to make the fit
  competitive. This places the open direction — a zero-sample analytic
  projection — inside the classical Fourier-extension problem, which is
  severely ill-conditioned but provably stable under regularization.
- **The scale floor is radiometric.** Clamping a thin axis while holding
  peak amplitude fixed raises each splat's integral, inflating total
  scene mass 11× on one capture. Measuring what the floor costs by point
  sampling is ill-posed: needles thinner than a pixel are nearly
  invisible to point evaluation, so we integrate over the pixel
  footprint instead — a convolution that is exact because covariances
  add. Matched to a pixel-integrated target the pipeline reports 11.5%
  where the sharp-vs-sharp pair reports 18.1%.
- **Storage has a rate-distortion boundary.** Phase-only codes floor
  amplitude fields at ~0.24 relative RMSE at any bit depth, because a
  bundle's magnitude carries information that unit-magnitude codes
  discard. Magnitude-preserving codes cross that boundary; a
  gamma-companded polar code is the faithful choice for
  wide-dynamic-range bundles.

## 7. What it costs

*(Draft density: numbers final, framing needs a pass.)*

We score every representation by one referee: reconstruct the field,
evaluate at the same query points, compare to the source's exact
mixture. Per-splat formats lose to quantization; bundles lose to
crosstalk.

| representation | MB | bytes/splat | field error |
|---|---:|---:|---:|
| PLY (SH-0) | 3.9 | 85 | 0.0% |
| SPZ v3 | 1.0 | 22 | 0.0% |
| SOG (SH palette) | 0.8 | 18 | 0.3% |
| holographic bundles (d=8,192) | 382.8 | 8,354 | 17.4% |
| same, 8-bit magnitude codec | 95.7 | 2,089 | 17.0% |

Two observations survive the loss. The 8-bit codec is *free*: four times
smaller at slightly **better** error, because max-scaled quantization
zeroes small components that on a forward-encoded bundle are mostly
crosstalk — an accidental shrinkage denoiser. And the two families scale
differently: bundle bytes follow occupied cells while per-splat bytes
follow content, so ten times the splats in a fixed volume costs a codec
10× and costs bundles 2.6×. The gap narrows fourfold per decade of
density and does not close on real captures.

## 8. Related work

*(To expand from PAPER.md §9, which lists each work with the sentence it
forces. Structure: foundations → the four works that bound our claims →
nearest neighbours → the graphics paragraph.)*

The graphics paragraph should say plainly: rasterization is better at
rasterizing; anti-aliasing in this family is solved by sampling-rate
filters and analytic pixel integration, and our scale floor is an ad-hoc
version of the former.

## 9. Limitations and future work

*(Draft density: list complete, prose needed.)*

Occlusion is outside the algebra. Storage is large in absolute terms.
Rotation is not a phase ramp — translation is, but rotation remixes
frequencies across the codebook and needs its own mechanism. Determinism
is semantic rather than bitwise. Banded and clustered readout require
structure in the data to pay off.

One generality result belongs here in a sentence rather than as a
claim: the same capacity law and the same partition-plus-locality remedy
transfer intact from geometric scenes to *rule tables*, where a
similarity-dispatch engine reaches 0.98 accuracy at 43× less compute
than exhaustive matching. That the law governs a domain with no
geometry in it is the strongest evidence we have that it is a property
of superposition rather than of splats.

## 10. Conclusion

*(To write once §§6–9 settle.)*
