# Paper skeleton — a scene can be a hypervector

Working plan for the publication pass
([#6](https://github.com/squatch-stack/hdc-holo/issues/6)). Three claims,
one law, and an honest boundary. Every number below is already measured
and in-tree; every figure already exists. Amend in place — this file is
the writeup's contract the way [SDK.md](SDK.md) is the SDK's.

**Thesis.** Store Gaussian splats as superposed random-Fourier features
and three capabilities fall out of one algebra: queries become inner
products, a whole rendered view becomes another bundle, and replicas
merge by addition. One capacity law, `sigma ~ sqrt(N R / 2d)`, governs
all three and states where each stops working.

**Scope discipline.** Three claims, not seven. What was deliberately
left out — and why — is at the bottom; that list is part of the plan,
not an oversight.

---

## 1. Introduction

The pitch in four moves: 3DGS made scenes explicit collections of
primitives, which made them fast to rasterize and awkward to *query*,
*merge*, or *carry*. Vector-symbolic architectures make composition
algebraic but are usually applied to symbols, not geometry. Fractional
power encoding bridges them — a point encoded as `e^{iWp}` with
`w ~ N(0, Sigma^-1)` makes inner products *equal* Gaussian kernels
(Bochner) — so a splat mixture is a random-feature bundle, and a scene
becomes one fixed-size complex vector.

State the boundary in the intro, not the conclusion: this is an
emission/tomographic representation. Alpha compositing is non-linear
and stays outside superposition. Saying so early is what makes the rest
credible.

## 2. Background

FHRR (bind = elementwise complex multiply, unbind = conjugate, bundle =
addition, permutation for order); the FPE/Bochner kernel bridge (Frady
et al., VFAs); Spatial Semantic Pointers for continuous space; 3DGS and
its compression/anti-aliasing literature. Keep it short — one page — and
carry the sweep's corrections *here* rather than defending them later
(§7).

---

## Claim 1 — A splat scene is one vector, and queries are algebra

**Assertion.** N anisotropic Gaussians encode into fixed-size complex64
bundles through importance-sampled mixture spectral codebooks, so each
splat keeps its own covariance inside a shared basis; evaluation
anywhere is an inner product, and role-filler payloads answer
`what_is_at(p)` / `where_is(label)` by unbinding.

| evidence | artifact |
|---|---|
| capacity fits `d^-0.50` exactly | `results/capacity_curve.png` |
| mixture codebooks: penalty 16-33x -> 2.4-3.2x | `results/mog_penalty.png` |
| four real captures, 519k-682k splats, exact-mixture referee | `results/real_redrock.png`, `real_scan-tucson.png`, `real_train.png`, `real_lidar-dense.png` |
| best Gaussian capture: 19%/22% slice error | `docs/real-scenes.md` |
| attribute queries on splats (SNR cliff, class filters) | `out/attribute_field.png` |
| scale bands x chunked cells; reach follows the band cap | `out/multiband.png`, `out/chunked3d.png` |
| codebook rule violated -> structured artifact, not noise | `results/failure_herringbone.png` |
| cross-hardware agreement to 2.5e-8 (M1 Max / RTX 5090 / CPU) | `bench/`, SDK log |

**Why it survives the sweep.** VSA-OGM is the closest published
relative and is scalar-occupancy only — no per-splat covariance, no
color, no rendering, no replication. GVKF independently validates the
"splatting is a kernel mixture" bridge from the graphics side but keeps
per-Gaussian parametric storage.

**Tests:** `test_spectral.py`, `test_capture.py`, `test_spatial.py`,
`test_attribute_field.py`.

---

## Claim 2 — A view is also just a bundle

**Assertion.** By the projection-slice theorem, folding one
per-frequency slice factor into `conj(S)` turns an entire orthographic
view into another bundle: every pixel is one inner product. No ray
marching, no depth sort, no geometry at render time. Blur is covariance
addition, so mip levels are free — which is also why projections need
their own dimension budget (a view uses only the spectrum slice
perpendicular to it).

| evidence | artifact |
|---|---|
| trefoil: 5-7% RMSE vs analytic line integrals | `out/ray_render.png`, `out/ray_render.gif` |
| real captures orbited from 135 cell bundles, no geometry | `results/real_turntable-scan-tucson.gif`, `real_turntable-redrock.gif` |
| X-ray views vs analytic mip on real captures | `results/real_*_xray.png` |
| color as amplitude on one shared basis | `out/color_knot.gif`, `out/color_photo.png` |
| translation as a phase ramp (closed form) | `results/translation.png` |

**The boundary, stated as a result.** Emission/tomography only. We do
not composite. The paper is stronger for drawing that line precisely
than for hedging it.

**Why it survives the sweep.** CryoSplat and R2-Gaussian do closed-form
Gaussian projection — per primitive. Folding a whole *view* into one
random-feature vector is unchallenged.

**Tests:** `test_render.py`, `test_color.py`, `test_capture.py`.

---

## Claim 3 — Superposition is almost a CRDT, and the "almost" is one axiom

**Assertion.** Bundling is commutative and associative but **not
idempotent**, so redelivered updates double-count. Writer-sharded
G-Counter bundles close accumulation; observed-remove tombstones close
deletion, because arithmetic retraction has PN-counter semantics —
concurrent duplicate removals over-cancel into negative phantoms.
Determinism is what makes it coordination-free: codewords and frequency
matrices are hash-derived from `(dim, seed)`, never sequential RNG
state, so any replica reconstructs them.

| evidence | artifact |
|---|---|
| two peers paint one scene offline, delta-sync into one | `out/crdt_scene.png` |
| attributed scenes merge; labels ride the registry | `out/crdt_attributes.png` |
| concurrent double-undo: clean vs negative phantom | `out/orset_undo.png` |
| two OS processes over TCP, matching state + render digests | `out/live_sync.png` |
| wire format v1: 12-byte headers, universe record validated | `holo/crdt.py`, SDK §5 |

**The framing the sweep handed us.** arXiv:2605.19373 reports that all
26 neural-network merge strategies tested fail commutativity,
associativity, or idempotency. Superposition gets the first two for
free and fails only the third — and that is exactly the gap these
recipes close. Different object, same shape; cite as nearest neighbour
and use the contrast.

**Tests:** `test_crdt.py`, `test_orset.py`, `test_live_sync.py`.

---

## 6. The law, and where it bends

Not a fourth claim — the spine that runs through the three, and the
section that turns a demo into engineering.

`sigma ~ sqrt(N R / 2d)`: capacity is a signal-to-noise budget, never a
table size; structures fail *soft*. Every demo prints the measured
curve against prediction.

Then the honest deviations, all measured:

- **Dense scenes correlate their noise** (~1.5-3x over the i.i.d. law).
  Coherent, not Monte-Carlo — doubling `d` bought 2-4% for +600 MB. A
  negative result with a diagnosis.
- **Per-cell ridge fitting is sampling-limited** at real capture
  density (0.72/0.53 vs forward's 0.52/0.38), after a spectral prior
  rescued it from 12x worse. Placed against Fourier-extension theory
  (Adcock; frames + regularization), which says the zero-sample route
  needs Tikhonov/TSVD from day one. `results/real_fit.png`
- **The scale floor is radiometric.** Clamping a thin axis while
  holding peak amplitude fixed inflates scene mass 11x on Red Rock.
  Related: measuring "what the floor costs" by point sampling is
  ill-posed — needles thinner than a pixel are invisible to point
  samples (footprint integration, §7).
- **Storage rate-distortion.** Phase-only codes floor amplitude fields
  at ~0.24 rel RMSE at any bit depth, which is why magnitude-preserving
  and gamma-companded codecs exist. `out/codec_curve.png`

## 7. Related work and adopted corrections

Carry the 2026-08-27 sweep's conclusions as *positioning*, not defence:

- **qFHRR** (arXiv:2604.25939) publishes quantized-phase FHRR at 3-4
  bits with integer-only binding — our phase codec from the hardware
  side. Cite as the representation; our contribution is the measured
  boundary where it fails (their bundling projects back to the unit
  circle, discarding the magnitude amplitude fields need).
- **Mip-Splatting** (CVPR 2024) and **Analytic-Splatting** (ECCV 2024)
  solve the sampling/anti-aliasing problem; our scale floor is an
  ad-hoc version of the former's 3D smoothing filter. Adopt and cite.
- **SSP shift property** and Voelker et al. (Neural Computation, 2021):
  motion as a phase ramp has precedent. Do not claim it.
- **VSA-OGM**, **HyperSpace**, **GVKF**, **CryoSplat/R2-Gaussian**,
  3DGS compression survey — nearest neighbours per strand.

## 8. Limitations

Occlusion and alpha compositing are outside linear superposition.
Determinism is semantic, not bitwise (~1 ulp). Storage is large in
absolute terms — cell bundles are hundreds of MB where the source PLY
is tens. Rotation is not a phase ramp. Clustered/banded readout needs
structure in the data (spatial locality for scenes) to pay off.

---

## Deliberately excluded, and why

- **Near-enough dispatch.** The most *surprising* result we hold — the
  same law and the same banding medicine transfer from geometric scenes
  to rule tables (0.98 accuracy at 43x less compute). Different
  audience; including it makes two papers stapled together. Cite in one
  sentence as evidence of generality, publish separately.
- **Codec strand as a headline.** qFHRR preempted the representation;
  our curve is supporting measurement (§6).
- **Dynamic holograms.** Mechanism has published precedent; the
  capture-scale combination is not finished.
- **Ridge-fitted holograms** (~70x better held-out than forward). Good,
  but a fourth idea competing for the same space, and its real-capture
  result is negative — it belongs in §6 as a limit, not as a claim.

## Before submission

1. ~~**Baseline table**~~ — DONE: [docs/baselines.md](docs/baselines.md).
   The result is a loss on that axis and the paper should say so
   plainly: a bundle is ~400x larger and ~50x less accurate at
   reproducing the field than SOG. Fold it into the introduction as
   the cost side of the trade — the three claims are about
   capabilities the codecs do not sell — and keep the two things it
   does say for us: HM-8 is free compression (4x smaller at slightly
   BETTER error, the shrinkage denoiser at capture scale), and the
   families scale differently (per-splat with content, bundles with
   occupied volume; the gap narrows 4x per decade of density but does
   not close).
2. ~~**Adopt the citations**~~ — DONE: §9 below carries the reference
   list, and each claim names the work that bounds it, so the draft is
   written from the corrected position rather than defending it in
   rebuttal.
3. ~~**Venue**~~ — DECIDED: §10.
4. Re-run the sweep at draft-freeze; log the delta with its date. That
   is the only item left, and it is a half-hour before submission.

---

## 9. References, and what each one bounds

Grouped by the job it does in the draft. The second column is the
sentence it forces us to write — adopting a citation means conceding
its ground up front.

**Foundations we build on**

| work | what it gives us |
|---|---|
| Plate, *Holographic Reduced Representations* | FHRR: bind/unbind/bundle, the algebra everything else is written in |
| Kanerva, *Sparse Distributed Memory*; *Hyperdimensional Computing* (2009) | the capacity-as-SNR framing, and the ancestor of the whole family |
| Frady, Kleyko, Kymn, Olshausen, Sommer, *Computing on Functions Using Randomized Vector Representations* (arXiv:2109.03429) | Vector Function Architectures — the FPE/Bochner bridge our field encoder instantiates |
| Rahimi & Recht, *Random Features for Large-Scale Kernel Machines* | random Fourier features; the reason an inner product equals a Gaussian kernel |
| Komer & Eliasmith, Spatial Semantic Pointers | continuous space in this algebra, and the shift property |
| Kleyko et al., HDC/VSA surveys I & II (arXiv:2111.06077, arXiv:2112.15424) | the field's own account of what is settled |

**Work that bounds our claims** — each of these removes something we
might otherwise have claimed

| work | the sentence it forces |
|---|---|
| **qFHRR** (arXiv:2604.25939) | quantized-phase FHRR at 3-4 bits is *published*; our phase codec is that representation reached from storage rather than hardware. We claim the measured boundary — their bundling projects back to the unit circle and drops the magnitude amplitude fields need — not the representation. |
| **Mip-Splatting** (arXiv:2311.16493, CVPR 2024) | our scale floor is an ad-hoc version of their 3D smoothing filter, which derives the same constraint from the input views' sampling rate. Adopt; do not re-derive. |
| **Analytic-Splatting** (arXiv:2403.11056, ECCV 2024) | pixel-area integration is theirs; our `footprint_blur` is the pre-projection form of it. Cite when the footprint referee appears. |
| **Voelker et al.**, *Simulating and Predicting Dynamical Systems With SSPs* (Neural Computation 33(8), 2021) | multi-object trajectories via the shift property already exist. `translate_bundle` is the shift theorem — we do not claim motion-as-phase-ramp. |
| **CRDTs for NN Model Merging** (arXiv:2605.19373) | nearest neighbour to our OR-Set layer; also our best contrast — all 26 of their merge strategies fail commutativity/associativity/idempotency, where superposition fails only the third. |
| **Adcock**, *On the numerical stability of Fourier extensions* (arXiv:1206.4111) + Adcock-Huybrechs frames theory | names the per-cell fit's failure as the Fourier-extension/plunge-region problem and says regularized least squares is the route. Our negative result lands in known theory rather than in the dark. |

**Nearest neighbours to position against**

| work | why it is not us |
|---|---|
| **VSA-OGM** (arXiv:2408.09066) | closest relative — SSP occupancy fields with sparse updates — but scalar occupancy only: no per-splat covariance, color, rendering, learning, or replication |
| **GVKF** (arXiv:2411.01853) | validates "splatting is a kernel mixture" from the graphics side, but keeps per-Gaussian parametric storage |
| **CryoSplat** (arXiv:2508.04929), **R2-Gaussian** (arXiv:2405.20693) | closed-form Gaussian projection under the Fourier slice theorem — per primitive, not a whole view folded into one vector |
| **HyperSpace** (arXiv:2604.15113) | the nearest "VSA SDK" effort; no scene, splat, or replication content |
| **3DGS compression survey** (arXiv:2502.19457) | the family our baseline table loses to, and should be seen to lose to |

## 10. Venue: the neuro-symbolic / HDC-VSA community

**Decision: submit to the HDC/VSA community, not to graphics.** arXiv
first (cs.NE, cross-listed cs.CV), then a venue in that community —
*npj Unconventional Computing* published VSA-OGM, *Neural Computation*
published Voelker, and the field has an active workshop track. Journal
if we want the negative results to have room; workshop if we want the
conversation sooner.

**Why not graphics, given that half the evidence is captures.** Three
reasons, and the third is decisive:

1. Graphics venues judge novel-view synthesis on photometric metrics
   against held-out images (PSNR/SSIM/LPIPS on Mip-NeRF 360, Tanks &
   Temples). **We do not do novel-view synthesis** — we do emission /
   X-ray projection, evaluated against an analytic referee. Our
   evidence answers a question that community does not ask.
2. Our baseline table is a loss on precisely the axis graphics
   reviewers weigh most (quality per byte). In an HDC venue that table
   is honest positioning; in a graphics venue it is the headline
   result, and the wrong one.
3. **All three claims are algebraic claims.** "A scene is a bundle,"
   "a view is a bundle," "merging is addition" — the contribution is
   what the algebra *buys*, and the capacity law is an HDC contract.
   The splats are the demanding test case that proves it survives real
   data, not the subject.

**What the decision changes in the draft.** Lead with the algebra and
introduce splats as the stress test, not the other way round: the
title and abstract should name the representation, not the capture.
Assume FHRR, FPE and cleanup as known and spend the space on what is
new — the scene encoder, the folded view, the CRDT recipes, and the
capacity budget that ties them. Keep every graphics figure: this
audience rarely sees a claim carried by 682k-splat captures and an
exact analytic referee, and that is a large part of why the paper is
worth reading. Address the graphics community in one paragraph of
related work rather than in the framing — say plainly that
rasterization is better at rasterizing, and point at
[docs/baselines.md](docs/baselines.md).

**Fallback.** If a reviewer's core objection is "why not just use
3DGS + SOG?", the answer is already in the paper and does not require
a rewrite — it is the capabilities list in §Baseline, and the
strongest single sentence for it is the CRDT contrast: superposition
gets two of three merge axioms for free, where 26 tuned strategies get
none.
