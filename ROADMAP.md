# Roadmap

How work gets picked here: items below are claimed in `SDK.md`'s
running log before anyone starts (see `CONTRIBUTING.md` for the
working agreements). Findings — positive or negative — land in the log
and in `docs/`, with figures. Nothing unproven enters the SDK surface.

## Shipped

- **0.1** — the charter executed end to end: FHRR core + data
  structures, splat fields (FPE / spectral / bands / cells / attributes
  / color), learning (ridge-fit holograms), closed-form X-ray
  rendering, CRDT replication with observed-remove deletion and live
  TCP sync, tagged wire/storage formats, GPU backend through every
  eval, docs-per-technique, gated CI. Tagged `v0.1.0`.
- **0.2 (in progress, findings in SDK.md log)** — codec
  rate-distortion (HP/HM/HG rules, measured on real captures),
  real-scene turntable, per-cell fitting of real scenes (spectral
  prior; honest sampling limit), Fourier-extension placement of the
  analytic-projection direction, fine-band reach split promoted to
  the capture default (reach follows the band cap; 33-44% slice-error
  cut on the saguaro at 1.5x storage — raising d instead bought 2-4%:
  dense-scene residual is coherent, not Monte-Carlo), measured codec
  rules on real bundles (HG-8 faithful; HM-4 an accidental denoiser),
  cross-hardware verified kernel bench (RTX 5090 ~9x M1 Max, >200x
  CPU, float64 checksums to 2.5e-8; the TF32 platform gotcha found and
  neutralized), native capture ingestion for iPhone LiDAR point
  clouds and raw 3DGS Gaussian PLYs (Scaniverse Red Rock, 547k splats:
  best Gaussian-capture slice numbers yet, 19%/22% — raw `.ply` is now
  the recommended interchange), and the first application-layer
  technique: near-enough dispatch (`holo/dispatch.py` — rules as
  similarity, abstention as policy, banded rulebooks; the capacity
  law and the banding medicine transfer from scenes to rule tables).

## 0.3 — what landed, and what is left

Tracked under the [0.3 milestone](https://github.com/squatch-stack/hdc-holo/milestone/1);
six of eight closed. Claims still go through SDK.md's log first.

**Closed.**

- **Principled shrinkage denoiser** ([#1](https://github.com/squatch-stack/hdc-holo/issues/1)) —
  soft shrinkage at the 25th magnitude percentile, 0.298/0.203 against an
  unshrunk 0.350/0.213, composing with HG-8 at 0.25x bytes.
- **Dense-scene coherent error** ([#3](https://github.com/squatch-stack/hdc-holo/issues/3)) —
  answered: the dense residual is NOT variance. Orthogonal coupling
  reduces variance and nothing else, and removes +18.4% on sparse saguaro
  against +1.9% on dense train — that split is the proof. Shrinkage takes
  +40.1% there and the analytic projection +60.7%.
- **Real-scene collaborative editing** ([#4](https://github.com/squatch-stack/hdc-holo/issues/4)) —
  cell-keyed epochs (one stroke, one tombstone; 10.0 MB of accumulator
  against 655.2 MB), HG-8 epoch blobs on the wire (84.0 MB -> 21.0 MB),
  and the memory pact as code. Found three silent failures on the way:
  a trimmed-history delta that leaves a peer holding nothing, generation
  loss in `compact()`, and codecs corrupting above 16 bits.
- **Publication pass** ([#6](https://github.com/squatch-stack/hdc-holo/issues/6)) —
  arXiv note written and 0.3.0 on PyPI. Submission is
  [#59](https://github.com/squatch-stack/hdc-holo/issues/59), owner action.

**Open.**

- **Analytic L2 projection for per-cell fits** ([#2](https://github.com/squatch-stack/hdc-holo/issues/2)) —
  works, and deliberately unpromoted. It is the largest lever measured
  (+59.3% on saguaro at keep=0.55, against the shipped keep=0.25's
  +38.0%), but the truncation is a knife edge: one step further, at 0.70,
  it is 37x WORSE than not projecting at all, and the edge moves with the
  capture. A fixed default is either conservative or catastrophic, and
  the failure is silent in the decode. **The path to promotion is
  automatic truncation selection**, and the signal it would use already
  exists — the solved/forward norm ratio separates working from broken by
  18x, and `run_projection_pipeline.py` checks it after every band.
  Details in [docs/fit.md](docs/fit.md).
- **Occlusion research spike** ([#5](https://github.com/squatch-stack/hdc-holo/issues/5)) —
  untouched, and unrelated to everything else in this milestone. Alpha
  compositing is outside linear superposition (a documented failure
  mode); the spike is to scope what a hybrid holographic-density plus
  classical-compositing pass would look like. The order-independent
  transparency literature is the right family: arXiv:2605.13855
  (SparseOIT), arXiv:2605.25345 (depth peeling for Gaussian surfels),
  arXiv:2305.10197 (learned OIT). **Consider moving to 0.4** so 0.3 can
  close on #2 alone.

- **Package name decision** ([#7](https://github.com/squatch-stack/hdc-holo/issues/7)) — `holo` is the charter's working name;
  rename happens once, before PyPI.
- **Dynamic holograms** (no issue yet; prototype
  `examples/dynamic_prototype.py`) — animation as algebra inside the bundle.
  Measured: rigid motion of an object's sub-bundle is one phase ramp
  (`translate_bundle`) — algebraically exact (bundle gap ~5e-7) and
  O(d) regardless of splat count (0.6 ms vs 200 ms re-encode at 20k
  splats, ~350x; the ramp never grows with N); binding frames with
  time codewords sums a whole animation into ONE vector where playback
  and `where_is(object, t)` are unbindings — median localization ~0.7
  splat scales across 5 objects x 10 query times, including BETWEEN
  stored frames, from a 128 KB vector. Stored-frame noise grows
  slower than the sqrt(T) guide (frame content is correlated);
  undersampled motion ghosts into motion blur rather than failing.
  Prior art, per the 2026-08-27 sweep: the shift property itself is
  standard in the Spatial Semantic Pointer literature, and Voelker et
  al. (Neural Computation 2021) already simulate multi-object
  trajectories in this algebra — `translate_bundle` is that shift
  theorem, not a new idea, and this lane must be claimed narrowly:
  the combination at capture scale, the one-vector animation with
  time-bound frames, `where_is(object, t)`, CRDT-merged animated
  scenes, and the cost model. Open: rotation is NOT a phase ramp
  (remixes frequencies — needs its own idea), object identity via
  OR-Set epochs, promotion to a module + docs page per the charter
  bar.
- **Interop** (LANDED) — `save_ply` + `save_spz` export, and
  `examples/run_viewer.py` renders any capture in real time through Spark.
  Measured: SPZ v2 is 16.4x smaller at 0.04-0.07% field error,
  because raw captures are already u8-quantized inside float32
  containers (SDK.md log). SOG export landed too (`holo/sog.py`): 19x
  smaller than the source PLY, smaller than SPZ, and the only one of
  our formats that carries higher-order SH — through a palette that
  keeps ~40% of it (SH is intrinsically hard to vector-quantize; 8x
  the palette buys 8%). Still open: SPZ v3/v4 parser bump (the
  ecosystem moved two versions); SOG's LOD/streamed variants; a
  hosted viewer link from the README.
- **Near-enough runtime** (no issue yet) — grow `holo/dispatch.py`
  from router to runtime: order-sensitive conditions (permuted
  position tags), record-valued conditions (role-filler payloads as
  rule state), and CRDT-merged rulebooks over the sync layer — the
  banded bundles already add; wire them through `holo/crdt.py` and
  measure merge behavior under concurrent rule edits.

## Support this work

Donation/sponsorship hooks are being set up (`.github/FUNDING.yml`
lands once accounts exist). If the roadmap above is useful to you —
research, robotics mapping, collaborative 3D, edge/analog hardware —
sponsoring specific items or funding issues is the most direct signal
for what gets built next.
