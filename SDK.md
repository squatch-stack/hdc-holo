# Toward an SDK: proven techniques, packaged

This document is the charter for turning this research repo into a
distributable, documented SDK. It inventories what is *proven* (with the
evidence), what the SDK's shape should be, and what has to happen before
anything is called 1.0. It is a working agreement between the humans and
agents editing this repo — amend it in place.

## Why an SDK

The repo has crossed from "experiments" to "capabilities that compose":
real captures encode into holographic bundles, semantic queries run by
unbinding, scenes replicate over CRDTs, images render straight from
vectors, and the hot kernels run 37-106x faster on the GPU. Each of
those was validated the same way — a capacity law measured, a figure
rendered against exact ground truth, a test pinned inside capacity — so
the criterion for SDK inclusion is already operational: **a technique is
"proven" when it has (a) a quantitative comparison against ground truth
or theory, (b) a deterministic test, and (c) a documented failure mode.**

## The proven inventory

| Technique | Where | Evidence |
|---|---|---|
| FHRR algebra (bind/unbind/bundle/permute, hash-derived codewords) | `holo/fhrr.py` | test suite; crosstalk matches `sqrt(N/2d)` throughout |
| Holographic data structures (map, sketch, record, sequence, ngram, graph, FSM, SDM) | `holo/*.py` | per-structure capacity tables in `examples/run_demos.py`; tests |
| Splat fields via fractional power encoding | `holo/field.py` | `out/field_comparison.png`; error ~ `1/sqrt(d)` test |
| Spectral encoder (per-splat anisotropic covariance, one codebook) | `holo/spectral.py` | capacity curves fit `d^-0.50` exactly (`results/capacity_curve.png`) |
| Mixture-of-Gaussians codebooks (multi-scale scenes) | `holo/spectral.py`, `examples/run_mog.py` | 3-10x noise cut, penalty 16-33x -> 2.4-3.2x (`results/mog_penalty.png`) |
| Scale bands + spatial chunking | `holo/spatial.py`, `examples/run_real_scene.py` | chunked beats global at equal d (test); locality = per-cell noise |
| Attribute/record payloads on splats (`what_is_at`, `where_is`) | `holo/attribute_field.py` | SNR cliff table; class-filter renders (`out/attribute_field.png`) |
| CRDT replication incl. attributed scenes + records (Loro) | `holo/crdt.py`, `examples/live_sync.py` | convergence tests; bit-identical merges; TCP demo |
| Ridge-fitting holograms (bundle = RFF weight vector) | `holo/fit.py` | fitted beats forward-encoded ~70x held-out (`out/fit_photo.png`) |
| Closed-form X-ray rendering (projection-slice) | `holo/render.py`, `examples/run_real_scene.py` | trefoil 5-7% RMSE; real-scene renders vs analytic mip |
| Real-capture pipeline (.splat / .spz v2 loaders, crop, clamp) | `holo/capture.py` | byte-verified parsers (synthetic round-trip tests); `results/real_scan-tucson*.png`, `results/real_train*.png` |
| Phase-only / quantized storage (2x/8x/16x) | `holo/phase.py` | round-trip similarity tests |
| GPU backend (MLX/Metal, real cos/sin formulation, batched cell decode) | `holo/accel.py` | 37x encode / 106x decode kernels on M1 Max; real-scene holographic stages 13 min -> 24 s end-to-end; matches NumPy to 1e-7 |
| Observed-remove deletion (OR-Set tombstones, epoch/stroke undo, owner compaction) | `holo/orset.py` | phantom-vs-clean demo (`out/orset_undo.png`); idempotence/add-wins/compaction tests |
| Capture export + real-time viewing (`save_ply` / `save_spz` / `save_sog`, Spark viewer) | `holo/capture.py`, `holo/sog.py`, `examples/viewer`, `examples/run_viewer.py` | lossless PLY round trip and SPZ-grid round trip (tests); 16.4x (SPZ) and 19x (SOG, keeps higher-order SH) compression, 0.04-0.07% field error vs the exact mixture (`docs/real-scenes.md`); SOG decoded back by the spec's own arithmetic (`tests/test_sog.py`) |
| Near-enough dispatch (similarity rule engine: matrix / one-vector bundle / banded+clustered routing, abstention as policy) | `holo/dispatch.py` | brittleness + banding-rescue + abstention tables (`hdc-demos dispatch`); capacity cliff and rescue pinned by test at N=d (`tests/test_dispatch.py`); `docs/dispatch.md` |

Documented failure modes that the SDK must carry in its docs, not bury:
crosstalk grows as `sqrt(N/2d)` and coherently worse in dense scenes;
alpha compositing / occlusion is outside linear superposition;
projections use only the spectrum slice perpendicular to the view
(renders need mip encodes and their own dimension budget); mixture
codebooks must reach the smallest axis scale present or importance
weights go heavy-tailed (stripes/herringbone); arithmetic retraction
(`HoloReplica.retract`) has PN-counter semantics — concurrent duplicate
retractions over-cancel into negative phantoms, which is what
`holo/orset.py`'s observed-remove tombstones exist to fix (removal as
set membership: idempotent, add-wins; item removal costs a re-encode
per read until the owner compacts) — so choose the deletion model:
arithmetic for single-owner retraction, OR-Set for multi-writer state;
concurrent same-key writes are multi-value. Determinism is semantic,
not bitwise: numpy's complex multiply reproduces only to ~1 ulp across
calls (alignment-dependent SIMD kernels), so recomputed vectors agree
to ~1e-7 — far under any decision threshold, but digests must hash blob
*bytes* or rounded values, and recomputed sums (`ORStore.merged`)
compare with `allclose`, never `array_equal`. Byte equality holds
exactly for state that travels as bytes (`HoloReplica.merged`).

## Target shape

```
holo/                       (working name; package rename happens once)
  core.py       FHRR algebra, spaces, codewords, cleanup memories
  encode.py     FPE + spectral encoders, mixture codebooks, bands/cells
  structures.py map/sketch/record/sequence/graph/fsm/... (thin, on core)
  scene.py      splat scenes: load (.splat/.spz), crop, clamp, attribute
  query.py      point/slice decode, what_is_at / where_is, records
  render.py     projection-slice views, mip encodes
  fit.py        ridge fitting (primal/dual), frequency bands
  sync.py       Loro replication (optional extra: holo[sync])
  storage.py    phase-only + quantized codecs
  backend.py    numpy | mlx dispatch (today's now holo/accel.py)
docs/           narrative docs: one page per technique = math, API,
                capacity budget, failure modes, the evidence figure
examples/       worked introductions (the algebra, dispatch, captures);
                run_*.py stay at the root as the evidence drivers
                that regenerate the docs figures
```

Principles:
- **Capacity is API.** Every constructor documents its noise budget in
  the docstring (`~sqrt(N R / 2d)` with the symbols defined) and every
  doc page shows the measured curve. No silent cliffs.
- **Determinism is a contract.** Codewords and frequency matrices are
  hash-derived from `(dim, seed)` — never sequential RNG state — so any
  replica reconstructs them; this is what makes sync coordination-free.
- **Backends are invisible.** Public APIs take/return NumPy; the
  cos/sin real formulation keeps every backend honest (no complex64 on
  Metal). `HDC_BACKEND` overrides; results match to float32 rounding.
- **Ground truth ships with the SDK.** Exact evaluators (mixtures, line
  integrals) stay in-tree; every technique remains re-verifiable.

## Path to 0.1

1. Freeze this inventory as the 0.1 surface (nothing unproven enters).
2. `pyproject.toml`, package layout above, `pip install -e .`; keep
   `hdc/` importable as a shim until callers migrate.
   *Done so far:* `pyproject.toml` with extras `[crdt]`, `[viz]`,
   `[gpu]`, `[dev]`; `pip install -e .` verified importable from
   anywhere; `__version__ = "0.1.0"`; console scripts `hdc-demos` /
   `holo-demos` (`holo/cli.py`; `examples/run_demos.py` is a shim — register new
   demos in `holo.cli.DEMOS`); `CHANGELOG.md` started. **The physical
   migration is done**: every implementation module lives in `holo/`
   (one file per concept, `git mv`'d with history) underneath the
   charter-named facade modules (core/encode/structures/scene/query/
   render/fit/sync/storage/backend), and `hdc/` is now purely the
   compatibility shim — edit `holo/*.py`, never the shims.
   `tests/test_holo_facade.py` pins facade == implementation == shim.
   The GPU backend is wired through the whole surface: every
   field/scene/render/fit eval dispatches via `accel.readout` (chunked
   scenes via `accel.cell_decode`), MLX/Metal when present with an
   identical-formulation NumPy fallback (~65x at render scale, float32
   agreement; `tests/test_accel.py`).
   *Done — migration complete:* the last two research strands are in
   the package: `holo/spectral.py` (spectral encoder + mixture
   codebooks; `hdc_splat.py` is now a shim) and `holo/capture.py`
   (.splat/.spz loaders with a `parse_spz` seam, mass-centered crop,
   banded cells, slice/X-ray decode + exact referees), exported via
   `holo.encode` / `holo.scene` and flat off `holo`;
   `examples/run_real_scene.py` is a thin example driver.
   `tests/test_spectral.py` + `tests/test_capture.py` cover them (the
   loader tests author both formats synthetically — CI needs no data
   files); docs pages moved off "pre-migration" wording; the stale
   pre-fix train figure was retired from evidence and repurposed as
   the labeled herringbone failure-mode exhibit
   (`results/failure_herringbone.png`, cited in `docs/spectral.md`);
   fresh `results/real_train*.png` ran through the fixed pipeline
   (slice alpha rel err 1.04 top-down / 0.58 side on this denser
   scene — the sqrt(local/2d) law, a candidate for fine-band cell or
   dimension tuning before 0.2).
3. Move module docstrings into `docs/` pages (math + budget + figure),
   one page per technique; docstrings keep the summary + budget.
   *Done:* `docs/` holds one page per inventory row (index in
   `docs/README.md`) — math, API snippet, measured budget, failure
   modes, and the evidence figures/tests for each, including the
   pre-migration strands (`docs/spectral.md`, `docs/real-scenes.md`).
   Docstrings were left rich rather than slimmed — the in-code
   narrative has earned its keep; trim opportunistically, never let the
   two contradict (pages cite measurements, docstrings the mechanism).
4. CI: pytest on macOS (mlx) and Linux (numpy-only) — the backend must
   degrade gracefully.
   *Done:* `.github/workflows/ci.yml`. Linux (3.9 + 3.12) installs
   `[dev]` only — no MLX exists there, so the job IS the fallback
   proof, with an explicit `backend_name() == 'numpy'` assert and a
   demo smoke run. macOS (arm64) installs `[dev,gpu]`, probes for a
   usable Metal device at runtime (VMs vary), runs the suite on the
   best backend, then ALWAYS re-runs it under `HDC_BACKEND=numpy`.
   Both paths verified green locally before committing.
5. Version the wire/storage formats: bundle blobs (`container::peer`),
   phase-quantized codes, and the label registry get format tags now,
   before anything external depends on them.
   *Done — wire format v1 (`WIRE_VERSION` in `holo/crdt.py`):*
   - Every bundle blob (G-Counter shard AND observed-remove epoch)
     carries a 12-byte header: magic `HB`, version u8, dtype u8
     (0 = complex64), channels u16, dim u32, reserved u16, then the
     payload. `pack_bundle`/`unpack_bundle` are the only blob codec;
     untagged or foreign bytes are refused with a clear error.
   - Every doc carries a format record (`format` map, key `holo`):
     JSON `{wire, dim, seed}` — the universe. Replicas validate it on
     every flush and after every `apply()`, refusing mismatches loudly
     instead of decoding garbage. The record is written in the SAME
     commit as the doc's first content ops (an empty flush creates no
     ops — otherwise a pre-write `version()` snapshot would exclude it
     from every delta and peers would queue on missing causal deps;
     the live-sync tests caught exactly that).
   - Phase-quantized codes get a storage envelope (`STORAGE_VERSION`
     in `holo/phase.py`): magic `HP`, version u8, bits u8, dim u32,
     then codes — nibble-packed two-per-byte at <= 4 bits, so a 4-bit
     codeword really is dim/2 bytes. `pack`/`unpack` on `holo.storage`.
   - Magnitude-preserving codec, magic `HM` (same storage version):
     version u8, bits u8, dim u32, scale f32, then scaled
     signed-integer re/im at `bits` per component (nibble-packed at
     <= 4). `pack_complex` writes it; `unpack` dispatches on magic.
     Motivated by the codec rate-distortion curve: phase-only codes
     floor amplitude fields at ~0.24 rel RMSE; 8-bit `HM` matches
     complex64 field fidelity at 4x fewer bytes. Use HP for
     codewords/symbol stores, HM for field bundles.
   - The key schemes (`container::peer`, `name/peer.epoch[/i]`,
     `namespace::label`) are part of v1; changing any layout bumps the
     version constant.

Non-goals for 0.1: alpha-composited rendering (outside superposition),
Python < 3.9, CUDA (the backend seam is where it would go later).

## 0.2 findings (running log)

- **Per-cell ridge fitting of real scenes** (`fit_cells` in
  `holo/capture.py`, `ridge_cell_fit(prior=)` in `holo/accel.py`,
  driver `examples/run_fit_real.py`, figure `results/real_fit.png`): two results.
  (1) Naive ridge under mixture codebooks FAILS (12x worse than
  forward) — minimum-norm regression spreads energy into the codebook's
  finest frequencies and memorizes samples as kernel-width bumps; a
  spectral prior `exp(-1/2 (0.35 cap)^2 |w|^2)` fixes it, tying forward
  encoding on sparse scenes (0.037 vs 0.036). (2) At real capture
  density the fit is SAMPLING-LIMITED and loses to forward (saguaro:
  0.72/0.53 vs 0.52/0.38): hundreds of floor-scale splats per cell need
  coverage at their own kernel width — tens of thousands of samples —
  beyond the dual solve. Open direction: the analytic L2 projection
  (region Gram = separable sincs, closed form, zero samples).
- **Codec split** (peer lane, step-5 note above): HP for symbols, HM
  for fields; 8-bit HM = complex64 fidelity at 4x fewer bytes.
- **Analytic projection has prior art** (peer lane, related-work 0.2
  delta): the box-region Gram of complex exponentials IS the classical
  Fourier extension problem — severely ill-conditioned (Slepian plunge
  region) but PROVEN stable under regularized least squares to
  ~sqrt(eps) (Adcock arXiv:1206.4111; Adcock-Huybrechs frames theory;
  AZ algorithm for fast solves). The empirical spectral prior maps to
  known frame-regularization theory; the zero-sample Gram route should
  start with truncated-SVD/Tikhonov, and docs/fit.md +
  docs/related-work.md carry the full placement.
- **Real-scene turntable done** (peer lane): `examples/run_turntable.py` orbits
  scan-tucson (519k splats) entirely from 135 cell bundles (135 MB) —
  36 frames at 5.2 s/frame, no geometry at render time, gamma-0.5
  tone map (X-ray ground-plane integrals crush linear exposure), high
  orbit default (a planar desert reads poorly edge-on). Honest
  assessment: evidence, not eye candy — an X-ray of a flat scene is
  structurally faithful but visually muted; emission-with-occlusion is
  the display-quality gap and stays out of scope (non-linear).
  Figures: `results/real_turntable-scan-tucson.gif` / `.png`.
- **Infrastructure hardening + docs refresh** (user-directed; completion
  note — claimed files released): CI now gates the macOS job to release
  tags/manual dispatch (10x billing on private repos), pins all actions
  to commit SHAs, adds least-privilege permissions, concurrency
  cancellation, timeouts, and a full-history gitleaks job
  (`.gitleaks.toml`). Docs refreshed for humans: `docs/README.md` is a
  navigational hub (Mermaid architecture map, grouped pages, reading
  paths), every page carries a breadcrumb and its evidence figures
  EMBEDDED inline, and diagrams use the GitHub-safe Mermaid subset
  (sequence diagrams for sync/orset, pipeline flowcharts, packet-beta
  byte layouts for HP/HM). CONTRIBUTING.md + SECURITY.md added.
  OPEN (owner decision): LICENSE — required before going public.
- **HG codec** (peer lane): gamma-companded polar coding
  (`pack_polar`, new magic — v0.1.0 froze HM's layout). Component-level
  win over linear HM at 4 bits (test-pinned); field-task TIE at demo
  dimensions (crosstalk dominates quantization there). Earns its keep
  on wide-dynamic-range bundles; open item: measure on real-capture
  premultiplied channels.
- **HG-on-capture measured** (`examples/run_codec_capture.py`, saguaro
  fine-band cells, d=8192, 4ch, both evidence slices): capture bundles
  ARE the wide-dynamic-range case — |S| spans 987x (p99.9/p50). At 8
  bits HG is the faithful codec: round-trip drift 0.013 vs HM's 0.124
  at identical bytes (0.25x of complex64), and vs-GT errors match
  complex64 exactly. The surprise is 4-bit HM: it BEATS the
  uncompressed bundle against ground truth (0.502/0.342 vs 0.522/0.379
  top-down/side) at 8x fewer bytes — max-scaling quantization zeroes
  the small components, which on a forward bundle are mostly crosstalk
  noise, i.e. the codec is an accidental shrinkage denoiser; HG
  preserves those components faithfully (drift 0.17 at 4 bits) and so
  keeps the noise. Rule refined: HG when fidelity TO THE BUNDLE matters
  (fitted holograms, sync payloads mid-edit); HM-4 when the bundle is a
  forward-encoded scene and GT fidelity per byte is the goal. NEW open
  item this spawns: deliberate component thresholding (soft/hard, at
  the crosstalk noise level) as a post-encode denoiser — if accidental
  truncation denoises, principled shrinkage should do better.
- **Fine-band reach split promoted to default** (capture lane): reach
  follows the band cap, and floor-scale splats — most of any real
  capture — were dragging 3x0.008 of reach they don't need. Splitting
  the fine band at 0.004 (same cell size, same d) cuts the in-reach
  crosstalk volume ~3x: saguaro slices 0.522/0.379 -> 0.350/0.213
  (33-44% error cut) at 1.5x storage; train 1.044/0.580 -> 0.976/0.499.
  NEGATIVE also logged: doubling the xfine band's d to 16384 bought
  only 2-4% on train for +600 MB — the dense ground plane's residual
  error is coherent, not Monte-Carlo, so more dimension cannot wash it
  out (connects to the shrinkage-denoiser item). BANDS default updated
  in holo/capture.py with the rationale; evidence figures regenerated;
  empty-band edge case fixed in decode_slice/render_xray. Historical
  note: earlier log entries and the fit experiment quote the pre-split
  baseline (0.52/0.38) — those numbers were correct at their date.
- **Cross-hardware bench + the TF32 finding** (capture/spectral lane +
  the job-runner session): bench/holo_bench_job.py runs the two dominant
  kernels from ONE file on torch-cuda/cupy/mlx/numpy with float64
  checksums for cross-hardware verification (examples/run_gpu_bench.py drives;
  local M1 Max: MLX 25x numpy on the capture-shaped workload). Recipe
  bring-up on the studio's RTX 5090 surfaced a platform gotcha worthy of
  the Accelerate-bug shelf: **fp32 matmuls default to TF32 tensor cores
  on Ampere+/Blackwell**, silently costing ~2 orders of magnitude
  (1e-7 -> ~1e-5 relative) — the job script now disables TF32 on torch
  and documents CUPY_TF32; caveat recorded in docs/backend.md. First
  5090 numbers (box's own sweep, TF32 off): d=8192 encode 18 ms, 31-44x
  its host CPU. VERIFIED three-way table (identical scene.npz, float64
  checksums agree to 2.5e-8 across hardware): RTX 5090 (cupy, TF32
  off) encode 0.20 s / readout 0.17 s; M1 Max (mlx) 0.85 / 2.4; CPU
  numpy ~42 / ~42 on an idle machine — the 5090 lands ~9x the M1 Max
  and >200x CPU on the capture-shaped workload. The CUDA seam the
  roadmap left open is now measured, remote, and sandboxed.
- **5090 productionization** (box session, published over the gpugate
  reports channel — ids 20260826-ed92e7 / 20260826-b978b2): the
  real-scene pipeline is now **6.92 s** end to end on the RTX 5090,
  from ~22 min upstream NumPy validate (CUDA validate 118 s ->
  production mode 13.1 s — skipping ground truth, the single biggest
  lever and not an optimisation at all -> binned decode_slice 8.25 s
  -> fused cell_decode 6.92 s). The instructive part is what lost:
  TF32 was strictly dominated (no wall win — the GEMM is only ~30% of
  the kernel — and slightly worse slice error), and a fused readout
  kernel won 5.54x on its kernel but ~0 on the pipeline, because the
  actual bottleneck was decode_slice's cell_mask scan: O(cells x
  points) on the CPU, 131.5M distance tests per Tucson slice, 82% of
  the call. Binning per the cuFINUFFT / 3DGS-rasterizer pattern
  (reach <= cell size, so 27 candidate cells instead of 2,621) is 19x
  on that step, with the binned pair set verified identical to
  cell_mask across all four bands. cell_decode fusion pays only 1.9x
  vs the readout's 5.5x for a structural reason worth remembering:
  each point sits in ~10 cells, the unfused path reuses its trig
  planes across all of them, and the fused kernel recomputes per pair
  — it wins on memory bandwidth alone while doing ~10x the trig; the
  proposed point-tile inversion that would recover both is estimated,
  not measured. Accuracy retraction recorded: "fused is 8x more
  accurate" was an all-positive-weights artifact — on zero-mean real
  payloads both paths land at 1.48e-05. Nothing upstream was edited
  (box-side import-time patches, holo_cuda.py / holo_bin.py); suite
  56 passed / 4 skipped, cross-backend checksums 4.5e-8 / 5.1e-8.
- **Facade binding gotcha** (surfaced by the box's patching route,
  caught by tests/test_structure of the shims): holo/backend.py and
  the hdc/* shims bind accel's function OBJECTS at import
  (`from .accel import readout, ...`), so replacing
  `holo.accel.readout` at runtime leaves every facade on the original
  — the GPU patch silently missed all facade-routed calls until the
  shim-resolution test failed. Filed as a 0.3 issue: either facades
  delegate at call time or the patch-before-import contract gets
  documented.
- **Near-enough dispatch — the first application-layer technique**
  (`holo/dispatch.py`, `tests/test_dispatch.py`, `docs/dispatch.md`):
  a rule engine where conditions are trigram-profile hypervectors,
  dispatch is similarity, and the threshold is POLICY (below it the
  engine abstains — inexpressible in a Boolean if-table). Three
  engines on the existing algebra: matrix (cosine argmax, O(N)),
  bundle (whole rulebook = ONE vector, O(K) readouts, pays the law),
  banded (random or k-means-clustered bands + top-r centroid routing).
  Measured (demo, d=4096): matrix holds 0.97 accuracy at 30% character
  typos where exact keyword-AND scores 0.00; at 2048 rules clustered
  top-1 routing answers from one band bundle at 0.99. The finding that
  earns the entry: **the one law and its one medicine transfer
  unchanged from geometric scenes to rule tables** — flat bundles pay
  sqrt(N/2d) (cliff pinned by test at N=d=1024), and the same
  partition-plus-locality-routing that fixed dense scenes (cells)
  fixes rulebooks (topic bands); clustered routing needs topic
  structure exactly as cells need spatial locality. Because bundles
  add, banded rulebooks MERGE by the writer-sharded CRDT recipe with
  no coordination. Known gap carried in the docs: trigram profiles are
  order-blind past the trigram horizon — order-sensitive conditions
  need the sequence recipe's permuted position tags (0.3 candidate).
- **examples/ landed; raw PLY promoted to recommended interchange**:
  the charter's examples/ slot is now real — three worked
  introductions (hello_hologram: the algebra + the law failing soft;
  near_enough_rules: messy dispatch, abstention, and the 4096-rule
  banding experiment that reproduces the module docstring's 43x
  claim; splats_from_ply: capture -> bundles -> verified slice ->
  X-ray in ~60 lines) with run_*.py staying at the root as evidence
  drivers. Capture docs and drivers now lead with the raw Gaussian
  `.ply` (full per-splat covariance, nothing quantized — `.spz` is a
  lossy export of the same scene) and Red Rock is the flagship
  capture; README gained an art showcase (turntable GIF) since the
  repo went public with zero renders visible on its front page.
- **Capture orientation normalized — 3DGS PLY and .splat were loading
  upside down** (user-caught: "the red rock PLY is upside down").
  Empirical audit via alpha-weighted side silhouettes of all five
  in-house captures: raw Gaussian `.ply` (COLMAP-convention world,
  right-down-front) and antimatter15 `.splat` arrive y-DOWN; `.spz`
  (specified right-up-back — the official PLY->SPZ conversion applies
  the flip, which is why the saguaro was always upright) and ARKit
  LiDAR clouds (gravity-aligned) arrive y-up. Fix: `capture._to_y_up`
  rotates 180 deg about x on load — positions (x,-y,-z), quaternions
  premultiplied by (0,1,0,0) — so every loader now emits the same
  y-up world; a covariance-congruence test pins sigma' = F sigma F
  (proper rotation, not a mirror). Fallout: every published train
  figure had been upside down and the slices were too abstract for
  anyone to notice — the upright silhouette is unmistakably a
  locomotive; train and Red Rock evidence figures regenerated
  (upright Red Rock slices land at 19%/22% vs the inverted run's
  23%/23% — the flip moves the mass-mode slice planes; live quotes
  updated, dated log entries left as records of their day).
  Lesson for the shelf: orientation is format METADATA the pipeline
  silently assumed away; a cheap ground-truth silhouette per new
  format would have caught this on day one.
- **Claims registry + stale-claim gate** (facts lane; `holo/facts/`,
  `claims/registry.jsonl`, `docs/facts.md`, `tests/test_claims.py`):
  every measured number in the prose is now a registered claim — typed
  value, supersession chain, citation sites, and (where possible) a
  derivation pinning it to code/tree ground truth. `holo-facts check`
  warns at pre-commit (`.githooks/`, opt-in) and blocks in CI
  (`--strict` step in linux-numpy). First run against HEAD caught
  real drift: test-count claims at 56/72+/actual, the CONTRIBUTING
  "LICENSE: none chosen yet" bullet outliving the license decision,
  the ~1.5-2x vs ~1.5-3x coherent-inflation split between the docs
  hub and fields.md, and — mid-build — its own derivation flagging
  the suite count moving 84 -> 85 -> 94 as tests landed. The
  13-min-vs-22-min ambiguity resolved as two DIFFERENT measurements
  (holographic stages vs full NumPy validate), now two claims with
  disambiguating notes. Old numbers are first-class: SDK.md's running
  log is a dated-record zone, CHANGELOG sections are version-scoped,
  and supersession (never deletion) is how values change. Phase 2
  (claimed, facts lane) dogfoods retrieval: per-chunk trigram-profile
  MATRIX via `holo.dispatch.FastNGramProfiler`, HG-8 persistence —
  never a bundle: at ~900 chunks the flat-bundle crosstalk floor
  sqrt(N/2d) ~ 0.45 at d=2048, forbidden by our own law; fuzzy recall
  is WARN-only since trigram cosine cannot tell a corrected
  restatement from a stale one. Phases 3-4 (claimed): minimal MCP
  server (`holo-facts mcp`; extra `[facts]`), and a sibling
  knowledge-base repo indexed by the same checker.
- **Fuzzy claims layer shipped** (facts lane, phase 2 complete): the
  stale-claim system now dogfoods retrieval — 274 doc chunks as
  L2-normalized trigram-profile rows (`holo.dispatch.FastNGramProfiler`
  at d=2048; matrix, never a bundle, per the capacity math in
  docs/facts.md), persisted through HG-8 with a plaintext-free
  sha256 sidecar; `holo-facts index / search / calibrate`, and
  `check --fuzzy` probes superseded claims for PARAPHRASED
  restatements as WARN-only findings. Calibrated on the real corpus:
  signal min/median 0.295/0.392 vs character-scrambled noise p95
  0.101 — threshold 0.18 sits in the gap. Two negative results worth
  the shelf: word-SHUFFLED probes are not noise (trigram profiles
  ignore word order — shuffles scored 0.54 p95, above real signal;
  noise must scramble characters), and consecutive markdown bullets
  were merging into one mega-paragraph, which had silently masked a
  real bug — the SDK dated-record zone never matched (absolute vs
  relative path keys) and was only passing via accidental marker
  proximity. Both fixed, test-pinned; the measured limitation stays
  by design: trigram cosine scores the CORRECTED license line 0.21
  against the retracted claim — it cannot tell corrected from stale,
  so exact matching owns the gate and fuzzy only warns.
- **Facts MCP server** (facts lane, phase 3 complete): the registry
  and fuzzy corpus are now queryable by any MCP client —
  `holo-facts mcp` (stdio) serves `search_claims` (registry keyword
  ranking unioned with fuzzy chunk hits mapped back through cite
  files), `get_claim` (record + supersession chain + LIVE derivation
  + cite sites), and `search_kb` (same matrix search over a knowledge-base
  checkout; honest not-configured payload until phase 4 exists). The
  `mcp` dependency is the `[facts]` extra with a python>=3.10 marker
  — this Mac's system python is 3.9, so the tool logic lives in
  holo/facts/query.py (stdlib, tested on 3.9) and the REAL stdio
  handshake is proven on CI's 3.12 leg
  (tests/test_facts_mcp.py: spawn server, initialize, list tools,
  call search_claims). Registration one-liner and .mcp.json example
  in docs/facts.md.
- **Front-matter validation tier** (facts lane; completes the facts
  plan): knowledge-base-style surfaces (config front_matter_surfaces
  globs) carry a validated contract — flat front-matter must parse,
  claims: ids must exist in the registry (cite-sites made
  cross-repo), arxiv: ids must be well-formed, and swept: dates must
  parse (the dated-sweep convention made structural). front_matter()
  moved to normalize.py where parsing lives. A seeded external
  knowledge base exercised the tier end-to-end — search_kb returned
  topic pages with arXiv ids attached, and its CI ran the identical
  gate at a pinned revision (0 FAIL) — and was retired the same day
  by owner decision: kb_path defaults to null and search_kb answers
  honestly when unconfigured. The tier stays for any future KB.
- **Export, compression, and a real viewer** (capture lane; the
  interop question settled with measurements). What a "raw" capture
  actually is, measured on Red Rock: 681,748 splats x 62 float32 =
  161 MB, but higher-order SH occupies 73% of the file while carrying
  9.9% of the color energy, the normals are all zero (8 MB of
  nothing), and the capture app had ALREADY quantized before export —
  208 distinct scale values and 252 distinct alphas across 682k
  splats, u8 grids inside float32 containers. So a raw 3DGS PLY is a
  lossless CONTAINER, not a high-precision measurement, which is
  exactly why compression is nearly free here: `save_spz` (SPZ v2)
  writes 10.3 MB — 16.4x smaller — and the field it reconstructs
  differs from the original mixture by 0.04-0.07% relative (exact
  referee, 20k subsample); the two are indistinguishable side by side
  in the viewer. Honest losses, both test-pinned: SPZ v2 is DC-only
  (drops the 9.9% view-dependent term) and its rotation error grows
  as ~grid/w near 180-degree rotations (storing xyz, recovering w).
  `save_ply` round-trips losslessly to float32 rounding on the full
  capture. `examples/run_viewer.py` + `examples/viewer` render any of it in
  real time via Spark (CDN, nothing installed): the display
  complement to the X-ray evidence renderer, with occlusion. Two
  gotchas worth the shelf: (1) a URL-constructed SplatMesh added to
  the scene before its bytes arrive can sit UNSORTED — a black first
  frame — so load PackedSplats first, then add, then
  `await spark.update({scene, camera})`; (2) auto-framing must target
  the SUBJECT, not the bounding box: Red Rock's 5-95% span is 46
  units around a ~1-unit subject, so a box-framed camera stares at
  empty sky (same medicine as the mass-centered crop: median center,
  low quantile of distance).
- **SOG export** (capture lane; `holo/sog.py`, `tests/test_sog.py`,
  driver `examples/export_formats.py`): the delivery format that keeps
  what SPZ throws away. SOG v2 is a zip of lossless WebP images —
  16-bit log-space positions split across two, codebook-indexed scales
  and DC color, smallest-three quaternions, and a PALETTE of
  higher-order SH. Red Rock: **8.3 MB, 19x** smaller than the source
  PLY and smaller than SPZ's 10.3 MB, while carrying the
  view-dependent term SPZ drops entirely. Both compression mechanisms
  are ones this SDK already believes in: Morton ordering so
  neighbouring pixels hold nearby splats (locality — unsorted, the
  images are noise and WebP buys little), and codebooks (the
  rate-distortion trade holo/phase.py makes for bundles, applied one
  layer down to per-splat attributes). Honest fidelity: a 1024-entry
  SH palette lands at 0.62 relative error on the SH term — it keeps
  ~40% of what SPZ discards (SH-attributable color error 9.9% ->
  ~6.1%), and that is near the FORMAT's ceiling here, not a tuning
  miss: 8x the palette only reaches 0.53, so this capture's SH
  residual is intrinsically hard to vector-quantize (a negative result
  worth remembering before anyone reaches for a bigger palette).
  Reader gotcha for the shelf, found by bisection: the spec allows
  palettes to 65536 and our writer is spec-correct at any size (the
  test decodes it with the spec's own arithmetic), but **Spark 2.1.0
  renders 256 and 1024 and shows NOTHING at 2048** — a reader ceiling,
  so 1024 is the tested default. Verification chain worth copying:
  independent spec decoder in tests + visual confirmation in a
  third-party renderer, which is what isolated the reader limit.
- Still queued: component-thresholding denoiser (new, unclaimed);
  dense-scene coherent error (see ROADMAP); box lane: render_xray
  binning (still scans, 0.73 s), point-tile cell_decode fusion,
  cuFINUFFT type-3 prototype.
