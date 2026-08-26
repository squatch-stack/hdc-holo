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
| Holographic data structures (map, sketch, record, sequence, ngram, graph, FSM, SDM) | `holo/*.py` | per-structure capacity tables in `run_demos.py`; tests |
| Splat fields via fractional power encoding | `holo/field.py` | `out/field_comparison.png`; error ~ `1/sqrt(d)` test |
| Spectral encoder (per-splat anisotropic covariance, one codebook) | `holo/spectral.py` | capacity curves fit `d^-0.50` exactly (`results/capacity_curve.png`) |
| Mixture-of-Gaussians codebooks (multi-scale scenes) | `holo/spectral.py`, `run_mog.py` | 3-10x noise cut, penalty 16-33x -> 2.4-3.2x (`results/mog_penalty.png`) |
| Scale bands + spatial chunking | `holo/spatial.py`, `run_real_scene.py` | chunked beats global at equal d (test); locality = per-cell noise |
| Attribute/record payloads on splats (`what_is_at`, `where_is`) | `holo/attribute_field.py` | SNR cliff table; class-filter renders (`out/attribute_field.png`) |
| CRDT replication incl. attributed scenes + records (Loro) | `holo/crdt.py`, `live_sync.py` | convergence tests; bit-identical merges; TCP demo |
| Ridge-fitting holograms (bundle = RFF weight vector) | `holo/fit.py` | fitted beats forward-encoded ~70x held-out (`out/fit_photo.png`) |
| Closed-form X-ray rendering (projection-slice) | `holo/render.py`, `run_real_scene.py` | trefoil 5-7% RMSE; real-scene renders vs analytic mip |
| Real-capture pipeline (.splat / .spz v2 loaders, crop, clamp) | `holo/capture.py` | byte-verified parsers (synthetic round-trip tests); `results/real_scan-tucson*.png`, `results/real_train*.png` |
| Phase-only / quantized storage (2x/8x/16x) | `holo/phase.py` | round-trip similarity tests |
| GPU backend (MLX/Metal, real cos/sin formulation, batched cell decode) | `holo/accel.py` | 37x encode / 106x decode kernels on M1 Max; real-scene holographic stages 13 min -> 24 s end-to-end; matches NumPy to 1e-7 |
| Observed-remove deletion (OR-Set tombstones, epoch/stroke undo, owner compaction) | `holo/orset.py` | phantom-vs-clean demo (`out/orset_undo.png`); idempotence/add-wins/compaction tests |

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
examples/       today's run_* scripts, curated
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
   `holo-demos` (`holo/cli.py`; `run_demos.py` is a shim — register new
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
   `run_real_scene.py` is a thin example driver.
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
  driver `run_fit_real.py`, figure `results/real_fit.png`): two results.
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
- **Real-scene turntable done** (peer lane): `run_turntable.py` orbits
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
- **HG-on-capture measured** (`run_codec_capture.py`, saguaro
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
  checksums for cross-hardware verification (run_gpu_bench.py drives;
  local M1 Max: MLX 25x numpy on the capture-shaped workload). Recipe
  bring-up on the studio's RTX 5090 surfaced a platform gotcha worthy of
  the Accelerate-bug shelf: **fp32 matmuls default to TF32 tensor cores
  on Ampere+/Blackwell**, silently costing ~2 orders of magnitude
  (1e-7 -> ~1e-5 relative) — the job script now disables TF32 on torch
  and documents CUPY_TF32; caveat recorded in docs/backend.md. First
  5090 numbers (box's own sweep, TF32 off): d=8192 encode 18 ms, 31-44x
  its host CPU. The aligned three-way comparison (same scene.npz, same
  checksums) runs once the box-side recipe executes the submitted
  payload through the job file — cupy backend added for exactly that.
- Still queued: component-thresholding denoiser (new, unclaimed);
  dense-scene coherent error (see ROADMAP).
