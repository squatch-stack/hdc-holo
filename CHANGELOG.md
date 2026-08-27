# Changelog

## Unreleased

- Export bridge: `save_ply` (raw Gaussian PLY, INRIA layout, SH
  degree 0, ecosystem y-down convention on disk — lossless to float32
  rounding) and `save_spz` (SPZ v2, 16.4x smaller at 0.04-0.07% field
  error on Red Rock). Anything the pipeline loads, crops, or merges
  now flows into splat-transform / SuperSplat / Spark and the rest of
  the display chain.
- `run_viewer.py` + `examples/viewer`: real-time splat rendering with
  occlusion (Spark/three.js from CDN, nothing installed locally) for
  any `.ply`/`.spz`/`.splat` — the display complement to the X-ray
  evidence renderer. Subject-aware auto-framing (captures put a long
  background tail at ~40x the subject, so a bounding box aims the
  camera at empty sky).
- Near-enough dispatch (`holo/dispatch.py`, demo `hdc-demos dispatch`,
  page `docs/dispatch.md`): a rule engine with no Boolean gates —
  conditions as trigram-profile hypervectors, dispatch as similarity,
  abstention as policy. Matrix / one-vector-bundle / banded engines;
  k-means-clustered top-r routing holds 0.98 accuracy on a 4096-rule
  book at 43x less compute than the matrix engine. The capacity law
  and the banding medicine transfer unchanged from scenes to rules.
- `examples/`: three worked introductions (the algebra in five
  minutes; near-enough rules incl. the reproducible 4096-rule banding
  experiment; capture -> bundles -> verified slice -> X-ray).
- Raw Gaussian `.ply` (INRIA 3DGS layout) promoted to the recommended
  capture interchange — full per-splat covariance, nothing quantized
  away; Red Rock is the flagship capture. README front page now shows
  the work (turntable showcase); `run_turntable.py --crop` for
  captures that concentrate mass in a small core.
- Capture orientation fix: 3DGS `.ply` and `.splat` load in the
  COLMAP y-down convention and were rendering upside down; loaders
  now normalize every format to a y-up world (180° rotation about x,
  positions and per-splat rotations together — covariance congruence
  test-pinned). Train and Red Rock evidence figures regenerated.

## 0.2.1 — 2026-08-26

- Launch polish: duet brand mark (sasquatch + saguaro, one interference
  field), full-bleed avatar, banner, social cards; brand-only identity
  throughout. First Zenodo-archived release (DOI).

## 0.2.0 — 2026-08-26

- Licensed under FSL-1.1-Apache-2.0 (LICENSE.md): free for everything
  except competing commercial offerings; every release converts to
  Apache-2.0 after two years.

- Codec rate-distortion curve (`hdc-demos codec`): symbols vs fields
  SPLIT — phase-only codes win 16x for retrieval, floor amplitude
  fields at ~0.24 rel RMSE at any bit depth.
- Magnitude-preserving `HM` codec (`pack_complex`; `unpack` dispatches
  on magic): scaled int re/im — 8-bit matches complex64 field fidelity
  at 4x fewer bytes, 4-bit beats equal-byte phase codes.
- Turntable demo (`hdc-demos turntable`): a 520-splat multi-object
  colored scene orbited from one 768KB hologram, 72 frames at
  ~360 ms/frame, 5.6-6.5% RMSE vs analytic projections.
- docs/related-work.md: dated arXiv positioning sweep (anchors,
  nearest neighbors, open claims, standing re-sweep practice).

## 0.1.0 — 2026-08-26

First packaged release (`pip install -e .`, import `hdc`, console
script `hdc-demos`). Everything below predates the version number and
ships in it:

- FHRR core (complex64 phasors; exact unbind; hash-derived codewords)
  and nine classical structures rebuilt holographically: hash map,
  membership/frequency sketches, role-filler records, stack/sequence,
  n-gram profiles, graph, FSM, Kanerva SDM.
- Splat fields via fractional power encoding (Bochner/RFF): scalar,
  attribute-carrying (what_is_at / where_is), multi-band covariance,
  chunked 3-D cells, RGB color channels; spectral strand for per-splat
  anisotropic covariance; real-capture pipeline (.splat/.spz).
- Learning: `HoloRegressor` — exact ridge regression whose weight
  vector IS the bundle (primal/dual, multi-channel RHS).
- Rendering: closed-form ray integrals (projection-slice); a view is a
  folded bundle; color renders; rotating-view figures.
- Replication on Loro: G-Counter-style writer-sharded bundles
  (`HoloReplica`, `Replicated*`), wire-protocol delta sync
  (`version()`/`updates_since()`), two-process TCP demo
  (`live_sync.py`), and observed-remove containers (`ORStore`,
  `ORHoloMap`, `ORStrokeScene`) with idempotent tombstone deletion,
  add-wins, epoch/stroke undo, and owner compaction.
- Storage: phase-only and 2-16x quantized codes.
- GPU: MLX/Metal backend (`hdc/accel.py`, cos/sin real formulation) —
  encode 37x / decode 106x on M1 Max, NumPy-identical to 1e-7.
- SDK surface: the `holo` package (core/encode/structures/scene/query/
  render/fit/sync/storage/backend), pinned consistent by
  `tests/test_holo_facade.py`; console scripts `hdc-demos` and
  `holo-demos`. Live sync carries stroke undo: both painters
  concurrently undo the same stroke over TCP.
- Physical migration: all implementation modules moved into `holo/`
  (history-preserving); `hdc/` reduced to compatibility shims.
- Backend everywhere: `accel.readout` (universal field readout, the
  conjugate twin of `decode`) and `accel.cell_decode` now carry every
  field/scene/render/fit eval — MLX/Metal ~65x at render scale, with an
  identical cos/sin NumPy fallback (float32 agreement).
- Docs: one page per proven technique under `docs/` (math, API,
  measured budgets, failure modes, evidence pointers), indexed by
  `docs/README.md`.
- CI: GitHub Actions — Linux 3.9/3.12 as the NumPy-fallback proof,
  Apple-silicon macOS with MLX + Metal runtime probe + forced-NumPy
  second pass.
- Format tags (wire v1 / storage v1): 12-byte `HB` headers on every
  bundle blob, a per-doc `{wire, dim, seed}` universe record validated
  on flush/apply (mismatches refused loudly), and an 8-byte `HP`
  envelope for quantized phase codes with true 4-bit nibble packing.
  The universe record rides in the same commit as first content ops —
  writing it any earlier strands delta sync on missing causal deps
  (caught by the live-sync tests).
- Test suite: one file per module (`tests/TESTING.md`), 56 tests.
- Platform: numpy pinned <2.0 (macOS Accelerate float32 GEMV corruption;
  OpenBLAS wheels are clean).
