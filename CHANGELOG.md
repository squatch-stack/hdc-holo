# Changelog

## Unreleased

- **`weighted_quantile` no longer depends on which way an axis points.**
  It interpolated against the inclusive cumulative weight, which credits a
  sample with the whole of its own weight; reversing the sort order turns
  that into an exclusive sum from the other end and shifts the grid by one
  entire sample. `build_scene` takes its crop centre as the per-axis
  weighted median and the PLY loaders negate y and z, so the centre moved
  under a frame flip by about two sample spacings. That is a bias in exact
  arithmetic, not rounding: at float64 it is 1.4e-05, nine orders above
  what rounding over 336,094 terms can produce. Interpolating against the
  midpoint grid removes it exactly, and the accumulator is now float64.
  Both halves are needed, and together they cost a few splats: measured
  through `clean_export.py` on three real captures at q0.55, the crop
  populations move 204,482 to 204,472, 753,314 to 753,238 and 120,391 to
  120,384. The alpha arm at q0.90 and q0.95 does not move, which is why an
  earlier draft of this entry called the change free — that arm is
  invariant to the grid, to the accumulator and to both together, so it
  cannot discriminate. Found by a peer lane reconciling two independent
  implementations of this crop, which also caught the false claim.

- **arXiv endorsement package** (`paper/ENDORSEMENT.md`): the 2026
  endorsement mechanics from arXiv's own pages, the message an
  endorser receives, and a pre-submission checklist (license, metadata,
  ancillary files, moderation). Submission itself is still owner action
  (issue #59).

- **numpy 2 allowed on Linux.** The `numpy<2.0` cap now applies to macOS
  only (`sys_platform == 'darwin'`), where Accelerate-backed numpy 2.x
  wheels still corrupt float32 GEMV with heap-layout-dependent non-finite
  results (re-tested 2026-09-04 on numpy 2.5.2: the full suite passes, and
  a stress test of 12,000 float32 GEMV products across 10 processes produced 3 non-finite results, all in one process). OpenBLAS and
  MKL wheels are clean; CI gains a Linux numpy-2 leg to keep them so. This
  unblocks installing `hdc-holo` into environments that already run numpy
  2 (a CUDA toolpack with numpy 2.5.2 was the case at hand).

## 0.3.0 — 2026-08-27

<!-- claims: allow project.license@0.2.1 -->
- **Relicensed to Apache-2.0** (LICENSE.md). FSL-1.1-Apache-2.0's
  non-compete clause is designed for products with hosted competitors;
  a library is a thing other code *depends on*, and the clause
  propagates to every dependent — the one cost that strikes at what
  this artifact is for. It also kept the project out of conda-forge and
  the distros, off the `License :: OSI Approved` classifier, and past
  many corporate review processes, all during the window a new library
  most needs adoption (FSL converts to Apache after two years anyway).
  Apache's patent grant is the real protection against someone gating
  the ideas. Releases 0.2.0 and 0.2.1 remain under FSL as published.

- **First PyPI release: `pip install hdc-holo`.** The distribution is
  `hdc-holo`; the import stays `holo`. That split is forced — `holo`
  on PyPI belongs to an unrelated 2020 project — and it is ordinary
  (`scikit-learn`/`sklearn`, `pillow`/`PIL`). One sharp edge, stated
  rather than discovered: that other project also installs a top-level
  `holo/`, so installing both puts two projects at one import path.
  Published by GitHub Actions through PyPI Trusted Publishing, so no
  API token exists to leak, and the workflow refuses to publish a tag
  whose name disagrees with `holo.__version__`.
- **Claims registry and stale-claim gate** (`holo/facts/`,
  `holo-facts`, [docs/facts.md](docs/facts.md)). Measured numbers in
  this repo live in many surfaces at once — SDK.md births them, then
  README, ROADMAP, docs, CHANGELOG and docstrings repeat them — and
  they had already drifted apart in seven places. Every number is now
  a typed claim in `claims/registry.jsonl`, checked against its
  citation sites and, where possible, re-derived from code or the tree
  itself, so a mismatch reads "the registry is stale" rather than
  leaving both numbers plausible. Superseded values are never deleted:
  they stay as the historical allowlist, which is what lets a
  changelog keep saying what was true at its date. CI blocks on
  `holo-facts check --strict`. A trigram-profile retrieval layer over
  the same corpus catches paraphrased restatements as WARN-only —
  built on this SDK's own machinery, as a matrix and deliberately not
  a bundle, because at that chunk count our own capacity law forbids
  the bundle. `holo-facts mcp` serves the registry to agent sessions.
- **Quality system** (`holo/quality/`, `holo-quality`,
  [docs/quality.md](docs/quality.md)). Structure rules (root clutter,
  module/test pairing, driver leaves, unreachable shims), a lint
  ratchet keyed by (file, rule) rather than line number so it survives
  refactors, a Kuzu/Cypher index of the codebase for structural
  queries, and a checked-in LSP configuration. Lint debt fell 293 -> 50
  across the campaign, with the survivors documented as deliberate
  rather than silenced.
- `holo/demokit.py`: `banner` and `Table`. Eighteen modules had each
  re-derived the same output formatting; none of them format their own
  output any more.
- Fixed [#10](https://github.com/squatch-stack/hdc-holo/issues/10):
  `holo/backend.py` and every `hdc/*` shim resolve through a module
  `__getattr__` per access instead of binding `accel`'s functions at
  import. An out-of-tree backend patched over `holo.accel.readout` had
  been silently ignored by every facade-routed call — results still
  correct, the GPU never engaged. Pinned by test.
- [PAPER.md](PAPER.md) and [paper/draft.md](paper/draft.md): the
  publication pass — three claims, one law, one boundary — drafted in
  a claims-gated surface so every number in it is drift-tested.
- Export bridge: `save_ply` (raw Gaussian PLY, INRIA layout, SH
  degree 0, ecosystem y-down convention on disk — lossless to float32
  rounding) and `save_spz` (SPZ v2, 16.4x smaller at 0.04-0.07% field
  error on Red Rock). Anything the pipeline loads, crops, or merges
  now flows into splat-transform / SuperSplat / Spark and the rest of
  the display chain.
- `save_sog` (`holo/sog.py`): SOG v2 export — Morton-ordered splats,
  codebook-indexed attributes, and an SH palette, written as lossless
  WebP images in a zip. Red Rock: 8.3 MB (19x smaller than source,
  smaller than SPZ) while carrying the view-dependent SH that SPZ
  drops; `load_ply_sh` reads that term out of a 3DGS PLY, and
  `examples/export_formats.py` writes all three formats side by side.
- `examples/run_viewer.py --compare B`: two splat files rendered side
  by side under one shared camera (`examples/viewer/compare.html`) —
  the rate-distortion look, for judging codecs by eye instead of by
  claim. Scene files are now served by explicit route, so the two may
  live in different directories and no directory is exposed.
- `examples/run_viewer.py` + `examples/viewer`: real-time splat rendering with
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
  the work (turntable showcase); `examples/run_turntable.py --crop` for
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
  (`examples/live_sync.py`), and observed-remove containers (`ORStore`,
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
