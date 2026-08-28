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

- **CLAIMED: memory/sync lane** — the headroom guard
  (`holo/budget.py`, `CONTRIBUTING.md`, `examples/run_projection_pipeline.py`)
  and then `holo/crdt.py` + `docs/sync.md` for the shallow-snapshot trim
  protocol. Both areas were unclaimed and untouched on main at 7bfda1f.
- **The >4 GB pact needed to be code, not a rule.** CONTRIBUTING.md
  already said to check `ps` before a heavy encode, because two
  concurrent real-scene runs had OOM-killed each other. It was then not
  checked twice more: a Tikhonov lambda sweep was launched as parallel
  processes next to a 15 GB splat trainer. The diagnosis is not only
  "forgot to look" — a sweep run as N processes rebuilds the SAME 537 MB
  band Gram N times and pays for the SAME O(d^3) eigendecomposition N
  times, so the parallel shape was slower as well as fatter. One process
  sharing the Gram (and one eigendecomposition across every TSVD
  truncation) is strictly better on both axes. `holo.budget` is
  deliberately off the public surface: it shells out to `ps` and reads
  free memory, which is developer behaviour, not library behaviour.
- **Three ways to lose a heavy run, and the guard covers one.** Memory
  is the documented one. The second is self-inflicted and cost a restart
  here: a full `pytest` launched while this lane's own 3-setting sweep
  was still resident took the machine down — the pact is about MY other
  jobs as much as anyone else's, and a test suite counts. The third is
  not memory at all: `holo/accel.py` selects MLX/Metal whenever mlx
  imports, so an encode shares the GPU with any splat training on the
  box, and when that faults the GPU, Metal's recovery discards this
  process's command buffer as an `InnocentVictim`. Measured here mid-run.
  `HDC_BACKEND=numpy` sidesteps it when only the number is wanted.
- **The guard caught a live collision on its first run**, which is the
  only reason it is worth its lines: it named a `msplat -n 3000` trainer
  climbing to 15 GB while a baseline pipeline held 4.8 GB, and reporting
  jobs by `comm` was useless for exactly the process that matters — this
  venv's interpreter is a symlink and `comm` resolves it to an Xcode
  framework path naming no job at all. Report by argv, basename-first.

- **Projection encode cost: 98% of it was one call, and the fix changed
  the accuracy too** (research lane; `examples/run_projection_pipeline.py`,
  `docs/fit.md`). Profiling put 106 s of a 108.6 s per-band fixed cost in
  `eigh`; per-cell solves were 27 ms. Two changes: batching cells 256 at a
  time turns hundreds of BLAS-2 matvecs into one BLAS-3 matmul —
  bit-identical (deviation 0.00e+00, checked with an uneven final chunk) and
  7.1x on that step — and Tikhonov needs no eigendecomposition at all, so an
  explicit inverse replaces `eigh` at ~6x less cost. 770 s -> ~250 s on
  train.
  *Tikhonov is also MORE accurate than truncation, at the right lambda, and
  the right lambda is not portable.* Saguaro wants 1e-6 (0.1227/0.0803,
  +65.0%/+62.3%); train wants 1e-1 or more (0.2986/0.1388, +68.9%/+71.9%,
  and the sweep had not turned). Each beats the truncated solve at its own
  setting. **Using saguaro's lambda on train is a 7.9% REGRESSION against
  forward encoding** — five orders of magnitude apart, so this is a per-scene
  knob, not a constant to inherit. Plausible mechanism, untested: a denser
  cell puts more energy where the Gram is near-singular and needs heavier
  damping.
  *A negative kept from the same pass:* randomised SVD is the wrong tool for
  this. It is 2.5x faster and its eigenvalues are excellent (1e-6 relative),
  but the pseudo-inverse is dominated by the SMALLEST kept eigenvalues, where
  a randomised range finder is worst — solve deviation 0.38 to 1120. Checking
  eigenvalue accuracy alone would have passed it straight through.

- **Shrinkage still helps a SOLVED bundle, contradicting the prediction**
  (research lane). Written before the test: a solved bundle is L2-optimal on
  its window and already regularised, so shrinkage should move it away from
  the optimum. Measured: `shrink` at p10 improves the solved saguaro bundle a
  further +6.4%/+3.6% (0.2031/0.1317 against 0.2170/0.1367). The prediction
  failed because the solve is optimal for the *windowed per-cell* objective,
  which is not the quantity being scored, and truncation means it is not
  optimal even for that. The gain shrinks where the solve is stronger (+1.4%
  on train under Tikhonov), which fits. Best combination measured to date:
  analytic projection + shrinkage, +42.0%/+38.2% on saguaro.

- **The analytic projection is a real solve, end to end** (research lane;
  `examples/run_projection_pipeline.py`, `docs/fit.md`; issue #2).
  Encoding EVERY cell analytically and decoding the same evidence
  slices against the same exact-mixture referee — the quantity the rest
  of the repo reports — gives saguaro 0.3501/0.2132 -> **0.2170/0.1367**
  (+38.0%/+35.9%) and train 0.9591/0.4948 -> **0.3765/0.1716**
  (+60.7%/+65.3%). It is **largest on the dense capture**, which is the
  case more dimension could not help and where orthogonal coupling
  bought 1.9%: train's top-down slice goes from an error as large as
  its signal to 0.38. Of the three levers this session produced —
  shrinkage, coupling, projection — this is the biggest by a wide
  margin, and the only one that attacks the error by SOLVING rather
  than by cleaning up after accumulation.
  *The per-cell study over-promised in one direction and under-promised
  in another.* It measured 2.5x on one cell's reconstruction, which is
  a different quantity; end to end that became +38%. But it also
  concluded the window "needs no truncation tuning", and that was an
  artifact of studying at d=2048. At the production d=8192 the window
  Gram's condition number is 1.6e20 to 3.2e20, past what float64 can
  invert, and a full-rank solve returns garbage of order 1e5 — not a
  degraded answer, a divergent one. Keeping 25% of the spectrum is safe
  on every band and sits near the ~3,300 space-bandwidth DOF a cell
  supports. `docs/fit.md`'s claim is corrected.
  *Cost:* encode ~15x slower (770 s against 51 s on train), dominated
  by the per-band eigendecomposition (63 s at d=8192) and the per-cell
  solve. Decode and storage unchanged — the output is an ordinary
  bundle, so codecs, replication and rendering are untouched.
  *Implementation note worth keeping:* the windowed right-hand side IS
  `spectral_bundle` applied to a modified scene — covariance shrunk by
  the window, mean pulled toward the cell centre, amplitude scaled —
  verified to 9e-7 against a direct per-splat loop. The existing fast
  path does the work.

- **Analytic L2 projection measured: the box is the better objective and
  the window is the usable one** (research lane;
  `examples/run_analytic_projection.py`, `docs/fit.md`; issue #2). The
  sampling limit in the per-cell fit is a property of fitting to
  SAMPLES; projecting the exact mixture onto the codebook needs none,
  and `spectral_bundle` already computes the right-hand side — it IS
  the mixture's Fourier transform — so the whole difference from
  forward encoding is replacing the diagonal importance weighting with
  a Gram solve. Median over the six most populated xfine cells of
  saguaro, d=2048: forward 0.1879 whole-cell, box **0.2052** (worse),
  Gaussian window s=h/2 **0.0739** (2.5x better). Restrict the splats
  to the cell interior and the ordering inverts: forward 0.1464, box
  **0.0273** (5.4x better), window 0.0609.
  *The inversion is the finding.* The box's right-hand side is the
  whole-space transform, correct only for splats well inside the cell.
  The exact box-restricted transform of an ANISOTROPIC Gaussian needs
  the complex error function, does not separate for non-diagonal
  covariance, and is not in numpy — so the box's advantage is real but
  not reachable without scipy or an approximation. The window has no
  such problem: window x Gaussian is another Gaussian, so its RHS is
  exact at any splat position, which is why it barely degrades between
  the two columns where the box collapses.
  *Two further practical results.* The window needs no truncation
  tuning — monotone and stable at full rank — where the box must be
  truncated and detonates if it is not (0.0273 at keep=1536, 8.6 at
  2048 interior; 677 at 2048 whole-cell). And the rank ceiling that
  might have killed the spike does NOT bind: space-bandwidth gives
  ~3,322 usable DOF per cell against ~330 splats, a 10x margin.
  *Not promoted.* This is a measured spike; the charter wants a
  deterministic test and a documented failure mode before an SDK API
  exists. The `G_c = D G0 D^H` factorisation is pinned by test, because
  one decomposition per band instead of per cell is the whole cost
  model.
  *Method note, since it cost three reversals.* The synthetic study
  said box 2.7x; real cells said box loses; the interior control said
  box is best-but-fragile. Each measurement overturned the previous
  reading, and only the third was actionable. Synthetic geometry chosen
  to be convenient — isotropic splats, interior-only evaluation — was
  what made the first two disagree.

- **Orthogonal coupling answers issue #3 by being an instrument, not a
  fix** (research lane; `holo/spectral.py` `sample_frequencies(coupling=)`,
  `examples/run_coupling.py`, `docs/spatial.md`). Orthogonal random
  features (Yu et al. 2016) reduce the VARIANCE of the kernel estimate
  and change nothing else — same d, same bytes, same decode path, only a
  different draw of W — so the share of a scene's error they remove IS
  the share that was variance. Saguaro: **+18.4%/+17.6%**. Train, the
  dense scene the issue is about: **+1.9%/+1.5%**. The dense residual is
  therefore not variance, which is what the d-doubling experiment could
  only hint at, obtained here without spending 600 MB.
  *And the interaction is the confirmation.* On saguaro, shrinkage's gain
  roughly halves once coupling has run (14.8% -> 7.4%, and -3.3% on the
  side): they compete for the same error. On train they do not interact
  at all (40.1% -> 40.5%), and shrinkage removes **+40.1%/+19.4%** —
  far more than its 14.8%/4.8% on saguaro. So shrinkage is the general
  tool and coupling is the specific one.
  *Two corrections to my own reasoning, both found by measuring.* First,
  I had written that ORF would help little at input dimension 3 because
  it only orthogonalises within blocks of that size. Exactly backwards:
  the gain is LARGEST in low dimension (43% at dim 3, 37% at 8, 0.1% at
  32) because high-dimensional Gaussian rows are already near-orthogonal.
  Second, the synthetic gain does not transfer — 46% kernel-estimate MSE
  reduction at d=8192 became 18% on a real sparse capture and ~2% on a
  dense one. Synthetic kernel MSE is not a proxy for pipeline error.
  *Implementation trap worth the line:* `np.linalg.qr` is not Haar —
  fold `sign(diag(R))` back into Q. Without it the row marginal drifts
  (per-axis KS 0.167 against a 0.0056 critical value) and
  `decode_weights`, which evaluates rho at each drawn frequency, goes
  quietly wrong everywhere while nothing raises. numpy has no built-in
  orthogonal sampler and scipy is not a dependency. Pinned by a test that
  fails against the uncorrected construction.
  *Default unchanged* (`coupling="iid"`), verified byte-identical against
  main including the rng stream position, because every committed number
  was taken under it.

- **A diagnostic that does not work, logged so it is not retried**
  (research lane). Spatial autocorrelation of the residual field looks
  like it should separate coherent error from Monte-Carlo error. It does
  not: correlation length is 3 px on saguaro (ac@1 0.715) and 1 px on
  train (ac@1 0.227) against a white-noise control at 1 px (ac@1 0.002),
  so the DENSE scene — the coherent one — has the whiter residual. The
  statistic tracks how much fine structure a scene has, not whether its
  error averages down with dimension. Coupling answers the intended
  question precisely because it varies variance alone.

- **Deliberate shrinkage beats the accidental one, and the accident had
  largely evaporated** (research lane; `holo/denoise.py`,
  `tests/test_denoise.py`, `docs/storage.md`; issue #1). Reproducing the
  codec baseline before building on it found that page's numbers stale:
  they were measured pre-reach-split and the bands moved underneath
  them. Current saguaro figures are uncompressed 0.350/0.213 and HM-4
  0.347/0.223 at a 28x dynamic range, against the logged 0.522/0.379,
  0.502/0.342 and 987x. HM-4 no longer beats the uncompressed decode on
  BOTH axes — it wins top-down and loses side — so the "accidental
  denoiser" framing was describing a configuration that no longer
  exists. The claims gate could not have caught this: those numbers were
  never registered, which is precisely the WARN-only
  `unregistered-number` category. Both are registered now.
  *The positive:* what the accident pointed at is real and taking it
  deliberately recovers much more. Soft shrinkage at the 25th magnitude
  percentile reaches **0.298/0.203**, better on both axes than any codec
  row, and composes cleanly — shrink then HG-8 preserves it exactly at
  0.25x bytes, where shrinking into HM-4 gives most of it back
  (0.339/0.230) because 4-bit quantization re-adds error of the order
  just removed.
  *The correction:* an early docstring claimed soft simply beats hard.
  It does not. The advantage is regime-dependent — hard wins where
  signal is sparse and strong against weak noise (a real gap to cut at),
  soft wins where magnitudes OVERLAP, which is the capture case because
  a mixture codebook leaves no gap. Caught by a synthetic test that
  failed for the right reason; both directions are now pinned by test.

- **Issue #2 (analytic L2 projection) reassigned** from the
  capture/spectral lane to the research lane, with that lane's explicit
  agreement — it had never been started there. Two results from the
  scoping, both verified independently by a second session:
  the per-cell Gram factorises as `G_c = D·G_0·D^H` with `D =
  diag(e^{i w_j·c})` unitary diagonal, so **one factorisation amortises
  over every cell in a band** (1,624 on the saguaro fine band) and cell
  position enters only as a diagonal phase; and conditioning is governed
  by the dimensionless `sigma_w·h`, on which all our real bands sit
  BENIGN (xfine/fine 7.8, mid 31, coarse 62). The plunge arrives with
  **d**, not with cell size: at the xfine band's real geometry cond(G_0)
  runs 7e1 (d=40) -> 5e4 (160) -> 8e13 (640) -> 4e19 (2560), with
  numerical rank falling well below d. A first attempt used a toy at
  `sigma_w·h = 0.5` and read 4.3e10, which is the ill-conditioned
  regime and would have misled the whole spike — **the diagnosis is
  cheap and needs no capture data, but it must be run at realistic d.**
  Open question worth checking before building the solver: if effective
  rank is below the splat count per cell, it bounds what ANY per-cell
  fit can recover, and the analytic route inherits the sampling-limited
  ceiling rather than escaping it.

- **Two findings logged for lanes not being worked** (research lane), so
  they outlive the session that found them. (1) Issue #4: on a
  capture-shaped replica (60 containers x 40 edit rounds, d=2048) a full
  `ExportMode.Snapshot()` is 40.3 MB where `ShallowSnapshot` and
  `StateOnly` are 1.0 MB — 2%, far past the 70-90% Loro advertises,
  because the payloads are dense bundles and history dominates.
  `holo/crdt.py`'s `snapshot()` uses the full mode; the installed
  loro-py already exposes both alternatives. (2) Issue #5: the right
  literature family for the occlusion hybrid is order-independent
  transparency — arXiv:2605.13855 (OIT for 3DGS via an active set,
  2026), arXiv:2605.25345 (depth peeling for Gaussian surfels, 2026),
  arXiv:2305.10197 (learned OIT). Those also belong in the paper's
  limitations section as evidence the boundary is being probed.

- **The SH flip table now has two derivations, not one**
  (`_SH_FLIP_X180` in `holo/capture.py`). It was derived numerically
  from the basis functions and pinned by a test that used *the same
  basis* — one implementation checking itself, which passes any
  convention error the two share. It is now also derived algebraically
  from monomial parity: a 180-degree turn about x sends
  (x, y, z) -> (x, -y, -z), so each coefficient picks up a sign fixed
  by how many flipped factors its monomial carries (xy and xz carry
  one, yz two, the even terms none). Normalisation constants cancel in
  the ratio whichever sign they have, so the second route touches
  neither the constants nor the convention that produced the table.
  Both agree on all 15 signs, and both fail if any one is corrupted
  (verified by flipping the yz entry, the subtlest of them).
  *Corroboration, not evidence:* an independent implementation in
  another language, written against these loaders, derived the same 15
  signs by a third route. That is worth recording and is deliberately
  NOT cited by the claim — it lives outside this tree, where
  `holo-facts` cannot check it, and a claim whose evidence a reader
  cannot open only looks substantiated.

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
- **Band assignment could lose splats silently** (capture lane; found
  by following a lint flag rather than a bug report). A B007 warning
  said `cap` was unused in `encode_bands`'s loop — cosmetically true,
  since the cap does its work upstream in `band_of` (which assigns the
  band) and `band_codebooks` (which builds the mixture), while the
  encode step needs only the codebook and the cell size, and every
  decode path applies `reach = 3 * cap` correctly. But checking WHY it
  was unused surfaced a real hole next to it: `band_of` is a
  `searchsorted`, so a splat larger than the LAST cap gets an index one
  past the end, matches no band, and is dropped — no error, no warning,
  and every downstream number (slice error, byte count, render) still
  looks healthy. Measured: a 60-splat scene with 10 oversize splats
  encoded 50 and lost 10 in silence. `build_scene` cannot reach that
  state (it clamps to S_HI and the coarse cap IS S_HI), and the render
  path clears it by 0.005 of arithmetic luck, so nothing shipped wrong
  — but any custom `bands` list narrower than its scene would have.
  `encode_bands` now refuses with the count and the offending scale.
  The generalizable bit: an "unused variable" is worth one minute of
  asking why it is unused, and silent data loss is the failure mode to
  hunt near index arithmetic.
- **Scale-clamp audit across all five captures** (capture lane;
  the follow-up to the band-assignment guard). Direct answer first:
  **no capture contains an unbandable splat** — `band_of` returns a
  valid index for every splat in redrock, wilsonscreek, lidar-dense,
  scan-tucson and train, so the new guard never fires on real data and
  nothing that shipped was losing content. Two things the audit turned
  up anyway:
  (1) The TOP clamp is nearly idle after cropping. Raw files carry
  11-26% of splats above S_HI with maxima of 18-37 units in a scene
  that normalizes to 1 — but those are background floaters, and the
  mass-centered crop removes essentially all of them: 0.00% of encoded
  splats sit at the cap (train, 0.05%, is the only capture where any
  survive). The crop, not the clamp, is what handles giant splats.
  (2) The BOTTOM clamp does enormous work: **99.4-99.7% of encoded
  splats have a thin axis at the S_LO floor** in every Gaussian
  capture (the LiDAR cloud, isotropic by construction, is the 0%
  control). Real 3DGS splats are extreme needles — redrock's median
  thin axis is ~35x below the floor. This is the quantitative backing
  for `band_codebooks`' rule that every band must sample out to the
  GLOBAL floor: needle-thin axes are not an edge case, they are the
  overwhelming majority.
  A caution recorded because it fooled two measurements before it was
  caught: "what does the floor cost?" is ILL-POSED under point
  sampling. Evaluating the exact mixture with raw (unclamped)
  covariances gives a field with 1/40th the energy of the clamped one
  and a nonsense relative difference (1155% at splat centers, 4365% on
  a slice grid), because a 5.8e-5-thick sheet is essentially invisible
  to point evaluation — it occupies vanishing volume. The floor is
  what makes a point-evaluated ground truth meaningful at all. Judging
  it honestly needs footprint/area integration (rasterization
  semantics) rather than point samples, which is a different evaluator
  and a real open question adjacent to the occlusion gap.
- **Footprint-integration evaluator** (capture lane;
  `footprint_blur`, `exact_slice(footprint=)`), answering the open
  question the scale-clamp audit raised: what does ground truth look
  like if you measure the way a rasterizer does — averaging over a
  pixel — instead of point-sampling? The convolution is exact and
  nearly free, because a Gaussian convolved with a Gaussian is a
  Gaussian: covariances add, so a box pixel of side `pix` is
  `render_mip` at sigma = pix/sqrt(12). Pinned against a 9^3
  supersample of the pixel volume (<2%), and against the case it
  exists for: a needle 100x thinner than a pixel reads ~0 to point
  samples and >10x that under integration.
  Two results. (1) **The floor is radiometric, not just geometric:**
  clamping a thin axis up while holding PEAK amplitude fixed raises
  each splat's integral, and across Red Rock that inflates total scene
  mass **11x**. Footprint integration does not wash this out (4365% ->
  4484% clamped-vs-raw) because the difference was never geometric.
  Worth a design decision someday: clamping could preserve mass
  instead of peak, which would keep radiometry at the cost of making
  needles fainter. (2) **Matched to a pixel-integrated target the
  pipeline does BETTER than its headline numbers:** encoding the
  footprint-blurred scene and refereeing against the footprint ground
  truth gives **11.5%** on Red Rock where the sharp-vs-sharp pair
  gives 18.1% — blurring concentrates each splat's spectrum, so the
  codebook covers it better, the same mechanism that makes the X-ray
  mip encode work.
  Caution recorded, because it produced a 71.7% scare first: the
  encode and the referee must describe the SAME field. Comparing a
  sharp encode against a footprint referee (or the reverse) measures
  the blur, not the hologram.
- **Publication-prep sweep** (2026-08-27, `docs/related-work.md` 0.3
  delta): four findings, two of which move our own framing rather than
  someone else's. (a) **qFHRR** (arXiv:2604.25939) publishes
  quantized-phase FHRR at 3-4 bits with integer-only binding — our HP
  codec from the hardware side — so phase-only quantized FHRR is no
  longer a novel representation for us to claim. It CORROBORATES the
  codec split, though: their bundling is not closed under quantized
  phase and projects back to the unit circle, discarding exactly the
  magnitude our measurements showed amplitude fields need (the ~0.24
  rel RMSE floor that motivated HM/HG). Two routes, one boundary; our
  contribution is the measured curve across it. (b) **Mip-Splatting**
  (CVPR 2024) and **Analytic-Splatting** (ECCV 2024) already solve the
  sampling problem the footprint evaluator rediscovered — S_LO is an
  ad-hoc version of their 3D smoothing filter — so that lane is adopt
  and cite, not research; the piece worth keeping is the radiometric
  one (peak-preserving clamp inflates scene mass 11x). (c) **Motion as
  a phase ramp has prior art**: the SSP shift property and Voelker et
  al. (Neural Computation 2021) simulating multi-object trajectories.
  An earlier session's framing of that lane as unclaimed was WRONG and
  ROADMAP now says so; what stays open is the capture-scale
  combination and the cost model. (d) **CRDT model merging**
  (arXiv:2605.19373) is the nearest neighbour to holo/orset.py, and
  gives a sharp contrast for a writeup: they find all 26 merge
  strategies fail commutativity/associativity/idempotency, where
  superposition hands us the first two free and fails only the third —
  the exact gap our G-Counter and observed-remove recipes close.
  Unchallenged and still the claims to lead with: splat scenes as
  holographic bundles with algebraic queries, projection-slice
  rendering folded into a bundle, mixture spectral codebooks for
  per-splat covariance, and a rule engine with a capacity contract.
- **Publication framing settled** (PAPER.md §9-10): citations adopted
  and venue chosen. Adoption here means conceding ground up front —
  each bounding work is listed with the sentence it forces us to write
  (qFHRR takes quantized-phase FHRR as a representation, Mip-Splatting
  takes the sampling filter our scale floor approximates,
  Analytic-Splatting takes pixel-area integration, Voelker et al. take
  motion-as-shift), so the draft is written from the corrected
  position instead of defending it in rebuttal. Venue: the **HDC/VSA
  community**, arXiv cs.NE first, not graphics — because we do not do
  novel-view synthesis (our referee is an analytic mixture, not
  held-out images), because the baseline table is a loss on exactly
  the axis graphics weighs most, and because all three claims are
  algebraic claims that happen to be tested on captures. Consequence
  for the draft: lead with the algebra, introduce splats as the stress
  test, keep every graphics figure, and answer "why not 3DGS+SOG?"
  with the capability list rather than a metric.
- Still queued: component-thresholding denoiser (new, unclaimed);
  dense-scene coherent error (see ROADMAP); box lane: render_xray
  binning (still scans, 0.73 s), point-tile cell_decode fusion,
  cuFINUFFT type-3 prototype.
