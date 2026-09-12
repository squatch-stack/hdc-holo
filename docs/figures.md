# Figure provenance

*[← docs index](README.md) · evidence*

This repo already insists a measured number be re-derivable from the
tree it was committed in ([facts.md](facts.md)). A figure is a measured
number that happens to be a picture, and the same rule applies: every
image under `results/` and `out/` records the command that regenerates
it, so *"can you show that at higher resolution?"* or *"does that still
hold after the loader change?"* is a command rather than an
archaeology problem.

`tests/test_figures.py` enforces the table below: every figure in the
tree appears here, every recorded path exists, and every named driver
exists. A new figure with no entry fails the suite.

**Run these as modules, not as paths.** `python -m examples.run_mog`
and not `python examples/run_mog.py`; `python -m holo.cli color` and
not `hdc-demos color`. From a worktree the path and console-script
forms silently import the SHARED checkout instead of your tree — the
editable install's meta-path finder outranks `sys.path`, and for a
script `sys.path[0]` is the script's own directory rather than the
repo root. Measured, both forms: run by path or via `hdc-demos` from a
worktree, the import resolves to the shared checkout; under `-m` it
resolves to the worktree. Since our protocol says all work happens in
worktrees, a figure "regenerated" in a lane would otherwise be
produced by main's code and look entirely fine. The module forms are
correct from the repo root too, so there is no reason to write the
other kind.

Why a record rather than a grep: several drivers build their output
names at runtime — `run_real_scene.py` writes `real_{stem}.png` from
its argument, `run_turntable.py` writes
`real_turntable-{name}.gif` — so the filename never appears in the
source and no search recovers the link.

## Real captures (`results/`)

| figure | regenerate with |
|---|---|
| `resonator_cliff.png` | `python -m bench.resonator_sweep` |
| `real_redrock.png`, `real_redrock_xray.png` | `python -m examples.run_real_scene data/iphone/redrock.ply` |
| `real_scan-tucson.png`, `real_scan-tucson_xray.png` | `python -m examples.run_real_scene data/scan-tucson.spz` |
| `real_train.png`, `real_train_xray.png` | `python -m examples.run_real_scene data/train.splat` |
| `real_lidar-dense.png`, `real_lidar-dense_xray.png` | `python -m examples.run_real_scene data/iphone/lidar-dense.ply` |
| `real_brookline-station.png`, `real_brookline-station_xray.png` | `python -m examples.run_real_scene data/brookline-station.ply` (run 20260904T073902) |
| `real_turntable-redrock.gif`, `.png` | `python -m examples.run_turntable data/iphone/redrock.ply --crop 0.5 --elev 0.7` |
| `real_turntable-scan-tucson.gif`, `.png` | `python -m examples.run_turntable data/scan-tucson.spz` |
| `real_fit.png` | `python -m examples.run_fit_real` |
| `capacity_curve.png`, `recon_2d.png`, `translation.png` | `python -m examples.run_prototype` |
| `mog_penalty.png` | `python -m examples.run_mog` |
| `baseline_table.md` (table, not image) | `python -m examples.run_baseline_table` |
| `gpu_sweep.md`, `gpu_sweep.json` (tables, not images) | `python -m bench.sweep_scenes results/gpu_sweep.json $GALLERY/scenes/*.spz` |
| `gpu_sweep_matched.json` (table, not an image) | `python -m bench.sweep_scenes results/gpu_sweep_matched.json --footprint $GALLERY/scenes/*.spz` |
| `gpu_sweep_both.json` (table, not an image) | `python -m bench.sweep_scenes results/gpu_sweep_both.json --footprint --budget 128 $GALLERY/scenes/*.spz` |
| `real_wilsons-creek.png` | `python -m bench.sweep_scenes /tmp/s.json --figures --dir results $GALLERY/scenes/wilsons-creek.spz` |
| `real_cannon.png` | `python -m bench.sweep_scenes /tmp/s.json --figures --dir results $GALLERY/scenes/cannon.spz` |
| `real_oak_xray.png` | `python -m bench.sweep_scenes /tmp/s.json --figures --dir results $GALLERY/scenes/oak.spz` |
| `failure_herringbone.png` | **not regenerable — archived exhibit** |

`failure_herringbone.png` is the one entry with no command, and
deliberately so: it is a *pre-fix* render preserved from before the
codebook rule was understood, showing what a band whose codebook does
not reach the global scale floor does to a capture. Regenerating it
would mean reintroducing the bug. It is evidence of a failure mode, and
[spectral.md](spectral.md) cites it as such.

The four `bench.sweep_scenes` rows take `$GALLERY` — the gallery
checkout whose `scenes/*.spz` are the inputs — because unlike every
other row here those captures are not in `data/`. They are the
published exports, and measuring *those* is the point: they are the
artefacts anyone else can obtain. The driver wants a CUDA device
(`bench/cuda_backend.py`); `--numpy` reproduces the same digits on CPU
about thirteen times slower.

Note what those rows do NOT keep. `--figures` writes both plates for
every scene it is given — `real_<stem>.png` and `real_<stem>_xray.png`
— but only three of the twenty-four are committed, the ones cited from
[gpu_sweep.md](../results/gpu_sweep.md). So running the oak row lands a
`real_oak.png` beside the X-ray plate this table names, and
`test_every_figure_records_its_provenance` will then fail on it: an
unrecorded figure is exactly what that test exists to catch, and here
it is catching a by-product rather than an omission. Delete it, or add
a row — do not silence the test. The remaining twenty-one are left out
deliberately: `gpu_sweep.json` carries every number they would show,
and the sweep costs six minutes to re-run in full.

Captures live in `data/`, which is gitignored — the real-capture rows
need the source files present. Everything else regenerates from a clean
checkout.

## Demos (`out/`)

| figure | regenerate with |
|---|---|
| `field_comparison.png` | `python -m holo.cli field` |
| `attribute_field.png` | `python -m holo.cli attribute` |
| `multiband.png`, `chunked3d.png` | `python -m holo.cli spatial` |
| `ray_render.png`, `ray_render.gif` | `python -m holo.cli render` |
| `color_knot.png`, `color_knot.gif`, `color_photo.png` | `python -m holo.cli color` |
| `turntable.png`, `turntable.gif` | `python -m holo.cli turntable` |
| `fit_photo.png` | `python -m holo.cli fit` |
| `codec_curve.png` | `python -m holo.cli codec` |
| `crdt_scene.png`, `crdt_attributes.png` | `python -m holo.cli crdt` |
| `orset_undo.png` | `python -m holo.cli orset` |
| `live_sync.png` | `python -m examples.live_sync` |
| `dynamic_prototype.png` | `python -m examples.dynamic_prototype` |
| `example_splats.png` | `python -m examples.splats_from_ply` |

## On the orphan-figure warning

`holo-facts check` reports figures that no surface cites. Six are
currently orphaned, and the warning is doing its job rather than
misfiring — each is a real decision, not an oversight:

- `real_turntable-redrock.png`, `real_turntable-scan-tucson.png` — the
  contact sheets. The animated `.gif` beside each is what the docs
  cite; the sheets exist for print, where an animation cannot go, and
  are the natural figure if the paper ever needs a still orbit.
- `real_lidar-dense_xray.png` — the LiDAR room's X-ray view.
  [real-scenes.md](real-scenes.md) cites the slice figure and gives the
  X-ray number in prose; the image is held for supplementary material.
- `turntable.png`, `turntable.gif` — the synthetic turntable demo,
  superseded as evidence by the real-capture orbits, kept because
  `python -m holo.cli turntable` still produces them.
- `dynamic_prototype.png` — the dynamic-holograms probe. The lane is a
  roadmap candidate rather than a proven technique, so the figure has
  no docs page to live on yet, by design.

The useful reading of that warning is "an image exists that no argument
depends on" — which is either a figure looking for its section, or a
result whose section has not been written. Both are worth knowing; this
page records which is which.

## For the paper

`paper/draft.md` cites ten figures and `paper/main.tex` is generated
from it, with `\\graphicspath` covering both `results/` and `out/` so
the same source compiles in-tree and inside the flattened arXiv bundle
that `paper/make_arxiv_bundle.py` assembles. A figure added to the
paper is therefore traceable end to end: **driver → file → this record
→ bundle**.

One figure the paper needs does not exist in this repo: a **pipeline
schematic** for §1, showing splats → scale bands × spatial cells →
mixture-codebook spectral encode → one vector per cell, with the three
consumers branching off it. The mermaid diagram in
[README.md](README.md) is the content; it needs redrawing as a vector
figure. That is drawing work rather than conversion work, which is why
it is named here rather than approximated.

| `out/place/similarity.png` | Phase correlation and radial power control for three synthetic places (12 descriptors). | `python -m bench.place_recognition /tmp/place.json --synthetic 3 --numpy --dim 512 --grid 9 --yaws 4 --scrambles 4 --figure out/place/similarity.png` |
| `out/place/similarity-corpus.png` | Raw phase correlation and radial control over the twelve gallery captures (5090, `bench/cuda_backend`); the four wide captures form a 0.70–0.96 block. | `python -m bench.place_recognition out/place/matrix.json $SCENES/brookline-station-2.spz $SCENES/brookline-station.spz $SCENES/cannon.spz $SCENES/oak.spz $SCENES/redrock-cairn.spz $SCENES/redrock.spz $SCENES/research-library-cannon.spz $SCENES/research-library.spz $SCENES/saguaro.spz $SCENES/springhouse-outside.spz $SCENES/wilsons-creek-gun.spz $SCENES/wilsons-creek.spz --dim 8192 --yaws 16 --grid 48 --scrambles 24 --partner 1 9 --partner 4 5 --partner 10 11 --figure out/place/similarity-corpus.png` |
| `out/place/similarity-corpus-whitened.png` | The same matrix with `--whiten 1` (PHAT); the block sharpens to 0.82 between wilsons-creek and redrock. | `python -m bench.place_recognition out/place/matrix.json $SCENES/brookline-station-2.spz $SCENES/brookline-station.spz $SCENES/cannon.spz $SCENES/oak.spz $SCENES/redrock-cairn.spz $SCENES/redrock.spz $SCENES/research-library-cannon.spz $SCENES/research-library.spz $SCENES/saguaro.spz $SCENES/springhouse-outside.spz $SCENES/wilsons-creek-gun.spz $SCENES/wilsons-creek.spz --dim 8192 --yaws 16 --grid 48 --scrambles 24 --partner 1 9 --partner 4 5 --partner 10 11 --whiten 1 --figure out/place/similarity-corpus-whitened.png` |
| `out/factorized_cleanup/crossover.png` | Synthetic factorized cleanup crossover. | `HDC_BACKEND=numpy MPLCONFIGDIR=/tmp/faccleanup-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m bench.factorized_cleanup --synthetic --backend numpy --d 1024,4096 --n 8,16,32 --loads 1,2,4,8,16 --Q 16 --reps 3 --max-gb 12 --out /tmp/faccleanup-numpy.json --figure out/factorized_cleanup/crossover.png` |
| `out/change/synthetic.png` | `python -m bench.change_detection /tmp/change-synthetic.json --synthetic --dim 4096 --seed 0 --figure out/change/synthetic.png` |
| `out/shape/synthetic.png` | Four shape similarity matrices under a synthetic class hypothesis. | `python -m bench.shape_descriptor /tmp/shape-synthetic.json --synthetic --dim 512 --yaws 8 --grid 3 --figure out/shape/synthetic.png` |
| `out/resonator_capture/where.png` | Synthetic spectral resonator positions and distractor recovery; failed attempts remain visible. | `python -m bench.resonator_capture /tmp/rescap-synthetic.json --synthetic --dim 4096 --values 16 --figure out/resonator_capture/where.png` |
| `out/place/tiles-synthetic.png` | `python -m bench.place_recognition /tmp/submap-tiles.json --synthetic 3 --numpy --tile 1 --overlap 0 --dim 1024 --grid 5 --limit 0.06 --yaws 4 --scrambles 4 --whiten 1 --prefilter 2 --figure out/place/tiles-synthetic.png` |
| `out/place/tiles-corpus-E8.png` | Capture matrix and tile-hit heatmap for 8-unit tiles over the twelve gallery captures (5090). | `python -m bench.place_recognition out/place/tiles-corpus-E8.json $SCENES/brookline-station-2.spz $SCENES/brookline-station.spz $SCENES/cannon.spz $SCENES/oak.spz $SCENES/redrock-cairn.spz $SCENES/redrock.spz $SCENES/research-library-cannon.spz $SCENES/research-library.spz $SCENES/saguaro.spz $SCENES/springhouse-outside.spz $SCENES/wilsons-creek-gun.spz $SCENES/wilsons-creek.spz --tile 8 --dim 8192 --yaws 4 --grid 24 --whiten 1 --prefilter 8 --min-mass 0.02 --scrambles 12 --partner 1 9 --partner 4 5 --partner 10 11 --figure out/place/tiles-corpus-E8.png` |
| `out/change/tiles-synthetic.png` | `python -m bench.change_detection /tmp/tiles4.json --synthetic --tile 4 --dim 4096 --seed 0 --figure out/change/tiles-synthetic.png` |

| `out/place/flatten-synthetic.png` | `python -m bench.place_recognition /tmp/flatten-place.json --synthetic 3 --numpy --dim 4096 --grid 9 --limit 0.12 --yaws 4 --scrambles 4 --whiten 1 --flatten-study --figure out/place/flatten-synthetic.png` |
| `out/merge/merge-fixture.png` | `python -m bench.merge_capture /tmp/merge-fixture.json data/fixtures/wilsons-creek-core.spz --dim 2048 --overlap 0.2 --drift 0,0.2,0.1 --figure out/merge/merge-fixture.png` | Merge fidelity by rule and region, wire bytes against the SPZ equivalent across the cell ladder, and state size against the number of contributions (`results/merge_capture.md`). |
