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

Why a record rather than a grep: several drivers build their output
names at runtime — `run_real_scene.py` writes `real_{stem}.png` from
its argument, `run_turntable.py` writes
`real_turntable-{name}.gif` — so the filename never appears in the
source and no search recovers the link.

## Real captures (`results/`)

| figure | regenerate with |
|---|---|
| `real_redrock.png`, `real_redrock_xray.png` | `python examples/run_real_scene.py data/iphone/redrock.ply` |
| `real_scan-tucson.png`, `real_scan-tucson_xray.png` | `python examples/run_real_scene.py data/scan-tucson.spz` |
| `real_train.png`, `real_train_xray.png` | `python examples/run_real_scene.py data/train.splat` |
| `real_lidar-dense.png`, `real_lidar-dense_xray.png` | `python examples/run_real_scene.py data/iphone/lidar-dense.ply` |
| `real_turntable-redrock.gif`, `.png` | `python examples/run_turntable.py data/iphone/redrock.ply --crop 0.5 --elev 0.7` |
| `real_turntable-scan-tucson.gif`, `.png` | `python examples/run_turntable.py data/scan-tucson.spz` |
| `real_fit.png` | `python examples/run_fit_real.py` |
| `capacity_curve.png`, `recon_2d.png`, `translation.png` | `python examples/run_prototype.py` |
| `mog_penalty.png` | `python examples/run_mog.py` |
| `baseline_table.md` (table, not image) | `python examples/run_baseline_table.py` |
| `failure_herringbone.png` | **not regenerable — archived exhibit** |

`failure_herringbone.png` is the one entry with no command, and
deliberately so: it is a *pre-fix* render preserved from before the
codebook rule was understood, showing what a band whose codebook does
not reach the global scale floor does to a capture. Regenerating it
would mean reintroducing the bug. It is evidence of a failure mode, and
[spectral.md](spectral.md) cites it as such.

Captures live in `data/`, which is gitignored — the real-capture rows
need the source files present. Everything else regenerates from a clean
checkout.

## Demos (`out/`)

| figure | regenerate with |
|---|---|
| `field_comparison.png` | `hdc-demos field` |
| `attribute_field.png` | `hdc-demos attribute` |
| `multiband.png`, `chunked3d.png` | `hdc-demos spatial` |
| `ray_render.png`, `ray_render.gif` | `hdc-demos render` |
| `color_knot.png`, `color_knot.gif`, `color_photo.png` | `hdc-demos color` |
| `turntable.png`, `turntable.gif` | `hdc-demos turntable` |
| `fit_photo.png` | `hdc-demos fit` |
| `codec_curve.png` | `hdc-demos codec` |
| `crdt_scene.png`, `crdt_attributes.png` | `hdc-demos crdt` |
| `orset_undo.png` | `hdc-demos orset` |
| `live_sync.png` | `python examples/live_sync.py` |
| `dynamic_prototype.png` | `python examples/dynamic_prototype.py` |
| `example_splats.png` | `python examples/splats_from_ply.py` |

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
  `hdc-demos turntable` still produces them.
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
