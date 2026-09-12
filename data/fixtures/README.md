# Two real crops, small enough to keep

Every lane in this repo has to answer one question before it spends GPU
time: does the thing work on a real capture? Six times this year a
synthetic ladder said yes and a capture said no — the box-versus-window
objective, the scramble null, block scaling, the bundle difference, the
resonator on spectral bundles, and density flattening. Synthetic
fixtures belong in `tests/`; the first pass of a lane belongs here.

These two files are that first pass. Each is a cube cut out of one of
the raw phone captures in `data/iphone/` with
`holo.capture.crop_scene_file`, so each is a genuine **subset** of its
parent: same positions, same scales, same rotations, same colours,
selected by their centres and nothing else.

| file | parent | box (parent world units) | splats | alpha floor |
|---|---|---|---|---:|
| `wilsons-creek-core.spz` | `data/iphone/wilsonscreek.ply` | lo `[-0.14, -1.90, -1.71]`, side 2.0 | 49,077 | 0.1 |
| `redrock-core.spz` | `data/iphone/redrock.ply` | lo `[-0.04, -0.51, -0.91]`, side 1.2 | 57,536 | 0.1 |

Both boxes are centred on their parent's mass core — the cube
`bench.place_recognition.crop_box` picks — which is where a phone
capture actually has detail, and both are under a megabyte as SPZ v3
(about 16 B per splat).

## Why subsets matter

The captures the public gallery ships are each an independent
480,000-splat subsample of a trained model. That means a gallery
"crop" and its gallery "parent" agree on only 0.488 and 0.638 of the
crop's positions (the gun against Wilson's Creek, the cairn against
redrock), at any tolerance that means anything —
so the difference between those two files is not the object, it is two
draws from the same distribution. Change detection's removal cases
could not be built for exactly that reason
(`results/change_detection.md`). A crop cut from the parent file has no
such problem, and `bbox_of` + `crop_scene_file` is the two-line way to
make one.

## Provenance and reuse

Both parents are Squatch Stack's own iPhone captures, and these crops
are distributed with the repository under its licence like every other
file in it. Nothing here is derived from third-party data.

They are **not** a benchmark. They are two arbitrary cubes of two
scenes, small enough to commit; no number measured on them belongs in
`results/` as a corpus result. The twelve gallery captures are the
corpus.

## Regenerating them

```python
from holo.capture import crop_scene_file
crop_scene_file("data/iphone/wilsonscreek.ply", [-0.14, -1.90, -1.71],
                [2.0, 2.0, 2.0], "data/fixtures/wilsons-creek-core.spz",
                alpha_min=0.1)
crop_scene_file("data/iphone/redrock.ply", [-0.04, -0.51, -0.91],
                [1.2, 1.2, 1.2], "data/fixtures/redrock-core.spz",
                alpha_min=0.1)
```
