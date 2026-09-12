# Change detection: synthetic drift ladder

Squatch Stack research tool: `bench/change_detection.py`. This is an ungated
CPU experiment, not a capture-scale result or a promoted accuracy claim.

## Result

The drift-only null does **not** provide reliable localization on this fixture.
The raw difference peaks at the removed cluster, but finite-codebook sidelobes
from that actual removal exceed the unchanged-scene null over much of the box.
The primitive baseline is substantially better at low drift. Increasing drift
sometimes improves bundle IoU because it raises the null threshold; this is
not evidence that drift improves the representation. No threshold was selected
using the true bbox, and no spatial mask suppresses the observed sidelobes.

| sigma_rec (scene units) | sigma_pos (scene units) | IoU bundle | IoU baseline | false mass |
|---|---|---|---|---|
| 0.25 | 0 | 0.0124 | 0.9259 | 0.8704 |
| 0.25 | 0.05 | 0.0201 | 0.3385 | 0.8349 |
| 0.25 | 0.1 | 0.0457 | 0.1875 | 0.7520 |
| 0.25 | 0.2 | 0.1459 | 0.1600 | 0.6076 |
| 0.5 | 0 | 0.0088 | 0.3068 | 0.9036 |
| 0.5 | 0.05 | 0.0095 | 0.2647 | 0.9009 |
| 0.5 | 0.1 | 0.0123 | 0.2842 | 0.8896 |
| 0.5 | 0.2 | 0.0255 | 0.2577 | 0.8451 |
| 1 | 0 | 0.0075 | 0.1120 | 0.9408 |
| 1 | 0.05 | 0.0076 | 0.1125 | 0.9405 |
| 1 | 0.1 | 0.0079 | 0.1286 | 0.9399 |
| 1 | 0.2 | 0.0089 | 0.1317 | 0.9379 |

![Synthetic difference, drift threshold and true bbox](../out/change/synthetic.png)

The figure is the first ladder cell (sigma_rec 0.25, sigma_pos 0), a y slice
through the removed bbox center. It decodes an 80 by 80 slice separately;
the table always evaluates the same 17 cubed 3-D grid, including endpoints.

## Reproduction and definitions

```sh
HDC_BACKEND=numpy OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/change-mpl .venv/bin/python -m bench.change_detection /tmp/change-synthetic.json --synthetic --dim 4096 --seed 0 --figure out/change/synthetic.png
```

The seeded eight-unit world contains four separated cubes, each with 128
uniformly sampled primitive centers in a cube of side 1.5 scene units. Native
Gaussian standard deviation is 0.12 units and alpha is one. The first cluster
is removed exactly. Its true bbox is the min/max of its unperturbed centers,
not an opacity isosurface or a blur-expanded label. Before and after receive
independent position and amplitude draws, with sigma_amp 0.2 and split_frac
0.1. Position jitter is isotropic Gaussian; lognormal amplitude multipliers
have mean one and relative standard deviation sigma_amp. Selected splats
split into two identical half-amplitude copies. This preserves the field and
tests count ambiguity, but is not a model of displaced, covariance-changing
splits from a real optimizer.

All positions and covariances use one fixed normalized parent frame, with
sigma_box = sigma_units / extent. One 4,096-row iid Gaussian codebook is
sampled per run at frequency standard deviation extent / min(sigma_rec).
Every ladder cell reuses it. Recognition blur is mass-preserving render_mip;
alpha spectra use spectral_bundle; maps use abs(decode_field_phasor).
The real phasor decoder adds the sampling kernel's blur (0.25 scene units
here), and finite sampling causes sidelobes. It is not an importance-weighted
inverse Fourier transform. Before-minus-after would reverse the signed
spectrum; the implementation uses after-minus-before and takes magnitude
only at the map stage.

For each cell, the observed before is fixed. Three independent unchanged
re-optimizations of the original before supply a pooled 99th-percentile
voxel threshold. The baseline uses those exact same null scene realizations,
reference capture and grid, with its own 99th-percentile threshold because
its scores have different units. This is an empirical voxelwise threshold,
not a familywise confidence level; three draws are only a CPU smoke calibration.

The baseline field is the absolute difference between nearest-center distance
fields, each capped at radius sigma_box. Binning gives exact radius-limited
nearest neighbors without an additional dependency; dense bins are processed
in bounded blocks. This control ignores alpha, covariance and duplicate
splits. It is our baseline, not a published method reproduced from code.

IoU uses strict score > threshold and an inclusive bbox. False mass is the
sum of threshold-surviving change magnitude outside the bbox divided by all
threshold-surviving change magnitude (zero when nothing survives). It is not
the fraction of voxels flagged, nor a false-positive rate on an unchanged
scene. JSON includes both raw maps in C-flattened ij grid order, thresholds,
settings and the bbox; the grid coordinates are linspace(0, 1, grid).

The focused tests are deliberately separate from this difficult ladder. The
localization test uses an isolated Gaussian object with a 0.4-unit annotation
and sigma_rec 0.1, and the agreement test uses a coarse Cartesian grid through
isolated centers. Both use drift-null thresholds. The held-out unchanged
capture test measures surviving false magnitude relative to the reference
field's total magnitude, requiring less than 10%; it does not redefine the
ladder's false-mass metric. These passing tests do not establish success on
the larger fixture.

## Capture commands for the maintainer

Set SCENES to the capture directory. The generated edits below omit the
optional AFTER positional; --remove and --insert construct it, replacing
AFTER if supplied. All observed captures must already share physical axes;
this tool does not register them. No real files were present in this worktree.

```sh
HDC_BACKEND=numpy .venv/bin/python -m bench.change_detection /tmp/change-wilsons-creek.json "$SCENES/wilsons-creek.spz" --frame "$SCENES/wilsons-creek.spz" --remove "$SCENES/wilsons-creek-gun.spz" --sigma-units 0.5 --grid 48 --drift 0,0.05,0.1,0.2 --dim 8192
HDC_BACKEND=numpy .venv/bin/python -m bench.change_detection /tmp/change-redrock.json "$SCENES/redrock.spz" --frame "$SCENES/redrock.spz" --remove "$SCENES/redrock-cairn.spz" --sigma-units 0.5 --grid 48 --drift 0,0.05,0.1,0.2 --dim 8192
HDC_BACKEND=numpy .venv/bin/python -m bench.change_detection /tmp/change-library-cannon.json "$SCENES/research-library.spz" --frame "$SCENES/research-library.spz" --insert "$SCENES/cannon.spz" --at 2 0 0 --sigma-units 0.5 --grid 48 --drift 0,0.05,0.1,0.2 --dim 8192
```

The chosen cannon translation is +2 scene units along x, with y/z unchanged.
The object is selected with crop_box/build_scene_fixed in its own physical
cube, restored to physical coordinates, then encoded in the parent's frame.
Its native covariance scales consistently. Check bbox_box in the output to
see how much of the translated object falls within the evaluation cube;
that placement cannot be verified without the captures. The generated after
retains inserted splats outside the cube, but the scored domain is [0,1]^3.
Removal verifies raw position overlap in scene units before alpha/cube
filtering (default tolerance 1e-5), refuses overlap below 0.99, and reports the
measured fraction. All parent positions within tolerance of the crop are
removed, preserving the parent's surviving attributes. A crop with no
eligible in-frame primitive is refused. No capture overlap is claimed here.

A plain BEFORE AFTER comparison needs no synthetic edit. Because it has no
known true bbox, its IoUs and false mass are null rather than invented labels.
The same requested drift ladder perturbs both supplied inputs. Real mode
uses sigma_rec 0.5 and grid 48 by default; synthetic mode uses the complete
three-scale ladder and grid 17. Explicit --sigma-units, --grid and --drift
can narrow either run. run_synthetic() defaults to dim 4096; the CLI defaults
to 8192. The CPU commands above are reproducible starting points; the
maintainer can use the existing CUDA patch environment for capture-scale
encoding/decoding. Baseline spatial binning remains CPU work.

## Limits and prior art

A change smaller than sigma_rec can vanish; drift above sigma_rec can read
as change everywhere. Alpha-only fingerprints cannot detect pure colour
changes; RGB channels are follow-up work. Non-subset crops are refused rather
than scored. Shared-frame crop/alpha/scale selection inherits
build_scene_fixed's behavior and must not be mistaken for full-capture
coverage. Fixed finite codebooks and bbox discretization are material limits.

The abstracts of [arXiv:2605.07203](https://arxiv.org/abs/2605.07203) and
[arXiv:2512.22830](https://arxiv.org/abs/2512.22830) were read before design.
The former motivates primitive comparison with drift; the latter uses
multi-view aggregation. Our isotropic null, duplicate splitting, baseline
and scoring definitions are ours; neither method's code was reproduced.
A follow-up outside this lane should investigate calibration for change-induced
spectral sidelobes and validate richer drift models on real recaptures before
making localization claims. No changes outside the exclusive lane files and
its one permitted figure-provenance row were made.
