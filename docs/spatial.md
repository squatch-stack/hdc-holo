# Covariance bands and spatial chunking

*[← docs index](README.md) · fields & scenes*

**What.** Two spatial organizations layered on the FPE field.

*Bands* (`MultiBandSplatField`): per-splat covariance, quantized. Each
band has its own `W ~ N(0, Sigma_b^-1)` and its own bundle; a splat
lands in the band matching its covariance; a query costs one inner
product per band. The continuum limit (every splat its own Sigma) is
the spectral encoder ([spectral.md](spectral.md)).

*Chunking* (`ChunkedSplatField`): one bundle per occupied grid cell. A
Gaussian kernel is ~zero a few sigma out, but in a single global bundle
every distant splat still contributes FULL-POWER noise to every query.
Chunking makes distant crosstalk exactly zero: a query consults only
cells whose box lies within `reach` of the point.

**Budget.** Chunked crosstalk is `sqrt(N_local/(2d))` instead of
`sqrt(N_total/(2d))`: 1500 splats in 456 cells at d=4096 reads at 4.7%
of peak where the global bundle reads at 19.5% — 4.2x lower error for
~9 consulted cells per query. The honest trade: at EQUAL total bytes a
single giant-d bundle is statistically stronger, but every query and
update then touches the whole hologram; cells keep compute, mutation,
and replication local (a cell is the CRDT sync unit — [sync.md](sync.md)).

## What kind of error is left, and what removes it

Doubling the finest band's dimension bought 2-4% for +600 MB, which said
the residual was not Monte-Carlo noise but could not say what it *was*.
Orthogonal frequency coupling settles it, by being a cleaner instrument
than dimension: it reduces the VARIANCE of the kernel estimate and
changes nothing else — same d, same bytes, same decode path, only a
different draw of `W` (`sample_frequencies(..., coupling="orthogonal")`).
So the fraction of a scene's error it removes IS the fraction that was
variance.

Measured on two captures (`python -m examples.run_coupling`), as
relative slice error against the exact mixture, top-down / side:

| | orthogonal coupling | shrinkage p25 | shrinkage AFTER coupling |
|---|---|---|---|
| saguaro, 519k splats | **+18.4% / +17.6%** | +14.8% / +4.8% | +7.4% / -3.3% |
| train, 504k splats (dense) | +1.9% / +1.5% | **+40.1% / +19.4%** | +40.5% / +17.6% |

Read the rows against each other and the picture is unambiguous. On
saguaro, coupling removes 18%, and shrinkage's remaining gain then
roughly halves — the two are competing for the same error, and coupling
got there first. On train, coupling removes almost nothing while
shrinkage removes 40%, and the two do not interact at all (40.1% before
coupling, 40.5% after). **The dense scene's residual is therefore not
variance** — a variance reducer that works elsewhere cannot touch it —
**and shrinkage is removing something else entirely.** That is a sharper
statement of the same fact the dimension experiment produced, arrived at
without spending 600 MB.

The practical rule: on sparse-to-moderate captures both tools help and
overlap, so pick one; on dense captures coupling is nearly free of
effect and shrinkage is the large win.

**A diagnostic that did NOT work, recorded so it is not retried.**
Spatial autocorrelation of the residual field looks like it should
separate coherent error from Monte-Carlo error, and it does not. Measured
correlation length is 3 px on saguaro (ac@1 0.715) and 1 px on train
(ac@1 0.227) against a white-noise control at 1 px (ac@1 0.002) — so the
DENSE scene, whose error is the coherent one, has the WHITER residual.
The statistic is real but it answers a different question: it tracks how
much fine structure the scene has, not whether the error averages down
with dimension. Coupling answers the intended question because it varies
variance alone.

**Failure modes.** Bands quantize covariance (a splat between bands
takes the nearest); cell size must exceed the kernel reach or queries
consult many cells; per-cell GPU dispatch is batched through
`accel.cell_decode` to amortize launches.

**API.**
```python
from holo import ChunkedSplatField, MultiBandSplatField
mb = MultiBandSplatField(4096, [sigma_broad, sigma_needle], seed=0)
mb.add_splat(mu, alpha, band=1)
ch = ChunkedSplatField(4096, sigma, cell_size=0.125)
ch.add_splat(mu, alpha); ch.eval(points)
```

**Evidence.** Global-bundle noise soup vs the clean chunked slice —
distant crosstalk zeroed by locality:

![1500 splats in 3D: global bundle vs chunked cells](../out/chunked3d.png)

![two covariance classes via frequency bands](../out/multiband.png)

`tests/test_spatial.py` (chunked beats global at equal d);
`holo-demos spatial`.
