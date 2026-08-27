# Fidelity per byte: what a bundle costs

*[← docs index](README.md) · positioning*

*(generator: `examples/run_baseline_table.py`; regenerate with
`python examples/run_baseline_table.py <capture>`)*

The comparison the 3DGS compression literature expects, run with one
referee so the two families are commensurable: every row is scored by
evaluating the field it reconstructs at the same query points, against
the exact Gaussian mixture of the source. Per-splat formats lose to
quantization, holographic bundles lose to crosstalk, and the referee
does not care which.

**Red Rock, 48,043 splats after crop, 6,000 query points.** (The
capture is subsampled because the referee is O(splats x points); every
row is scored on the *same* subsample, so bytes and error stay
comparable.)

| representation | MB | B/splat | field err | loss source |
|---|---:|---:|---:|---|
| PLY (SH-0, our writer) | 3.9 | 85 | 0.0% | lossless |
| SPZ v3 | 1.0 | 22 | 0.0% | quantized |
| SOG (SH palette) | 0.8 | 18 | 0.3% | quantized + palette |
| holographic bundles (d=2,048) | 95.7 | 2,088 | 33.4% | crosstalk |
| holographic bundles (d=8,192) | 382.8 | 8,354 | 17.4% | crosstalk |
| same, HM-8 codec | 95.7 | 2,089 | 17.0% | crosstalk + quantization |
| same, HM-4 codec | 47.9 | 1,045 | 25.8% | crosstalk + quantization |

## Read this honestly

**On fidelity per byte, the per-splat codecs win, and it is not close.**
A bundle is roughly **400x larger** and **50x less accurate** at
reproducing the same field. If the job is to store a scene and
rasterize it later, use SOG and stop reading.

That number is the point of publishing this table rather than avoiding
it. A bundle is not a compression format, and the claims made for it
are not compression claims — it is a *queryable, mergeable,
renderable-without-geometry* field. What the bytes buy:

- **Query by algebra** — `what_is_at(p)`, `where_is(label)`, a slice, a
  point, at a cost independent of splat count within the cell
  ([attributes.md](attributes.md)).
- **Views without geometry** — a whole orthographic render is another
  bundle, no ray marching and no sort ([render.md](render.md)).
- **Merge without coordination** — replicas add
  ([sync.md](sync.md), [orset.md](orset.md)).

A per-splat codec sells none of those; it sells bytes, and it sells
them very well.

## Two things the table does say in our favour

**HM-8 is free compression.** Four times smaller at *slightly better*
error (17.0% vs 17.4% uncompressed) — max-scaled quantization zeroes
the small components, which on a forward-encoded bundle are mostly
crosstalk. The accidental shrinkage denoiser from
[storage.md](storage.md), reproduced at capture scale. HM-4 goes 8x
smaller and does finally cost accuracy (25.8%), so the useful setting
is 8 bits.

**The two families scale differently with density.** Bundle bytes are
fixed per occupied cell however many splats land inside it; every
per-splat format grows linearly with content. At a fixed crop:

| splats | SPZ MB | bundle MB (d=8,192) | ratio |
|---:|---:|---:|---:|
| 4,778 | 0.10 | 149 | 1,490x |
| 11,985 | 0.25 | 226 | 904x |
| 24,061 | 0.50 | 294 | 588x |
| 48,043 | 1.00 | 383 | 383x |

Ten times the splats costs SPZ 10x and costs bundles 2.6x — the gap
narrows fourfold over one decade of density. **It does not close on
real captures**, and honesty requires saying so: reaching parity would
take orders of magnitude more density than a phone scan produces. What
is true is the shape — per-splat formats scale with *content*, bundles
scale with *occupied volume* — which is why the bundle's cost is
predictable from the scene's extent rather than its detail.

## Caveats a reviewer will raise

- The bundle rows use the pipeline's own band/cell configuration; a
  configuration tuned for bytes rather than slice fidelity would land
  elsewhere on the curve, and this table does not sweep that.
- Fitted holograms beat forward-encoded ones by ~70x held-out on
  synthetic mixtures ([fit.md](fit.md)) but are sampling-limited at
  real capture density, so the forward numbers above are what a
  capture actually gets today.
- SOG's 0.3% is against *this* referee (the alpha field); its SH
  palette carries a separate 0.54 relative error on the
  view-dependent term ([real-scenes.md](real-scenes.md)).
