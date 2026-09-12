# Error locality

*[← docs index](README.md) · fields and scenes*

**What.** The share of squared error carried by the worst fraction of
probe points, with per-cell enrichment to locate the concentration.
Keep the headline relative L2 and the locality share together: never rank
captures by the headline without the locality share beside it.

**Theory.** For residuals `r`, sort `r²` in descending order and take
`k = max(1, floor(n f))` points. Their share is `sum(top k r²) / sum(r²)`.
Constant-magnitude residuals give `k/n`. For iid Gaussian residuals,
the squares follow chi-squared with one degree of freedom. The worst
fraction `f` carries population error mass

    z = Phi^-1(1 - f/2)
    phi(z) = exp(-z²/2) / sqrt(2 pi)
    null(f) = f + 2 z phi(z)

Integration by parts gives the extra tail term: the reference is not `f`.
The Gaussian null is **8.4492% at f = 1%** and **1.2696% at f = 0.1%**.
The existing seeded measurement at n = 50,000 gives 8.5088% and 1.3016%;
`tests/test_locality.py` checks these against the population null within
four asymptotic standard errors. An earlier brief's ~5.5% reference was
inconsistent with this null. Evidence: [error locality](../results/error_locality.md).

For a cell covering index set `I`, its share is `sum(r[I]²) / sum(r²)`
and its enrichment is `share / (len(I)/n)`. Uniform error gives enrichment
one. Overlapping cells receive full credit, so shares may sum above one;
duplicate indices within a cell count once. Empty and zero-error cells
are omitted. These diagnostic definitions are ours.

**What it is for.** In the [existing GPU sweep](../results/gpu_sweep.md),
Wilson's Creek at 176.7% relative L2 has 91% of squared error in the worst
1% of pixels and 32% in the worst 0.1%: one blown cell in a faithful
landscape. The cannon at 150.3% has 16% / 3%: plane-wave herringbone across
the plane. The oak's good-looking 17.9% is 97% localised in the worst 1%,
so concentration is not confined to large headline errors. These are
existing capture observations, not a new measurement or a causal test.

**Failure modes.** Salt-and-pepper outliers can read as a spike with
enrichment near one everywhere. Shares depend on `n`; report it when
comparing probe grids. The Gaussian null is a population tail, not a
finite-`n` order statistic; `null_share` validates `n` but does not use it
to adjust the tail. Supplying `k/n` accounts only for pixel rounding.
The diagnostic says where error is, never why. Zero residuals give zero
shares; zero truth gives relative L2 zero for exact reconstruction and
infinity otherwise.

**API.** All names are available flat from `holo` and through `holo.scene`.

```python
from holo.scene import locality_report, sweep_row_fields

report = locality_report(truth, recon, membership=cell_labels)
row.update(sweep_row_fields(report, "top_down"))
```

`truth` and `recon` are matching nonempty finite real vectors. Membership
is either one integer cell label per point or a list of index arrays
(which may overlap). `LocalityReport` carries `rel_l2`, `norm_truth`,
`fill`, `peak_ratio`, `shares`, `null_shares`, `cells`, and `reading`.
`error_shares(resid, fracs=(0.01, 0.001))`, `null_share(frac, n)`, and
`enrichment(resid2, membership)` expose the components; enrichment takes
already-squared residuals. Shares are fractions, not percentages.
`locality_report` defaults to `floor=0.01`, `fracs=(0.01, 0.001)`, and
`spike_share=0.5`; it always includes the 1% share for its reading.
The reading is spike above or at `spike_share`, distributed at or below
twice the Gaussian null, and localised otherwise. These are concentration
labels. `sweep_row_fields(report, key)` requires both default fractions
and emits the sweep's five diagnostic keys with their existing rounding.
It does not emit relative L2. The recorded JSON round-trip and the seeded
float32/float64 arithmetic tests pin that export contract.
