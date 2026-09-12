# Error locality across field techniques

[The GPU sweep](gpu_sweep.md) reports Wilson's Creek at 176.7% relative
L2 with 91% / 32% of squared error in the worst 1% / 0.1% of pixels;
cannon's side slice is 150.3% with only 16% / 3%. Similar headlines,
opposite diagnoses: concentrated cell failures versus broad herringbone.
`bench.locality` makes the diagnostic reusable with NumPy alone. It retains
truth norm, occupied fraction, and peak ratio alongside relative L2, so a
small denominator or sparse reference remains visible. It preserves the
existing sweep's five diagnostic keys and rounding without changing either
driver. It does not encode, decode, or depend on a field representation.

**Two nulls.** Constant-magnitude random-sign residuals put exactly k/n of
squared error into any k pixels. Gaussian residuals have varying magnitudes:
their squares follow chi-square with one degree of freedom. With
z = Phi-inverse(1 − f/2), the population mass above the upper-f quantile is
f + 2 z phi(z), from integration by parts. This gives **8.4492%** at f=1%
and **1.2696%** at f=0.1%; the brief's approximate 5.5% is inconsistent with
its specified chi-square null. Gaussian variation alone therefore creates
substantial concentration; f is an unfair reference for such noise.
At n=50,000, seed 19, measured shares are 8.5088% and 1.3016% (the seeded
test checks both within four asymptotic standard errors for the empirical
top-fraction ratio). Constant-magnitude tests return 1% and 0.1% exactly.
The default reading is spike at ≥50% in the worst 1%, distributed at
≤2 times the Gaussian reference (16.8983%), and localised otherwise.
These thresholds describe concentration, not a causal classification.

**Enrichment.** A cell receives the squared error at every pixel inside its
reach, divided by total image error. Overlapping cells each receive full
credit, so shares can sum above one; credit is not divided by overlap count.
Enrichment is this share divided by the cell's pixel count / n. Uniform error
gives one regardless of reach. Raw share instead promotes wide bands: the
original cell diagnostic found a one-member mid cell holding 99.6% of error
while covering 13,989 of 50,176 pixels. Empty and zero-error cells are omitted,
as in that driver. Labels retain integer IDs; lists of index arrays use list
positions as IDs, with duplicate indices within a cell counted once.

**Limits and matched references.** Scattered salt-and-pepper outliers can
produce a spike reading with enrichment near one everywhere. Share alone
cannot distinguish one blown cell from scattered outliers; membership is
required for a cause reading and still does not prove a mechanism. Pixel
count matters: the sweep selects k=max(1, floor(n f)), so tiny images have
coarse fractions. Coverage is normalised by n; report n when comparing grids.
`null_share(f, n)` validates n but returns the specified population tail mass,
not a finite-sample expected order-statistic ratio. Pass k/n as f to adjust
for pixel rounding. Zero residuals have zero shares; zero truth yields zero
relative L2 if reconstruction is also zero, infinity otherwise.

The abstract of [Spectral Prefiltering of Neural Fields,
arXiv:2510.08394](https://arxiv.org/abs/2510.08394), checked online, describes
scaling Fourier features by a filter's frequency response and training on
filtered signals, including filters beyond Gaussian. This is a neural-field
analogue of the repo's matched-referee principle: compare at the intended
filter footprint. That analogy does not prove the cause of these errors;
a filtered reference also changes the task, so its error should be labelled.

```python
import numpy as np
from bench.locality import locality_report, sweep_row_fields
truth = np.ones(10_000)
recon = truth + np.random.default_rng(19).normal(0, 0.01, truth.size)
report = locality_report(truth, recon, np.repeat(np.arange(10), 1000))
print(truth.size, report.reading, report.cells[0], sweep_row_fields(report, "demo"))
```

Validation: `python -m ruff check bench/locality.py tests/test_locality.py`,
`python -m pytest tests/test_locality.py -q`, `python -m pytest tests -q`,
`python -m holo.quality.cli check`, and `python -m holo.facts.cli check --strict`.
Checks passed: 14 focused tests; full suite 255 passed, 9 skipped in 60.27 s;
lint clean; quality debt 50 (baseline 50); strict facts 0 FAIL, 23 WARN.
Real captures and GPU hardware are unavailable here; no new capture measurements
were attempted. The capture numbers above are the existing sweep's results.
