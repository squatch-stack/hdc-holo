# Operator benchmark: CPU template, GPU measurements pending

HyperSpace code was not accessible; the operators and timing protocol here are ours.
The research pass checked [arXiv:2604.15113](https://arxiv.org/abs/2604.15113),
GitHub searches for the arXiv id and `hyperspace hyperdimensional`, and the
[repository linked by the paper](https://github.com/Parsa-Research-Laboratory/HyperSpace),
which returned HTTP 404 on 2026-09-12 (including the GitHub tree API). No code,
sizes or timing implementation could be mirrored. The paper describes iterative
cleanup; our single-pass nearest-codeword search is a distinct workload.

Float32 HRR at d stores 4d bytes; two-plane FHRR at d stores 8d bytes.
FHRR at d/2 therefore matches HRR at d in codeword bytes and dense dot-product
arithmetic. The same-d FHRR control necessarily has twice the bytes: all three
rows cannot truthfully have identical `bytes_per_codeword`. Storage equality is
not a proof of equal capacity: a unit phasor has one independent phase despite
occupying two real planes. A negative result would be reproducibly lower recovery
for FHRR at d/2 at loads where HRR at d maintains useful recovery.

## Protocol

Measured on arm64 Darwin with Python 3.12.14 and NumPy 1.26.4; BLAS/OpenMP
thread limits were set to one in the command below. No GPU was used.

- NumPy float32, seed 1729, Q=128, K chunks of 256, one warmup, best of three
  synchronised wall-clock repetitions; generation and initial transfers excluded.
- Bind processes Q vectors against one broadcast codeword. HRR uses rFFT
  circular convolution; NumPy FFT promotes internally, then returns float32.
- Bundle uses K synthetic splats, four channels, three position coordinates and
  six covariance coefficients. It transcribes the nested `holo_bench_job.encode`
  closure. Spatial similarity transcribes readout with unit frequency weights;
  those closures are not importable, so backend selection is the imported helper.
- Cleanup uses independent random Q queries and K codewords. Scores are raw
  real dot products, without division by d. Each tile is reduced on the host in
  float64, and ties select the first codeword. GPU timing includes those diagnostic
  transfers; it is not a pure device-GEMM measurement.
- Memory is a conservative whole-run live-array prediction, not measured RSS or
  operator-local allocation. It includes host/device copies, score tiles, trig/FFT
  temporaries and accuracy arrays, but excludes runtime and allocator overhead.
  Codebook-only bytes are 8Kd for FHRR and 4Kd for HRR. Large d may be refused
  because the accuracy experiment itself scales quadratically in storage.

## Measured operator rows

Times are milliseconds. Bytes are the shared whole-run predicted budget. Accuracy
is shown on cleanup rows; see the separate equal-bytes table for the matched pair.

| Operator | Backend | d | K | Time (ms) | Predicted bytes | Top-1 at 50% load |
|---|---|---:|---:|---:|---:|---:|
| bind | numpy | 256 | 100 | 0.038 | 17805824 | — |
| bundle | numpy | 256 | 100 | 0.159 | 17805824 | — |
| similarity | numpy | 256 | 100 | 0.134 | 17805824 | — |
| cleanup | numpy | 256 | 100 | 0.198 | 17805824 | 0.273438 |
| cleanup_hrr | numpy | 256 | 100 | 0.105 | 17805824 | 0.328125 |
| bind_hrr | numpy | 256 | 100 | 0.160 | 17805824 | — |
| bind | numpy | 256 | 1000 | 0.037 | 33305600 | — |
| bundle | numpy | 256 | 1000 | 3.886 | 33305600 | — |
| similarity | numpy | 256 | 1000 | 0.133 | 33305600 | — |
| cleanup | numpy | 256 | 1000 | 1.697 | 33305600 | 0.273438 |
| cleanup_hrr | numpy | 256 | 1000 | 0.866 | 33305600 | 0.328125 |
| bind_hrr | numpy | 256 | 1000 | 0.157 | 33305600 | — |
| bind | numpy | 1024 | 100 | 0.123 | 132821504 | — |
| bundle | numpy | 1024 | 100 | 0.588 | 132821504 | — |
| similarity | numpy | 1024 | 100 | 0.508 | 132821504 | — |
| cleanup | numpy | 1024 | 100 | 0.728 | 132821504 | 0.179688 |
| cleanup_hrr | numpy | 1024 | 100 | 0.368 | 132821504 | 0.253906 |
| bind_hrr | numpy | 1024 | 100 | 0.742 | 132821504 | — |
| bind | numpy | 1024 | 1000 | 0.164 | 192558080 | — |
| bundle | numpy | 1024 | 1000 | 22.933 | 192558080 | — |
| similarity | numpy | 1024 | 1000 | 0.516 | 192558080 | — |
| cleanup | numpy | 1024 | 1000 | 6.905 | 192558080 | 0.179688 |
| cleanup_hrr | numpy | 1024 | 1000 | 6.970 | 192558080 | 0.253906 |
| bind_hrr | numpy | 1024 | 1000 | 0.711 | 192558080 | — |
| bind | cupy | TBD | TBD | pending | pending | pending |
| bundle | cupy | TBD | TBD | pending | pending | pending |
| similarity | cupy | TBD | TBD | pending | pending | pending |
| cleanup | cupy | TBD | TBD | pending | pending | pending |
| cleanup_hrr | cupy | TBD | TBD | pending | pending | pending |
| bind_hrr | cupy | TBD | TBD | pending | pending | pending |
| bind | mlx | TBD | TBD | pending | pending | pending |
| bundle | mlx | TBD | TBD | pending | pending | pending |
| similarity | mlx | TBD | TBD | pending | pending | pending |
| cleanup | mlx | TBD | TBD | pending | pending | pending |
| cleanup_hrr | mlx | TBD | TBD | pending | pending | pending |
| bind_hrr | mlx | TBD | TBD | pending | pending | pending |

## Accuracy at equal bytes

Each row uses seed 1730 and an independent role/item associative memory. Sum
bound pairs, unbind every role, then select the highest score among the N items.
FHRR uses uniform unit phasors; HRR uses unit-norm Gaussian vectors and circular
correlation for unbinding. The explicit nominal-load convention is FHRR capacity
`dim`, HRR capacity `dim/2`; half load is respectively `N=dim/2` and `N=dim/4`.
Thus the matched pair has the same N. Accuracy is independent of timing K and Q;
there are N queries and N candidates. Full-d FHRR is a larger-load control.
The `sqrt(N/(2*dim))` noise law alone does not define top-1 capacity, and these
loads are clearly beyond a high-reliability regime. No normalisation or denoising
is applied to the composite memory.

| Base d | Representation | Actual dim | N | Bytes/codeword | Correct/N | Top-1 |
|---:|---|---:|---:|---:|---:|---:|
| 256 | FHRR@d | 256 | 128 | 2048 | 35/128 | 0.273438 |
| 256 | FHRR@d/2 | 128 | 64 | 1024 | 26/64 | 0.406250 |
| 256 | HRR@d | 256 | 64 | 1024 | 21/64 | 0.328125 |
| 1024 | FHRR@d | 1024 | 512 | 8192 | 92/512 | 0.179688 |
| 1024 | FHRR@d/2 | 512 | 256 | 4096 | 60/256 | 0.234375 |
| 1024 | HRR@d | 1024 | 256 | 4096 | 65/256 | 0.253906 |

At base d=1024 the matched FHRR row recovers 60/256 and HRR 65/256; at base
d=256 they recover 26/64 and 21/64. These small, single-seed reversals do not
establish an advantage or a negative result: neither backend holds high accuracy.
Multiple independent seeds and a load sweep would be needed for a capacity claim.

## Reproduction and verification

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m bench.operator_bench --backend numpy --d 256,1024 --K 100,1000 --Q 128 --reps 3 --verify --out gpubench/opbench-numpy.json
python -m bench.operator_bench --backend cupy --d 4096,8192,32768 --K 100,1000,10000,100000 --Q 4096 --reps 3 --max-gb 12 --verify --out gpubench/opbench-cupy.json
python -m bench.operator_bench --backend mlx --d 4096 --K 100,1000 --Q 4096 --reps 3 --max-gb 12 --verify --out gpubench/opbench-mlx.json
python -m ruff check bench/operator_bench.py tests/test_operator_bench.py
python -m pytest tests/test_operator_bench.py -q
python -m pytest tests -q
python -m holo.quality.cli check
python -m holo.facts.cli check --strict
```

The measured NumPy verification corners (K=1000, d=256 and 1024) have zero
relative checksum differences for all six operators across repeat runs. Verification
uses float64 sums of absolute values and a 2e-6 relative threshold; any nonfinite
or excessive difference fails. The sum is of absolute values because a signed sum
of near-zero-mean scores cancels: the first CUDA run on the 5090 failed the cleanup
corner at 1.7e-5 with every element agreeing to float32 rounding. Memory-refused verification corners are explicitly marked refused,
not passed. These CPU self-checks do not validate another backend. CUDA/Metal were
unavailable, so GPU rows remain a template. CUDA refuses `CUPY_TF32=1`.
Full per-query argmax histograms and checksums are in the gitignored JSON output.

Validation: focused operator tests 10 passed; full suite 251 passed, 9 skipped
in 165.83 seconds. Ruff reports no violations in the two new Python files.
Quality debt remains 50 (baseline 50); strict facts reports 0 FAIL, 23 WARN.
The large d=32768, K=100000, Q=4096 command was also exercised: both its run
and its K=1000 verification corner were refused under the 12 GB budget before
workload allocation, and the refusal report was written successfully.
