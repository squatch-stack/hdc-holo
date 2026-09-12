# Operator benchmark: CPU template and 5090 rows

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
  Codebook-only bytes are 8Kd for FHRR and 4Kd for HRR. Accuracy storage is
  capped by `max_items`; the specified conservative budget still refuses some
  large-d rows (see the memory table below).

## Measured operator rows

Times are milliseconds. These historical timing rows and their original byte
predictions are retained unchanged. Their old 50%-load accuracy annotations are
superseded by the shared-N curves below, not recomputed timing measurements.

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
| bind | cupy | 4096 | 100 | 0.575 | 2589995520 | — |
| bundle | cupy | 4096 | 100 | 0.092 | 2589995520 | — |
| similarity | cupy | 4096 | 100 | 0.757 | 2589995520 | — |
| cleanup | cupy | 4096 | 100 | 0.554 | 2589995520 | 0.0903 (FHRR@d, N=2048) / 0.1191 (FHRR@d/2, N=1024) |
| cleanup_hrr | cupy | 4096 | 100 | 0.445 | 2589995520 | 0.1035 (HRR@d, N=1024) |
| bind_hrr | cupy | 4096 | 100 | 0.369 | 2589995520 | — |
| bind | cupy | 4096 | 1000 | 0.574 | 2846487552 | — |
| bundle | cupy | 4096 | 1000 | 0.341 | 2846487552 | — |
| similarity | cupy | 4096 | 1000 | 0.753 | 2846487552 | — |
| cleanup | cupy | 4096 | 1000 | 4.123 | 2846487552 | 0.0903 (FHRR@d, N=2048) / 0.1191 (FHRR@d/2, N=1024) |
| cleanup_hrr | cupy | 4096 | 1000 | 3.396 | 2846487552 | 0.1035 (HRR@d, N=1024) |
| bind_hrr | cupy | 4096 | 1000 | 0.370 | 2846487552 | — |
| bind | cupy | 4096 | 10000 | 0.579 | 5206935552 | — |
| bundle | cupy | 4096 | 10000 | 3.205 | 5206935552 | — |
| similarity | cupy | 4096 | 10000 | 0.770 | 5206935552 | — |
| cleanup | cupy | 4096 | 10000 | 40.854 | 5206935552 | 0.0903 (FHRR@d, N=2048) / 0.1191 (FHRR@d/2, N=1024) |
| cleanup_hrr | cupy | 4096 | 10000 | 33.565 | 5206935552 | 0.1035 (HRR@d, N=1024) |
| bind_hrr | cupy | 4096 | 10000 | 0.368 | 5206935552 | — |
| bind | cupy | 8192 | 100 | 1.220 | 7850701312 | — |
| bundle | cupy | 8192 | 100 | 0.086 | 7850701312 | — |
| similarity | cupy | 8192 | 100 | 0.947 | 7850701312 | — |
| cleanup | cupy | 8192 | 100 | 0.743 | 7850701312 | 0.0593 (FHRR@d, N=4096) / 0.0903 (FHRR@d/2, N=2048) |
| cleanup_hrr | cupy | 8192 | 100 | 0.554 | 7850701312 | 0.0850 (HRR@d, N=2048) |
| bind_hrr | cupy | 8192 | 100 | 1.066 | 7850701312 | — |
| bind | cupy | 8192 | 1000 | 1.221 | 8343122944 | — |
| bundle | cupy | 8192 | 1000 | 0.399 | 8343122944 | — |
| similarity | cupy | 8192 | 1000 | 0.946 | 8343122944 | — |
| cleanup | cupy | 8192 | 1000 | 5.464 | 8343122944 | 0.0593 (FHRR@d, N=4096) / 0.0903 (FHRR@d/2, N=2048) |
| cleanup_hrr | cupy | 8192 | 1000 | 4.069 | 8343122944 | 0.0850 (HRR@d, N=2048) |
| bind_hrr | cupy | 8192 | 1000 | 1.060 | 8343122944 | — |
| bind | cupy | 8192 | 10000 | 1.225 | 13062866944 | — |
| bundle | cupy | 8192 | 10000 | 3.906 | 13062866944 | — |
| similarity | cupy | 8192 | 10000 | 0.951 | 13062866944 | — |
| cleanup | cupy | 8192 | 10000 | 54.053 | 13062866944 | 0.0593 (FHRR@d, N=4096) / 0.0903 (FHRR@d/2, N=2048) |
| cleanup_hrr | cupy | 8192 | 10000 | 40.420 | 13062866944 | 0.0850 (HRR@d, N=2048) |
| bind_hrr | cupy | 8192 | 10000 | 1.062 | 13062866944 | — |
| bind | mlx | TBD | TBD | pending | pending | pending |
| bundle | mlx | TBD | TBD | pending | pending | pending |
| similarity | mlx | TBD | TBD | pending | pending | pending |
| cleanup | mlx | TBD | TBD | pending | pending | pending |
| cleanup_hrr | mlx | TBD | TBD | pending | pending | pending |
| bind_hrr | mlx | TBD | TBD | pending | pending | pending |

## Accuracy at equal bytes

Each curve point uses seed 1730 and an independent role/item associative memory.
Sum bound pairs, unbind every role, then select the highest score among N items.
FHRR uses uniform unit phasors; HRR uses unit-norm Gaussian vectors and circular
correlation for unbinding. These are our definitions, not a reproduction of
HyperSpace's iterative cleanup. No normalisation or denoising is applied to the
composite memory.

The shared nominal is `d // 4`. At each requested load all three rows use
`N = max(1, min(max_items, round(load * (d // 4))))`, including the full-d
FHRR control. Defaults are loads 0.02, 0.05, 0.1, 0.2, 0.5 and max_items=4096.
The nominal is a plotting convention, not a proven capacity boundary. Rounded
or capped loads may repeat N. Accuracy is independent of timing K and Q: each
point has N queries and N candidates. Changing N regenerates the seeded arrays;
the points are not nested subsets of one memory.

JSON now stores `accuracy.loads` and `accuracy.rows[label]`, with aligned `N`
and `top1` arrays and scalar `dim`, `bytes_per_codeword`, `accuracy_backend`.
FHRR@d/2 and HRR@d have equal codeword bytes; FHRR@d uses twice those bytes.

Verdict rule: compare FHRR@d/2 against HRR@d at every load, and highlight the
largest tested load where HRR@d recovers at least 0.9. A lower FHRR@d/2 top-1
there is a negative result. If no tested load qualifies, no useful-recovery
verdict is available. A single seeded curve does not establish a capacity claim.

NumPy synthetic runs: K=1000, Q=128, best of three timing repetitions, one BLAS
thread. The complete three-d run including K=1000 verification corners took
3.218480 seconds. All six operator checksum errors were zero at every corner.

| Base d | Representation | N at loads 0.02 / 0.05 / 0.1 / 0.2 / 0.5 | Bytes/codeword | Top-1 at 0.02 | 0.05 | 0.1 | 0.2 | 0.5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 256 | FHRR@d | 1 / 3 / 6 / 13 / 32 | 2048 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.968750 |
| 256 | FHRR@d/2 | 1 / 3 / 6 / 13 / 32 | 1024 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.781250 |
| 256 | HRR@d | 1 / 3 / 6 / 13 / 32 | 1024 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.718750 |
| 1024 | FHRR@d | 5 / 13 / 26 / 51 / 128 | 8192 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.890625 |
| 1024 | FHRR@d/2 | 5 / 13 / 26 / 51 / 128 | 4096 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.601562 |
| 1024 | HRR@d | 5 / 13 / 26 / 51 / 128 | 4096 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.601562 |
| 4096 | FHRR@d | 20 / 51 / 102 / 205 / 512 | 32768 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.794922 |
| 4096 | FHRR@d/2 | 20 / 51 / 102 / 205 / 512 | 16384 | 1.000000 | 1.000000 | 1.000000 | 0.946341 | 0.408203 |
| 4096 | HRR@d | 20 / 51 / 102 / 205 / 512 | 16384 | 1.000000 | 1.000000 | 1.000000 | 0.946341 | 0.417969 |

The largest qualifying load is 0.2 in all three dimensions. The matched pair
ties at 1.0 for d=256 and 1024 and at 0.946341 for d=4096: no negative at the
qualifying load. At loads 0.02, 0.05 and 0.1 every row is perfect. At load 0.5,
FHRR@d/2 is higher than HRR@d for d=256, tied for d=1024, and lower by 0.009766
for d=4096; HRR recovery is below 0.9 at all three of those points.

## Reproduction and verification

```sh
HDC_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m bench.operator_bench --backend numpy --d 256,1024,4096 --K 1000 --Q 128 --reps 3 --loads 0.02,0.05,0.1,0.2,0.5 --max-items 4096 --verify --out /tmp/oploadcurve-numpy.json
python -m bench.operator_bench --backend cupy --d 4096,8192,32768 --K 100,1000,10000 --Q 4096 --reps 3 --max-gb 26 --verify --out gpubench/opbench-cupy.json
python -m bench.operator_bench --backend mlx --d 4096 --K 100,1000 --Q 4096 --reps 3 --max-gb 12 --verify --out gpubench/opbench-mlx.json
python -m ruff check bench/operator_bench.py tests/test_operator_bench.py
python -m pytest tests/test_operator_bench.py -q
python -m pytest tests -q
python -m holo.quality.cli check
python -m holo.facts.cli check --strict
```

The measured NumPy verification corners (K=1000, d=256, 1024 and 4096) have zero
relative checksum differences for all six operators across repeat runs. Verification
uses float64 sums of absolute values and a 2e-6 relative threshold; any nonfinite
or excessive difference fails. The sum is of absolute values because a signed sum
of near-zero-mean scores cancels: the first CUDA run on the 5090 failed the cleanup
corner at 1.7e-5 with every element agreeing to float32 rounding. Memory-refused verification corners are explicitly marked refused,
not passed. These CPU self-checks do not validate another backend. This lane used only NumPy; the historical 5090 timing rows are retained and
new GPU curves remain a template. CUDA refuses `CUPY_TF32=1`.
Full per-query argmax histograms and checksums are in the gitignored JSON output.

Load-curve validation: focused tests 22 passed, 1 xfailed in 0.10 seconds;
slowest fixture setup 0.02 seconds. Full suite: 3 failed, 362 passed, 9 skipped,
1 xfailed in 20.52 seconds. All three failures are the same `tests.count`
registry mismatch (321 registered versus 328 derived). The registry is outside
this lane. The strict expected failure records the contradictory 12 GB target.
Ruff: All checks passed! Quality: lint debt: 50 (baseline 50).
Strict facts: 1 FAIL, 25 WARN; the only FAIL is `tests.count`.

The measured total runtime above wraps `main` with `time.perf_counter`, including
all three NumPy runs, verification reruns and JSON writing. Reproduce it with:

```sh
HDC_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python - <<'PYTHON'
import time
from bench.operator_bench import main
start = time.perf_counter()
main(['--backend', 'numpy', '--d', '256,1024,4096', '--K', '1000',
      '--Q', '128', '--reps', '3', '--loads', '0.02,0.05,0.1,0.2,0.5',
      '--max-items', '4096', '--verify', '--out', '/tmp/oploadcurve-numpy.json'])
print('Total runtime including verification:', time.perf_counter() - start)
PYTHON
```

## 5090 rows (2026-09-12)

RTX 5090, CuPy 13.6, TF32 off, best of 3, `--verify` corners at d=4096 and
8192 (K=1000) agreeing with NumPy to 2e-7 or better on every operator once
the checksum summed absolute values (see the verification paragraph). K=10⁴
at d=8192 needed `--max-gb 26`; K=10⁵ at any d and every d=32768 row were
refused by the byte prediction (28.8 GB, 60 GB, and 96–305 GB) — the
historical refusals used the old quadratic accuracy budget.

- **Cleanup dominates, as HyperSpace reports.** At K=10⁴ cleanup is
  40.9 ms (FHRR) and 33.6 ms (HRR) at d=4096, 54.1 and 40.4 ms at
  d=8192, against sub-millisecond bind, similarity and bind_hrr and a
  3–4 ms bundle. HRR cleanup is 18–25% cheaper at the same d (one real
  GEMM against two); at equal bytes (FHRR at d/2) the FLOPs are equal by
  construction.
- **The equal-bytes accuracy row leans FHRR on the GPU**, the opposite
  of the CPU corner: at 16 KB/codeword FHRR@d/2 recovers 11.9% against
  HRR@d 10.4% (d=4096); at 32 KB, 9.0% against 8.5% (d=8192). Do not
  read those historical failure-regime values as a capacity result; use the
  shared-N curves and verdict rule above.

The accuracy probe now sweeps the shared-N loads described above, with a capped
item count. The timing rows above stand as historical measurements. Fill this
curve template from the maintainer's new 5090 JSON, once per load and dimension:

| Backend | Base d | Load | Shared N | FHRR@d top-1 | FHRR@d/2 top-1 | HRR@d top-1 | FHRR@d/2 bytes | HRR@d bytes | Verdict at largest HRR >=0.9 load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cupy | 4096 / 8192 / 32768 | pending | pending | pending | pending | pending | pending | pending | pending |

The prescribed memory replacement is `20*d*min(d,max_items)` inside the
itemsize multiplier, leaving all other terms unchanged. At d=32768, Q=4096,
float32 and max_items=4096:

| K | Predicted bytes | Decimal GB | Fits 12 GB | Fits 26 GB |
|---:|---:|---:|---|---|
| 100 | 20624454144 | 20.624454144 | no | yes |
| 1000 | 22532453376 | 22.532453376 | no | yes |
| 10000 | 41407973376 | 41.407973376 | no | no |

The brief's under-12-GB target is incompatible with its formula. The accuracy
term alone is 10,737,418,240 bytes, and the K=10000 resident-plane term alone
exceeds 12 GB even with Q=1. The formula was retained exactly; its under-12-GB
test is a strict expected failure. The maintainer's 26 GB command will still
refuse d=32768, K=10000, though its K=1000 verification corner fits the
prediction. Reaching that budget requires a separately reviewed allocation
model or workload change, not simply the requested accuracy cap.

Reproduce the byte table without allocating workloads:

```sh
HDC_BACKEND=numpy .venv/bin/python - <<'PYTHON'
from bench.operator_bench import predict_bytes
for k in (100, 1000, 10000):
    print(k, predict_bytes(32768, k, 4096, max_items=4096))
PYTHON
```
