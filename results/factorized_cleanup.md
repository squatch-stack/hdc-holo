# Factorized cleanup: NumPy crossover and GPU template

Squatch Stack, seeded synthetic measurements, 2026-09-12. These are CPU results,
not predictions of RTX 5090 timings. Python 3.12, NumPy 1.26.4, Darwin arm64;
BLAS/OpenMP thread limits one. Q=16 for every setting (reduced from the CLI
256 default for a CPU smoke matrix), seed 1729, one warmup and best of three,
40 resonator iterations maximum, score floor 0.2, coarse=8, top-r=4.
Single-seed recovery has 1/16 resolution; no confidence interval or universal
capacity threshold is inferred. Initial generation/transfers are not timed.

## Definitions and fairness

Use spectral sign -1, integer axis coordinates 0..n-1 and independent uniform
frequencies in [-pi,pi]. The expected kernel is sinc, so distinct integer rows
are approximately orthogonal. This is a favorable discrete synthetic fixture,
not a dense correlated real-capture grid. Product rows follow np.ndindex order.
For each of Q independent bundles, stream N independent random item-position
pairs, sum their bindings, and unbind the first item's phasor. By exchangeability
this samples the query for any pair j. Different loads reuse the same pair
prefixes, positions and target items. Positions may repeat; item phasors are
independent. Queries are not normalized and truth never initializes cleanup.

Brute force imports the operator benchmark's chunked real dot scoring (256-row
chunks), timing, synchronization, random planes, backend and budget helpers.
Coarse-to-fine partitions each axis into contiguous groups (unequal partitions
are supported). Cell representatives are products of L2-normalized axis sums,
equivalent to normalized centroids of the product rows in each cell. Stage one
ranks absolute complex overlap, matching BandedDispatcher's selection shape;
stage two generates fine rows from the factor books and calls the same real
scorer as brute force. Candidate generation, sorting and host reductions are
included in time. Precomputed centroids are included in held codebook bytes.
Using representative fine rows alone would miss the many approximately
orthogonal positions within a cell. Even centroids dilute signal as cells grow.

Resonator calls holo.resonator.resonator without restarts or truth-derived
initialization: synchronous associative projections, default sum initialization,
phase projection, magnitude readout. Its factor metric differs from the real
scene score used by brute force. Convergence and correctness are reported
separately, including correct but unconverged readouts. Each iteration has
projection, reconstruction and readout work proportional to 3nd; the count of
3n inner products alone omits reconstruction and convergence-readout constants.

`bytes_held` counts the resident numeric dictionary plus the Q complex queries
for each isolated method. It excludes transient candidate arrays, score tiles,
resonator conjugates/estimates and Python metadata; it is not process RSS. The
whole-harness preallocation guard conservatively includes those temporaries and
host/device copies, using the imported operator budget plus factor/query scratch.
It retains that helper's conservative d-squared allowance although items here
are streamed. The shared timing harness holds all methods, so do not interpret
one method's bytes as the harness's total. Allocator and BLAS caches are excluded.

## Measured matrix

Seconds per Q-query batch. Recovery, convergence, spurious convergence and true
cell selection are fractions. A dash means the diagnostic does not apply.
All execution backends in this table are NumPy.

| d | n | K | N | N K/d | Cleanup | Seconds | Bytes held | Codebook bytes | Top-1 | Mean iterations | Converged | Spurious | True cell selected |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 512 | 1 | 0.5 | brute | 0.000585 | 4325376 | 4194304 | 1.0000 | — | — | — | — |
| 1024 | 8 | 512 | 1 | 0.5 | coarse_fine | 0.002528 | 4521984 | 4390912 | 1.0000 | — | — | — | 1.0000 |
| 1024 | 8 | 512 | 1 | 0.5 | resonator | 0.011480 | 327680 | 196608 | 1.0000 | 5.2500 | 1.0000 | 0.0000 | — |
| 1024 | 8 | 512 | 2 | 1 | brute | 0.000586 | 4325376 | 4194304 | 1.0000 | — | — | — | — |
| 1024 | 8 | 512 | 2 | 1 | coarse_fine | 0.002532 | 4521984 | 4390912 | 1.0000 | — | — | — | 1.0000 |
| 1024 | 8 | 512 | 2 | 1 | resonator | 0.011607 | 327680 | 196608 | 1.0000 | 5.4375 | 1.0000 | 0.0000 | — |
| 1024 | 8 | 512 | 4 | 2 | brute | 0.000586 | 4325376 | 4194304 | 1.0000 | — | — | — | — |
| 1024 | 8 | 512 | 4 | 2 | coarse_fine | 0.002689 | 4521984 | 4390912 | 1.0000 | — | — | — | 1.0000 |
| 1024 | 8 | 512 | 4 | 2 | resonator | 0.013063 | 327680 | 196608 | 1.0000 | 6.1875 | 1.0000 | 0.0000 | — |
| 1024 | 8 | 512 | 8 | 4 | brute | 0.000584 | 4325376 | 4194304 | 1.0000 | — | — | — | — |
| 1024 | 8 | 512 | 8 | 4 | coarse_fine | 0.002464 | 4521984 | 4390912 | 1.0000 | — | — | — | 1.0000 |
| 1024 | 8 | 512 | 8 | 4 | resonator | 0.015604 | 327680 | 196608 | 1.0000 | 7.3125 | 1.0000 | 0.0000 | — |
| 1024 | 8 | 512 | 16 | 8 | brute | 0.000582 | 4325376 | 4194304 | 1.0000 | — | — | — | — |
| 1024 | 8 | 512 | 16 | 8 | coarse_fine | 0.002436 | 4521984 | 4390912 | 1.0000 | — | — | — | 1.0000 |
| 1024 | 8 | 512 | 16 | 8 | resonator | 0.019607 | 327680 | 196608 | 1.0000 | 9.6250 | 1.0000 | 0.0000 | — |
| 1024 | 16 | 4096 | 1 | 4 | brute | 0.005087 | 33685504 | 33554432 | 1.0000 | — | — | — | — |
| 1024 | 16 | 4096 | 1 | 4 | coarse_fine | 0.004489 | 4718592 | 4587520 | 1.0000 | — | — | — | 1.0000 |
| 1024 | 16 | 4096 | 1 | 4 | resonator | 0.017215 | 524288 | 393216 | 1.0000 | 6.3125 | 1.0000 | 0.0000 | — |
| 1024 | 16 | 4096 | 2 | 8 | brute | 0.004977 | 33685504 | 33554432 | 1.0000 | — | — | — | — |
| 1024 | 16 | 4096 | 2 | 8 | coarse_fine | 0.004542 | 4718592 | 4587520 | 1.0000 | — | — | — | 1.0000 |
| 1024 | 16 | 4096 | 2 | 8 | resonator | 0.018602 | 524288 | 393216 | 1.0000 | 6.9375 | 1.0000 | 0.0000 | — |
| 1024 | 16 | 4096 | 4 | 16 | brute | 0.004920 | 33685504 | 33554432 | 1.0000 | — | — | — | — |
| 1024 | 16 | 4096 | 4 | 16 | coarse_fine | 0.004436 | 4718592 | 4587520 | 1.0000 | — | — | — | 1.0000 |
| 1024 | 16 | 4096 | 4 | 16 | resonator | 0.030984 | 524288 | 393216 | 1.0000 | 12.1250 | 1.0000 | 0.0000 | — |
| 1024 | 16 | 4096 | 8 | 32 | brute | 0.005118 | 33685504 | 33554432 | 1.0000 | — | — | — | — |
| 1024 | 16 | 4096 | 8 | 32 | coarse_fine | 0.004377 | 4718592 | 4587520 | 1.0000 | — | — | — | 1.0000 |
| 1024 | 16 | 4096 | 8 | 32 | resonator | 0.053476 | 524288 | 393216 | 0.8750 | 21.3125 | 0.8750 | 0.0625 | — |
| 1024 | 16 | 4096 | 16 | 64 | brute | 0.005190 | 33685504 | 33554432 | 1.0000 | — | — | — | — |
| 1024 | 16 | 4096 | 16 | 64 | coarse_fine | 0.004413 | 4718592 | 4587520 | 0.9375 | — | — | — | 0.9375 |
| 1024 | 16 | 4096 | 16 | 64 | resonator | 0.083753 | 524288 | 393216 | 0.4375 | 33.6250 | 0.4375 | 0.0000 | — |
| 1024 | 32 | 32768 | 1 | 32 | brute | 0.042622 | 268566528 | 268435456 | 1.0000 | — | — | — | — |
| 1024 | 32 | 32768 | 1 | 32 | coarse_fine | 0.016600 | 5111808 | 4980736 | 1.0000 | — | — | — | 1.0000 |
| 1024 | 32 | 32768 | 1 | 32 | resonator | 0.040784 | 917504 | 786432 | 1.0000 | 10.1875 | 1.0000 | 0.0000 | — |
| 1024 | 32 | 32768 | 2 | 64 | brute | 0.042369 | 268566528 | 268435456 | 1.0000 | — | — | — | — |
| 1024 | 32 | 32768 | 2 | 64 | coarse_fine | 0.016638 | 5111808 | 4980736 | 0.8125 | — | — | — | 0.8125 |
| 1024 | 32 | 32768 | 2 | 64 | resonator | 0.087517 | 917504 | 786432 | 0.8750 | 23.3125 | 0.8750 | 0.0000 | — |
| 1024 | 32 | 32768 | 4 | 128 | brute | 0.042821 | 268566528 | 268435456 | 1.0000 | — | — | — | — |
| 1024 | 32 | 32768 | 4 | 128 | coarse_fine | 0.016490 | 5111808 | 4980736 | 0.4375 | — | — | — | 0.4375 |
| 1024 | 32 | 32768 | 4 | 128 | resonator | 0.126439 | 917504 | 786432 | 0.3125 | 33.7500 | 0.2500 | 0.0000 | — |
| 1024 | 32 | 32768 | 8 | 256 | brute | 0.042588 | 268566528 | 268435456 | 1.0000 | — | — | — | — |
| 1024 | 32 | 32768 | 8 | 256 | coarse_fine | 0.016190 | 5111808 | 4980736 | 0.1875 | — | — | — | 0.1875 |
| 1024 | 32 | 32768 | 8 | 256 | resonator | 0.150607 | 917504 | 786432 | 0.0000 | 40.0000 | 0.0000 | 0.0000 | — |
| 1024 | 32 | 32768 | 16 | 512 | brute | 0.042694 | 268566528 | 268435456 | 1.0000 | — | — | — | — |
| 1024 | 32 | 32768 | 16 | 512 | coarse_fine | 0.016545 | 5111808 | 4980736 | 0.0625 | — | — | — | 0.0625 |
| 1024 | 32 | 32768 | 16 | 512 | resonator | 0.149856 | 917504 | 786432 | 0.0000 | 40.0000 | 0.0000 | 0.0000 | — |
| 4096 | 8 | 512 | 1 | 0.125 | brute | 0.003558 | 17301504 | 16777216 | 1.0000 | — | — | — | — |
| 4096 | 8 | 512 | 1 | 0.125 | coarse_fine | 0.006973 | 18087936 | 17563648 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 8 | 512 | 1 | 0.125 | resonator | 0.027886 | 1310720 | 786432 | 1.0000 | 4.6250 | 1.0000 | 0.0000 | — |
| 4096 | 8 | 512 | 2 | 0.25 | brute | 0.003434 | 17301504 | 16777216 | 1.0000 | — | — | — | — |
| 4096 | 8 | 512 | 2 | 0.25 | coarse_fine | 0.007030 | 18087936 | 17563648 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 8 | 512 | 2 | 0.25 | resonator | 0.028188 | 1310720 | 786432 | 1.0000 | 4.7500 | 1.0000 | 0.0000 | — |
| 4096 | 8 | 512 | 4 | 0.5 | brute | 0.003578 | 17301504 | 16777216 | 1.0000 | — | — | — | — |
| 4096 | 8 | 512 | 4 | 0.5 | coarse_fine | 0.007165 | 18087936 | 17563648 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 8 | 512 | 4 | 0.5 | resonator | 0.031035 | 1310720 | 786432 | 1.0000 | 5.2500 | 1.0000 | 0.0000 | — |
| 4096 | 8 | 512 | 8 | 1 | brute | 0.003521 | 17301504 | 16777216 | 1.0000 | — | — | — | — |
| 4096 | 8 | 512 | 8 | 1 | coarse_fine | 0.007212 | 18087936 | 17563648 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 8 | 512 | 8 | 1 | resonator | 0.032049 | 1310720 | 786432 | 1.0000 | 5.4375 | 1.0000 | 0.0000 | — |
| 4096 | 8 | 512 | 16 | 2 | brute | 0.003429 | 17301504 | 16777216 | 1.0000 | — | — | — | — |
| 4096 | 8 | 512 | 16 | 2 | coarse_fine | 0.007248 | 18087936 | 17563648 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 8 | 512 | 16 | 2 | resonator | 0.037080 | 1310720 | 786432 | 1.0000 | 6.5000 | 1.0000 | 0.0000 | — |
| 4096 | 16 | 4096 | 1 | 1 | brute | 0.029029 | 134742016 | 134217728 | 1.0000 | — | — | — | — |
| 4096 | 16 | 4096 | 1 | 1 | coarse_fine | 0.014069 | 18874368 | 18350080 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 16 | 4096 | 1 | 1 | resonator | 0.050286 | 2097152 | 1572864 | 1.0000 | 5.5000 | 1.0000 | 0.0000 | — |
| 4096 | 16 | 4096 | 2 | 2 | brute | 0.029009 | 134742016 | 134217728 | 1.0000 | — | — | — | — |
| 4096 | 16 | 4096 | 2 | 2 | coarse_fine | 0.013601 | 18874368 | 18350080 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 16 | 4096 | 2 | 2 | resonator | 0.054376 | 2097152 | 1572864 | 1.0000 | 6.0000 | 1.0000 | 0.0000 | — |
| 4096 | 16 | 4096 | 4 | 4 | brute | 0.029564 | 134742016 | 134217728 | 1.0000 | — | — | — | — |
| 4096 | 16 | 4096 | 4 | 4 | coarse_fine | 0.013536 | 18874368 | 18350080 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 16 | 4096 | 4 | 4 | resonator | 0.060005 | 2097152 | 1572864 | 1.0000 | 6.7500 | 1.0000 | 0.0000 | — |
| 4096 | 16 | 4096 | 8 | 8 | brute | 0.030253 | 134742016 | 134217728 | 1.0000 | — | — | — | — |
| 4096 | 16 | 4096 | 8 | 8 | coarse_fine | 0.014230 | 18874368 | 18350080 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 16 | 4096 | 8 | 8 | resonator | 0.070875 | 2097152 | 1572864 | 1.0000 | 8.0625 | 1.0000 | 0.0000 | — |
| 4096 | 16 | 4096 | 16 | 16 | brute | 0.029533 | 134742016 | 134217728 | 1.0000 | — | — | — | — |
| 4096 | 16 | 4096 | 16 | 16 | coarse_fine | 0.014413 | 18874368 | 18350080 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 16 | 4096 | 16 | 16 | resonator | 0.126047 | 2097152 | 1572864 | 0.9375 | 14.8125 | 0.9375 | 0.0000 | — |
| 4096 | 32 | 32768 | 1 | 8 | brute | 0.237589 | 1074266112 | 1073741824 | 1.0000 | — | — | — | — |
| 4096 | 32 | 32768 | 1 | 8 | coarse_fine | 0.057450 | 20447232 | 19922944 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 32 | 32768 | 1 | 8 | resonator | 0.100737 | 3670016 | 3145728 | 1.0000 | 6.6250 | 1.0000 | 0.0000 | — |
| 4096 | 32 | 32768 | 2 | 16 | brute | 0.225040 | 1074266112 | 1073741824 | 1.0000 | — | — | — | — |
| 4096 | 32 | 32768 | 2 | 16 | coarse_fine | 0.055905 | 20447232 | 19922944 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 32 | 32768 | 2 | 16 | resonator | 0.107005 | 3670016 | 3145728 | 1.0000 | 7.2500 | 1.0000 | 0.0000 | — |
| 4096 | 32 | 32768 | 4 | 32 | brute | 0.226043 | 1074266112 | 1073741824 | 1.0000 | — | — | — | — |
| 4096 | 32 | 32768 | 4 | 32 | coarse_fine | 0.057379 | 20447232 | 19922944 | 1.0000 | — | — | — | 1.0000 |
| 4096 | 32 | 32768 | 4 | 32 | resonator | 0.131965 | 3670016 | 3145728 | 1.0000 | 9.1875 | 1.0000 | 0.0000 | — |
| 4096 | 32 | 32768 | 8 | 64 | brute | 0.230649 | 1074266112 | 1073741824 | 1.0000 | — | — | — | — |
| 4096 | 32 | 32768 | 8 | 64 | coarse_fine | 0.055651 | 20447232 | 19922944 | 0.8750 | — | — | — | 0.8750 |
| 4096 | 32 | 32768 | 8 | 64 | resonator | 0.288624 | 3670016 | 3145728 | 0.9375 | 20.8750 | 0.9375 | 0.0000 | — |
| 4096 | 32 | 32768 | 16 | 128 | brute | 0.228137 | 1074266112 | 1073741824 | 1.0000 | — | — | — | — |
| 4096 | 32 | 32768 | 16 | 128 | coarse_fine | 0.058509 | 20447232 | 19922944 | 0.3125 | — | — | — | 0.3125 |
| 4096 | 32 | 32768 | 16 | 128 | resonator | 0.517162 | 3670016 | 3145728 | 0.1875 | 38.5000 | 0.1250 | 0.0000 | — |

## Observed crossovers

First sampled K faster than brute, independently for each load. These are
observed grid points, not an interpolated crossover or a guarantee of accuracy.

| d | Load N | Coarse-to-fine first faster K | Resonator first faster K |
|---:|---:|---:|---:|
| 1024 | 1 | 4096 | 32768 |
| 1024 | 2 | 4096 | none through 32768 |
| 1024 | 4 | 4096 | none through 32768 |
| 1024 | 8 | 4096 | none through 32768 |
| 1024 | 16 | 4096 | none through 32768 |
| 4096 | 1 | 4096 | 32768 |
| 4096 | 2 | 4096 | 32768 |
| 4096 | 4 | 4096 | 32768 |
| 4096 | 8 | 4096 | none through 32768 |
| 4096 | 16 | 4096 | none through 32768 |

First tested load where resonator recovery is strictly below brute force:

| d | n | K | First lower-recovery N | N K/d |
|---:|---:|---:|---:|---:|
| 1024 | 8 | 512 | none through 16 | — |
| 1024 | 16 | 4096 | 8 | 32.0 |
| 1024 | 32 | 32768 | 2 | 64.0 |
| 4096 | 8 | 512 | none through 16 | — |
| 4096 | 16 | 4096 | 16 | 16.0 |
| 4096 | 32 | 32768 | 8 | 64.0 |

Brute force wins at K=512 for every tested load and d. Coarse-to-fine overtakes
at K=4096 but its time advantage can mask missed cells: for d=1024, n=32,
N=16 it selects the true cell for only 1/16 queries, and recovers exactly that
one. Full cell coverage matches brute force in the seeded test, including an
uneven partition. In this matrix all coarse recovery losses coincide with
failure to shortlist the true cell.

The resonator cliff is visible as N K/d increases, but this ratio is a workload
coordinate, not a proved universal capacity law. At d=1024, n=16 recovery first
falls at N=8 (ratio 32); at n=32 it first falls at N=2 (ratio 64). At d=4096,
n=16 it first falls at N=16 (ratio 16), and at n=32 at N=8 (ratio 64). Frequency
correlation, finite iterations, initialization and finite sampling also matter.
Brute recovery is 1.0 throughout the requested load matrix; the test's wider
N=1,32,256 sweep demonstrates degradation of every method.

Spurious fixed points are not hidden: the table reports convergence, correctness
and the converged-but-incorrect fraction separately. Retained examples:

- d=1024, n=16, N=8: correct 0.8750, converged 0.8750, spurious 0.0625.
- d=1024, n=32, N=4: correct 0.3125, converged 0.2500, spurious 0.0000.
- d=4096, n=32, N=16: correct 0.1875, converged 0.1250, spurious 0.0000.

Repeated process runs exposed float32 sensitivity near the resonator cliff:
terminal readouts sometimes change despite the same seed, because small numeric
perturbations can change the iterative trajectory. Timing is best of three;
checksum, argmax, accuracy and convergence all describe the last timed call,
consistent with the imported clock's diagnostic convention. The deterministic
unit fixtures are intentionally in a stable small-grid regime. Recovery
numbers near failure are illustrative single-run outcomes, not bitwise portable
predictions. More seeds, larger Q and backend comparisons are needed for a
capacity estimate. Increasing iterations alone cannot be assumed to eliminate
stable wrong solutions.

## Prior art and the rotation-product control

Read the abstracts of [HyperSpace, arXiv:2604.15113](https://arxiv.org/abs/2604.15113),
[in-memory factorization, arXiv:2211.05052](https://arxiv.org/abs/2211.05052),
[linearithmic cleanup, arXiv:2506.15793](https://arxiv.org/abs/2506.15793), and
[the nonlinear cleanup comparison](https://doi.org/10.3389/frai.2026.1793314).
HyperSpace motivates operator-level measurement; the in-memory work uses
hardware stochasticity that the existing NumPy resonator does not reproduce.
The nonlinear comparison motivates separating terminal outcomes. These are
our workload and cell definitions, not mirrored experimental definitions.

Read the [rotation-product paper](https://arxiv.org/pdf/2506.15793), particularly
Eq. 5 and Algorithms 1–2. Its construction is reproducible: `krop_row` and
`cleanup_krop` implement the row reconstruction and butterfly, and a seeded
1024-dimensional test compares argmax, score and checksum against a dense
rotation-product dictionary. The implementation follows the paper; no source
implementation was copied. It is a real square K=d codebook of rotation/reflection
products, not a Cartesian product of three length-d phasor axis books. Therefore
it is a separate control, excluded from this same-dictionary crossover. A useful
capacity/timing comparison would need HRR random keys, circular convolution and
correlation, power-of-two dimensions, and explicit K=d versus our independent
K=n³. Substituting its butterfly for this spectral dictionary is unsupported.

## Reproduction and GPU template

Run from the worktree root:

```sh
HDC_BACKEND=numpy MPLCONFIGDIR=/tmp/faccleanup-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m bench.factorized_cleanup --synthetic --backend numpy --d 1024,4096 --n 8,16,32 --loads 1,2,4,8,16 --Q 16 --reps 3 --max-gb 12 --out /tmp/faccleanup-numpy.json --figure out/factorized_cleanup/crossover.png
HDC_BACKEND=numpy MPLCONFIGDIR=/tmp/faccleanup-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest tests/test_factorized_cleanup.py -q --durations=5
HDC_BACKEND=numpy MPLCONFIGDIR=/tmp/faccleanup-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest tests -q
.venv/bin/ruff check bench/factorized_cleanup.py tests/test_factorized_cleanup.py
.venv/bin/holo-quality check
.venv/bin/holo-facts check --strict
```

Figure: [crossover](../out/factorized_cleanup/crossover.png). It shows the first
and last loads; the complete five-load matrix is above. JSON contains raw
argmax outputs, checksums, histograms, diagnostics and conservative byte bounds.
The JSON goes outside the repository to keep lane edits within the file matrix.

Maintainer GPU command (not run here):

```sh
HDC_BACKEND=numpy CUPY_TF32=0 python -m bench.factorized_cleanup --synthetic --backend cupy --d 4096,8192 --n 8,16,32,64 --loads 1,2,4,8,16 --Q 256 --reps 3 --max-gb 12 --out gpubench/faccleanup-cupy.json
```

CuPy brute/coarse rows run on device, with host diagnostic transfers included.
The existing holo resonator is NumPy-only and its row explicitly reports
`execution_backend=numpy`, using pretransferred host inputs. No GPU factorized
crossover claim is valid from that mixed run. A true GPU resonator requires
backend dispatch in `holo/resonator.py`, outside this lane. No copy of that
inner loop was introduced. The maintainer must address this before comparing
three RTX 5090 methods. TF32 is refused when CUPY_TF32=1. Large settings are
recorded as refused before materialization; n=64 and default Q may need a
larger budget or smaller Q. No GPU or real capture was available here.

The refusal CLI was also exercised before allocation:

```sh
HDC_BACKEND=numpy MPLCONFIGDIR=/tmp/faccleanup-mpl .venv/bin/python -m bench.factorized_cleanup --synthetic --d 8192 --n 64 --loads 1 --Q 256 --reps 1 --max-gb 0.001 --out /tmp/faccleanup-refused.json
```

Result: zero runs, one refusal; predicted 143883534336 bytes exceeds 0.001 GB.

## Validation and handoff

Pre-flight output (inside this worktree):

```text
<worktree>/holo/__init__.py
```

- Focused tests: 14 passed in 0.41s; slowest test 0.09s.
- Full suite: 367 passed, 9 skipped in 21.01s.
- Ruff: All checks passed! (both lane Python files).
- holo-quality: lint debt: 50 (baseline 50).
- holo-facts --strict: 0 FAIL, 25 WARN; no tests.count failure.
- Figure rendered and visually inspected; docs/figures.md contains one new row.
- git diff --check passed. No commits, pushes, branch changes or installs.

Two PLR0913 suppressions retain the requested benchmark entry-point signatures
with search configuration; all other lane lint rules pass without suppression.
Only the four exclusive artifacts and the allowed figure-registry row changed.
The suggested GPU-dispatch change to holo/resonator.py was not made.
