# Storage position: bundle payloads and compressed splat assets

## 1. What the 384× compares

The [baseline table](baseline_table.md) compares complex64 cell bundles at
d=8192 with an SPZ v3 asset for the same scene. Its approximately 48k-splat
row is 383 MB versus 1 MB, reported as 384× using the underlying sizes;
the reconstruction table reports 17.4% bundle field error versus rounded
0.0% SPZ error. This is field reconstruction error, not a universal visual
quality loss or codec distortion metric. At 4,778 splats the ratio is
1475×; at 11,985 it is 904×; at 24,061 it is 589×. Cell occupancy, partition,
and scene size determine the ratio. The README headline describes that
baseline, not a constant cost of the representation.

## 2. The asset side

The same baseline measures PLY (SH-0, this repository's writer) at
85 B/splat, SPZ v3 at 22 B/splat, and SOG at 18 B/splat. We use the measured
22 here, rather than `save_spz`'s approximate 20. These are fixture-specific
rates, not format guarantees; SH content and codec settings matter.
The gallery ships SOG; [real-scenes documentation](../docs/real-scenes.md)
and [the SDK](../SDK.md) describe that export/viewer path.
[`holo/sog.py`](../holo/sog.py) is writer-only: ecosystem readers consume it,
and the tests decode with specification arithmetic.

Abstracts read online for this note:

- [EntropyGS, arXiv:2508.10227](https://arxiv.org/abs/2508.10227): models
  attribute distributions for adaptive quantization and entropy coding,
  reporting about 30× rate reduction with similar rendering quality.
- [Compression in 3D Gaussian Splatting, arXiv:2502.19457](https://arxiv.org/abs/2502.19457):
  surveys compression through a topology-based taxonomy, comparing fidelity,
  rate, and computational efficiency.
- [SUCCESS-GS, arXiv:2512.07197](https://arxiv.org/abs/2512.07197): organizes
  static and dynamic Gaussian compression into parameter compression and
  restructuring, with benchmarks and evaluation metrics.

EntropyGS's approximately 30× is relative to its input 3DGS data, not an
additional factor over SPZ or SOG. Dividing our 85 B/splat SH-0 fixture by
30 would mix baselines. No entropy codec was implemented or benchmarked
here; the accounting definitions below are Squatch Stack's own.

## 3. Bundle bytes at the knee

Source: [`gpu_sweep_both.json`](gpu_sweep_both.json), all twelve rows.
The exact keys used are `scene`, `splats_encoded`, `cells_per_band.xfine`,
`cells_per_band.fine`, `cells_per_band.mid`, `cells_per_band.coarse`, and
`cells`. Sum the four per-band counts and require equality with `cells`.
Missing counts cause an error naming the missing key; inconsistent counts
also fail. `splats_encoded`, rather than `splats_loaded`, keeps the asset
comparison on the same cropped scene. The SPZ sizes below are arithmetic
models at 22 B/splat, not new file-size measurements. The older
`gpu_sweep.json` has different cell counts and is not mixed into this table.
The tool accepts that file separately with `--sweep`.

`holo/capture.py` defines four `BANDS` with cell sizes 1/32, 1/32, 1/8,
and 1/4, and `DIM=8192`. This calculation preserves the recorded partition
at all three rate points. One complex64 vector at d=8192 costs 65,536 B;
HG-8 (8/8) at d=8192 costs 16,384 B; 4/4 at d=16384 also costs 16,384 B.
The latter two coincide in payload bytes and differ in error. All MB below
are decimal (1,000,000 B); the knee is 16 KiB per cell. The ratio column is
knee/SPZ; HG-8 has the same ratio and complex64 has four times that ratio.

This is one vector payload per recorded cell, excluding codec headers,
codebooks, cell metadata, extra attribute channels, and CRDT history or
writer shards. It is not a full archive or resident-memory measurement.
Nor does it add the separate `xray_cells` representation in the sweep.

| capture | splats | cells | SPZ MB | complex64 MB | HG-8 MB | 4/4 knee MB | knee/SPZ |
|---|---:|---:|---:|---:|---:|---:|---:|
| brookline-station-2 | 379489 | 11452 | 8.35 | 750.52 | 187.63 | 187.63 | 22.47x |
| brookline-station | 435295 | 14041 | 9.58 | 920.19 | 230.05 | 230.05 | 24.02x |
| cannon | 141735 | 5566 | 3.12 | 364.77 | 91.19 | 91.19 | 29.25x |
| oak | 1158774 | 37288 | 25.49 | 2443.71 | 610.93 | 610.93 | 23.96x |
| redrock-cairn | 368486 | 12793 | 8.11 | 838.40 | 209.60 | 209.60 | 25.86x |
| redrock | 388654 | 11708 | 8.55 | 767.30 | 191.82 | 191.82 | 22.43x |
| research-library-cannon | 311357 | 10906 | 6.85 | 714.74 | 178.68 | 178.68 | 26.09x |
| research-library | 497544 | 15969 | 10.95 | 1046.54 | 261.64 | 261.64 | 23.90x |
| saguaro | 370345 | 9791 | 8.15 | 641.66 | 160.42 | 160.42 | 19.69x |
| springhouse-outside | 622242 | 16318 | 13.69 | 1069.42 | 267.35 | 267.35 | 19.53x |
| wilsons-creek-gun | 428631 | 11630 | 9.43 | 762.18 | 190.55 | 190.55 | 20.21x |
| wilsons-creek | 388238 | 12094 | 8.54 | 792.59 | 198.15 | 198.15 | 23.20x |

Median ratio at the knee: 23.55x SPZ.

The [D2 capture ladder](quant_lowbit.md#real-cells-2026-09-12-two-captures)
is the reason to position bytes at this knee. At the 16,384 B budget,
Cannon's 4/4 at d=16384 has error 0.1175 (max scale, no shrink), while
2/2 at d=32768 reaches only 0.2688 even with p99.9 scaling. Wilson's Creek
has 0.1329 versus 0.2432. Thus 2/2 at 4d is approximately twice as bad as
4/4 at 2d at equal bytes (2.3× and 1.8×). HG-8's corresponding errors are
0.1556 and 0.1882, illustrating why equal bytes do not mean equal error.
This supports the present floor of useful bundle payload bytes on those
two captures, not a universal optimum for every cell or query. The table
extrapolates that rate point across recorded cell counts; it does not
claim new reconstruction measurements for the other ten captures.
D3 block scaling is pending; there is no demonstrated lower-bit win here.

Reproduce the real table and export exact integer bytes and all three
ratios (the JSON output is a temporary artifact):

```sh
HDC_BACKEND=numpy .venv/bin/python -m bench.storage_position --sweep results/gpu_sweep_both.json --json /tmp/storage-position-real.json
```

Seeded synthetic end-to-end arithmetic, with no capture files:

```sh
HDC_BACKEND=numpy .venv/bin/python -m bench.storage_position --synthetic --seed 7 --json /tmp/storage-position-synthetic.json
```

## 4. Bytes per query capability

The [SDK's implemented-feature inventory](../SDK.md) supports a
computational representation argument, not a storage compression win:

- **Algebraic query:** role/label unbinding and spatial field evaluation
  act on bundles. See [attributes](../docs/attributes.md),
  [structures](../docs/structures.md), and [fields](../docs/fields.md).
  Labels and roles must have been encoded; a raw capture bundle does not
  acquire semantic annotations merely by changing its storage precision.
- **Translation as a multiply:** the Fourier shift theorem translates a
  spectral bundle by an elementwise phase multiply; see
  [spectral encoding](../docs/spectral.md). This is the bundle operation;
  maintaining a spatial partition after movement can require more work.
- **CRDT merge:** writer-sharded accumulators and causal versioning make
  replicated updates converge; see [sync](../docs/sync.md) and
  [observed-remove semantics](../docs/orset.md). Addition alone is not
  idempotent. The payload table does not budget these replication layers.
- **Locality diagnostics:** [spatial chunking](../docs/spatial.md) describes
  local query/update scope; [error locality](error_locality.md) documents
  worst-fraction error shares and cell enrichment used to diagnose field
  failures. These diagnostics require decoded fields and a reference;
  they are representation-independent, not an exclusive bundle operation
  or a measurement obtainable from compressed bytes alone.

An asset supports the analogous workflows after decode plus an appropriate
index, with metadata, query logic, and replication machinery where needed.
This is a qualitative cost comparison; no latency or amortization ratio is
measured here. Asset formats can also be wrapped in CRDTs or evaluated with
locality diagnostics. Bundles provide the tested algebraic substrate;
these features are not evidence that assets are incapable of the tasks.

## 5. What would change the picture

A D3 result showing block-scaled magnitudes below four bits preserving
query and reconstruction quality at equal total bytes could lower the
floor. Block scales and headers must count toward that budget; a payload
bit count alone would not demonstrate a win.

Tiles as the unit of storage could make working-set and transfer costs
follow the queried region. Evaluate tile metadata, boundary overlap,
codebook sharing, cache behavior, and update/replication traffic alongside
quality. This could improve bytes touched per query without reducing the
whole-scene archive. It needs measurements, not a replacement constant
for the old 384× headline. The maintainer can follow up on README and
gated documentation through the registered `capture.spz_compression`,
`capture.sog_compression`, `capture.bundle_vs_codec`, and
`precision.rank_vs_bits_*` claims; this lane changes none of those surfaces.
