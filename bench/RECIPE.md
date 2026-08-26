# holo-bench: installing the benchmark recipe on a job runner

*This brief is addressed to whoever (human or agent) administers a
sandboxed GPU job runner and wants to expose holo's kernel benchmark as
a recipe. It states the contract; wire it into your runner's own recipe
format — you have that source in front of you, this document does not
guess at it.*

## What the recipe runs

One self-contained script from this repo: [`holo_bench_job.py`](holo_bench_job.py).
It reads `scene.npz` from the working directory, runs holo's two
dominant kernels (spectral encode + field readout — identical math to
`holo/accel.py`, see `docs/backend.md`) on the best available backend,
and writes `bench.json`. On an NVIDIA host it wants the `torch-cuda`
backend.

## Contract

- **Command**: `python3 holo_bench_job.py` (add `--backend torch-cuda`
  to fail loudly rather than fall back to CPU if CUDA is missing).
- **Environment**: python3 with `numpy` and CUDA-enabled `torch`
  preinstalled — the sandbox has no network, so nothing can be
  installed at job time.
- **Input**: one file, `scene.npz` (~12 MB), submitted by the client.
- **Artifact**: `bench.json` (~1 KB), written to the working directory.
- **Budgets**: the reference workload (131,072 splats / 262,144 points
  at d = 32,768, chunked at 16,384) peaks around **9-10 GB of VRAM**
  in transients — request **12 GB** to be comfortable. Wall time on a
  modern discrete GPU should be seconds; a **300 s timeout** is
  generous. CPU-side memory stays under 1 GB. No GPU? Don't bother —
  the NumPy fallback exists for verification, not benchmarking.
- **Isolation**: the script is stdlib+numpy(+torch); it touches only
  `./scene.npz` and `./bench.json`, needs no network, no HOME, no
  devices beyond the GPU. It is designed for exactly the blank-HOME,
  no-net, seccomp'd sandbox a good runner provides.

## Verifying the install

Submit any `scene.npz` produced by this repo's `run_gpu_bench.py
--make-payload`; the returned `bench.json` carries float64 checksums
(`check_bundle_sum`, `check_readout_sum`) that must match the values
computed locally on any other backend for the same payload — same
numbers first, then compare clocks.
