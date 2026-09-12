"""Seeded Stage 0 capacity measurements; no real captures or GPU required.

Run python -m bench.resonator_sweep. Defaults cover all 120 conditions with
30 trials each. On slower CPUs, --large-trials 10 reduces dimensions >=8192
to ten trials per condition. Use OPENBLAS_NUM_THREADS=1
for these small matrix-vector products to avoid thread launch overhead.
Recovery counts exact identity/x/y tuples per object, including failures as
misses. Convergence counts accepted attempts per requested object; scene
recovery requires every object. The load ratio is descriptive, not a universal
capacity law: grid correlations and scene multiplicity also affect recovery.

Measured 2026-09-12 with the default sweep: 120 conditions, 3600 trials,
144 seconds. At d=4096/grid=16, recovery was 100% for one-object scenes
and 90% for three-object scenes. Recovery
crossed 50% between load 9.75 (57.78%) and 11.375 (41.43%). Across grids and
dimensions the first below-half points ranged from .5078 to 208: this is not
one universal cliff. There were 411 accepted incorrect tuples, predominantly
on the correlated 32-point grids; convergence is explicitly not correctness.
A 50-trial follow-up (--dims 4096 1024 --grids 16 --objects 1 2 3 4 8
--trials 50 --no-plot) measured 100% single-object and 84.67% three-object
recovery at d=4096, with 5.08 and 6.20 mean accepted iterations respectively.
The d=1024/eight-object case recovered 48.25% (34% complete scenes), versus
52.08% with 30 trials: the half-recovery boundary has sampling uncertainty.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from holo.attribute_field import AttributeSplatField
from holo.fhrr import FHRR
from holo.resonator import factorize_all, grid_codebook


def setup(dim, grid, seed):
    field = AttributeSplatField(FHRR(dim, seed), .04)
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        field.attrs.get(label)
    values = np.linspace(0, 1, grid)
    books = [field.attrs.matrix(), *(grid_codebook(field.W[:, k], values)
                                    for k in range(2))]
    return field, values, books


def measure(dim, grid, n_objects, trials=30, seed=0):
    """Independent trial streams make filtered sweeps reproduce full runs."""
    field, values, books = setup(dim, grid, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, dim, grid, n_objects]))
    recovered = converged = complete = spurious = 0
    iterations = []
    for _ in range(trials):
        flat = rng.choice(26 * grid * grid, n_objects, replace=False)
        truth = {tuple(map(int, np.unravel_index(i, (26, grid, grid)))) for i in flat}
        field.S.fill(0)
        field.splats.clear()
        for label, x, y in sorted(truth):
            field.add_splat([values[x], values[y]], field.attrs.labels[label])
        results = factorize_all(field.S, books, n_objects)
        found = {tuple(r.indices) for r in results if r.converged}
        recovered += len(truth & found)
        complete += truth <= found
        converged += sum(r.converged for r in results)
        spurious += sum(tuple(r.indices) not in truth for r in results if r.converged)
        iterations.extend(r.n_iters for r in results if r.converged)
    return {"dim": dim, "grid": grid, "n_objects": n_objects, "trials": trials,
                "load": n_objects * 26 * grid * grid / dim,
                "recovery": recovered / (trials * n_objects),
                "scene_recovery": complete / trials,
                "convergence": converged / (trials * n_objects),
                "mean_iters": float(np.mean(iterations)) if iterations else 0.,
                "max_iters": max(iterations, default=0), "spurious": spurious}


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    dimensions = sorted({r["dim"] for r in rows})
    styles = {8: "-", 16: "--", 32: ":"}
    for dim, grid in sorted({(r["dim"], r["grid"]) for r in rows}):
        group = [r for r in rows if r["dim"] == dim and r["grid"] == grid]
        ax.plot([r["load"] for r in group], [r["recovery"] for r in group],
                marker=".", label=f"d={dim}, grid={grid}",
                color=plt.get_cmap("tab10")(dimensions.index(dim)),
                linestyle=styles.get(grid, "-"))
    ax.axhline(.5, color="gray", linestyle="--", linewidth=.8)
    ax.set(xscale="log", xlabel="objects × (26 × grid²) / dimension",
           ylabel="Exact object recovery", ylim=(-.03, 1.03),
           title="Synchronous resonator and deflation: synthetic capacity")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    Path("out").mkdir(exist_ok=True)
    fig.savefig("out/resonator_cliff.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[1024, 2048, 4096,
                                                              8192, 16384])
    parser.add_argument("--grids", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--objects", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--large-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    rows = []
    print("dim grid objects trials load recovery scene convergence mean_iters "
          "max_iters spurious", flush=True)
    for dim in args.dims:
        for grid in args.grids:
            for n_objects in args.objects:
                trials = min(args.trials, args.large_trials) if dim >= 8192 \
                    else args.trials
                row = measure(dim, grid, n_objects, trials, args.seed)
                rows.append(row)
                print(" ".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                               for v in row.values()), flush=True)
    if not args.no_plot:
        plot(rows)
    print(f"Elapsed: {time.monotonic() - started:.2f}s")


if __name__ == "__main__":
    main()
