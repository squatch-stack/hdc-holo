"""Squatch Stack synthetic separable cleanup benchmark (negative spectral sign).

Prior art read: arXiv:2604.15113, arXiv:2211.05052, arXiv:2506.15793
and doi:10.3389/frai.2026.1793314. Workload and coarse-cell definitions are
ours; this is not a reproduction of those implementations. The rotation-product
paper uses a different square codebook, not this rectangular phasor dictionary.

Integer coordinates 0..n-1 and independent uniform frequencies [-pi, pi]
give a sinc spatial kernel with approximately orthogonal integer grid rows.
Coarse cells partition each axis into contiguous groups; their representatives
are normalized sums of fine rows. Stage one uses absolute complex overlap;
stage two and brute force use real overlap, via imported op_cleanup. Resonator
readout uses absolute overlap and does not certify the real-overlap optimum.

Timing imports the operator benchmark's warmup/best-of-reps clock. Generation
and initial transfers are excluded, candidate generation and diagnostic host
transfers are included. The existing holo resonator is NumPy-only: even for
--backend cupy its row is explicitly a CPU reference, not a CUDA measurement.
Bytes held describe each method in isolation, not RSS or the shared harness;
peak prediction includes host/device copies but excludes allocator/BLAS caches.
All runs are synthetic; --synthetic makes that choice explicit.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bench.operator_bench import (
    CHUNK,
    SEED,
    _clock,
    _finish,
    _host,
    _lib,
    _planes,
    op_cleanup,
    pick_backend,
    predict_bytes,
)
from holo.fhrr import FHRR
from holo.resonator import grid_codebook, resonator


def krop_row(angles, index):
    """arXiv:2506.15793 Algorithm 1, a separate real square codebook.

    This reconstructs the paper's rotation/reflection row, not a spectral
    position. Angles are ordered from least to most significant index bit.
    """
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError('finite one-dimensional angles required')
    if not 0 <= index < 2 ** len(angles):
        raise ValueError('index outside rotation codebook')
    row = np.ones(1)
    for bit, angle in enumerate(angles):
        c, s = np.cos(angle), np.sin(angle)
        row = np.concatenate((s * row, -c * row) if (index >> bit) & 1
                             else (c * row, s * row))
    return row


def cleanup_krop(query, angles):
    """Algorithm 2 butterfly control: O(Q*d*log d), K=d=2**len(angles).

    Deliberately excluded from the same-dictionary crossover: replacing the
    value code changes the task and requires a separate HRR capacity study.
    """
    angles = np.asarray(angles, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if (angles.ndim != 1 or query.ndim != 2
            or query.shape[1] != 2 ** len(angles)
            or not np.isfinite(angles).all() or not np.isfinite(query).all()):
        raise ValueError('finite (Q,2**len(angles)) queries required')
    values = query.copy()
    for level, angle in enumerate(angles[::-1]):
        blocks = values.reshape(len(query), 2 ** level, -1)
        left, right = np.split(blocks, 2, axis=-1)
        c, s = np.cos(angle), np.sin(angle)
        values = np.stack((c * left + s * right, s * left - c * right),
                          axis=2).reshape(query.shape)
    ids = values.argmax(axis=1)
    return {'argmax': ids, 'scores': values[np.arange(len(query)), ids],
            'checksum': float(np.abs(values).sum(dtype=np.float64))}


def axis_codebooks(freqs, n, sign=-1):
    """Three (n,d) complex64 books, at unit-spaced integer coordinates."""
    freqs = np.asarray(freqs)
    if freqs.ndim != 2 or freqs.shape[1] != 3 or n < 1 or sign not in (-1, 1):
        raise ValueError('require (d,3) frequencies, positive n and sign +/-1')
    return [grid_codebook(freqs[:, j], np.arange(n), sign) for j in range(3)]


def product_codebook(books):
    """Materialize in np.ndindex order (last factor varies fastest)."""
    return (books[0][:, None, None, :] * books[1][None, :, None, :]
            * books[2][None, None, :, :]).reshape(-1, books[0].shape[1])


def predict_bytes_product(d, n, Q, dtype=np.complex64):
    """Conservative preallocation bound; runtime/allocator caches excluded.

    Reuse the operator benchmark budget (including its conservative d² term),
    and add factors and query-generation scratch. Items are streamed, so the
    peak does not scale with load. Count complex dtype as two real planes.
    """
    if min(d, n, Q) < 1 or np.dtype(dtype).kind != 'c':
        raise ValueError('positive d,n,Q and complex dtype required')
    real = np.empty((), dtype=dtype).real.dtype
    return predict_bytes(d, n ** 3, Q, real) + np.dtype(dtype).itemsize * (
        24 * n * d + 32 * Q * d)


def cleanup_brute(query_planes, product_planes, chunk=CHUNK):
    return op_cleanup(*query_planes, *product_planes, chunk)


def _coarse_grid(books, coarse):
    n = len(books[0])
    if coarse < 1:
        raise ValueError('coarse must be positive')
    groups = np.array_split(np.arange(n), min(n, coarse))
    lib = _lib(books[0])
    axes = []
    for book in books:
        rows = lib.stack([book[g].sum(axis=0) for g in groups])
        norms = lib.sqrt(lib.sum(lib.abs(rows) ** 2, axis=1, keepdims=True))
        axes.append(rows / lib.maximum(norms, 1e-12) * np.sqrt(book.shape[1]))
    return groups, product_codebook(axes)


def cleanup_coarse_fine(query, books, coarse, top_r):
    """Coarse may be a cell count or a precomputed (groups, centroids) tuple."""
    if top_r < 1:
        raise ValueError('top_r must be positive')
    groups, centroids = (_coarse_grid(books, coarse)
                         if isinstance(coarse, int) else coarse)
    lib = _lib(query)
    scores = lib.abs(query @ centroids.conj().T)
    selected = _host(lib.argsort(-scores, axis=1)[:, :top_r])
    n, nc = len(books[0]), len(groups)
    ids, best, candidates = [], [], []
    checksum = 0.0
    for q, cells in zip(query, selected):
        choices = []
        for cell in cells:
            a, b, c = np.unravel_index(cell, (nc, nc, nc))
            mesh = np.meshgrid(groups[a], groups[b], groups[c], indexing='ij')
            choices.extend(np.ravel_multi_index(mesh, (n, n, n)).ravel())
        ix = np.unique(choices)
        a, b, c = np.unravel_index(ix, (n, n, n))
        memory = books[0][a] * books[1][b] * books[2][c]
        result = cleanup_brute((q[None].real, q[None].imag),
                               (memory.real, memory.imag))
        ids.append(ix[result['argmax'][0]])
        best.append(result['scores'][0])
        checksum += result['checksum']
        candidates.append(ix)
    return {'argmax': np.asarray(ids), 'scores': np.asarray(best),
            'checksum': checksum, 'candidates': candidates}


def cleanup_resonator(query, books, iters=40, score_floor=0.2):
    """Call the existing CPU resonator; retain unsuccessful final readouts."""
    query = _host(query)
    books = [_host(b) for b in books]
    results = [resonator(q, books, iters=iters, score_floor=score_floor)
               for q in query]
    ids = [np.ravel_multi_index(r.indices, tuple(map(len, books)))
           for r in results]
    scores = np.asarray([min(r.scores) for r in results])
    return {'argmax': np.asarray(ids), 'scores': scores,
            'checksum': float(np.abs(scores).sum(dtype=np.float64)),
            'iterations': np.asarray([r.n_iters for r in results]),
            'converged': np.asarray([r.converged for r in results])}


def _queries(books, load, Q, rng):
    """One independently seeded bundle per query, querying its first item.

    Streaming pairs bounds memory independently of N. Duplicate positions are
    allowed, items remain independent. A reset RNG couples different loads by
    identical prefixes of item-position pairs.
    """
    d, n = books[0].shape[1], len(books[0])
    bundle = np.zeros((Q, d), np.complex64)
    for j in range(load):
        indices = rng.integers(n, size=(3, Q))
        real, imag = _planes(rng, Q, d)
        item = (real + 1j * imag).astype(np.complex64)
        position = FHRR.bind(*(b[ix] for b, ix in zip(books, indices)))
        bundle += FHRR.bind(item, position)
        if j == 0:
            target = item
            truth = np.ravel_multi_index(indices, (n, n, n))
    return FHRR.unbind(bundle, target), truth


def workload(lib, d, n, loads, Q, rng, *, coarse=8, top_r=4, iters=40):  # noqa: PLR0913
    """Zero-arg closures for one load N (the scalar `loads` argument).

    Results include identical truth_index arrays for auditing recovery. Each
    closure draws no randomness; no initial transfer is timed. Near the
    resonator cliff float32 reduction differences can alter terminal readouts.
    """
    if min(d, n, loads, Q, coarse, top_r, iters) < 1:
        raise ValueError('dimensions, load, Q and search limits must be positive')
    freqs = rng.uniform(-np.pi, np.pi, (d, 3))
    host_books = axis_codebooks(freqs, n)
    query, truth = _queries(host_books, loads, Q, rng)
    books = [lib.asarray(b) for b in host_books]
    device_query = lib.asarray(query)
    product = product_codebook(books)
    # Contiguous float32 planes match the imported operator's GEMMs.
    planes = (lib.ascontiguousarray(product.real),
              lib.ascontiguousarray(product.imag))
    del product
    qp = (lib.ascontiguousarray(device_query.real),
          lib.ascontiguousarray(device_query.imag))
    coarse_grid = _coarse_grid(books, coarse)
    _finish((*planes, *qp, coarse_grid[1]))

    def brute():
        return dict(cleanup_brute(qp, planes), truth_index=truth)

    def coarse_fine():
        return dict(cleanup_coarse_fine(device_query, books, coarse_grid, top_r),
                    truth_index=truth)

    def factorized():
        return dict(cleanup_resonator(query, host_books, iters), truth_index=truth)

    return {'brute': brute, 'coarse_fine': coarse_fine, 'resonator': factorized}


def recover(kind_result, truth_index):
    truth = np.asarray(truth_index)
    if truth.ndim != 1 or not truth.size or truth.shape != kind_result['argmax'].shape:
        raise ValueError('truth must match nonempty query results')
    return float(np.mean(kind_result['argmax'] == truth))


def _measure(fn, reps, held, backend):
    latest = {}

    def timed():
        result = fn()
        latest['result'] = result
        return result

    timing = _clock(timed, reps)
    result = latest['result']
    truth = result['truth_index']
    row = dict(timing, bytes_held=held, recovery=recover(result, truth),
               execution_backend=backend, argmax=result['argmax'].tolist())
    if 'converged' in result:
        correct = result['argmax'] == truth
        row.update(mean_iterations=float(result['iterations'].mean()),
                   converged_rate=float(result['converged'].mean()),
                   spurious_rate=float(np.mean(result['converged'] & ~correct)))
    if 'candidates' in result:
        row['true_cell_selected_rate'] = float(np.mean([
            t in candidates for t, candidates in zip(truth, result['candidates'])]))
    return row


def run_matrix(backend, d, n, loads, Q, reps, max_gb, *,  # noqa: PLR0913
               coarse=8, top_r=4, iters=40):
    loads = list(loads)
    if (not loads or min(*loads, reps, coarse, top_r, iters) < 1
            or not np.isfinite(max_gb) or max_gb <= 0):
        raise ValueError('positive loads, reps, search limits and budget required')
    if backend not in ('numpy', 'cupy'):
        raise ValueError('backend must be numpy or cupy')
    if backend == 'cupy' and os.environ.get('CUPY_TF32') == '1':
        raise ValueError('CUPY_TF32=1 refused: true float32 GEMMs required')
    predicted = predict_bytes_product(d, n, Q)
    if predicted > max_gb * 1e9:
        raise MemoryError(f'predicted {predicted} bytes exceeds {max_gb} GB budget')
    name, lib = pick_backend(backend)
    nc = min(n, coarse)
    held = {'brute': 8 * d * (n ** 3 + Q),
            'coarse_fine': 8 * d * (3 * n + nc ** 3 + Q),
            'resonator': 8 * d * (3 * n + Q)}
    rows = []
    for load in loads:
        ops = workload(lib, d, n, load, Q, np.random.default_rng(SEED),
                       coarse=coarse, top_r=top_r, iters=iters)
        methods = {key: _measure(fn, reps, held[key],
                                'numpy' if key == 'resonator' else name)
                   for key, fn in ops.items()}
        for method in methods.values():
            method['codebook_bytes'] = method['bytes_held'] - 8 * Q * d
        rows.append({'load': load, 'load_K_over_d': load * n ** 3 / d,
                     'methods': methods})
        del ops
    return {'backend': name, 'd': d, 'n': n, 'K': n ** 3, 'Q': Q,
            'reps': reps, 'seed': SEED, 'sign': -1, 'coarse': nc,
            'top_r': min(top_r, nc ** 3), 'iters': iters,
            'predicted_bytes': predicted, 'rows': rows}


def _figure(report, path):
    plt.switch_backend('Agg')

    dims = sorted({run['d'] for run in report['runs']})
    if not dims:
        return
    fig, axes = plt.subplots(2, len(dims), figsize=(6 * len(dims), 8), squeeze=False)
    for col, d in enumerate(dims):
        runs = sorted((r for r in report['runs'] if r['d'] == d), key=lambda r: r['K'])
        for method in ('brute', 'coarse_fine', 'resonator'):
            for load in (runs[0]['rows'][0]['load'], runs[0]['rows'][-1]['load']):
                rows = [next(row for row in r['rows'] if row['load'] == load)
                        for r in runs]
                label = f'{method}, N={load}'
                for ax, metric in zip(axes[:, col], ('seconds', 'recovery')):
                    ax.plot([r['K'] for r in runs],
                            [r['methods'][method][metric] for r in rows],
                            marker='o', label=label)
                    ax.set_xscale('log')
                    ax.set_xlabel('K = n³')
                    ax.set_ylabel(metric)
                    ax.grid(alpha=.25)
        axes[0, col].set_yscale('log')
        axes[0, col].set_title(f"d={d}; Q={runs[0]['Q']}; {report['backend']}")
        axes[1, col].set_ylim(-.03, 1.03)
        axes[1, col].legend(fontsize=8)
    fig.suptitle('Synthetic cleanup: batch time and exact position recovery')
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=('numpy', 'cupy'), default='numpy')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--d', default='4096,8192')
    ap.add_argument('--n', default='8,16,32,64')
    ap.add_argument('--loads', default='1,2,4,8,16')
    ap.add_argument('--Q', type=int, default=256)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--max-gb', type=float, default=12)
    ap.add_argument('--coarse', type=int, default=8)
    ap.add_argument('--top-r', type=int, default=4)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--out')
    ap.add_argument('--figure')
    args = ap.parse_args(argv)
    report = {'backend': args.backend, 'synthetic': True, 'runs': [], 'refused': []}
    for d in map(int, args.d.split(',')):
        for n in map(int, args.n.split(',')):
            try:
                run = run_matrix(args.backend, d, n, map(int, args.loads.split(',')),
                                 args.Q, args.reps, args.max_gb, coarse=args.coarse,
                                 top_r=args.top_r, iters=args.iters)
                report['runs'].append(run)
                print(f'Finished d={d}, n={n}', flush=True)
            except MemoryError as exc:
                report['refused'].append({'d': d, 'n': n, 'reason': str(exc)})
    path = Path(args.out or f'gpubench/faccleanup-{args.backend}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    if args.figure:
        _figure(report, args.figure)
    print(f'Wrote {path}: {len(report["runs"])} runs, {len(report["refused"])} refused')
    return report


if __name__ == '__main__':
    main()
