"""HyperSpace code was not accessible; these are our operator benchmarks.

Research checked arXiv:2604.15113 (https://arxiv.org/abs/2604.15113),
GitHub searches for the id and 'hyperspace hyperdimensional', and the paper's
https://github.com/Parsa-Research-Laboratory/HyperSpace link (HTTP 404).
Thus sizes and timing are ours, not a reproduction of unavailable code.
The paper's iterative cleanup differs from our one-pass nearest-codeword GEMM.

Bind is phasor multiplication or HRR circular convolution. Bundle and spatial
similarity transcribe the nested encode/readout closures in holo_bench_job
(they cannot be imported); backend selection is imported from that module.
Bundle uses four channels, Gaussian envelopes and the negative spectral sign;
similarity uses unit readout weights. Cleanup uses unnormalised real dot scores.
All inputs are seeded float32, with one warmup and best-of-reps synchronised
wall time, excluding generation/transfers. Cleanup includes chunkwise host
float64 score reduction, so its timing includes diagnostic transfer overhead.
NumPy FFT internally promotes to double; outputs are cast back to float32.

Accuracy uses independent random role/item associations: sum bind(role,item),
unbind each role and clean up against all N stored items. Uniform phasors and
unit-norm Gaussian HRR items are used; HRR unbind is circular correlation.
Nominal load is dim for FHRR, dim/2 for HRR; 50% means N=dim/2 and dim/4.
This explicit convention is not a proven capacity boundary: the sqrt(N/2dim)
FHRR noise law does not fix top-1 capacity without specifying dictionary size.
FHRR@d/2 and HRR@d have equal bytes and equal N; FHRR@d is the same-d control
with twice the bytes and load. Unit phasors have one independent phase, despite
two stored real planes; equal storage alone does not prove equal capacity.
"""

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np

from bench.holo_bench_job import pick_backend

CHUNK = 256
SEED = 1729


def _lib(a):
    root = type(a).__module__.split('.')[0]
    return importlib.import_module({'mlx': 'mlx.core'}.get(root, root))


def _host(a):
    return a.get() if type(a).__module__.startswith('cupy') else np.asarray(a)


def _finish(values):
    lib = _lib(values[0])
    if lib.__name__ == 'mlx.core':
        lib.eval(*values)
    elif lib.__name__ == 'cupy':
        lib.cuda.Stream.null.synchronize()


def op_bind(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


def op_bundle(mu, cov6, amp_norm, W, wq, chunk=CHUNK):
    """Use encode's chunked envelope-times-phasor GEMM, four channels in runs."""
    lib = _lib(mu)
    sr = lib.zeros((amp_norm.shape[1], W.shape[0]), dtype=mu.dtype)
    si = lib.zeros_like(sr)
    for lo in range(0, len(mu), chunk):
        ph = mu[lo:lo + chunk] @ W.T
        env = lib.exp(-0.5 * (cov6[lo:lo + chunk] @ wq.T))
        a = amp_norm[lo:lo + chunk].T
        sr = sr + a @ (env * lib.cos(ph))
        si = si - a @ (env * lib.sin(ph))
        _finish((sr, si))
    return sr, si


def op_similarity(points, W, Sr, Si):
    """Readout with unit weights, chunking points to bound trig temporaries."""
    lib = _lib(points)
    outputs = []
    for lo in range(0, len(points), CHUNK):
        ph = points[lo:lo + CHUNK] @ W.T
        out = lib.cos(ph) @ Sr.T - lib.sin(ph) @ Si.T
        _finish((out,))
        outputs.append(out)
    return lib.concatenate(outputs, axis=0)


def _cleanup(Qr, Mr, chunk, Qi=None, Mi=None):
    if chunk < 1 or len(Mr) < 1:
        raise ValueError('cleanup requires positive chunk and nonempty memory')
    best = np.full(len(Qr), -np.inf)
    ids = np.zeros(len(Qr), dtype=np.int64)
    total = 0.0
    for lo in range(0, len(Mr), chunk):
        scores = Qr @ Mr[lo:lo + chunk].T
        if Qi is not None:
            scores = scores + Qi @ Mi[lo:lo + chunk].T
        s = _host(scores)
        total += float(np.abs(s).sum(dtype=np.float64))
        ix = s.argmax(axis=1)
        val = s[np.arange(len(s)), ix]
        better = val > best
        ids[better] = lo + ix[better]
        best = np.maximum(best, val)
    # Absolute values: a signed sum of zero-mean scores cancels and turns
    # float32 GEMM rounding into a false verification failure.
    return {'argmax': ids, 'scores': best, 'checksum': total}


def op_cleanup(Qr, Qi, Mr, Mi, chunk):
    """Stream K without storing Q by K scores; ties choose the first index."""
    return _cleanup(Qr, Mr, chunk, Qi, Mi)


def op_cleanup_hrr(Q, M):
    return _cleanup(Q, M, CHUNK)


def op_bind_hrr(a, b):
    lib = _lib(a)
    return lib.fft.irfft(lib.fft.rfft(a) * lib.fft.rfft(b),
                         n=a.shape[-1]).astype(a.dtype)


def predict_bytes(d, K, Q, dtype=np.float32):
    """Conservative live-array budget, including host/device copies and accuracy.

    Counts all six resident planes, generation/FFT/trig temporaries, bounded
    score tiles and the largest association experiment. It excludes library
    allocator caches, BLAS workspaces and runtime overhead; not measured RSS.
    """
    if min(d, K, Q) < 1:
        raise ValueError('d, K and Q must be positive')
    s = np.dtype(dtype).itemsize
    return int(s * (16 * (K + Q) * d + 32 * CHUNK * d
                    + 8 * Q * min(K, CHUNK) + 20 * d * d + 32 * (K + Q)))


def _planes(rng, n, d):
    ph = rng.uniform(-np.pi, np.pi, (n, d)).astype(np.float32)
    return np.cos(ph), np.sin(ph)


def _accuracy(kind, dim, lib):
    rng = np.random.default_rng(SEED + 1)
    n = max(1, dim // (2 if kind == 'FHRR' else 4))
    if kind == 'FHRR':
        mr, mi = _planes(rng, n, dim)
        rr, ri = _planes(rng, n, dim)
        mr, mi, rr, ri = (lib.array(a) for a in (mr, mi, rr, ri))
        br, bi = op_bind(rr, ri, mr, mi)
        sr, si = br.sum(axis=0), bi.sum(axis=0)
        qr, qi = op_bind(sr, si, rr, -ri)
        result = op_cleanup(qr, qi, mr, mi, CHUNK)
    else:
        m = rng.standard_normal((n, dim)).astype(np.float32)
        r = rng.standard_normal((n, dim)).astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        m, r = lib.array(m), lib.array(r)
        bundle = op_bind_hrr(r, m).sum(axis=0)
        query = lib.fft.irfft(lib.fft.rfft(bundle)
                              * lib.conj(lib.fft.rfft(r)), n=dim).astype(m.dtype)
        result = op_cleanup_hrr(query, m)
    return {'N': n, 'top1': float(np.mean(result['argmax'] == np.arange(n))),
            'bytes_per_codeword': dim * (8 if kind == 'FHRR' else 4),
            'dim': dim, 'accuracy_backend': lib.__name__}


def _clock(fn, reps):
    result = fn()
    _sync_result(result)
    best = float('inf')
    for _ in range(reps):
        start = time.perf_counter()
        result = fn()
        _sync_result(result)
        best = min(best, time.perf_counter() - start)
    if isinstance(result, dict):
        return {'seconds': best, 'checksum': result['checksum'],
                'argmax_histogram': np.bincount(result['argmax']).tolist()}
    arrays = result if isinstance(result, tuple) else (result,)
    return {'seconds': best,
            'checksum': sum(float(np.abs(_host(a)).sum(dtype=np.float64))
                            for a in arrays)}


def _sync_result(result):
    if not isinstance(result, dict):
        _finish(result if isinstance(result, tuple) else (result,))


def _workload(lib, d, K, Q):
    rng = np.random.default_rng(SEED)
    mr, mi = (lib.array(a) for a in _planes(rng, K, d))
    qr, qi = (lib.array(a) for a in _planes(rng, Q, d))
    m, q = (lib.array(rng.standard_normal((n, d)).astype(np.float32))
            for n in (K, Q))
    mu, cov, amp, W, wq, points = (
        lib.array(rng.random(shape, dtype=np.float32)) for shape in
        ((K, 3), (K, 6), (K, 4), (d, 3), (d, 6), (Q, 3)))
    sr, si = op_bundle(mu, cov, amp, W, wq)
    _finish((mr, mi, qr, qi, m, q, sr, si))
    return {
        'bind': lambda: op_bind(qr, qi, mr[0], mi[0]),
        'bundle': lambda: op_bundle(mu, cov, amp, W, wq),
        'similarity': lambda: op_similarity(points, W, sr, si),
        'cleanup': lambda: op_cleanup(qr, qi, mr, mi, CHUNK),
        'cleanup_hrr': lambda: op_cleanup_hrr(q, m),
        'bind_hrr': lambda: op_bind_hrr(q, m[0]),
    }


def run_matrix(backend, d, K, Q, reps, max_gb):
    predicted = predict_bytes(d, K, Q, np.float32)
    if d < 2 or d % 2 or reps < 1 or not np.isfinite(max_gb) or max_gb <= 0:
        raise ValueError('require even d >= 2, reps >= 1 and finite positive max_gb')
    if predicted > max_gb * 1e9:
        raise MemoryError(f'predicted {predicted} bytes exceeds {max_gb} GB budget')
    if backend not in ('numpy', 'cupy', 'mlx'):
        raise ValueError('backend must be numpy, cupy or mlx')
    if backend == 'cupy' and os.environ.get('CUPY_TF32') == '1':
        raise ValueError('CUPY_TF32=1 refused: true float32 GEMMs required')
    name, lib = pick_backend(backend)
    ops = _workload(lib, d, K, Q)
    timings = {key: _clock(fn, reps) for key, fn in ops.items()}
    del ops
    rows = {label: _accuracy(kind, dim, lib) for label, kind, dim in
            [('FHRR@d', 'FHRR', d), ('FHRR@d/2', 'FHRR', d // 2),
             ('HRR@d', 'HRR', d)]}
    return {'backend': name, 'd': d, 'K': K, 'Q': Q, 'reps': reps,
            'seed': SEED, 'chunk': CHUNK, 'predicted_bytes': predicted,
            'codebook_bytes': 8 * K * d, 'operators': timings,
            'accuracy': rows}


def _verify(reference, chosen):
    errors = {}
    for key, op in reference['operators'].items():
        a, b = op['checksum'], chosen['operators'][key]['checksum']
        error = abs(a - b) / max(abs(a), np.finfo(float).tiny)
        errors[key] = error
    if not all(np.isfinite(e) and e <= 2e-6 for e in errors.values()):
        raise ValueError(f'checksum verification failed: {errors}')
    return errors


def _verification_corner(args, d):
    try:
        ref = run_matrix('numpy', d, 1000, args.Q, args.reps, args.max_gb)
        chosen = run_matrix(args.backend, d, 1000, args.Q,
                            args.reps, args.max_gb)
    except MemoryError as exc:
        return {'d': d, 'K': 1000, 'status': 'refused', 'reason': str(exc)}
    return {'d': d, 'K': 1000, 'status': 'passed',
            'relative_errors': _verify(ref, chosen)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=['numpy', 'cupy', 'mlx'], default='numpy')
    ap.add_argument('--d', default='4096,8192,32768')
    ap.add_argument('--K', default='100,1000,10000,100000')
    ap.add_argument('--Q', type=int, default=4096)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--max-gb', type=float, default=12)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--out')
    args = ap.parse_args(argv)
    report = {'runs': [], 'refused': [], 'verification': []}
    for d in map(int, args.d.split(',')):
        for K in map(int, args.K.split(',')):
            try:
                report['runs'].append(run_matrix(
                    args.backend, d, K, args.Q, args.reps, args.max_gb))
            except MemoryError as exc:
                report['refused'].append({'d': d, 'K': K, 'reason': str(exc)})
        if args.verify:
            report['verification'].append(_verification_corner(args, d))
    path = Path(args.out or f'gpubench/opbench-{args.backend}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {path}: {len(report["runs"])} runs, '
          f'{len(report["refused"])} refused')
    return report


if __name__ == '__main__':
    main()
