"""NumPy resonators for separable position and identity factors.

The caller chooses the sign: sign=+1 matches attribute_field's e^{+iWp};
sign=-1 matches spectral's e^{-iw·mu}. No encoder conversion is implicit.
The abstract, Eq. 5 and Methods of arXiv:2208.12880 (checked 2026-09-12)
provide the associative projection, phasor dynamics and outer deflation loop.
This implementation uses synchronous updates, without the paper's added noise
or sparsifying nonlinearities. Stable indices alone do not certify recovery:
spurious fixed points are possible, especially with correlated grid rows.
The seeded 50-trial 16x16-grid measurement at d=4096 recovered 100% of single
objects (5.08 mean iterations) and 84.67% of objects in three-object scenes;
see python -m bench.resonator_sweep for the reproducible capacity experiment.
Contrary to the initial brief, torchhd functional.py has a resonator step,
but restricts it to MAPTensor (checked on GitHub 2026-09-12); it does not
implement this complex FHRR loop.
FTO caveat supplied for this research prototype: US patent 12,014,263 covers
VSA encoding of continuous spaces; this is research-only.
"""

from dataclasses import dataclass

import numpy as np

from .fhrr import FHRR


@dataclass
class ResonatorResult:
    indices: list[int]
    scores: list[float]
    estimates: list[np.ndarray]
    converged: bool
    n_iters: int
    trace: list[list[float]]


def grid_codebook(W_col, values, sign=+1) -> np.ndarray:
    """Keep the encoder's chosen phase sign when discretizing one axis."""
    return np.exp(sign * 1j * np.outer(values, W_col)).astype(np.complex64)


def _inputs(s, codebooks, init):
    s = np.asarray(s, dtype=np.complex64)
    books = [np.asarray(m, dtype=np.complex64) for m in codebooks]
    if s.ndim != 1 or not s.size or not books:
        raise ValueError("a nonempty vector and codebooks are required")
    for m in books:
        if m.ndim != 2 or not len(m) or m.shape[1] != s.size:
            raise ValueError("codebooks must have shape (n, len(s)) with n > 0")
        if not np.isfinite(m).all():
            raise ValueError("codebooks must be finite")
    if not np.isfinite(s).all():
        raise ValueError("signal must be finite")
    if init is not None and (
        len(init) != len(books)
        or any(np.shape(v) != s.shape or not np.isfinite(v).all() for v in init)
    ):
        raise ValueError("init must contain one finite vector per factor")
    return s, books


def _readout(books, estimates):
    # Complex projections have a free global phase per factor. Magnitude
    # identifies the row independently of that gauge; reconstruction below
    # uses the original rows and a real scene amplitude.
    scores = [np.abs(m.conj() @ v) / m.shape[1]
              for m, v in zip(books, estimates)]
    indices = [int(np.argmax(a)) for a in scores]
    return indices, [float(a[i]) for a, i in zip(scores, indices)]


def resonator(s, codebooks, iters=40, hysteresis=0.0, init=None,
              score_floor=0.2, rng=None) -> ResonatorResult:
    """Iterate AA† and phasor projections, retaining unsuccessful attempts.

    Default initialization is the normalized sum of every candidate row.
    Passing an RNG instead supplies random positive mixture weights. Updates
    are synchronous; hysteresis is the fraction of the previous phasor retained.
    Three unchanged transitions plus the score floor declare convergence.
    Scores are phase-invariant normalized estimate/codeword overlaps.
    """
    s, books = _inputs(s, codebooks, init)
    if iters < 0 or not 0 <= hysteresis <= 1 or score_floor < 0:
        raise ValueError("invalid iteration count, hysteresis or score floor")
    if init is None:
        init = [m.sum(axis=0) if rng is None else rng.uniform(.5, 1.5, len(m)) @ m
                for m in books]
    estimates = [FHRR.normalize(np.asarray(v)) for v in init]
    conjugates = [m.conj() for m in books]
    indices, scores = _readout(books, estimates)
    stable, trace, converged = 0, [], False
    for _ in range(iters):
        previous = estimates.copy()
        for j, m in enumerate(books):
            q = s.copy()
            for k, v in enumerate(previous):
                if k != j:
                    q *= v.conj()
            projected = FHRR.normalize(m.T @ (conjugates[j] @ q))
            estimates[j] = FHRR.normalize(
                (1 - hysteresis) * projected + hysteresis * estimates[j])
        chosen, scores = _readout(books, estimates)
        stable = stable + 1 if chosen == indices else 0
        indices = chosen
        trace.append(scores.copy())
        if stable >= 3 and min(scores) >= score_floor:
            converged = True
            break
    return ResonatorResult(indices, scores, estimates, converged, len(trace), trace)


def deflate(s, result, codebooks) -> np.ndarray:
    """Subtract the chosen binding with its real least-squares amplitude."""
    chosen = FHRR.bind(*(m[i] for m, i in zip(codebooks, result.indices)))
    alpha = float(np.vdot(chosen, s).real) / len(s)
    return np.asarray(s - alpha * chosen, dtype=np.complex64)


def factorize_all(s, codebooks, max_objects, iters=40,
                  score_floor=0.2) -> list[ResonatorResult]:
    """Return attempts, including the first failure; never deflate a failure.

    A stable candidate must also explain at least score_floor real amplitude
    in the residual. This rejects stable but unsupported factor combinations.
    """
    residual = np.asarray(s, dtype=np.complex64).copy()
    results = []
    for _ in range(max_objects):
        result = resonator(residual, codebooks, iters=iters, score_floor=score_floor)
        chosen = FHRR.bind(*(m[i] for m, i in zip(codebooks, result.indices)))
        supported = np.vdot(chosen, residual).real / len(residual) >= score_floor
        result.converged = bool(result.converged and supported)
        results.append(result)
        if not result.converged:
            break
        residual = deflate(residual, result, codebooks)
    return results


def demo(dim=4096, seed=0, save_png=True) -> dict:
    """Report a three-object synthetic scene without importing bench code."""
    from .attribute_field import AttributeSplatField

    space = FHRR(dim, seed)
    field = AttributeSplatField(space, .04)
    values = np.linspace(0, 1, 16)
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        field.attrs.get(label)
    books = [field.attrs.matrix(), *(grid_codebook(field.W[:, k], values)
                                    for k in range(2))]
    rng = np.random.default_rng(seed)
    truth = [tuple(map(int, (rng.integers(26), *rng.integers(16, size=2))))
             for _ in range(3)]
    for label, x, y in truth:
        field.add_splat([values[x], values[y]], field.attrs.labels[label])
    results = factorize_all(field.S, books, len(truth))
    found = {tuple(r.indices) for r in results if r.converged}
    numbers = {"recovery": len(found.intersection(truth)) / len(truth),
               "iterations": [r.n_iters for r in results],
               "converged": [r.converged for r in results]}
    print(numbers)
    if save_png:
        print("For the capacity figure run: python -m bench.resonator_sweep")
    return numbers
