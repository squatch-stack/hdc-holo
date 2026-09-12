# Resonator factorization

*[← docs index](README.md) · foundations*

**What.** Given a bundle believed to contain `identity (x) position`,
with a codebook for each factor, alternate cleanup of the factors until
the estimates stop moving. This follows the resonator-network approach
of [Renner et al., arXiv:2208.12880](https://arxiv.org/abs/2208.12880):
factor a sum of vector products into objects and poses. This implementation
uses synchronous complex FHRR updates, without the paper's added noise or
sparsifying nonlinearities; it is not a reproduction of the full
hierarchical neuromorphic architecture.

**Budget.** Codebooks of sizes `n_j` represent `prod(n_j)` candidate
tuples with `sum(n_j)` codeword comparisons per iteration (each over `d`
components). This representational capacity is not a recovery guarantee.
For each factor, unbind the other current estimates, project through its
codebook `M.T @ (M.conj() @ q)`, and normalize to unit phasors. Updates
use the previous iteration's estimates for every factor.

Existing measurements at d=4096 over 50 seeded trials give 100% single-object
and 84.67% three-object recovery. Recovery crosses 50% between load
**9.75 and 11.4** in `n_obj * prod(n_j) / d` for the reported crossing.
The upper endpoint is rounded from 11.375. The recorded sweep explicitly
finds that this is **not one universal cliff across grid shapes**: grid
correlations and object multiplicity also affect recovery. See
[the sweep figure](../out/resonator_cliff.png) and
[the sweep's recorded findings](../bench/resonator_sweep.py).

**Sign convention — match the encoder explicitly.** `holo/spectral.py`
encodes position as `e^{-i w.mu}`; `holo/attribute_field.py` uses
`e^{+i W p}`. Pass `sign=-1` or `sign=+1` to
`grid_codebook(W_col, values, sign)` accordingly, even though the default
is positive. The wrong sign silently prevents convergence to the intended
factorization; no encoder conversion is implicit.

**Failure modes.**

- The resonator **does not converge on spectral bundles of real captures
  at any setting tried**. The working alternative was one-shot whitened
  correlation. The module is proven synthetically and negative on captures;
  see [the measured 5090 results](../results/resonator_capture.md).
- `converged=True` is not `correct`: the capacity sweep accepted 411
  incorrect tuples. Read the scores, not just the flag, and validate against
  ground truth when available. Scores measure estimate/codeword overlap;
  even a stable high-scoring tuple need not be the right object.
- Near the cliff, terminal readout can differ between identical CPU runs:
  iterative argmax can flip on approximately one ulp of arithmetic variation.
  [The SDK's determinism contract](../SDK.md) is semantic, not bitwise.
  Tests stay in the stable small-grid regime for this reason.

**API.** Flat exports from `holo` and the `holo.structures` facade are
`resonator`, `ResonatorResult`, `factorize_all`, `deflate`, and `grid_codebook`.

```python
from holo.structures import grid_codebook, resonator

books = [identity_book, grid_codebook(W[:, 0], values, sign=+1)]
result = resonator(bundle, books)
```

`resonator(s, codebooks, iters=40, hysteresis=0.0, init=None,
score_floor=0.2, rng=None)` accepts a vector and `(n_j, d)` codebooks.
It returns `ResonatorResult(indices, scores, estimates, converged,
n_iters, trace)`, retaining unsuccessful attempts. Three unchanged index
transitions plus the score floor declare convergence. Default initialization
normalizes the sum of candidate rows; an RNG supplies random positive
mixture weights instead.

`deflate(s, result, codebooks)` subtracts the selected binding with its
real least-squares amplitude. `factorize_all(s, codebooks, max_objects,
iters=40, score_floor=0.2)` returns attempts including the first failure;
it checks residual amplitude support and never deflates a failed attempt.
Run the packaged example with `HDC_BACKEND=numpy python -m holo.cli resonator`.
Evidence and stable-regime checks live in `tests/test_resonator.py`.
