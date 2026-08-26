"""Gaussian splat fields via fractional power encoding."""

import numpy as np

from holo import GaussianSplatField


def test_field_matches_mixture_and_improves_with_dim():
    rng = np.random.default_rng(7)
    mus = rng.uniform(0.2, 0.8, size=(30, 2))
    alphas = rng.uniform(0.5, 1.0, size=30)
    P = rng.uniform(0, 1, size=(500, 2))
    rmses = []
    for d in [1024, 16384]:
        field = GaussianSplatField(d, np.eye(2) * 0.05 ** 2, seed=0)
        for mu, a in zip(mus, alphas):
            field.add_splat(mu, a)
        rmses.append(float(np.sqrt(np.mean(
            (field.eval(P) - field.exact(P)) ** 2))))
    assert rmses[1] < rmses[0]          # error shrinks ~1/sqrt(d)
    assert rmses[1] < 0.08              # and is small in absolute terms
