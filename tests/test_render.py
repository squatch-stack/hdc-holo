"""Closed-form ray rendering against analytic projections."""

import numpy as np
import pytest

from holo import (GaussianSplatField, exact_projection, render_orthographic,
                 view_bundle)


def test_render_matches_analytic_projection():
    field = GaussianSplatField(8192, np.eye(3) * 0.05 ** 2, seed=0)
    rng = np.random.default_rng(9)
    for _ in range(25):
        field.add_splat(rng.uniform(0.25, 0.75, 3), 1.0)
    center, half, res = np.array([0.5, 0.5, 0.5]), 0.45, 48
    for view in ([1, 0, 0], [1, 1, 1]):
        truth = exact_projection(field.splats, field.sigma_inv, view,
                                 center, half, res)
        holo = render_orthographic(field.S, field.W, view, center, half,
                                   res, t_extent=1.8)
        rmse = float(np.sqrt(np.mean((holo - truth) ** 2)))
        assert rmse < 0.12 * truth.max()


def test_view_bundle_zero_frequency_limit():
    # frequencies exactly perpendicular to the view must get factor T,
    # not a 0/0: build a W row with w.v == 0 and check finiteness
    W = np.array([[0.0, 5.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    S = np.ones(2, dtype=np.complex64)
    view, T = [1, 0, 0], 2.0
    Sv = view_bundle(S, W, view, T)
    assert np.all(np.isfinite(Sv))
    assert Sv[0] == pytest.approx(T)   # w.v = 0 -> integral = T exactly
