"""RGB amplitude channels: point queries, color renders, replication."""

import numpy as np
import pytest

from holo import ColorSplatField, exact_projection, render_orthographic


def test_color_field_rgb_point_queries():
    field = ColorSplatField(4096, np.eye(2) * 0.05 ** 2, seed=0)
    rng = np.random.default_rng(21)
    for _ in range(30):
        field.add_splat(rng.uniform(0.1, 0.9, 2), rng.uniform(0.2, 1.0, 3))
    P = rng.uniform(0, 1, size=(400, 2))
    err = field.eval_rgb(P) - field.exact_rgb(P)
    # crosstalk ~ sqrt(sum(w^2)/(2d)) ~ 0.04, inflated ~1.6x by nearby
    # splats sharing frequencies (correlated noise); measured 0.065
    assert np.sqrt(np.mean(err ** 2)) < 0.09


def test_color_render_matches_analytic_per_channel():
    field = ColorSplatField(8192, np.eye(3) * 0.05 ** 2, seed=0)
    rng = np.random.default_rng(22)
    for _ in range(20):
        field.add_splat(rng.uniform(0.25, 0.75, 3), rng.uniform(0.3, 1.0, 3))
    center, half, res = np.array([0.5, 0.5, 0.5]), 0.45, 40
    holo = render_orthographic(field.S, field.W, [1, 1, 0], center, half,
                               res, t_extent=1.8)
    assert holo.shape == (res, res, 3)
    truth = np.stack([exact_projection(field.channel_splats(c),
                                       field.sigma_inv, [1, 1, 0], center,
                                       half, res)
                      for c in range(3)], axis=-1)
    rmse = float(np.sqrt(np.mean((holo - truth) ** 2)))
    assert rmse < 0.12 * truth.max()


def test_replicated_color_scene_converges():
    pytest.importorskip("loro", reason="Loro CRDT bindings not installed")
    from holo import FHRR, HoloReplica, ReplicatedColorScene
    A = HoloReplica(FHRR(2048, seed=0))
    B = HoloReplica(FHRR(2048, seed=0))
    sa = ReplicatedColorScene(A, np.eye(2) * 0.04 ** 2)
    sb = ReplicatedColorScene(B, np.eye(2) * 0.04 ** 2)
    rng = np.random.default_rng(23)
    for _ in range(30):
        sa.add_splat(rng.uniform([0.1, 0.1], [0.5, 0.9]), [1.0, 0.3, 0.1])
        sb.add_splat(rng.uniform([0.5, 0.1], [0.9, 0.9]), [0.1, 0.3, 1.0])
    A.sync(B)
    P = rng.uniform(0, 1, size=(300, 2))
    va, vb = sa.eval_rgb(P), sb.eval_rgb(P)
    assert va.shape == (300, 3)
    assert np.allclose(va, vb)
    # A's warm half stays red-dominant, B's cool half blue-dominant
    left, right = P[:, 0] < 0.4, P[:, 0] > 0.6
    assert va[left, 0].max() > va[left, 2].max()
    assert va[right, 2].max() > va[right, 0].max()
