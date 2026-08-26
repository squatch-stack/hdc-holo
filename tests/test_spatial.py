"""Multi-band covariance classes and spatially chunked fields."""

import numpy as np

from holo import ChunkedSplatField, GaussianSplatField, MultiBandSplatField


def test_multiband_covers_two_covariance_classes():
    sigmas = [np.eye(2) * 0.05 ** 2, np.eye(2) * 0.01 ** 2]
    mb = MultiBandSplatField(8192, sigmas, seed=0)
    mb.add_splat([0.3, 0.3], 1.0, band=0)
    mb.add_splat([0.7, 0.7], 1.0, band=1)
    rng = np.random.default_rng(11)
    P = rng.uniform(0, 1, size=(400, 2))
    assert np.sqrt(np.mean((mb.eval(P) - mb.exact(P)) ** 2)) < 0.05


def test_chunked_beats_global_at_same_dim():
    rng = np.random.default_rng(13)
    sigma = np.eye(3) * 0.03 ** 2
    chunked = ChunkedSplatField(2048, sigma, cell_size=0.125, seed=0)
    glob = GaussianSplatField(2048, sigma, seed=0)
    for _ in range(800):
        mu, a = rng.uniform(0.05, 0.95, 3), float(rng.uniform(0.5, 1))
        chunked.add_splat(mu, a)
        glob.add_splat(mu, a)
    P = rng.uniform(0.1, 0.9, size=(400, 3))
    truth = chunked.exact(P)
    rmse_c = np.sqrt(np.mean((chunked.eval(P) - truth) ** 2))
    rmse_g = np.sqrt(np.mean((glob.eval(P) - truth) ** 2))
    assert rmse_c < rmse_g / 2       # local crosstalk only
    assert rmse_c < 0.25
