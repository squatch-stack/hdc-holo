"""Ridge-fitting holograms from raw samples of a target field."""

import numpy as np

from holo import FrequencyBands, GaussianSplatField, HoloRegressor


def test_fit_recovers_mixture_from_samples_only():
    rng = np.random.default_rng(5)
    truth = GaussianSplatField(2048, np.eye(2) * 0.05 ** 2, seed=0)
    for _ in range(40):
        truth.add_splat(rng.uniform(0.1, 0.9, 2), float(rng.uniform(0.5, 1)))
    Ptr = rng.uniform(0, 1, size=(6000, 2)).astype(np.float32)
    Pte = rng.uniform(0, 1, size=(1500, 2)).astype(np.float32)
    reg = HoloRegressor(FrequencyBands([2048], [0.05], seed=0)) \
        .fit(Ptr, truth.exact(Ptr))
    y_te = truth.exact(Pte)
    fit_rmse = float(np.sqrt(np.mean((reg.eval(Pte) - y_te) ** 2)))
    bundle_rmse = float(np.sqrt(np.mean((truth.eval(Pte) - y_te) ** 2)))
    assert fit_rmse < 0.05             # accurate from samples alone
    assert fit_rmse < bundle_rmse      # optimal S beats the forward bundle


def test_fit_denoises_noisy_samples():
    rng = np.random.default_rng(6)
    truth = GaussianSplatField(2048, np.eye(2) * 0.05 ** 2, seed=0)
    for _ in range(40):
        truth.add_splat(rng.uniform(0.1, 0.9, 2), float(rng.uniform(0.5, 1)))
    Ptr = rng.uniform(0, 1, size=(8000, 2)).astype(np.float32)
    Pte = rng.uniform(0, 1, size=(1500, 2)).astype(np.float32)
    noisy = truth.exact(Ptr) + rng.normal(0, 0.1, len(Ptr))
    reg = HoloRegressor(FrequencyBands([2048], [0.05], seed=0)) \
        .fit(Ptr, noisy, lam=1e-2)
    rmse = float(np.sqrt(np.mean((reg.eval(Pte) - truth.exact(Pte)) ** 2)))
    assert rmse < 0.1                  # below the per-sample noise level


def test_fit_multichannel_shares_one_solve():
    # two channels of one target: same design matrix, two right-hand
    # sides; both channels must decode independently
    rng = np.random.default_rng(8)
    t0 = GaussianSplatField(2048, np.eye(2) * 0.05 ** 2, seed=0)
    t1 = GaussianSplatField(2048, np.eye(2) * 0.05 ** 2, seed=0)
    for _ in range(20):
        t0.add_splat(rng.uniform(0.1, 0.9, 2), 1.0)
        t1.add_splat(rng.uniform(0.1, 0.9, 2), 1.0)
    Ptr = rng.uniform(0, 1, size=(6000, 2)).astype(np.float32)
    Pte = rng.uniform(0, 1, size=(1000, 2)).astype(np.float32)
    Y = np.stack([t0.exact(Ptr), t1.exact(Ptr)], axis=1)
    reg = HoloRegressor(FrequencyBands([2048], [0.05], seed=0)).fit(Ptr, Y)
    pred = reg.eval(Pte)
    assert pred.shape == (1000, 2)
    assert np.sqrt(np.mean((pred[:, 0] - t0.exact(Pte)) ** 2)) < 0.05
    assert np.sqrt(np.mean((pred[:, 1] - t1.exact(Pte)) ** 2)) < 0.05
