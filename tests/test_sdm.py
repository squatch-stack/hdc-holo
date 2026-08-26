"""Kanerva's Sparse Distributed Memory: recall from noisy addresses."""

import numpy as np

from holo import SparseDistributedMemory


def test_sdm_noisy_recall():
    rng = np.random.default_rng(42)
    sdm = SparseDistributedMemory(5000, 256, 108, seed=0)
    patterns = rng.choice([-1, 1], size=(50, 256)).astype(np.int8)
    for p in patterns:
        sdm.write(p)
    recovered = 0
    for p in patterns:
        noisy = p.copy()
        idx = rng.choice(256, size=26, replace=False)  # 10% flipped
        noisy[idx] *= -1
        recovered += bool((sdm.read(noisy, iters=3) == p).all())
    assert recovered / len(patterns) >= 0.95
