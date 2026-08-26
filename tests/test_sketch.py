"""Membership and frequency sketches by bundling."""

import numpy as np
import pytest

from holo import FrequencySketch, MembershipFilter


def test_membership_filter(space):
    f = MembershipFilter(space)
    for i in range(200):
        f.add(f"in{i}")
    assert all(f"in{i}" in f for i in range(200))  # no false negatives
    fpr = sum(f"out{i}" in f for i in range(1000)) / 1000
    assert fpr < 0.05


def test_frequency_sketch(space):
    sk = FrequencySketch(space)
    counts = {"a": 7, "b": 30, "c": 1, "d": 112}
    for w, c in counts.items():
        sk.add(w, c)
    for w, c in counts.items():
        # crosstalk noise std = sqrt(sum of OTHER counts^2 / (2d))
        sigma = np.sqrt(sum(v ** 2 for k, v in counts.items() if k != w)
                        / (2 * space.dim))
        assert sk.estimate(w) == pytest.approx(c, abs=4 * sigma)
