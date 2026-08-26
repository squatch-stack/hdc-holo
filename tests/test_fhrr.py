"""Core FHRR algebra: bind/unbind exactness, superposition, codewords."""

import numpy as np
import pytest

from holo import FHRR, ItemMemory


def test_bind_unbind_is_exact(space):
    a, b = space.random(), space.random()
    assert space.sim(FHRR.unbind(FHRR.bind(a, b), b), a) == pytest.approx(
        1.0, abs=1e-3)
    assert abs(space.sim(a, b)) < 0.1  # random vectors are quasi-orthogonal


def test_bundle_preserves_members(space):
    items = [space.random() for _ in range(20)]
    bundle = FHRR.bundle(*items)
    for v in items:
        assert space.sim(bundle, v) > 0.7
    assert abs(space.sim(bundle, space.random())) < 0.3


def test_label_vectors_are_deterministic_and_order_free():
    s1, s2 = FHRR(2048, seed=3), FHRR(2048, seed=3)
    m1, m2 = ItemMemory(s1), ItemMemory(s2)
    for lab in ["alice", "bob", "carol"]:      # one replica, one order
        m1.get(lab)
    for lab in ["carol", "alice", "bob"]:      # another replica, another
        m2.get(lab)
    for lab in ["alice", "bob", "carol"]:
        assert np.array_equal(m1.get(lab), m2.get(lab))
    assert not np.array_equal(FHRR(2048, seed=4).label_vector("alice"),
                              s1.label_vector("alice"))  # seed-scoped
