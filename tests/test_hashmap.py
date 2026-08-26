"""HoloMap: key/value pairs in superposition."""

import pytest

from holo import HoloMap


def test_holomap_round_trip(space):
    m = HoloMap(space)
    pairs = {f"k{i}": f"v{i % 50}" for i in range(200)}
    for k, v in pairs.items():
        m.put(k, v)
    for k, v in pairs.items():
        label, score = m.get(k)
        assert label == v
        assert score > 0.5


def test_holomap_delete(space):
    m = HoloMap(space)
    m.put("a", "x")
    m.put("b", "y")
    assert m.delete("a") == ("x", pytest.approx(1.0, abs=0.1))
    assert m.get("b")[0] == "y"
    assert m.get("a")[1] < 0.5  # only noise remains under key a
