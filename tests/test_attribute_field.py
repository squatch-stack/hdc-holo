"""Attribute-carrying splats: what_is_at, where_is, record payloads."""

import numpy as np

from holo import AttributeSplatField, RecordSpace


def test_attribute_field_what_and_where(space):
    field = AttributeSplatField(space, 0.04)
    labels = ["tree", "rock", "water"]
    # centers on a grid spaced 0.15 (~4 sigma) so classes never overlap
    centers, assigned = [], []
    for i in range(6):
        for j in range(5):
            mu = np.array([0.1 + 0.15 * i, 0.15 + 0.15 * j])
            lab = labels[(i * 5 + j) % 3]
            field.add_splat(mu, lab, alpha=1.0)
            centers.append(mu)
            assigned.append(lab)
    # "what is at p" is exact well inside capacity (noise ~ sqrt(30/2d))
    for mu, lab in zip(centers, assigned):
        got, score = field.what_is_at(mu)
        assert got == lab
        assert score > 0.6
    # joint existence test
    assert field.is_there(assigned[0], centers[0]) > 0.6
    others = [l for l in labels if l != assigned[0]]
    assert field.is_there(others[0], centers[0]) < 0.3
    # "where is": the class hologram renders high at its centers, low at rest
    holo = field.where_is("tree")
    own = [mu for mu, lab in zip(centers, assigned) if lab == "tree"]
    rest = [mu for mu, lab in zip(centers, assigned) if lab != "tree"]
    assert field.eval_positions(holo, np.array(own)).min() > 0.7
    assert field.eval_positions(holo, np.array(rest)).max() < 0.3


def test_attribute_field_record_payloads(space):
    rs = RecordSpace(space)
    field = AttributeSplatField(space, 0.04)
    rng = np.random.default_rng(3)
    stored = []
    for i in range(20):
        mu = rng.uniform(0.1, 0.9, size=2)
        kind, color = f"kind{i % 4}", f"color{i % 5}"
        field.add_splat(mu, rs.encode({"kind": kind, "color": color}))
        stored.append((mu, kind, color))
    # two exact unbinds deep: position, then role
    for mu, kind, color in stored:
        payload = field.at(mu)
        assert rs.get(payload, "kind")[0] == kind
        assert rs.get(payload, "color")[0] == color
