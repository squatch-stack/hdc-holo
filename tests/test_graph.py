"""Holographic graph: edge queries and neighbor recovery."""

from holo import HoloGraph


def test_graph_edges_and_neighbors(space):
    g = HoloGraph(space, directed=True)
    edges = [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")]
    for u, v in edges:
        g.add_edge(u, v)
    for u, v in edges:
        assert g.has_edge(u, v)[0]
    assert not g.has_edge("d", "a")[0]
    assert {l for l, _ in g.neighbors("a")} == {"b", "c"}
