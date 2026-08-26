"""Holographic graph: the whole edge set as one vector (GrapHD-style).

    G = sum over edges (u, v) of bind(U, rho(V))

Edge membership is one inner product: ~1 if present, 0 +- sqrt(m/(2d))
otherwise. Unbinding a node U (then undoing rho) yields the superposition
of its out-neighbors, recovered by thresholded cleanup — an adjacency
list read out by correlation instead of pointer chasing.

The permutation rho on the target slot matters because binding is
COMMUTATIVE: with plain bind(U, V) an edge is an unordered pair, so a
directed graph can't tell (u, v) from (v, u), and neighbors(u) returns
in-neighbors mixed in with out-neighbors as *exact* aliases, not noise.
rho makes source and target distinct roles. Undirected graphs simply
bundle both directions.
"""

import numpy as np

from .fhrr import FHRR, ItemMemory, Permutation


class HoloGraph:
    def __init__(self, space, directed=True, threshold=0.5, perm_seed=1):
        self.space = space
        self.nodes = ItemMemory(space, "nodes")
        self.rho = Permutation(space, seed=perm_seed)
        self.G = space.zeros()
        self.directed = directed
        self.threshold = threshold
        self.m = 0

    def add_edge(self, u, v):
        U, V = self.nodes.get(u), self.nodes.get(v)
        self.G += FHRR.bind(U, self.rho(V))
        if not self.directed:
            self.G += FHRR.bind(V, self.rho(U))
        self.m += 1

    def has_edge(self, u, v):
        s = self.space.sim(self.G, FHRR.bind(self.nodes.get(u),
                                             self.rho(self.nodes.get(v))))
        return s >= self.threshold, s

    def neighbors(self, u):
        """Out-neighbors of u, as (label, score), best first."""
        x = self.rho(FHRR.unbind(self.G, self.nodes.get(u)), power=-1)
        return self.nodes.matches(x, self.threshold)


def demo(dim=4096, seed=0):
    print(f"== HoloGraph: edge set in superposition (d={dim}) ==")
    n_nodes = 60
    print(f"{'edges m':>8} {'edge precision':>15} {'edge recall':>12} "
          f"{'neighbor-set exact':>19}")
    for m in [100, 400, 1000, 2000]:
        space = FHRR(dim, seed=seed)
        g = HoloGraph(space, directed=True)
        rng = np.random.default_rng(seed + 4)
        edges = set()
        while len(edges) < m:
            u, v = rng.integers(n_nodes, size=2)
            if u != v:
                edges.add((int(u), int(v)))
        adj = {}
        for u, v in edges:
            g.add_edge(f"n{u}", f"n{v}")
            adj.setdefault(u, set()).add(v)
        # positive probes: every edge; negative probes: absent pairs
        tp = sum(g.has_edge(f"n{u}", f"n{v}")[0] for u, v in edges)
        neg, fp = 0, 0
        while neg < m:
            u, v = rng.integers(n_nodes, size=2)
            if u != v and (int(u), int(v)) not in edges:
                fp += g.has_edge(f"n{u}", f"n{v}")[0]
                neg += 1
        precision = tp / (tp + fp) if tp + fp else 1.0
        exact = sum(
            {lbl for lbl, _ in g.neighbors(f"n{u}")} ==
            {f"n{v}" for v in vs}
            for u, vs in adj.items()) / len(adj)
        print(f"{m:>8} {precision:>15.1%} {tp/m:>12.1%} {exact:>19.1%}")

    space = FHRR(dim, seed=seed)
    g = HoloGraph(space, directed=False)
    for u, v in [("a", "b"), ("a", "c"), ("c", "d")]:
        g.add_edge(u, v)
    print(f"undirected toy graph a-b, a-c, c-d: "
          f"neighbors(a) = {[l for l, _ in g.neighbors('a')]}, "
          f"neighbors(c) = {[l for l, _ in g.neighbors('c')]}")
    print()
