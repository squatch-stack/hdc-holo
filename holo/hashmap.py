"""Holographic hash map: every key/value pair lives superposed in ONE vector.

    M = sum_i  bind(K_i, V_i)

Lookup unbinds the key — the matching pair yields its value codeword
exactly, every other pair contributes noise — then cleanup snaps the
noisy result to the nearest known value. There is no addressing, no
buckets, no collisions: capacity is a signal-to-noise budget. Retrieval
noise std is sqrt((N-1)/(2d)) for N pairs in dimension d.

Deletion shows the WORM-like grain of superposition: you can only
subtract a pair exactly, so delete() must first *retrieve* the value.
A wrong retrieval makes deletion corrupt the store instead of shrinking it.
"""

import numpy as np

from .demokit import Table, banner
from .fhrr import FHRR, ItemMemory


class HoloMap:
    def __init__(self, space):
        self.space = space
        self.keys = ItemMemory(space, "keys")
        self.values = ItemMemory(space, "values")
        self.M = space.zeros()
        self.n = 0

    def put(self, key, value):
        self.M += FHRR.bind(self.keys.get(key), self.values.get(value))
        self.n += 1

    def get(self, key):
        """Returns (value_label, score). Score near 1 means confident."""
        v_hat = FHRR.unbind(self.M, self.keys.get(key))
        return self.values.cleanup(v_hat)

    def delete(self, key):
        """Retrieve-then-subtract. Only exact if get() was correct."""
        value, score = self.get(key)
        self.M -= FHRR.bind(self.keys.get(key), self.values.get(value))
        self.n -= 1
        return value, score


def demo(dim=4096, seed=0):
    banner("HoloMap: hash map in superposition", dim)
    table = Table(("pairs N", 8), ("load N/d", 9, ".2f"),
                  ("pred noise", 11, ".3f"), ("accuracy", 9, ".1%"))
    table.header()
    n_values = 256
    for n_pairs in [100, 500, 1000, 2000, 4000]:
        space = FHRR(dim, seed=seed)
        m = HoloMap(space)
        rng = np.random.default_rng(seed + 1)
        pairs = [(f"key{i}", f"val{rng.integers(n_values)}")
                 for i in range(n_pairs)]
        for k, v in pairs:
            m.put(k, v)
        correct = sum(m.get(k)[0] == v for k, v in pairs)
        noise = np.sqrt((n_pairs - 1) / (2 * dim))
        table.row(n_pairs, n_pairs / dim, noise, correct / n_pairs)

    space = FHRR(dim, seed=seed)
    m = HoloMap(space)
    for k, v in [("alice", "eng"), ("bob", "sales"), ("carol", "legal")]:
        m.put(k, v)
    # scores rounded deliberately: the post-delete score is CROSSTALK,
    # and numpy's complex multiply reproduces only to ~1 ulp across
    # runs (SDK.md's determinism caveat), so printing 17 digits of it
    # made this demo's output differ run to run for no information
    def show(label, hit):
        print("%s -> (%r, %.3f)" % (label, hit[0], hit[1]))

    show("get('bob')", m.get("bob"))
    show("delete('bob')", m.delete("bob"))
    show("get('bob') after delete", m.get("bob"))
    print("   (the last score is noise: ~0)")
    print()
