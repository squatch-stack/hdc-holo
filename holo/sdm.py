"""Kanerva's Sparse Distributed Memory: RAM with hypervector addresses.

The ancestor of all of this (1988). M hard storage locations get random
binary addresses in {-1,+1}^n. A write activates every location within
Hamming radius r of the target address and increments/decrements its
counters with the data word; a read sums the counters of the activated
locations and takes the sign. Every pattern is smeared across ~p*M
locations and every location holds pieces of many patterns — so reads
work from *approximate* addresses, and iterating a read (feed the output
back in as the address) converges to the stored pattern like a Hopfield
attractor. Used autoassociatively here: data == address.
"""

import numpy as np

from .demokit import Table


class SparseDistributedMemory:
    def __init__(self, n_locations=5000, dim=256, radius=108, seed=0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.radius = radius
        self.addresses = rng.choice([-1, 1], size=(n_locations, dim)) \
                            .astype(np.int8)
        self.counters = np.zeros((n_locations, dim), dtype=np.float32)

    def _active(self, addr):
        # Hamming distance = (dim - dot)/2 for bipolar vectors
        dots = self.addresses @ addr.astype(np.int32)
        return (self.dim - dots) // 2 <= self.radius

    def write(self, addr, data=None):
        data = addr if data is None else data
        self.counters[self._active(addr)] += data

    def read(self, addr, iters=3):
        for _ in range(iters):
            s = self.counters[self._active(addr)].sum(axis=0)
            addr = np.where(s >= 0, 1, -1).astype(np.int8)
        return addr


def demo(dim=4096, seed=0):  # dim arg unused; SDM has its own geometry
    n, M, r = 256, 5000, 108
    print(f"== Sparse Distributed Memory: {M} locations, "
          f"{n}-bit addresses, radius {r} ==")
    rng = np.random.default_rng(seed + 6)
    sdm = SparseDistributedMemory(M, n, r, seed=seed)
    patterns = rng.choice([-1, 1], size=(150, n)).astype(np.int8)
    for p in patterns:
        sdm.write(p)
    active = sdm._active(patterns[0]).sum()
    print(f"stored 150 random patterns autoassociatively "
          f"(~{active} locations activated per write)")
    # the row format used width 13 for a column the header set to 14,
    # so the numbers sat one space left of their heading — the drift
    # this library exists to stop. One table now defines both.
    table = Table(("address noise", 14, ".0%"), ("exact recovery", 15, ".1%"),
                  ("avg bit errors", 15, ".1f"))
    table.header()
    for flip_frac in [0.05, 0.10, 0.20, 0.30, 0.40]:
        exact, bit_errs = 0, 0
        for p in patterns:
            noisy = p.copy()
            idx = rng.choice(n, size=int(flip_frac * n), replace=False)
            noisy[idx] *= -1
            out = sdm.read(noisy, iters=3)
            errs = int((out != p).sum())
            exact += errs == 0
            bit_errs += errs
        table.row(flip_frac, exact / len(patterns),
                  bit_errs / len(patterns))
    print()
