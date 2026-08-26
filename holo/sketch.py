"""Sketches: Bloom-filter and count-min-style structures as plain bundling.

A binary Bloom filter marks k hash positions per item; a bundle adds the
item's whole random phasor vector. Membership testing is one inner
product: members score ~1, non-members score 0 +- sqrt(N/(2d)). The
false-positive rate is the Gaussian tail beyond the threshold — the same
trade a Bloom filter makes, but real-valued and mergeable by addition.

The frequency sketch is the same vector read differently: adding an item
c times (or with weight c) makes its inner product estimate c directly,
like a count-min sketch whose 'rows' are the d phasor dimensions.
"""

import numpy as np

from .fhrr import FHRR, ItemMemory


class MembershipFilter:
    def __init__(self, space, threshold=0.5):
        self.space = space
        self.items = ItemMemory(space, "members")
        self.M = space.zeros()
        self.threshold = threshold
        self.n = 0

    def add(self, item):
        self.M += self.items.get(item)
        self.n += 1

    def score(self, item):
        return self.space.sim(self.M, self.items.get(item))

    def __contains__(self, item):
        return self.score(item) >= self.threshold


class FrequencySketch:
    def __init__(self, space):
        self.space = space
        self.items = ItemMemory(space, "counted")
        self.M = space.zeros()

    def add(self, item, count=1.0):
        self.M += np.complex64(count) * self.items.get(item)

    def estimate(self, item):
        return self.space.sim(self.M, self.items.get(item))


def demo(dim=4096, seed=0):
    print(f"== Sketches: membership + frequency by bundling (d={dim}) ==")
    print(f"{'members N':>10} {'pred FPR':>9} {'meas FPR':>9} {'meas FNR':>9}")
    n_probes = 4000
    for n_members in [100, 200, 500, 1000]:
        space = FHRR(dim, seed=seed)
        f = MembershipFilter(space)
        for i in range(n_members):
            f.add(f"member{i}")
        fnr = sum(f"member{i}" not in f for i in range(n_members)) / n_members
        fpr = sum(f"other{i}" in f for i in range(n_probes)) / n_probes
        # non-member score ~ N(0, sqrt(N/(2d))); FPR = tail beyond 0.5
        from math import erfc, sqrt
        z = f.threshold / np.sqrt(n_members / (2 * dim))
        pred = 0.5 * erfc(z / sqrt(2))
        print(f"{n_members:>10} {pred:>9.2%} {fpr:>9.2%} {fnr:>9.2%}")

    space = FHRR(dim, seed=seed)
    sk = FrequencySketch(space)
    rng = np.random.default_rng(seed + 2)
    true = {f"word{i}": int(c) for i, c in
            enumerate(rng.zipf(1.8, 40).clip(1, 200))}
    for w, c in true.items():
        sk.add(w, c)
    errs = [abs(sk.estimate(w) - c) for w, c in true.items()]
    print(f"frequency sketch: 40 items, counts 1..200, "
          f"mean |err| = {np.mean(errs):.2f}, max |err| = {np.max(errs):.2f}")
    some = list(true.items())[:4]
    for w, c in some:
        print(f"  count({w}) true={c:<4d} est={sk.estimate(w):.1f}")
    print()
