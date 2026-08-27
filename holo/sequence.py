"""Sequences and stacks via permutation: order as a spatial transform.

A fixed random permutation rho acts like a position tag that composes:

    S = item_0 + rho(item_1) + rho^2(item_2) + ...

Random permutations map codewords to (nearly) orthogonal vectors, so each
position is a separate 'channel' in the same superposition. A stack is
the same trick used destructively: push shifts everything one position
deeper and adds the new top; pop cleans up the top, subtracts it exactly,
and shifts back. Pops stay *exact* as long as cleanup classifies the top
correctly — the failure mode is a misclassification at depth, after which
the subtraction corrupts the store (there is no rollback in a hologram).
"""

import numpy as np

from .demokit import Table, banner
from .fhrr import FHRR, ItemMemory, Permutation


class HoloStack:
    def __init__(self, space, perm_seed=1):
        self.space = space
        self.items = ItemMemory(space, "stack-items")
        self.rho = Permutation(space, seed=perm_seed)
        self.S = space.zeros()
        self.depth = 0

    def push(self, label):
        self.S = self.rho(self.S) + self.items.get(label)
        self.depth += 1

    def peek(self):
        return self.items.cleanup(self.S)

    def pop(self):
        label, score = self.peek()
        self.S = self.rho(self.S - self.items.get(label), power=-1)
        self.depth -= 1
        return label, score


class SequenceMemory:
    """Whole sequence in one vector, random access to any position."""

    def __init__(self, space, perm_seed=1):
        self.space = space
        self.items = ItemMemory(space, "seq-items")
        self.rho = Permutation(space, seed=perm_seed)

    def encode(self, labels):
        s = self.space.zeros()
        for i, label in enumerate(labels):
            s += self.rho(self.items.get(label), power=i)
        return s

    def decode(self, s, position):
        return self.items.cleanup(self.rho(s, power=-position))


def demo(dim=4096, seed=0):
    banner("Stack & sequence via permutation", dim)
    table = Table(("depth", 7), ("LIFO pop accuracy", 18, ".1%"),
                  ("first error at", 15))
    table.header()
    alphabet = 64
    for depth in [50, 200, 800, 1600]:
        space = FHRR(dim, seed=seed)
        st = HoloStack(space)
        rng = np.random.default_rng(seed + 3)
        pushed = [f"sym{rng.integers(alphabet)}" for _ in range(depth)]
        for label in pushed:
            st.push(label)
        first_err = None
        correct = 0
        for i, expect in enumerate(reversed(pushed)):
            got, _ = st.pop()
            if got == expect:
                correct += 1
            elif first_err is None:
                first_err = i + 1
        table.row(depth, correct / depth,
                  first_err if first_err else "-")

    space = FHRR(dim, seed=seed)
    seq = SequenceMemory(space)
    word = list("holographic")
    s = seq.encode(word)
    decoded = "".join(seq.decode(s, i)[0] for i in range(len(word)))
    print(f'sequence: encoded "holographic" as ONE vector, '
          f'decoded per-position -> "{decoded}"')
    print()
