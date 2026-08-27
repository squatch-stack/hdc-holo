"""Finite state machine in superposition (Osipov/Kleyko-style).

The whole transition table is one vector:

    T = sum over (s, a) of  bind(S(s), A(a), rho(S(delta(s, a))))

Stepping unbinds current state and input symbol, undoes the permutation,
and cleanup snaps the result to a legal state.

Why the permutation rho on the next-state slot: binding is COMMUTATIVE,
so without it a transition (t, a, s) *entering* the queried state s on
the queried symbol a unbinds to S(t)*S(s)*conj(S(s)) = S(t) — a perfect
alias, indistinguishable from the true answer. Permuting the target slot
gives 'current state' and 'next state' distinct roles, turning that
alias back into noise. (First version of this file had the bug; the
capacity demo caught it at ~67% accuracy where theory predicted ~100%.)
"""

import numpy as np

from .demokit import banner
from .fhrr import FHRR, ItemMemory, Permutation


class HoloFSM:
    def __init__(self, space, perm_seed=1):
        self.space = space
        self.states = ItemMemory(space, "states")
        self.symbols = ItemMemory(space, "symbols")
        self.rho = Permutation(space, seed=perm_seed)
        self.T = space.zeros()

    def add_transition(self, state, symbol, next_state):
        self.T += FHRR.bind(self.states.get(state),
                            self.symbols.get(symbol),
                            self.rho(self.states.get(next_state)))

    def step(self, state, symbol):
        q = FHRR.unbind(self.T, FHRR.bind(self.states.get(state),
                                          self.symbols.get(symbol)))
        return self.states.cleanup(self.rho(q, power=-1))

    def run(self, start, inputs):
        state, trajectory = start, [start]
        for symbol in inputs:
            state, _ = self.step(state, symbol)
            trajectory.append(state)
        return trajectory


def demo(dim=4096, seed=0):
    banner("HoloFSM: transition table in one vector", dim)
    # divisibility-by-3 automaton over binary strings: state = value mod 3
    space = FHRR(dim, seed=seed)
    mod3 = HoloFSM(space)
    for s in range(3):
        for bit in "01":
            mod3.add_transition(f"r{s}", bit, f"r{(2 * s + int(bit)) % 3}")
    for word in ["110", "1011", "1100", "111"]:
        final = mod3.run("r0", word)[-1]
        print(f"  {word} (={int(word, 2)}): final state {final} "
              f"-> divisible by 3: {final == 'r0'}")

    # capacity: random DFAs of growing size, per-step agreement w/ ground truth
    print(f"  {'|Q| x |A|':>10} {'transitions':>12} {'per-step accuracy':>18}")
    rng = np.random.default_rng(seed + 5)
    for n_states, n_syms in [(10, 5), (20, 10), (40, 20), (80, 20)]:
        space = FHRR(dim, seed=seed)
        fsm = HoloFSM(space)
        delta = {(s, a): int(rng.integers(n_states))
                 for s in range(n_states) for a in range(n_syms)}
        for (s, a), s2 in delta.items():
            fsm.add_transition(f"q{s}", f"a{a}", f"q{s2}")
        steps, ok = 0, 0
        for _ in range(50):
            s = 0
            for a in rng.integers(n_syms, size=30):
                expect = delta[(s, int(a))]
                got, _ = fsm.step(f"q{s}", f"a{a}")
                ok += got == f"q{expect}"
                steps += 1
                s = expect  # follow ground truth so errors don't compound
        print(f"  {n_states:>4} x {n_syms:<4} {n_states * n_syms:>12} "
              f"{ok/steps:>18.1%}")
    print()
