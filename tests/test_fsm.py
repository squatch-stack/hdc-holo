"""Finite state machine in superposition."""

from holo import HoloFSM


def test_fsm_mod3(space):
    fsm = HoloFSM(space)
    for s in range(3):
        for bit in "01":
            fsm.add_transition(f"r{s}", bit, f"r{(2 * s + int(bit)) % 3}")
    for value in range(1, 32):
        word = bin(value)[2:]
        final = fsm.run("r0", word)[-1]
        assert (final == "r0") == (value % 3 == 0)
