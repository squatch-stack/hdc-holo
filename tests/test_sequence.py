"""Permutation-tagged stacks and sequences."""

from holo import HoloStack, SequenceMemory


def test_stack_lifo(space):
    st = HoloStack(space)
    pushed = [f"s{i % 32}" for i in range(100)]
    for label in pushed:
        st.push(label)
    popped = [st.pop()[0] for _ in range(100)]
    assert popped == pushed[::-1]


def test_sequence_random_access(space):
    seq = SequenceMemory(space)
    word = list("holographic")
    s = seq.encode(word)
    assert [seq.decode(s, i)[0] for i in range(len(word))] == word
