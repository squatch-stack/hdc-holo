"""holo/dispatch.py — near-enough dispatch, banding, abstention."""

import numpy as np

from holo import FHRR
from holo.dispatch import BandedDispatcher, FastNGramProfiler, NearEnoughDispatcher


def _mkword(rng):
    return "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"),
                              rng.integers(4, 10)))


def _rules(rng, n_rules, n_actions=8, vocab=None, per_topic=None):
    """Rules over a shared vocab, or topic-structured when per_topic."""
    if per_topic:
        topics = [[_mkword(rng) for _ in range(per_topic)]
                  for _ in range(n_actions)]
        vocab = [w for t in topics for w in t]
        rules = [(" ".join(rng.choice(topics[i % n_actions], 5,
                                      replace=False)),
                  f"route-{i % n_actions}") for i in range(n_rules)]
    else:
        vocab = vocab or [_mkword(rng) for _ in range(400)]
        rules = [(" ".join(rng.choice(vocab, 5, replace=False)),
                  f"route-{i % n_actions}") for i in range(n_rules)]
    return rules, vocab


def _typo(rng, text, rate):
    return "".join(chr(ord("a") + rng.integers(26))
                   if c != " " and rng.random() < rate else c
                   for c in text)


def test_fast_profiler_matches_ngram_encoder():
    space = FHRR(1024, seed=0)
    for n in (2, 3, 4):                            # n is honored, not
        fast = FastNGramProfiler(space, n=n)       # hardcoded trigrams
        slow = fast.enc.profile("the quick brown fox")
        quick = fast.profile("the quick brown fox")
        assert np.allclose(quick, slow, atol=1e-3)  # identical math


def test_matrix_dispatch_survives_typos():
    rng = np.random.default_rng(1)
    rules, _vocab = _rules(rng, 64)
    d = NearEnoughDispatcher(rules, dim=2048, seed=0)
    # 15% character typos on every condition: trigram cosine margins
    # sit far above cross-rule similarity of disjoint random keywords
    for cond, action in rules[:32]:
        got, score = d.dispatch_matrix(_typo(rng, cond, 0.15))
        assert got == action
        assert score > 0.3


def test_exactness_is_brittle_where_matrix_is_not():
    rng = np.random.default_rng(2)
    rules, vocab = _rules(rng, 32)
    d = NearEnoughDispatcher(rules, dim=2048, seed=0)
    cond, action = rules[0]
    words = cond.split()[1:]                        # drop one keyword
    text = " ".join([*words, vocab[0]])
    # Boolean AND cannot fire with a keyword missing...
    toks = set(text.split())
    fired = [a for c, a in rules if all(k in toks for k in c.split())]
    assert action not in fired
    # ...the graded engine still routes it
    assert d.dispatch_matrix(text)[0] == action


def test_bundle_dispatch_within_capacity():
    rng = np.random.default_rng(3)
    rules, _vocab = _rules(rng, 48)
    # crosstalk sqrt(N/2d) = sqrt(48/8192) ~ 0.077 vs signal ~1: >4 sigma
    d = NearEnoughDispatcher(rules, dim=4096, seed=0)
    for cond, action in rules[:24]:
        assert d.dispatch_bundle(cond)[0] == action


def test_flat_bundle_pays_the_law_and_banding_rescues():
    rng = np.random.default_rng(4)
    # a wide vocab keeps rules non-confusable, so the ONLY error source
    # is bundle crosstalk — the quantity under test
    vocab = [_mkword(rng) for _ in range(3000)]
    rules, vocab = _rules(rng, 1024, n_actions=16, vocab=vocab)
    d = NearEnoughDispatcher(rules, dim=1024, seed=0)   # floor ~ 0.71
    banded = BandedDispatcher(d, 64)                    # floor ~ 0.089
    flat_hits = band_hits = 0
    probes = [rules[i] for i in rng.integers(0, 1024, 60)]
    for cond, action in probes:
        flat_hits += d.dispatch_bundle(cond)[0] == action
        band_hits += banded.dispatch(cond)[0] == action
    # bounds carry slack because a handful of probes sit near argmax
    # ties, and numpy's ~1-ulp alignment-dependent variance (see
    # SDK.md's determinism caveat) can flip those across processes —
    # the SEPARATION is the assertion, not an exact count
    assert band_hits >= 52            # per-band load 16 at d=1024: clean
    assert flat_hits <= 42            # load 1024 at d=1024: past budget
    assert band_hits - flat_hits >= 15


def test_clustered_routing_finds_the_band():
    rng = np.random.default_rng(5)
    rules, _vocab = _rules(rng, 512, n_actions=8, per_topic=40)
    d = NearEnoughDispatcher(rules, dim=2048, seed=0)
    cl = BandedDispatcher(d, 16, clustered=True)
    hits = 0
    probes = [rules[i] for i in rng.integers(0, 512, 60)]
    for cond, action in probes:
        got, _ = cl.dispatch(_typo(rng, cond, 0.1), top_r=1)
        hits += got == action
    assert hits >= 54                 # topic structure makes top-1 route


def test_abstention_threshold_is_policy():
    rng = np.random.default_rng(6)
    rules, _vocab = _rules(rng, 32)
    d = NearEnoughDispatcher(rules, dim=2048, seed=0)
    # gibberish scores near the noise floor -> abstain
    got, score = d.dispatch_matrix("zzzz qqqq xxxx vvvv", threshold=0.3)
    assert got is None and score < 0.3
    # a clean condition clears the same threshold
    got, score = d.dispatch_matrix(rules[0][0], threshold=0.3)
    assert got == rules[0][1] and score > 0.9


def test_dispatchers_are_deterministic():
    rng = np.random.default_rng(7)
    rules, _ = _rules(rng, 16)
    a = NearEnoughDispatcher(rules, dim=1024, seed=3)
    b = NearEnoughDispatcher(rules, dim=1024, seed=3)
    # semantic, not bitwise: recomputed sums agree to ~1 ulp only
    # (alignment-dependent SIMD — SDK.md's determinism caveat), so
    # allclose on state and exact equality on decisions
    assert np.allclose(a.R, b.R, atol=1e-4)
    ra, rb = a.dispatch_matrix(rules[5][0]), b.dispatch_matrix(rules[5][0])
    assert ra[0] == rb[0]
    assert abs(ra[1] - rb[1]) < 1e-5
