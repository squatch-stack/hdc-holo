"""A rule engine with no Boolean gates.

Conditions are trigram-profile hypervectors over messy text, dispatch
is similarity, and the acceptance threshold is POLICY — below it the
engine abstains ("route to a human"), which an if-table cannot say.
Part 2 reproduces the scaling claim in holo/dispatch.py's docstring:
a topic-structured 4096-rule book routed by clustered band bundles at
a fraction of the matrix engine's compute. See docs/dispatch.md.

    python examples/near_enough_rules.py [--scale]
"""

import sys

import numpy as np

from holo import BandedDispatcher, NearEnoughDispatcher

# -- part 1: a support desk that survives real typing ----------------

rules = [
    ("password reset locked out login access", "auth"),
    ("balance statement invoice billing charge", "billing"),
    ("shipping delivery tracking package late", "logistics"),
    ("refund return broken damaged replacement", "returns"),
    ("cancel subscription downgrade plan", "retention"),
]
d = NearEnoughDispatcher(rules, dim=4096)

messy = [
    "hi im lokced out cant reset my passwrd",
    "wheres my pakage?? tracking says late",
    "u charged me twice, need that invoce checked",
    "it arrived brokn i want a replacment",
    "asdf qwerty zzzz",                       # gibberish: should abstain
]
print("input -> action (score), threshold 0.25 is policy:")
for text in messy:
    action, score = d.dispatch_matrix(text, threshold=0.25)
    label = action if action else "ABSTAIN -> human"
    print(f"  {text!r:<48} {label}  ({score:.2f})")

# -- part 2: the law and the medicine at 4096 rules ------------------
# A flat one-vector rulebook pays crosstalk ~sqrt(N/2d); k-means
# banding + top-r centroid routing restores accuracy while reading
# only the bands the query is near — the same partition+locality
# medicine that fixes dense scenes (docs/spatial.md).

if "--scale" not in sys.argv:
    print("\n(run with --scale for the 4096-rule banding experiment)")
    sys.exit(0)

rng = np.random.default_rng(0)
n_rules, n_actions, dim = 4096, 32, 2048
topics = [["".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"),
                              rng.integers(4, 10))) for _ in range(60)]
          for _ in range(n_actions)]
big = [(" ".join(rng.choice(topics[i % n_actions], 6, replace=False)),
        f"route-{i % n_actions}") for i in range(n_rules)]
vocab = [w for t in topics for w in t]

dt = NearEnoughDispatcher(big, dim=dim)
clustered = BandedDispatcher(dt, n_bands=64, clustered=True)


def corrupt(text):
    words = text.split()[1:] + list(rng.choice(vocab, 2))
    rng.shuffle(words)
    return " ".join("".join(chr(ord("a") + rng.integers(26))
                            if rng.random() < 0.1 else c for c in w)
                    for w in words)


engines = [
    ("matrix (ceiling)", lambda t: dt.dispatch_matrix(t), n_rules),
    ("flat bundle", lambda t: dt.dispatch_bundle(t), n_actions),
    ("clustered top-1", lambda t: clustered.dispatch(t, top_r=1),
     64 + n_actions),
    ("clustered top-4", lambda t: clustered.dispatch(t, top_r=4),
     64 + 4 * n_actions),
]
print(f"\nN={n_rules} rules, d={dim}, typo 10%, one keyword dropped"
      f"  (flat-bundle floor sqrt(N/2d) = {np.sqrt(n_rules/(2*dim)):.2f})")
print(f"  {'engine':>18} {'acc':>6}   inner products / dispatch")
probes = [big[i] for i in rng.integers(0, n_rules, 300)]
for label, fn, ips in engines:
    hits = sum(fn(corrupt(c))[0] == a for c, a in probes)
    note = "" if ips == n_rules else f"  ({n_rules/ips:.0f}x less than matrix)"
    print(f"  {label:>18} {hits/len(probes):>6.2f}   {ips:>5}{note}")
