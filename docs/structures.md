# Classical data structures as holograms

*[← docs index](README.md) · foundations*

**What.** Address-based structures rebuilt as correlation-based ones:
the whole structure lives superposed in one vector, lookups are inner
products, and there are no buckets, pointers, or collisions — only an
SNR budget. One implementation file per structure under `holo/`.

| structure | encoding | readout |
|---|---|---|
| `HoloMap` | `sum bind(K_i, V_i)` | unbind key, cleanup value |
| `MembershipFilter` | `sum item_i` (a real-valued Bloom filter) | `sim >= 0.5` |
| `FrequencySketch` | `sum count_i * item_i` (count-min cousin) | `sim` estimates the count |
| `RecordSpace` | `sum bind(role_i, filler_i)` | unbind role; analogies by algebra |
| `HoloStack` / `SequenceMemory` | permutation powers as position tags | cleanup at `rho^-i` |
| `NGramEncoder` | bound, permuted trigrams | cosine between profiles |
| `HoloGraph` | `sum bind(U, rho(V))` | unbind node -> neighbor superposition |
| `HoloFSM` | `sum bind(S, A, rho(S'))` | unbind state & symbol, cleanup |
| `SparseDistributedMemory` | Kanerva 1988: RAM with hypervector addresses | activated-counter sums, iterated |

**Budgets** (d = 4096 measured points, from the demo tables):
HoloMap holds ~500 pairs before errors appear (noise `sqrt(N/2d)`
against a 256-value codebook); records support ~320 fields; the stack
pops reliably to depth ~200; membership FPR is the Gaussian tail
`P(N(0, sqrt(N/2d)) > 0.5)` and matches prediction to a few tenths of a
percent; the frequency sketch's error is `sqrt(sum other counts^2/2d)`.
SDM's operating point matters: activation radius 108 (~30 of 5000
locations) recovers 96% of 150 patterns at 5% address noise — radius 104
activates ~3 locations and misses, 116+ activates hundreds and drowns.

**Failure modes.** Deletion needs the exact addend (retrieve-then-
subtract corrupts on a wrong retrieval); sequences/stacks corrupt
irreversibly after a cleanup miss (no rollback in a hologram); analogy
queries (`"dollar of Mexico"`) pay second-order noise.

**API.**
```python
from holo import FHRR, HoloMap, RecordSpace
space = FHRR(4096, seed=0)
m = HoloMap(space); m.put("alice", "eng")
value, score = m.get("alice")            # ('eng', ~1.0)
rs = RecordSpace(space)
usa = rs.encode({"capital": "washington", "currency": "dollar"})
mex = rs.encode({"capital": "cdmx", "currency": "peso"})
rs.analogy(usa, mex, "dollar")           # ('peso', ~1.0)
```

**Evidence.** `tests/test_hashmap.py` ... `tests/test_sdm.py`;
capacity tables from `holo-demos hashmap sketch record sequence ngram
graph fsm sdm` — each demo pushes past its cliff on purpose.

Built on these: [dispatch.md](dispatch.md) turns trigram profiles +
bind/bundle/cleanup into a rule engine — the SDK's first
application-layer technique.
