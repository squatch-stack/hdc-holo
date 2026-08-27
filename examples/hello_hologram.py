"""The algebra in five minutes.

Everything in this SDK is one trick applied carefully: state lives
superposed in a d-dimensional complex vector of unit phasors, binding
is elementwise multiply (exactly invertible by the conjugate),
bundling is addition, and reading anything back is one inner product
against noise of std ~sqrt(N/(2d)) — the "one law" every docs page
budgets against.

    python examples/hello_hologram.py
"""

import numpy as np

from holo import FHRR, HoloMap, RecordSpace

space = FHRR(dim=4096, seed=0)

# -- bind / unbind: an exact inverse ---------------------------------
country = space.label_vector("mexico")      # hash-derived: any process
currency = space.label_vector("peso")       # with (dim, seed) gets the
pair = space.bind(country, currency)        # same vector — no registry
back = space.unbind(pair, country)
print("unbind recovers the bound partner:",
      f"sim(back, peso) = {space.sim(back, currency):.3f}")

# -- a hash map that is one vector -----------------------------------
m = HoloMap(space)
for k, v in [("name", "ada"), ("lang", "python"), ("year", "1843")]:
    m.put(k, v)
print("HoloMap.get('lang') ->", m.get("lang")[0])

# -- records and the analogy query -----------------------------------
# Kanerva's "what is the dollar of mexico?": build two role-filler
# records, then transform one whole record by the other.
rs = RecordSpace(space)
usa = rs.encode({"country": "usa", "capital": "washington",
                 "currency": "dollar"})
mex = rs.encode({"country": "mexico", "capital": "cdmx",
                 "currency": "peso"})
label, score = rs.analogy(usa, mex, "dollar")
print(f"the dollar of mexico is: {label}  (score {score:.2f})")

# -- the one law, watched failing SOFT -------------------------------
# Load a single map far past its budget: accuracy degrades smoothly
# with sqrt(N/2d) — there is no allocation cliff, only rising noise.
print("\n     N   sqrt(N/2d)   get() accuracy")
for n in [32, 256, 1024, 4096]:
    big = HoloMap(space)
    keys = [f"k{i}" for i in range(n)]
    for i, k in enumerate(keys):
        big.put(k, f"v{i % 500}")
    probe = np.random.default_rng(1).choice(n, 200)
    acc = np.mean([big.get(keys[i])[0] == f"v{i % 500}" for i in probe])
    print(f"  {n:>4}        {np.sqrt(n / (2 * space.dim)):.2f}"
          f"           {acc:.2f}")

print("\nEvery structure in holo/ is this pattern with one twist each —"
      "\nsee docs/README.md and `hdc-demos` for the capacity tables.")
