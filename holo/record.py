"""Role-filler records: a struct/DB row as one hypervector.

    R = bind(role_1, filler_1) + bind(role_2, filler_2) + ...

Field access unbinds the role. Because binding distributes over the
bundle, records also support *analogical* queries with pure algebra
(Kanerva's "What is the Dollar of Mexico?"):

    role_hat = unbind(R_usa, dollar)      ~ 'currency' + noise
    answer   = unbind(R_mex, role_hat)    ~ 'peso' + more noise
    cleanup(answer) -> peso

No schema lookup, no join — the query vector is *constructed*, not learned.
"""

from .fhrr import FHRR, ItemMemory


class RecordSpace:
    """Shared role/filler codebooks + encode/decode for records."""

    def __init__(self, space):
        self.space = space
        self.roles = ItemMemory(space, "roles")
        self.fillers = ItemMemory(space, "fillers")

    def encode(self, fields):
        r = self.space.zeros()
        for role, filler in fields.items():
            r += FHRR.bind(self.roles.get(role), self.fillers.get(filler))
        return r

    def get(self, record_hv, role):
        return self.fillers.cleanup(FHRR.unbind(record_hv, self.roles.get(role)))

    def analogy(self, record_a, record_b, filler_in_a):
        """What plays in B the role that filler_in_a plays in A?"""
        role_hat = FHRR.unbind(record_a, self.fillers.get(filler_in_a))
        return self.fillers.cleanup(FHRR.unbind(record_b, role_hat))


def demo(dim=4096, seed=0):
    print(f"== Records: role-filler bindings (d={dim}) ==")
    space = FHRR(dim, seed=seed)
    rs = RecordSpace(space)
    usa = rs.encode({"name": "USA", "capital": "Washington",
                     "currency": "dollar"})
    mex = rs.encode({"name": "Mexico", "capital": "CDMX",
                     "currency": "peso"})
    for role in ["name", "capital", "currency"]:
        print(f"  usa.{role:<9}-> {rs.get(usa, role)}")
    label, score = rs.analogy(usa, mex, "dollar")
    print(f'  "the dollar of Mexico" -> ({label!r}, {score:.3f})')
    label, score = rs.analogy(mex, usa, "CDMX")
    print(f'  "the CDMX of the USA"  -> ({label!r}, {score:.3f})')

    # capacity: fields per record before access degrades
    print(f"  {'fields':>7} {'access accuracy':>16}")
    import numpy as np
    for n_fields in [5, 20, 80, 320, 1280]:
        space = FHRR(dim, seed=seed)
        rs = RecordSpace(space)
        fields = {f"role{i}": f"filler{i % 64}" for i in range(n_fields)}
        hv = rs.encode(fields)
        ok = sum(rs.get(hv, r)[0] == f for r, f in fields.items())
        print(f"  {n_fields:>7} {ok/n_fields:>16.1%}")
    print()
