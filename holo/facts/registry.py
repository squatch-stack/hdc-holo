"""The claims registry: every measured number in the prose, as data.

`claims/registry.jsonl` holds one JSON object per line (# comments and
blank lines allowed). A claim is a typed value with provenance and a
supersession chain; superseded entries are never deleted — they ARE the
historical allowlist that lets "Historical note" prose keep its old
numbers legally. This mirrors the repo's own conventions: date-scoped
supersession, explicit retraction entries, and the proven bar of
(a) quantitative comparison, (b) deterministic test, (c) documented
failure mode.

Statuses: current | superseded | retracted. Superseded generations get
`base.id@<version>` ids. `kind` gives value semantics: count (integer,
exact), floor ("N+" citations pass if N <= current), measurement
(tolerance or accepted spellings), identifier (exact string, accepted
spellings), text (pattern presence is the claim).
"""

import json
from dataclasses import dataclass, field

__all__ = ["Claim", "load_registry", "validate", "base_id"]

_STATUSES = {"current", "superseded", "retracted"}
_KINDS = {"count", "floor", "measurement", "identifier", "text"}


@dataclass
class Claim:
    id: str
    statement: str = ""
    value: object = None
    units: str = ""
    kind: str = "text"
    status: str = "current"
    supersedes: str = None
    superseded_by: str = None
    as_of: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    patterns: list = field(default_factory=list)
    accepted: list = field(default_factory=list)
    context_any: list = field(default_factory=list)
    cites: list = field(default_factory=list)
    allow_historical_in: list = field(default_factory=list)
    check: dict = field(default_factory=dict)
    tolerance: object = None
    lane: str = ""
    notes: str = ""

    def accepted_values(self):
        vals = [self.value] if self.value is not None else []
        return vals + list(self.accepted)


def base_id(claim_id):
    return claim_id.split("@", 1)[0]


def load_registry(path):
    claims = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError("%s:%d: bad JSON: %s" % (path, lineno, e))
            known = {k: v for k, v in obj.items()
                     if k in Claim.__dataclass_fields__}
            claims.append(Claim(**known))
    return claims


def validate(claims):
    """Structural errors in the registry itself (empty list = valid)."""
    errors = []
    ids = {}
    for c in claims:
        if c.id in ids:
            errors.append("duplicate id: %s" % c.id)
        ids[c.id] = c
        if c.status not in _STATUSES:
            errors.append("%s: bad status %r" % (c.id, c.status))
        if c.kind not in _KINDS:
            errors.append("%s: bad kind %r" % (c.id, c.kind))
        if c.status != "current" and "@" not in c.id and \
                c.status != "retracted":
            errors.append("%s: superseded claims need @version ids" % c.id)
    for c in claims:
        for link, name in ((c.supersedes, "supersedes"),
                           (c.superseded_by, "superseded_by")):
            if link and link not in ids:
                errors.append("%s: %s -> unknown id %s" % (c.id, name, link))
        if c.status == "superseded" and c.superseded_by is None:
            # the current generation carries the same base id
            if base_id(c.id) not in ids:
                errors.append("%s: no current generation %s"
                              % (c.id, base_id(c.id)))
    return errors
