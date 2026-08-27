"""facts: the claims registry and stale-claim checker.

Every measured number in this repo's prose is a registered claim in
`claims/registry.jsonl` — typed value, provenance, supersession chain,
citation sites. `holo-facts check` verifies the prose against the
registry (and the registry against code/tree ground truth), warning at
pre-commit and blocking in CI. See docs/facts.md.
"""

from .check import CheckResult, Finding, run  # noqa: F401
from .registry import Claim, load_registry, validate  # noqa: F401

__all__ = ["Claim", "CheckResult", "Finding", "load_registry", "run",
           "validate"]
