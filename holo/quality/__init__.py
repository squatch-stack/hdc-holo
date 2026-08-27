"""quality: lint ratchet, structure rules, and the code graph.

Repo-maintenance tooling that ships with the package so CI, the
pre-commit hook, and contributors all run the identical checks. Its
dependencies (ruff, kuzu) live in the `quality` extra — the core SDK
stays stdlib+numpy.
"""

from .ratchet import collect, compare, load_baseline, save_baseline
from .structure import check_structure

__all__ = ["check_structure", "collect", "compare", "load_baseline",
           "save_baseline"]
