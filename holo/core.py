"""FHRR algebra: spaces, bind/unbind/bundle/permute, cleanup memories.

The determinism contract lives here: codewords are hash-derived from
(dim, seed, label) — see SDK.md, "the determinism contract".
"""

from .fhrr import FHRR, ItemMemory, Permutation

__all__ = ["FHRR", "ItemMemory", "Permutation"]
