"""Compatibility shim: the implementation lives in holo/hashmap.py."""

from ._shim import delegate

__getattr__, __dir__ = delegate("holo.hashmap")
