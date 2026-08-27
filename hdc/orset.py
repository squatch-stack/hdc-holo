"""Compatibility shim: the implementation lives in holo/orset.py."""

from ._shim import delegate

__getattr__, __dir__ = delegate("holo.orset")
