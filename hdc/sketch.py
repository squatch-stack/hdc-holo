"""Compatibility shim: the implementation lives in holo/sketch.py."""

from ._shim import delegate

__getattr__, __dir__ = delegate("holo.sketch")
