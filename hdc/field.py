"""Compatibility shim: the implementation lives in holo/field.py."""

from ._shim import delegate

__getattr__, __dir__ = delegate("holo.field")
