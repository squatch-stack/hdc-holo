"""Compatibility shim: the implementation lives in holo/attribute_field.py."""

from ._shim import delegate

__getattr__, __dir__ = delegate("holo.attribute_field")
