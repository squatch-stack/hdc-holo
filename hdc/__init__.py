"""Compatibility shim: the implementation migrated to the ``holo``
package (the SDK surface — see SDK.md). Import from ``holo``; this
package re-exports the same objects so existing callers keep working.
Edit holo/*.py, never these shims."""

from holo import __all__ as _holo_all
from holo import __version__  # noqa: F401

from ._shim import delegate

__all__ = list(_holo_all)
__getattr__, __dir__ = delegate("holo")
