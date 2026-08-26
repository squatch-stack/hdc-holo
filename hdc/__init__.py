"""Compatibility shim: the implementation migrated to the ``holo``
package (the SDK surface — see SDK.md). Import from ``holo``; this
package re-exports the same objects so existing callers keep working.
Edit holo/*.py, never these shims."""

from holo import *              # noqa: F401,F403
from holo import __all__ as _holo_all, __version__  # noqa: F401

__all__ = list(_holo_all)
