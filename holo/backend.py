"""Compute backend dispatch: NumPy everywhere, MLX/Metal where present
(install extra: ``holo[gpu]``; force with HDC_BACKEND=mlx|numpy).

Public APIs take and return NumPy arrays; phasors are carried as
cos/sin planes on the device so no backend needs complex64. Results
match NumPy to float32 rounding.
"""

from . import accel as _accel

# F822: ruff cannot see that __getattr__ below supplies these — PEP 562
# module attributes are invisible to static analysis. The names ARE the
# surface; `from holo.backend import readout` works, and the facade test
# exercises each one.
__all__ = [  # noqa: F822
    "active",
    "backend_name",
    "cell_decode",
    "decode",
    "readout",
    "spectral_bundle",
]


def __getattr__(name):
    """Resolve against holo.accel on every access.

    Binding the kernels at import (`from .accel import readout`) made
    this facade hold whatever existed then — so an out-of-tree backend
    patching `holo.accel.readout` at runtime left every facade-routed
    call on the original NumPy path, silently, with results still
    correct and the GPU never engaged. Late binding removes the trap
    rather than documenting it.
    """
    if name in __all__:
        return getattr(_accel, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(__all__)
