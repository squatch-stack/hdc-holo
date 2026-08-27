"""Compute backend dispatch: NumPy everywhere, MLX/Metal where present
(install extra: ``holo[gpu]``; force with HDC_BACKEND=mlx|numpy).

Public APIs take and return NumPy arrays; phasors are carried as
cos/sin planes on the device so no backend needs complex64. Results
match NumPy to float32 rounding.
"""

from .accel import (
                    active,
                    backend_name,
                    cell_decode,
                    decode,
                    readout,
                    spectral_bundle,
)

__all__ = [
                    "active",
                    "backend_name",
                    "cell_decode",
                    "decode",
                    "readout",
                    "spectral_bundle",
]
