"""Tagged storage codecs: phase-only, magnitude-preserving, companded.

Three codecs, one reader (`unpack` dispatches on the blob's magic):
`pack` (HP, phase-only — symbols), `pack_complex` (HM, linear re/im),
`pack_polar` (HG, gamma-companded polar — wide-dynamic-range bundles).
Choose per task: see docs/storage.md for the measured rules. Quantized
codes are a wire/storage format — always tagged (SDK.md, format
versioning).
"""

from .phase import (STORAGE_VERSION, dequantize, from_phases,  # noqa: F401
                    pack, pack_complex, pack_polar, quantize, to_phases,
                    unpack)

__all__ = ["to_phases", "from_phases", "quantize", "dequantize",
           "pack", "pack_complex", "pack_polar", "unpack",
           "STORAGE_VERSION"]
