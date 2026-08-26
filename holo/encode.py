"""Continuous-domain encoders: fractional power encoding and its
spatial organizations.

Point -> phasor codeword e^{i W p}; frequency rows drawn from
N(0, Sigma^-1) make inner products equal Gaussian kernels (Bochner).
Bands quantize per-splat covariance; cells localize crosstalk and give
replication its sync unit. The spectral encoder (holo/spectral.py)
drops the shared-covariance restriction entirely: a splat's hypervector
is its Fourier spectrum sampled at a shared (optionally
mixture-of-Gaussians) codebook, so every splat keeps its own
anisotropic Sigma inside one bundle.
"""

from .field import GaussianSplatField  # noqa: F401
from .spatial import ChunkedSplatField, MultiBandSplatField  # noqa: F401
from .spectral import (SplatScene, decode_field,  # noqa: F401
                       decode_field_phasor, decode_weights, phasor_bundle,
                       random_scene, sample_frequencies, spectral_bundle,
                       translate_bundle)

__all__ = ["GaussianSplatField", "MultiBandSplatField", "ChunkedSplatField",
           "SplatScene", "sample_frequencies", "decode_weights",
           "spectral_bundle", "phasor_bundle", "decode_field",
           "decode_field_phasor", "translate_bundle", "random_scene"]
