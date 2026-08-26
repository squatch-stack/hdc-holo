"""Splat scenes with payloads: symbolic attributes, RGB color, and
real captures.

Attribute scenes answer what_is_at(p) / where_is(label) by unbinding;
color rides as amplitude channels on one frequency basis. The
real-capture pipeline (holo/capture.py) loads pretrained scenes
(.splat / antimatter15, .spz v2 / Niantic — byte-verified parsers),
crops to the mass center, clamps scales, and encodes through
scale-banded cells with mixture codebooks; slices and X-ray views
decode straight from the cell bundles.
"""

from .attribute_field import AttributeSplatField  # noqa: F401
from .color import ColorSplatField  # noqa: F401
from .capture import (band_codebooks, build_scene,  # noqa: F401
                      decode_slice, encode_bands, exact_slice, exact_xray,
                      load_scene_file, load_splat, load_spz, render_mip,
                      render_xray)

__all__ = ["AttributeSplatField", "ColorSplatField",
           "load_splat", "load_spz", "load_scene_file", "build_scene",
           "render_mip", "band_codebooks", "encode_bands",
           "decode_slice", "exact_slice", "render_xray", "exact_xray"]
