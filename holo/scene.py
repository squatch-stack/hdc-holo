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

from .attribute_field import AttributeSplatField
from .capture import (
                      band_codebooks,
                      build_scene,
                      decode_slice,
                      encode_bands,
                      exact_slice,
                      exact_xray,
                      load_ply_sh,
                      load_scene_file,
                      load_splat,
                      load_spz,
                      render_mip,
                      render_xray,
                      save_ply,
                      save_spz,
)
from .color import ColorSplatField
from .sog import save_sog

__all__ = [
                      "AttributeSplatField",
                      "ColorSplatField",
                      "band_codebooks",
                      "build_scene",
                      "decode_slice",
                      "encode_bands",
                      "exact_slice",
                      "exact_xray",
                      "load_ply_sh",
                      "load_scene_file",
                      "load_splat",
                      "load_spz",
                      "render_mip",
                      "render_xray",
                      "save_ply",
                      "save_sog",
                      "save_spz",
]
