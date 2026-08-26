#!/usr/bin/env python3
"""Generate the hdc-holo brand assets from one geometry spec.

The mark extends the squatch.cc "resonance" system: a two-source
interference field (gold rings) with the project's canon silhouette at
the constructive node — here, the saguaro from the Tucson scan this
repo actually encodes. Palette bridges the family sites:

    deep forest  #274f42   (squatch.cc ground)
    gold rings   #C9A84C / #B8860B (squatch.cc resonance field)
    warm cream   #FAF6EF   (squatch.cc paper)
    desert rust  #C2562E   (squatch.cc accent)

Outputs: logo.svg, logo-512.png, logo-1024.png, social-preview.png.
Run: python assets/make_brand.py
"""

import os

import numpy as np

FOREST = "#274f42"
FOREST_DEEP = "#1c3a30"
GOLD = "#C9A84C"
GOLD_DIM = "#B8860B"
CREAM = "#FAF6EF"
SAND = "#c8bca7"
RUST = "#C2562E"

# geometry in a 256x256 badge frame
CENTER, R_BADGE = (128, 128), 118
SOURCES = [(96, 150), (166, 108)]          # ring sources near the arm tips
RING_STEP, RING_MAX = 16, 176
TRUNK = [(128, 206), (128, 76)]            # round-capped strokes
ARM_L = [(123, 156), (100, 156), (100, 116)]
ARM_R = [(133, 130), (157, 130), (157, 94)]
FLOOR = [(76, 206), (180, 206)]
W_TRUNK, W_ARM, W_FLOOR = 23, 15, 4


def svg_mark():
    rings = []
    for i, (sx, sy) in enumerate(SOURCES):
        color = GOLD if i == 0 else GOLD_DIM
        for r in range(RING_STEP, RING_MAX, RING_STEP):
            rings.append(
                f'<circle cx="{sx}" cy="{sy}" r="{r}" fill="none" '
                f'stroke="{color}" stroke-width="1.2" opacity="0.28"/>')

    def stroke(pts, width, color, opacity=1.0):
        d = "M " + " L ".join(f"{x},{y}" for x, y in pts)
        return (f'<path d="{d}" fill="none" stroke="{color}" '
                f'stroke-width="{width}" stroke-linecap="round" '
                f'stroke-linejoin="round" opacity="{opacity}"/>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256" role="img" aria-label="hdc-holo mark">
<title>holo — a saguaro at the constructive node of a two-source interference field. Same resonance, desert canon.</title>
<defs><clipPath id="badge"><circle cx="{CENTER[0]}" cy="{CENTER[1]}" r="{R_BADGE}"/></clipPath></defs>
<circle cx="{CENTER[0]}" cy="{CENTER[1]}" r="{R_BADGE}" fill="{FOREST}"/>
<g clip-path="url(#badge)">
{chr(10).join(rings)}
{stroke(FLOOR, W_FLOOR, RUST, 0.85)}
{stroke(TRUNK, W_TRUNK, CREAM)}
{stroke(ARM_L, W_ARM, CREAM)}
{stroke(ARM_R, W_ARM, CREAM)}
</g>
<circle cx="{CENTER[0]}" cy="{CENTER[1]}" r="{R_BADGE}" fill="none" stroke="{GOLD}" stroke-width="2" opacity="0.55"/>
</svg>
"""


def draw_mark(ax, cx=0.0, cy=0.0, scale=1.0):
    """The same geometry on a matplotlib axes (unit = badge pixels)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    def T(x, y):
        return cx + (x - 128) * scale, cy - (y - 128) * scale

    badge = Circle((cx, cy), R_BADGE * scale, facecolor=FOREST,
                   edgecolor="none", zorder=1)
    ax.add_patch(badge)
    clip = Circle((cx, cy), R_BADGE * scale, transform=ax.transData)
    for i, (sx, sy) in enumerate(SOURCES):
        color = GOLD if i == 0 else GOLD_DIM
        for r in range(RING_STEP, RING_MAX, RING_STEP):
            ring = Circle(T(sx, sy), r * scale, facecolor="none",
                          edgecolor=color, linewidth=1.2 * scale,
                          alpha=0.28, zorder=2)
            ring.set_clip_path(clip)
            ax.add_patch(ring)
    for pts, w, color, alpha in [(FLOOR, W_FLOOR, RUST, 0.85),
                                 (TRUNK, W_TRUNK, CREAM, 1.0),
                                 (ARM_L, W_ARM, CREAM, 1.0),
                                 (ARM_R, W_ARM, CREAM, 1.0)]:
        xs, ys = zip(*(T(x, y) for x, y in pts))
        line = Line2D(xs, ys, linewidth=w * scale, color=color,
                      alpha=alpha, solid_capstyle="round",
                      solid_joinstyle="round", zorder=3)
        line.set_clip_path(clip)
        ax.add_line(line)
    rim = Circle((cx, cy), R_BADGE * scale, facecolor="none",
                 edgecolor=GOLD, linewidth=2 * scale, alpha=0.55, zorder=4)
    ax.add_patch(rim)


def render_logo_png(path, px):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-128, 128)
    ax.set_ylim(-128, 128)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_mark(ax)
    fig.savefig(path, dpi=px / (px / 100) / 100 * 100, transparent=True)
    plt.close(fig)


def render_social(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 640)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1280, 640, facecolor=FOREST_DEEP))
    # faint full-canvas resonance echo
    for r in range(60, 1500, 60):
        ax.add_patch(plt.Circle((330, 320), r, facecolor="none",
                                edgecolor=GOLD, linewidth=1.0, alpha=0.06))
    ax.set_aspect("equal")

    class _Shim:
        transData = ax.transData

        @staticmethod
        def add_patch(p):
            return ax.add_patch(p)

        @staticmethod
        def add_line(ln):
            return ax.add_line(ln)

    draw_mark(_Shim, cx=300, cy=320, scale=1.8)
    font = {"fontfamily": "Helvetica Neue", "color": CREAM}
    ax.text(590, 420, "holo", fontsize=80, fontweight="bold", **font)
    ax.text(592, 358, "holographic computing on FHRR hypervectors",
            fontsize=22, color=GOLD, fontfamily="Helvetica Neue")
    ax.text(592, 278,
            "data structures  ·  splat scenes  ·  learning  ·  rendering\n"
            "CRDT sync — superposed in one complex vector",
            fontsize=19, color=SAND, fontfamily="Helvetica Neue",
            linespacing=1.7, va="top")
    ax.text(592, 152, "squatch-stack/hdc-holo",
            fontsize=18, color=CREAM, alpha=0.9,
            fontfamily="Menlo")
    ax.text(592, 96, "FSL-1.1, converting to Apache-2.0   ·   squatch.cc"
            "   ·   squatch.cc", fontsize=15, color=GOLD_DIM,
            fontfamily="Helvetica Neue")
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "logo.svg"), "w") as f:
        f.write(svg_mark())
    render_logo_png(os.path.join(here, "logo-512.png"), 512)
    render_logo_png(os.path.join(here, "logo-1024.png"), 1024)
    render_social(os.path.join(here, "social-preview.png"))
    print("wrote logo.svg, logo-512.png, logo-1024.png, social-preview.png")
