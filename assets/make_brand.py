#!/usr/bin/env python3
"""Generate the hdc-holo brand assets from one geometry spec.

The mark extends the squatch.cc "resonance" system: a two-source
interference field (gold rings) with the project's canon silhouette at
the constructive node — here, the saguaro from the Tucson scan this
repo actually encodes. Palette bridges the family sites:

    deep forest  #274f42   (squatch.cc ground)
    gold rings   #C9A84C / #B8860B (squatch.cc resonance field)
    warm cream   #FAF6EF   (desert paper)
    desert rust  #C2562E   (desert accent)

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
    ax.text(592, 96, "FSL-1.1, converting to Apache-2.0   ·   "
            "squatch.cc", fontsize=15, color=GOLD_DIM,
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


# -- the duet mark: studio + project in ONE field ---------------------------
#
# The combined mark makes squatch.cc's metaphor literal: one
# interference field, two sources, and each source IS a canon
# silhouette — the sasquatch (studio, verbatim path from squatch.cc)
# on the left, the saguaro (holo) on the right, their ring systems
# interfering in the middle. Two signals find the shift that lets
# them resonate: the studio and the project.

SASQ_W, SASQ_H = 203.2, 279.4


def _sasquatch_d():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "sasquatch-path.txt")) as f:
        return f.read().strip()


def _sasquatch_polys(n_per_seg=8):
    """Studio path -> polygon subpaths in its native 203x279 coords."""
    from svgelements import Close, Line, Move
    from svgelements import Path as SvgPath
    polys, cur = [], []
    for seg in SvgPath(_sasquatch_d()):
        if isinstance(seg, Move):
            if len(cur) > 2:
                polys.append(cur)
            cur = [(seg.end.x, seg.end.y)]
        elif isinstance(seg, (Line, Close)):
            cur.append((seg.end.x, seg.end.y))
        else:
            for t in np.linspace(0, 1, n_per_seg + 1)[1:]:
                p = seg.point(t)
                cur.append((p.x, p.y))
    if len(cur) > 2:
        polys.append(cur)
    return polys


def _saguaro_strokes(tx, ty, s):
    """The saguaro strokes transformed: x' = tx + x*s, y' = ty + y*s."""
    out = []
    for pts, w in [(TRUNK, W_TRUNK), (ARM_L, W_ARM), (ARM_R, W_ARM)]:
        out.append(([(tx + x * s, ty + y * s) for x, y in pts], w * s))
    return out


# duet layout (256 badge, SVG coordinates, y down)
D_SASQ = dict(s=0.60, tx=17.0, ty=42.4)     # base lands on the floor
D_SAG = dict(s=0.80, tx=79.6, ty=45.2)
D_FLOOR = [(34, 210), (222, 210)]
D_SOURCES = [(80, 122), (182, 128)]          # sasquatch heart, saguaro crown

# banner layout (1280x400)
B_SASQ = dict(s=0.93, tx=236.0, ty=70.2)
B_SAG = dict(s=1.30, tx=783.6, ty=62.2)
B_FLOOR = [(150, 330), (1130, 330)]
B_SOURCES = [(330, 200), (950, 205)]


def _svg_rings(sources, step, rmax, width=1.2, op=0.26):
    parts = []
    for i, (sx, sy) in enumerate(sources):
        color = GOLD if i == 0 else GOLD_DIM
        for r in range(step, rmax, step):
            parts.append(f'<circle cx="{sx}" cy="{sy}" r="{r}" fill="none" '
                         f'stroke="{color}" stroke-width="{width}" '
                         f'opacity="{op}"/>')
    return parts


def _svg_saguaro(layout):
    parts = []
    for pts, w in _saguaro_strokes(layout["tx"], layout["ty"], layout["s"]):
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<path d="{d}" fill="none" stroke="{CREAM}" '
                     f'stroke-width="{w:.1f}" stroke-linecap="round" '
                     f'stroke-linejoin="round"/>')
    return parts


def _svg_sasquatch(layout):
    return (f'<g transform="translate({layout["tx"]},{layout["ty"]}) '
            f'scale({layout["s"]})"><path fill="{CREAM}" '
            f'd="{_sasquatch_d()}"/></g>')


def _svg_floor(pts, width):
    (x1, y1), (x2, y2) = pts
    return (f'<path d="M {x1},{y1} L {x2},{y2}" stroke="{RUST}" '
            f'stroke-width="{width}" stroke-linecap="round" '
            f'opacity="0.85" fill="none"/>')


def svg_duet():
    body = "\n".join(_svg_rings(D_SOURCES, 15, 200)
                     + [_svg_floor(D_FLOOR, 4),
                        _svg_sasquatch(D_SASQ)]
                     + _svg_saguaro(D_SAG))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256" role="img" aria-label="Squatch Stack duet mark">
<title>Squatch Stack — sasquatch and saguaro as the two sources of one interference field. Two signals find the shift that lets them resonate.</title>
<defs><clipPath id="duet"><circle cx="128" cy="128" r="118"/></clipPath></defs>
<circle cx="128" cy="128" r="118" fill="{FOREST}"/>
<g clip-path="url(#duet)">
{body}
</g>
<circle cx="128" cy="128" r="118" fill="none" stroke="{GOLD}" stroke-width="2" opacity="0.55"/>
</svg>
"""


def svg_banner():
    body = "\n".join(_svg_rings(B_SOURCES, 34, 1100, width=1.4, op=0.22)
                     + [_svg_floor(B_FLOOR, 6),
                        _svg_sasquatch(B_SASQ)]
                     + _svg_saguaro(B_SAG))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 400" width="1280" height="400" role="img" aria-label="Squatch Stack banner">
<title>Squatch Stack — the studio and the project resonate in one field.</title>
<defs><clipPath id="ban"><rect x="0" y="0" width="1280" height="400" rx="20"/></clipPath></defs>
<rect x="0" y="0" width="1280" height="400" rx="20" fill="{FOREST_DEEP}"/>
<g clip-path="url(#ban)">
{body}
</g>
</svg>
"""


def render_duet_png(path, px):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, PathPatch
    from matplotlib.path import Path as MplPath

    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)          # SVG orientation: y down
    ax.set_aspect("equal")
    ax.axis("off")
    badge = Circle((128, 128), 118, facecolor=FOREST, edgecolor="none",
                   zorder=1)
    ax.add_patch(badge)
    clip = Circle((128, 128), 118, transform=ax.transData)
    for i, (sx, sy) in enumerate(D_SOURCES):
        color = GOLD if i == 0 else GOLD_DIM
        for r in range(15, 200, 15):
            ring = Circle((sx, sy), r, facecolor="none", edgecolor=color,
                          linewidth=1.2 * px / 256, alpha=0.26, zorder=2)
            ring.set_clip_path(clip)
            ax.add_patch(ring)
    (fx1, fy1), (fx2, fy2) = D_FLOOR
    floor = Line2D([fx1, fx2], [fy1, fy2], linewidth=4 * px / 256,
                   color=RUST, alpha=0.85, solid_capstyle="round", zorder=3)
    floor.set_clip_path(clip)
    ax.add_line(floor)
    verts, codes = [], []
    s, tx, ty = D_SASQ["s"], D_SASQ["tx"], D_SASQ["ty"]
    for poly in _sasquatch_polys():
        pts = [(tx + x * s, ty + y * s) for x, y in poly]
        verts += pts + [pts[0]]
        codes += ([MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1)
                  + [MplPath.CLOSEPOLY])
    sasq = PathPatch(MplPath(verts, codes), facecolor=CREAM,
                     edgecolor="none", zorder=4)
    sasq.set_clip_path(clip)
    ax.add_patch(sasq)
    for pts, w in _saguaro_strokes(D_SAG["tx"], D_SAG["ty"], D_SAG["s"]):
        xs, ys = zip(*pts)
        line = Line2D(xs, ys, linewidth=w * px / 256, color=CREAM,
                      solid_capstyle="round", solid_joinstyle="round",
                      zorder=5)
        line.set_clip_path(clip)
        ax.add_line(line)
    rim = Circle((128, 128), 118, facecolor="none", edgecolor=GOLD,
                 linewidth=2 * px / 256, alpha=0.55, zorder=6)
    ax.add_patch(rim)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def make_duet():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "duet.svg"), "w") as f:
        f.write(svg_duet())
    with open(os.path.join(here, "banner.svg"), "w") as f:
        f.write(svg_banner())
    render_duet_png(os.path.join(here, "duet-512.png"), 512)
    render_duet_png(os.path.join(here, "duet-1024.png"), 1024)
    print("wrote duet.svg, banner.svg, duet-512.png, duet-1024.png")


if __name__ == "__main__":
    make_duet()


def render_profile_social(path):
    """1280x640 social card for the squatch-stack PROFILE repo: the duet
    flanking the studio wordmark. (hdc-holo's own card is
    social-preview.png; this one is the account's.)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(640, 0)                      # SVG orientation, y down
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1280, 640, facecolor=FOREST_DEEP))
    sasq = dict(s=1.15, cx=210)
    sag = dict(s=1.70, cx=1070)
    floor_y = 520
    sources = [(sasq["cx"], 380), (sag["cx"], 350)]
    for i, (sx, sy) in enumerate(sources):
        color = GOLD if i == 0 else GOLD_DIM
        for r in range(40, 1400, 40):
            ax.add_patch(plt.Circle((sx, sy), r, facecolor="none",
                                    edgecolor=color, linewidth=1.3,
                                    alpha=0.14))
    ax.add_line(Line2D([120, 1160], [floor_y, floor_y], linewidth=6,
                       color=RUST, alpha=0.8, solid_capstyle="round"))
    s, tx = sasq["s"], sasq["cx"] - SASQ_W * sasq["s"] / 2
    ty = floor_y - SASQ_H * s
    verts, codes = [], []
    for poly in _sasquatch_polys():
        pts = [(tx + x * s, ty + y * s) for x, y in poly]
        verts += pts + [pts[0]]
        codes += ([MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1)
                  + [MplPath.CLOSEPOLY])
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=CREAM,
                           edgecolor="none"))
    sg_s = sag["s"]
    sg_tx, sg_ty = sag["cx"] - 128 * sg_s, floor_y - 206 * sg_s
    for pts, w in _saguaro_strokes(sg_tx, sg_ty, sg_s):
        xs, ys = zip(*pts)
        ax.add_line(Line2D(xs, ys, linewidth=w, color=CREAM,
                           solid_capstyle="round",
                           solid_joinstyle="round"))
    ax.text(640, 305, "Squatch Stack", fontsize=62, fontweight="bold",
            color=CREAM, fontfamily="Helvetica Neue", ha="center")
    ax.text(640, 375, "Software that finds harmony, shipped.",
            fontsize=25, color=GOLD, fontfamily="Helvetica Neue",
            ha="center")
    ax.text(640, 432, "squatch.cc   ·   github.com/squatch-stack",
            fontsize=16, color=SAND,
            fontfamily="Helvetica Neue", ha="center")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    render_profile_social(os.path.join(here, "profile-social.png"))
    print("wrote profile-social.png")
