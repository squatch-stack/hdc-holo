"""One capture, every delivery format — sizes and what each costs.

    python examples/export_formats.py [scene.ply] [--out data/export]

Writes PLY (lossless interchange), SPZ v2 (small, DC-only), and SOG
(smallest, keeps the view-dependent SH through a palette), then prints
the size table. View any of the results with:

    python examples/run_viewer.py data/export/<name>.sog

See docs/real-scenes.md for the measured fidelity of each.
"""

import argparse
import os

from holo.capture import load_ply_sh, load_scene_file, save_ply, save_spz
from holo.sog import save_sog


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", nargs="?",
                    default=os.path.join("data", "iphone", "redrock.ply"))
    ap.add_argument("--out", default=os.path.join("data", "export"))
    ap.add_argument("--sh-clusters", type=int, default=1024,
                    help="SOG palette size (Spark 2.1.0 tops out at 1024)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.scene))[0]
    pos, scale, rgba, quat = load_scene_file(args.scene)
    sh = load_ply_sh(args.scene) if args.scene.endswith(".ply") else None
    src_mb = os.path.getsize(args.scene) / 2**20
    print(f"{len(pos):,} splats from {os.path.basename(args.scene)} "
          f"({src_mb:.0f} MB)"
          + (f", SH bands {sh.shape[2]}" if sh is not None else ", no SH"))

    outs = []
    for name, fn in [
            (f"{stem}.ply", lambda p: save_ply(p, pos, scale, rgba, quat)),
            (f"{stem}.spz", lambda p: save_spz(p, pos, scale, rgba, quat)),
            (f"{stem}.sog", lambda p: save_sog(p, pos, scale, rgba, quat,
                                               sh=sh,
                                               sh_clusters=args.sh_clusters))]:
        path = os.path.join(args.out, name)
        fn(path)
        outs.append((name, os.path.getsize(path) / 2**20))

    print(f"\n  {'file':<22}{'size':>9}{'vs source':>12}   carries")
    notes = {".ply": "everything except higher-order SH (lossless)",
             ".spz": "DC color only — SH dropped",
             ".sog": "DC + higher-order SH (palette)"}
    for name, mb in outs:
        ext = os.path.splitext(name)[1]
        print(f"  {name:<22}{mb:>7.1f} MB{src_mb / mb:>10.0f}x   {notes[ext]}")


if __name__ == "__main__":
    main()
