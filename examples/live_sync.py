#!/usr/bin/env python3
"""Two OS processes co-painting ONE holographic scene, live over TCP —
with observed-remove undo.

    python examples/live_sync.py                 # driver: spawns painters A and B
    python examples/live_sync.py --rounds 6      # shorter session
    python examples/live_sync.py --undo-round -1 # disable the undo event

Painter A (warm palette) listens; painter B (cool palette) connects.
Each round every painter lays one brush stroke — an ORStrokeScene epoch
— into its replica, then the two exchange Loro update deltas as
length-prefixed frames. Nothing but delta bytes crosses the socket:
codewords and the frequency basis W are hash/seed-derived, so both
processes agree on the algebra with no coordination, and Loro's version
vectors make redelivery harmless.

At --undo-round, BOTH painters concurrently undo the SAME stroke (the
earliest live one — a deterministic choice both can make locally before
exchanging). Undo is observed-remove by exclusion (hdc/orset.py): the
duplicate tombstones collapse to one set entry, the stroke vanishes
exactly once, and no negative phantom appears — the PN-counter anomaly,
exercised over a real wire and passing by construction.

The wire protocol is HoloReplica.version() / updates_since(): after each
exchange a painter snapshots its version vector; the next round it sends
only ops not covered by that snapshot. At the end both processes
independently render the merged scene and print digests of their CRDT
map bytes and of the render. The driver compares them: identical, or it
exits nonzero. Snapshots become out/live_sync.png.
"""

import argparse
import hashlib
import os
import socket
import struct
import subprocess
import sys

import numpy as np

PORT_PREFIX = "PORT "
DIGEST_PREFIX = "DIGEST "
OR_MAPS = ("bundles", "or-bundles", "or-index", "or-tombs", "or-folded")


def snap_rounds(args):
    if 0 <= args.undo_round < args.rounds:
        return sorted({0, max(args.undo_round - 1, 0), args.undo_round})
    return [0, min(4, args.rounds - 1)]


def send_frame(sock, payload):
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def recv_frame(sock):
    header = recv_exact(sock, 4)
    return recv_exact(sock, struct.unpack(">I", header)[0])


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def stroke(rng, x_range, hue_range, scene):
    """One brush stroke: 12 splats along a drifting arc, one hue; the
    stroke is sealed into its own epoch, so it is one undoable unit."""
    import colorsys
    p = np.array([rng.uniform(*x_range), rng.uniform(0.08, 0.92)])
    ang = rng.uniform(0, 2 * np.pi)
    curve = rng.uniform(-0.6, 0.6)
    rgb = colorsys.hsv_to_rgb(rng.uniform(*hue_range) % 1.0, 0.85, 1.0)
    for _ in range(12):
        scene.add_splat(np.clip(p, 0.03, 0.97), rgb,
                        alpha=float(rng.uniform(0.6, 1.0)))
        p = p + 0.035 * np.array([np.cos(ang), np.sin(ang)])
        ang += curve * 0.3
    return scene.end_stroke()


def render(scene, res):
    xs = np.linspace(0, 1, res, dtype=np.float32)
    P = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    return scene.eval_rgb(P).reshape(res, res, 3)


def save_png(img, path, scale):
    from PIL import Image
    shown = np.clip(img / scale, 0, 1)
    Image.fromarray((shown * 255).astype(np.uint8)[::-1]).save(path)


def state_digest(replica):
    """Hash every CRDT map's sorted (key, value) bytes — blobs, indexes,
    tombstones, folds. Bytes only: recomputed float sums are not
    bitwise-reproducible (see SDK.md)."""
    h = hashlib.sha256()
    for map_name in OR_MAPS:
        m = replica.doc.get_map(map_name)
        for k in sorted(m.keys()):
            v = m.get(k).value
            h.update(k.encode())
            h.update(v if isinstance(v, (bytes, bytearray))
                     else str(v).encode())
    return h.hexdigest()[:16]


def _connect(args):
    """A listens and announces its port; B dials it. Returns the socket."""
    if args.role == "A":
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        print(f"{PORT_PREFIX}{srv.getsockname()[1]}", flush=True)
        srv.settimeout(60)
        conn, _ = srv.accept()
    else:
        conn = socket.create_connection(("127.0.0.1", args.port),
                                        timeout=60)
    conn.settimeout(60)
    return conn


def _pick_undo(scene, r, args):
    """At the undo round, both peers tombstone the SAME stroke.

    They pick the earliest live one: their pre-exchange views differ
    only in each side's NEWEST stroke, which can never be the minimum,
    so the choice agrees without coordination — that is the point being
    demonstrated. NOTE that WHICH stroke that is varies run to run:
    stroke ids are ordered by Loro peer id, which is random per
    process, so the demo converges to one of two possible images. Both
    painters always agree (the invariant the test checks); the image is
    a coin flip.
    """
    if r != args.undo_round:
        return None
    live = scene.strokes()
    if not live:
        return None
    scene.undo_stroke(live[0])
    return live[0]


def _exchange(conn, replica, last, role):
    """One lockstep frame each way; returns the new version marker."""
    delta = replica.updates_since(last)
    if role == "A":                         # lockstep: A sends first
        send_frame(conn, delta)
        peer = recv_frame(conn)
    else:
        peer = recv_frame(conn)
        send_frame(conn, delta)
    replica.apply(peer)
    return delta, replica.version()


def worker(args):
    from holo import FHRR, HoloReplica, ORStrokeScene
    role = args.role
    space = FHRR(args.dim, seed=args.seed)     # both painters: same space
    replica = HoloReplica(space)
    scene = ORStrokeScene(replica, np.eye(2) * 0.03 ** 2)
    conn = _connect(args)

    rng = np.random.default_rng(0xA1CE if role == "A" else 0xB0B)
    x_range = (0.05, 0.55) if role == "A" else (0.45, 0.95)
    hue_range = (-0.06, 0.14) if role == "A" else (0.50, 0.72)  # warm/cool
    os.makedirs("out/live", exist_ok=True)
    snaps = snap_rounds(args)

    last = replica.version()
    for r in range(args.rounds):
        stroke(rng, x_range, hue_range, scene)
        undone = _pick_undo(scene, r, args)
        delta, last = _exchange(conn, replica, last, role)
        note = f" undo {undone}" if undone else ""
        print(f"ROUND {r} sent {len(delta)} bytes{note}", flush=True)
        if r in snaps and not args.no_montage:
            save_png(render(scene, args.res),
                     f"out/live/{role}_r{r}.png", scale=1.2)

    img = render(scene, args.res)
    if not args.no_montage:
        save_png(img, f"out/live/{role}_final.png", scale=1.2)
    render_hash = hashlib.sha256(
        np.round(img, 4).astype(np.float32).tobytes()).hexdigest()[:16]
    print(f"{DIGEST_PREFIX}state={state_digest(replica)} "
          f"render={render_hash}", flush=True)
    conn.close()


def _spawn_pair(args):
    """Start painter A, read the port it announces, then start B on it.
    Returns (proc_a, proc_b); exits if A never announces a port."""
    base = [sys.executable, os.path.abspath(__file__),
            "--rounds", str(args.rounds), "--dim", str(args.dim),
            "--res", str(args.res), "--seed", str(args.seed),
            "--undo-round", str(args.undo_round)]
    if args.no_montage:
        base.append("--no-montage")
    a = subprocess.Popen([*base, "--role", "A"], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    port = None
    for line in a.stdout:
        if line.startswith(PORT_PREFIX):
            port = int(line[len(PORT_PREFIX):])
            break
    if port is None:
        print(a.communicate()[0] or "painter A produced no port")
        sys.exit(1)
    b = subprocess.Popen([*base, "--role", "B", "--port", str(port)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    return a, b


def _parse_painter_output(out_a, out_b):
    """(digests, bytes per round from A, undo notices) — the painters
    report over stdout because they are separate OS processes."""
    digests, bytes_per_round, undos = {}, [], []
    for role, out in (("A", out_a), ("B", out_b)):
        for line in out.splitlines():
            if line.startswith(DIGEST_PREFIX):
                digests[role] = line[len(DIGEST_PREFIX):]
            elif line.startswith("ROUND"):
                if role == "A":
                    bytes_per_round.append(int(line.split()[3]))
                if " undo " in line:
                    undos.append((role, line.split(" undo ")[1]))
    return digests, bytes_per_round, undos


def _report(args, digests, bytes_per_round, undos, returncodes):
    """Print the convergence verdict; returns True when it converged."""
    print(f"two processes, {args.rounds} rounds, "
          f"delta frames from A: {bytes_per_round} bytes")
    if undos:
        same = len({u for _, u in undos}) == 1
        print(f"concurrent undo of {undos[0][1]} by "
              f"{', '.join(r for r, _ in undos)} "
              f"(same stroke chosen independently: {same})")
    print(f"painter A: {digests.get('A', 'MISSING')}")
    print(f"painter B: {digests.get('B', 'MISSING')}")
    ok = ("A" in digests and digests.get("A") == digests.get("B")
          and all(code == 0 for code in returncodes))
    print("CONVERGED: independent processes, identical holograms"
          if ok else "DIVERGED")
    return ok


def _montage(args):
    """Both painters' rounds side by side, ending in identical frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    snaps = snap_rounds(args)
    cols = [*(f"r{r}" for r in snaps), "final"]
    labels = {f"r{args.undo_round}": "after concurrent undo",
              "final": "final (identical)"}
    if 0 <= args.undo_round < args.rounds and args.undo_round > 0:
        labels[f"r{args.undo_round - 1}"] = "before undo"
    fig, axes = plt.subplots(2, len(cols), figsize=(3.4 * len(cols), 7.2))
    for row, role in enumerate("AB"):
        for col, tag in enumerate(cols):
            ax = axes[row, col]
            ax.imshow(plt.imread(f"out/live/{role}_{tag}.png"))
            ax.set_xticks([])
            ax.set_yticks([])
            label = labels.get(tag, "round " + tag[1:])
            ax.set_title(f"painter {role} — {label}", fontsize=10)
    fig.suptitle("Live Loro sync with observed-remove undo: both "
                 "painters undo the same stroke, it vanishes once",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("out/live_sync.png", dpi=110)
    print("saved out/live_sync.png")


def driver(args):
    a, b = _spawn_pair(args)
    out_a = a.stdout.read()
    out_b, _ = b.communicate(timeout=600)
    a.wait(timeout=60)

    digests, bytes_per_round, undos = _parse_painter_output(out_a, out_b)
    if not _report(args, digests, bytes_per_round, undos,
                   (a.returncode, b.returncode)):
        print(out_a, out_b)
        sys.exit(1)
    if not args.no_montage:
        _montage(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", choices=["A", "B"])
    ap.add_argument("--port", type=int)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--res", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--undo-round", type=int, default=6,
                    help="round at which BOTH painters undo the earliest "
                         "stroke (-1 disables)")
    ap.add_argument("--no-montage", action="store_true")
    args = ap.parse_args()
    worker(args) if args.role else driver(args)


if __name__ == "__main__":
    main()
