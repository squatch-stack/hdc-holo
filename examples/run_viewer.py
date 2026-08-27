#!/usr/bin/env python3
"""Serve splat files in the real-time viewer (examples/viewer).

    examples/run_viewer.py                       # data/iphone/redrock.ply
    examples/run_viewer.py data/scan-tucson.spz
    examples/run_viewer.py out/mine.ply --port 8200 --no-browser

    # two codecs, one camera, side by side — the rate-distortion look
    examples/run_viewer.py data/export/redrock.spz \
        --compare data/export/redrock.sog

Rendering is Spark (three.js) with occlusion-correct compositing —
the display complement to holo/render.py's X-ray evidence renderer.
Spark and three.js load from their CDNs in the browser; nothing is
installed or executed locally. Files this pipeline writes
(`holo.capture.save_ply` / `save_spz` / `holo.sog.save_sog`) all view
the same way.

Pages are served from memory and only the named scene files are
routed — no directory is exposed.
"""

import argparse
import http.server
import os
import shutil
import threading
import typing
import urllib.parse
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.join(HERE, "viewer", "index.html")
COMPARE = os.path.join(HERE, "viewer", "compare.html")


class Handler(http.server.BaseHTTPRequestHandler):
    """Two page routes and an explicit file allowlist."""

    page = b""
    files: typing.ClassVar[dict] = {}   # url path -> absolute file path

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.page)))
            self.end_headers()
            self.wfile.write(self.page)
            return
        target = self.files.get(path)
        if not target:
            self.send_error(404, "not served")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(os.path.getsize(target)))
        self.end_headers()
        with open(target, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def log_message(self, fmt, *args):        # keep the console quiet
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


def _label(path):
    mb = os.path.getsize(path) / 2**20
    return f"{os.path.basename(path)} · {mb:.1f} MB"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", nargs="?",
                    default=os.path.join("data", "iphone", "redrock.ply"))
    ap.add_argument("--compare", metavar="SCENE",
                    help="second file: render both side by side under one "
                         "camera (codec A vs codec B on the same capture)")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    scenes = [args.scene] + ([args.compare] if args.compare else [])
    paths = []
    for s in scenes:
        p = os.path.abspath(s)
        if not os.path.exists(p):
            raise SystemExit(f"no such scene file: {p}")
        paths.append(p)

    # distinct route per file, so the two may live in different
    # directories (and may share a basename)
    slots = ["a", "b"][:len(paths)]
    Handler.files = {f"/{slot}/{os.path.basename(p)}": p
                     for slot, p in zip(slots, paths)}
    src = list(Handler.files)

    with open(COMPARE if args.compare else VIEWER, "rb") as f:
        Handler.page = f.read()
    q = {"src": src[0]} if not args.compare else {
        "a": src[0], "b": src[1],
        "la": _label(paths[0]), "lb": _label(paths[1])}
    url = f"http://localhost:{args.port}/?{urllib.parse.urlencode(q)}"

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("serving " + " vs ".join(_label(p) for p in paths)
          + f"\n  {url}\nCtrl-C to stop")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
