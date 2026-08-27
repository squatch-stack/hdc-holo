#!/usr/bin/env python3
"""Serve any splat file in the real-time viewer (examples/viewer).

    run_viewer.py                       # data/iphone/redrock.ply
    run_viewer.py data/scan-tucson.spz
    run_viewer.py out/mine.ply --port 8200 --no-browser

Rendering is Spark (three.js) with occlusion-correct compositing —
the display complement to holo/render.py's X-ray evidence renderer.
Spark and three.js load from their CDNs in the browser; nothing is
installed or executed locally. Files this pipeline writes
(`holo.capture.save_ply` / `save_spz`) view the same way.

The page is served from memory and the scene straight out of its own
directory (no temp copies): only that one directory is exposed.
"""

import argparse
import functools
import http.server
import os
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.join(HERE, "examples", "viewer", "index.html")


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serve the viewer page at /, everything else from the scene dir."""

    page = b""

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.page)))
            self.end_headers()
            self.wfile.write(self.page)
            return
        super().do_GET()

    def log_message(self, fmt, *args):        # keep the console quiet
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", nargs="?",
                    default=os.path.join("data", "iphone", "redrock.ply"))
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    scene = os.path.abspath(args.scene)
    if not os.path.exists(scene):
        raise SystemExit(f"no such scene file: {scene}")

    for ext in (".ply", ".spz", ".splat", ".ksplat", ".sog"):
        Handler.extensions_map[ext] = "application/octet-stream"
    with open(VIEWER, "rb") as f:
        Handler.page = f.read()

    handler = functools.partial(Handler, directory=os.path.dirname(scene))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    name = os.path.basename(scene)
    url = f"http://localhost:{args.port}/?src={name}"
    print(f"serving {name} ({os.path.getsize(scene) / 2**20:.0f} MB) "
          f"at {url}\nCtrl-C to stop")
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
