"""Serve this folder for local testing.

A plain ``python -m http.server`` maps some systems' ``.js`` to ``text/plain``,
so the ES module loads fail and the page comes up blank. This server pins the
MIME types itself.

It also sends the COOP/COEP pair, so a local page is cross-origin isolated and
gets multi-threaded WASM without the ``coi-serviceworker.js`` detour the
deployed page needs (``--no-coi`` turns that off, to reproduce a plain host).
Threaded, because one blocking request otherwise stalls the whole page.

    python demo/serve.py [--port 8000] [--no-coi]
"""
from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path

TYPES = {
    ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
    ".wasm": "application/wasm", ".bin": "application/octet-stream",
    ".onnx": "application/octet-stream", ".html": "text/html; charset=utf-8",
    ".css": "text/css",
}

ISOLATION = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "cross-origin",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    isolate = True

    def guess_type(self, path):
        return TYPES.get(Path(path).suffix.lower(), super().guess_type(path))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        if self.isolate:
            for k, v in ISOLATION.items():
                self.send_header(k, v)
        super().end_headers()

    def log_message(self, fmt, *a):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-coi", action="store_true",
                    help="drop the COOP/COEP headers, like a plain static host")
    args = ap.parse_args()
    root = str(Path(__file__).resolve().parent)
    handler = functools.partial(Handler, directory=root)
    Handler.isolate = not args.no_coi
    # Deliberately NOT SO_REUSEADDR: on Windows it lets this server share a port
    # that another process already owns, and then half the requests are answered
    # by that other process. Fail loudly on a busy port instead.
    http.server.ThreadingHTTPServer.allow_reuse_address = False
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        iso = "no COOP/COEP" if args.no_coi else "cross-origin isolated"
        print(f"serving {root} at http://127.0.0.1:{args.port}/  ({iso}, ctrl-c to stop)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
