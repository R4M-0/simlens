"""Minimal, dependency-free HTTP serving of an Explainer over stdlib http.server.

    python -m simlens.serve --bundle path/to/bundle.simlens --port 8008

POST /v1/explain  { "query": [...], "candidates": [[...], ...], "level": "feature",
                    "top_k": 8 }
→ { "bundle_hash": "...", "results": [ <attribution dict>, ... ] }

This is the reference sidecar; a Rust `simlens-serve` binary is on the roadmap for the
hot path.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from .explain import Explainer


def make_handler(explainer: Explainer):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok", "bundle_hash": explainer._hash})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/explain":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                q = req["query"]
                level = req.get("level")
                top_k = int(req.get("top_k", 8))
                results = [
                    explainer.explain(q, c, level=level, top_k=top_k).to_dict()
                    for c in req.get("candidates", [])
                ]
                self._send(200, {"bundle_hash": explainer._hash, "results": results})
            except Exception as e:  # noqa: BLE001
                self._send(400, {"error": str(e)})

        def log_message(self, *_):  # silence
            pass

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="SimLens reference serving sidecar")
    ap.add_argument("--bundle", default=None, help="path to a .simlens bundle (optional)")
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--port", type=int, default=8008)
    args = ap.parse_args(argv)

    explainer = Explainer(bundle=args.bundle, metric=args.metric)
    server = HTTPServer(("0.0.0.0", args.port), make_handler(explainer))
    print(f"simlens serving on :{args.port}  (bundle_hash={explainer._hash})")
    server.serve_forever()


if __name__ == "__main__":
    main()
