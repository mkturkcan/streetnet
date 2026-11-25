"""Serve the viewer and a local tile archive with HTTP range requests (PMTiles reads byte ranges).

    python tools/viewer/serve.py --pmtiles streetnet-nyc.pmtiles      then open http://localhost:8000/viewer/

Serves docs/ at /, and the given .pmtiles at /data/streetnet-nyc.pmtiles; the viewer is opened with
?data=/data/streetnet-nyc.pmtiles automatically (a redirect from /). Standard library only.
"""
import argparse
import http.server
import os
import re
from functools import partial
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"


class Handler(http.server.SimpleHTTPRequestHandler):
    pmtiles = None

    def translate_path(self, path):
        if path.split("?")[0] == "/data/streetnet-nyc.pmtiles":
            return str(self.pmtiles)
        return super().translate_path(path)

    def do_GET(self):
        if self.path in ("/", "/viewer", "/viewer/"):
            self.send_response(302)
            self.send_header("Location", "/viewer/?data=/data/streetnet-nyc.pmtiles")
            self.end_headers()
            return
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().do_GET()
        m = re.match(r"bytes=(\d+)-(\d*)", rng)
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(end - start + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmtiles", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    Handler.pmtiles = Path(a.pmtiles).resolve()
    srv = http.server.ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=str(DOCS)))
    print(f"[viewer] http://{a.host}:{a.port}/viewer/  (tiles: {Handler.pmtiles})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
