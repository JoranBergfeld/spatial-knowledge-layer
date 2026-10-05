"""Stand-ins for what the build talks to."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs

from mini_campus import overpass_response


@dataclass
class Overpass:
    """A stand-in Overpass endpoint on localhost."""

    url: str = ""
    status: int = 200
    queries: list[str] = field(default_factory=list)


@contextmanager
def serve_overpass() -> Iterator[Overpass]:
    state = Overpass()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            form = self.rfile.read(int(self.headers["Content-Length"])).decode()
            state.queries.append(parse_qs(form)["data"][0])
            body = overpass_response() if state.status == 200 else b"rate limited"
            self.send_response(state.status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_port}/api/interpreter"
    try:
        yield state
    finally:
        server.shutdown()
