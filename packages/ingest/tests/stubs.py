"""Stand-ins for what the build talks to."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs

import prose
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


@contextmanager
def _serve(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


def _reply(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


@dataclass
class Site:
    """A stand-in for the sites the corpus links to: the venue's pages and the manuals."""

    url: str = ""
    pages: dict[str, tuple[bytes, str]] = field(default_factory=lambda: dict(prose.PAGES))
    # Paths that answer 404, as a page taken down would.
    offline: set[str] = field(default_factory=set)
    requests: list[str] = field(default_factory=list)


@contextmanager
def serve_site() -> Iterator[Site]:
    state = Site()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state.requests.append(self.path)
            if self.path in state.offline or self.path not in state.pages:
                _reply(self, 404, b"not here", "text/plain")
                return
            body, content_type = state.pages[self.path]
            _reply(self, 200, body, content_type)

        def log_message(self, *args: Any) -> None:
            pass

    with _serve(Handler) as url:
        state.url = url
        yield state


@dataclass
class OpenAI:
    """A stand-in Azure OpenAI resource: it translates from a table and embeds by hash."""

    url: str = ""
    status: int = 200
    # (deployment, operation, request body), in the order they came.
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def made(self, operation: str) -> list[dict[str, Any]]:
        return [body for _, called, body in self.calls if called == operation]


@contextmanager
def serve_openai() -> Iterator[OpenAI]:
    state = OpenAI()
    route = re.compile(r"/openai/deployments/([^/]+)/(chat/completions|embeddings)\?api-version=")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            found = route.match(self.path)
            if state.status != 200 or not found or not self.headers.get("api-key"):
                _reply(
                    self, state.status if state.status != 200 else 404, b"{}", "application/json"
                )
                return
            deployment, operation = found.groups()
            state.calls.append((deployment, operation, body))
            if operation == "embeddings":
                (text,) = body["input"]
                words = len(text.split())
                answer = {
                    "model": prose.EMBEDDING_MODEL,
                    "data": [{"index": 0, "embedding": prose.embedding(text)}],
                    "usage": {"prompt_tokens": words, "total_tokens": words},
                }
            else:
                system, user = body["messages"]
                english = prose.translation(user["content"])
                answer = {
                    "model": prose.TRANSLATION_MODEL,
                    "choices": [{"message": {"role": "assistant", "content": english}}],
                    "usage": {
                        "prompt_tokens": len((system["content"] + " " + user["content"]).split()),
                        "completion_tokens": len(english.split()),
                    },
                }
            _reply(self, 200, json.dumps(answer).encode(), "application/json")

        def log_message(self, *args: Any) -> None:
            pass

    with _serve(Handler) as url:
        state.url = url
        yield state
