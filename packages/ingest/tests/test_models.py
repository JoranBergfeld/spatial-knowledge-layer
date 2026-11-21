"""The model client tries a call again on a rate limit, a server error or a timeout. Only then."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from spatial_ingest import models

ANSWER = {
    "choices": [{"message": {"content": "A urinal."}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    "model": "gpt-5.4-mini",
}


def client(waits: list[float]) -> models.AzureOpenAI:
    return models.AzureOpenAI(
        endpoint="https://example.invalid",
        api_key="key",
        api_version="2024-10-21",
        translation_deployment="compose",
        embedding_deployment="embed",
        sleep=waits.append,
    )


def answering(monkeypatch: pytest.MonkeyPatch, *replies: Callable[[], httpx.Response]) -> list[int]:
    """httpx.post gives these replies in turn. Returns the call count, as a one-item list."""
    calls = [0]
    request = httpx.Request("POST", "https://example.invalid")

    def post(url: str, **_: Any) -> httpx.Response:
        reply = replies[min(calls[0], len(replies) - 1)]
        calls[0] += 1
        response = reply()
        response.request = request
        return response

    monkeypatch.setattr(models.httpx, "post", post)
    return calls


def status(code: int, **headers: str) -> Callable[[], httpx.Response]:
    return lambda: httpx.Response(code, json={"error": {}}, headers=headers)


def ok() -> httpx.Response:
    return httpx.Response(200, json=ANSWER)


def timeout() -> httpx.Response:
    raise httpx.ReadTimeout("The read operation timed out")


@pytest.mark.parametrize("first", [status(429), status(500), status(503), timeout])
def test_a_passing_failure_is_tried_again(
    monkeypatch: pytest.MonkeyPatch, first: Callable[[], httpx.Response]
) -> None:
    waits: list[float] = []
    calls = answering(monkeypatch, first, ok)
    assert client(waits).summarise("Summarise.", "Urinal P-8150").text == "A urinal."
    assert calls[0] == 2
    assert len(waits) == 1


def test_retry_after_is_honoured_up_to_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "MAX_WAIT_SECONDS", 60.0)
    waits: list[float] = []
    answering(
        monkeypatch, status(429, **{"retry-after": "7"}), status(429, **{"retry-after": "900"}), ok
    )
    client(waits).summarise("Summarise.", "text")
    assert waits == [7.0, 60.0]


def test_without_retry_after_it_backs_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "MAX_WAIT_SECONDS", 60.0)
    waits: list[float] = []
    answering(monkeypatch, status(500), status(500), status(500), ok)
    client(waits).summarise("Summarise.", "text")
    assert waits == [2.0, 4.0, 8.0]


def test_it_gives_up_after_the_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    calls = answering(monkeypatch, status(429))
    with pytest.raises(models.ModelError, match="429"):
        client(waits).summarise("Summarise.", "text")
    assert calls[0] == models.ATTEMPTS
    assert len(waits) == models.ATTEMPTS - 1


def test_a_timeout_on_every_attempt_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = answering(monkeypatch, timeout)
    with pytest.raises(models.ModelError, match="timed out"):
        client([]).embed("text")
    assert calls[0] == models.ATTEMPTS


@pytest.mark.parametrize("code", [400, 401, 404])
def test_a_refusal_fails_at_once(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    # A content filter is a 400. Asking again gets the same answer.
    waits: list[float] = []
    calls = answering(monkeypatch, status(code), ok)
    with pytest.raises(models.ModelError, match=str(code)):
        client(waits).summarise("Summarise.", "text")
    assert calls[0] == 1
    assert waits == []


def test_a_v1_server_embeds_in_place_of_the_azure_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def post(
        url: str, *, headers: dict[str, Any], json: dict[str, Any], **_: Any
    ) -> httpx.Response:
        sent.append((url, headers, json))
        answer = {"data": [{"embedding": [0.6, 0.8]}], "usage": {"prompt_tokens": 2}}
        return httpx.Response(200, json=answer, request=httpx.Request("POST", url))

    monkeypatch.setattr(models.httpx, "post", post)
    env = {
        models.ENDPOINT_ENV: "https://example.invalid",
        models.API_KEY_ENV: "azure-key",
        models.TRANSLATION_DEPLOYMENT_ENV: "compose",
        models.EMBEDDING_BASE_URL_ENV: "http://spark:8103/v1/",
        models.EMBEDDING_API_KEY_ENV: "spark-key",
        models.EMBEDDING_MODEL_ENV: "google/embeddinggemma-2",
        models.EMBEDDING_DIMENSIONS_ENV: "512",
    }
    client = models.from_env(env)

    embedded = client.embed("a passage")

    assert client.embedding_deployment == "google/embeddinggemma-2"
    assert models.configured_embedding(env) == "google/embeddinggemma-2"
    assert embedded.deployment == embedded.model == "google/embeddinggemma-2"
    assert embedded.vector == (0.6, 0.8)
    (url, headers, body) = sent[0]
    assert url == "http://spark:8103/v1/embeddings"
    assert headers == {"Authorization": "Bearer spark-key"}
    assert body == {"model": "google/embeddinggemma-2", "input": ["a passage"], "dimensions": 512}


def test_a_v1_server_needs_its_key_and_model_not_an_azure_embedding_deployment() -> None:
    env = {
        models.ENDPOINT_ENV: "https://example.invalid",
        models.API_KEY_ENV: "azure-key",
        models.TRANSLATION_DEPLOYMENT_ENV: "compose",
        models.EMBEDDING_BASE_URL_ENV: "http://spark:8103/v1",
    }
    with pytest.raises(
        models.ModelError, match="SPATIAL_EMBEDDING_API_KEY, SPATIAL_EMBEDDING_MODEL"
    ):
        models.from_env(env)
    assert models.configured_embedding({models.EMBEDDING_DEPLOYMENT_ENV: "small"}) == "small"
    assert models.configured_embedding({}) is None
