"""The model calls build makes: translating a passage, summarising a document, embedding.

Translation and summaries go to the composition deployment on the Azure OpenAI
resource the service uses, named by the same environment variables. Embedding
goes to the same resource unless SPATIAL_EMBEDDING_BASE_URL names an
OpenAI-compatible /v1 server instead (the Spark's vLLM, serving
google/embeddinggemma-2). Then the model's name stands where a deployment's
would, in the vectors and the manifest, and the service reads it back from there.

Every call reports the model that served it and its token usage, which is where
the manifest's ingestion spend comes from.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

ENDPOINT_ENV = "AZURE_OPENAI_ENDPOINT"
API_KEY_ENV = "AZURE_OPENAI_API_KEY"
API_VERSION_ENV = "AZURE_OPENAI_API_VERSION"
TRANSLATION_DEPLOYMENT_ENV = "SPATIAL_COMPOSITION_DEPLOYMENT"
EMBEDDING_DEPLOYMENT_ENV = "SPATIAL_EMBEDDING_DEPLOYMENT"
# An OpenAI-compatible embedding server, in place of the Azure deployment above.
EMBEDDING_BASE_URL_ENV = "SPATIAL_EMBEDDING_BASE_URL"
EMBEDDING_API_KEY_ENV = "SPATIAL_EMBEDDING_API_KEY"
EMBEDDING_MODEL_ENV = "SPATIAL_EMBEDDING_MODEL"
# Optional. A Matryoshka cut the server makes (512 of embeddinggemma's 768). Unset is native.
EMBEDDING_DIMENSIONS_ENV = "SPATIAL_EMBEDDING_DIMENSIONS"
DEFAULT_API_VERSION = "2024-10-21"

# Bump the version whenever the prompt's words change. The manifest records it
# for every translation, and a stored translation made under another version is
# made again.
PROMPT_VERSION = "translate-to-english@1"
PROMPT = (
    "Translate the user's text into English. It is a passage from a venue's accessibility "
    "page or from an equipment manual, written in the language tagged {language}. Translate "
    "all of it and add nothing: no notes, no explanations. Keep the line breaks. Keep names "
    "of buildings, rooms and people, numbers, units and codes exactly as written. Return "
    "only the translation."
)


# Tries per call, and the longest wait between two. Guessed, not measured.
ATTEMPTS = 5
MAX_WAIT_SECONDS = 60.0


class ModelError(Exception):
    pass


def _retryable(response: httpx.Response) -> bool:
    return response.status_code == 429 or response.status_code >= 500


def _wait(response: httpx.Response | None, attempt: int) -> float:
    """Retry-After when the service says, else 2, 4, 8, 16 seconds. Never over the cap."""
    said = response.headers.get("retry-after") if response is not None else None
    try:
        wait = float(said) if said is not None else 2.0**attempt
    except ValueError:
        wait = 2.0**attempt
    return min(max(wait, 0.0), MAX_WAIT_SECONDS)


@dataclass(frozen=True)
class Translated:
    text: str
    deployment: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Embedded:
    vector: tuple[float, ...]
    deployment: str
    model: str
    input_tokens: int


# A summary comes back in the same shape: text, who wrote it, what it cost.
Summarised = Translated


class Models(Protocol):
    translation_deployment: str
    embedding_deployment: str

    def translate(self, text: str, language: str) -> Translated: ...

    def summarise(self, instruction: str, text: str) -> Summarised:
        """A summary of ``text``. The instruction is spatial_ingest.summaries' business."""

    def embed(self, text: str) -> Embedded: ...


def _post(
    url: str,
    *,
    params: Mapping[str, str],
    headers: Mapping[str, str],
    body: dict[str, Any],
    what: str,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """One call, tried again on a rate limit, a server error or a timeout.

    A build makes hundreds of these in a row, and one 429 used to throw the
    whole build away. Anything else (a 400, a content filter) fails at once.
    """
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = httpx.post(url, params=params, headers=headers, json=body, timeout=120)
            if _retryable(response) and attempt < ATTEMPTS:
                sleep(_wait(response, attempt))
                continue
            response.raise_for_status()
            answer: dict[str, Any] = response.json()
        except httpx.TransportError as error:
            if attempt < ATTEMPTS:
                sleep(_wait(None, attempt))
                continue
            raise ModelError(f"{what} failed: {error}") from error
        except (httpx.HTTPError, ValueError) as error:
            raise ModelError(f"{what} failed: {error}") from error
        return answer
    raise AssertionError("unreachable: the last attempt returns or raises")


def _embedded(answer: dict[str, Any], deployment: str) -> Embedded:
    try:
        return Embedded(
            vector=tuple(float(value) for value in answer["data"][0]["embedding"]),
            deployment=deployment,
            model=str(answer.get("model") or deployment),
            input_tokens=int(answer["usage"]["prompt_tokens"]),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ModelError(f"{deployment} answered something unreadable: {error!r}") from error


class OpenAIEmbeddings:
    """Embeddings from an OpenAI-compatible /v1 server. The model's name is the deployment.

    It sends the text it is given. What goes in front of it (the model's document
    prompt) is the caller's: spatial_ingest.documents adds it, so it is in the
    hash a stored vector is matched by.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._api_key = api_key
        self._dimensions = dimensions
        self._sleep = sleep

    def embed(self, text: str) -> Embedded:
        body: dict[str, Any] = {"model": self.model, "input": [text]}
        if self._dimensions is not None:
            body["dimensions"] = self._dimensions
        answer = _post(
            self._url,
            params={},
            headers={"Authorization": f"Bearer {self._api_key}"},
            body=body,
            what=f"embeddings on {self.model}",
            sleep=self._sleep,
        )
        return _embedded(answer, self.model)


class AzureOpenAI:
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str,
        translation_deployment: str,
        embedding_deployment: str,
        sleep: Callable[[float], None] = time.sleep,
        embeddings: OpenAIEmbeddings | None = None,
    ) -> None:
        self.translation_deployment = translation_deployment
        self.embedding_deployment = embeddings.model if embeddings else embedding_deployment
        self._embeddings = embeddings
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._api_version = api_version
        self._sleep = sleep

    def _post(self, deployment: str, operation: str, body: dict[str, Any]) -> dict[str, Any]:
        return _post(
            f"{self._endpoint}/openai/deployments/{deployment}/{operation}",
            params={"api-version": self._api_version},
            headers={"api-key": self._api_key},
            body=body,
            what=f"{operation} on {deployment}",
            sleep=self._sleep,
        )

    def _chat(self, system: str, user: str, what: str) -> Translated:
        deployment = self.translation_deployment
        answer = self._post(
            deployment,
            "chat/completions",
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            },
        )
        try:
            written = answer["choices"][0]["message"]["content"]
            usage = answer["usage"]
            said = Translated(
                text=str(written).strip(),
                deployment=deployment,
                model=str(answer.get("model") or deployment),
                input_tokens=int(usage["prompt_tokens"]),
                output_tokens=int(usage["completion_tokens"]),
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelError(f"{deployment} answered something unreadable: {error!r}") from error
        if not said.text:
            raise ModelError(f"{deployment} returned an empty {what}")
        return said

    def translate(self, text: str, language: str) -> Translated:
        return self._chat(PROMPT.format(language=language), text, "translation")

    def summarise(self, instruction: str, text: str) -> Summarised:
        return self._chat(instruction, text, "summary")

    def embed(self, text: str) -> Embedded:
        if self._embeddings is not None:
            return self._embeddings.embed(text)
        deployment = self.embedding_deployment
        return _embedded(self._post(deployment, "embeddings", {"input": [text]}), deployment)


def configured_embedding(env: Mapping[str, str] | None = None) -> str | None:
    """The deployment (or /v1 model) the environment would embed with. None when it names none.

    Needs no key, so build can ask before it knows whether it will embed anything.
    """
    env = os.environ if env is None else env
    if env.get(EMBEDDING_BASE_URL_ENV):
        return env.get(EMBEDDING_MODEL_ENV) or None
    return env.get(EMBEDDING_DEPLOYMENT_ENV) or None


def from_env(env: Mapping[str, str] | None = None) -> Models:
    """The real client, from the environment. Build only asks once it has a call to make."""
    env = os.environ if env is None else env
    v1 = bool(env.get(EMBEDDING_BASE_URL_ENV))
    embedding = (EMBEDDING_API_KEY_ENV, EMBEDDING_MODEL_ENV) if v1 else (EMBEDDING_DEPLOYMENT_ENV,)
    names = (ENDPOINT_ENV, API_KEY_ENV, TRANSLATION_DEPLOYMENT_ENV, *embedding)
    missing = [name for name in names if not env.get(name)]
    if missing:
        raise ModelError(
            f"the corpus needs translating, summarising and embedding, so set {', '.join(missing)}"
        )
    embeddings = None
    if v1:
        dimensions = env.get(EMBEDDING_DIMENSIONS_ENV)
        try:
            size = int(dimensions) if dimensions else None
        except ValueError as error:
            raise ModelError(f"{EMBEDDING_DIMENSIONS_ENV} is not a number") from error
        embeddings = OpenAIEmbeddings(
            base_url=env[EMBEDDING_BASE_URL_ENV],
            api_key=env[EMBEDDING_API_KEY_ENV],
            model=env[EMBEDDING_MODEL_ENV],
            dimensions=size,
        )
    return AzureOpenAI(
        endpoint=env[ENDPOINT_ENV],
        api_key=env[API_KEY_ENV],
        api_version=env.get(API_VERSION_ENV) or DEFAULT_API_VERSION,
        translation_deployment=env[TRANSLATION_DEPLOYMENT_ENV],
        embedding_deployment=env.get(EMBEDDING_DEPLOYMENT_ENV, ""),
        embeddings=embeddings,
    )
