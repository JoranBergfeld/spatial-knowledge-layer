"""The model calls build makes: translating a passage and embedding one.

Both go to deployments on the Azure OpenAI resource the service uses, named by
the same environment variables. Translation is the composition deployment's job.
The embedding deployment is the RAG pipeline's, for now.

Every call reports the model that served it and its token usage, which is where
the manifest's ingestion spend comes from.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

ENDPOINT_ENV = "AZURE_OPENAI_ENDPOINT"
API_KEY_ENV = "AZURE_OPENAI_API_KEY"
API_VERSION_ENV = "AZURE_OPENAI_API_VERSION"
TRANSLATION_DEPLOYMENT_ENV = "SPATIAL_COMPOSITION_DEPLOYMENT"
EMBEDDING_DEPLOYMENT_ENV = "SPATIAL_EMBEDDING_DEPLOYMENT"
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


class ModelError(Exception):
    pass


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


class Models(Protocol):
    translation_deployment: str
    embedding_deployment: str

    def translate(self, text: str, language: str) -> Translated: ...

    def embed(self, text: str) -> Embedded: ...


class AzureOpenAI:
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str,
        translation_deployment: str,
        embedding_deployment: str,
    ) -> None:
        self.translation_deployment = translation_deployment
        self.embedding_deployment = embedding_deployment
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._api_version = api_version

    def _post(self, deployment: str, operation: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._endpoint}/openai/deployments/{deployment}/{operation}"
        try:
            response = httpx.post(
                url,
                params={"api-version": self._api_version},
                headers={"api-key": self._api_key},
                json=body,
                timeout=120,
            )
            response.raise_for_status()
            answer: dict[str, Any] = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ModelError(f"{operation} on {deployment} failed: {error}") from error
        return answer

    def translate(self, text: str, language: str) -> Translated:
        deployment = self.translation_deployment
        answer = self._post(
            deployment,
            "chat/completions",
            {
                "messages": [
                    {"role": "system", "content": PROMPT.format(language=language)},
                    {"role": "user", "content": text},
                ]
            },
        )
        try:
            english = answer["choices"][0]["message"]["content"]
            usage = answer["usage"]
            translated = Translated(
                text=str(english).strip(),
                deployment=deployment,
                model=str(answer.get("model") or deployment),
                input_tokens=int(usage["prompt_tokens"]),
                output_tokens=int(usage["completion_tokens"]),
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelError(f"{deployment} answered something unreadable: {error!r}") from error
        if not translated.text:
            raise ModelError(f"{deployment} returned an empty translation")
        return translated

    def embed(self, text: str) -> Embedded:
        deployment = self.embedding_deployment
        answer = self._post(deployment, "embeddings", {"input": [text]})
        try:
            return Embedded(
                vector=tuple(float(value) for value in answer["data"][0]["embedding"]),
                deployment=deployment,
                model=str(answer.get("model") or deployment),
                input_tokens=int(answer["usage"]["prompt_tokens"]),
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelError(f"{deployment} answered something unreadable: {error!r}") from error


def from_env(env: Mapping[str, str] | None = None) -> Models:
    """The real client, from the environment. Build only asks once it has a call to make."""
    env = os.environ if env is None else env
    names = (ENDPOINT_ENV, API_KEY_ENV, TRANSLATION_DEPLOYMENT_ENV, EMBEDDING_DEPLOYMENT_ENV)
    missing = [name for name in names if not env.get(name)]
    if missing:
        raise ModelError(f"the corpus needs translating and embedding, so set {', '.join(missing)}")
    return AzureOpenAI(
        endpoint=env[ENDPOINT_ENV],
        api_key=env[API_KEY_ENV],
        api_version=env.get(API_VERSION_ENV) or DEFAULT_API_VERSION,
        translation_deployment=env[TRANSLATION_DEPLOYMENT_ENV],
        embedding_deployment=env[EMBEDDING_DEPLOYMENT_ENV],
    )
