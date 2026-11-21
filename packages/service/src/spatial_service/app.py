"""The service: one turn operation over HTTP.

    POST /turn   a persona, a snapshot (or none, for the current one), prior state
                 and a question; returns the turn with its trace
    GET /venue   the snapshot served now and the one before, the sourced clearance
                 share, the personas and the route constants (venue_facts.py)
    GET /places  every building a persona can see in a named snapshot, with its
                 equipment types; /places/{building_id} one building's floors, rooms
                 and which rooms hold what (listing.py)
    GET /eval    the eval snapshot the last evaluation run wrote, or the one empty
                 answer when there has been no run (evaluation.py)
    GET /health/live    200 while the process answers HTTP, touching nothing
    GET /health/ready   200 when the store answers and a snapshot is current, 503
                        otherwise (health.py)

Configuration is from the environment:

    SPATIAL_SERVICE_DATABASE_URL     a login that is a member of spatial_reader, and no more
    SPATIAL_LLM_BASE_URL             optional: an OpenAI-compatible /v1 server that plans and
                                     composes, the Spark's, with SPATIAL_LLM_API_KEY. Then
                                     the two deployments below are model names on it, and
                                     may be the same one. Unset, both are on Azure.
    AZURE_OPENAI_ENDPOINT            the Azure OpenAI resource; needed when no /v1 server
                                     plans, and for snapshots embedded on Azure
    AZURE_OPENAI_API_KEY             its key
    AZURE_OPENAI_API_VERSION         optional, 2024-10-21 by default
    SPATIAL_PLANNING_DEPLOYMENT      the deployment that plans
    SPATIAL_COMPOSITION_DEPLOYMENT   the deployment that composes; a different one on Azure
    SPATIAL_LLM_TIMEOUT_SECONDS      optional, 30 by default
    SPATIAL_TREE_NODE_CAP            optional: how many spaces of the collapsed tree the
                                     planner is shown, 200 by default
    OTEL_EXPORTER_OTLP_ENDPOINT      optional; where metrics are pushed (metrics.py).
                                     Unset, none are.
    SPATIAL_EMBEDDING_BASE_URL       optional: an OpenAI-compatible /v1 server that embeds
                                     queries for snapshots whose chunks SPATIAL_EMBEDDING_MODEL
                                     embedded, with SPATIAL_EMBEDDING_API_KEY. Any other
                                     snapshot's queries go to the Azure resource
    AZURE_SEARCH_ENDPOINT            optional: the Azure AI Search service the snapshots'
                                     indexes are in, with AZURE_SEARCH_API_KEY
    SPATIAL_SEARCH_DIR               optional: the stand-in index directory, when no
                                     Azure service is named. With neither, the indexes
                                     are tables in the service's own database.
    SPATIAL_SEMANTIC_RANKER          optional: "on" reranks retrieved passages with
                                     Azure's semantic ranker. Off by default.
    SPATIAL_EVAL_SNAPSHOT            optional: the file an evaluation run wrote its eval
                                     snapshot to. Unset or not there yet, /eval says so.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from spatial_core.records import Persona
from spatial_service import evaluation as eval_contract
from spatial_service import health, listing, metrics, search, venue_facts
from spatial_service.listing import BuildingPlaces, Places, PlacesRefused
from spatial_service.llm import Deployment, Llm
from spatial_service.metrics import Metrics
from spatial_service.retrieval import Documents, Retriever
from spatial_service.store import PostgresStore, Store
from spatial_service.tree import NODE_CAP
from spatial_service.turn import TurnService
from spatial_service.turn_models import Turn, TurnKind, TurnRefused, TurnRequest
from spatial_service.venue_facts import VenueFacts

SEMANTIC_RANKER_ENV = "SPATIAL_SEMANTIC_RANKER"
LLM_BASE_URL_ENV = "SPATIAL_LLM_BASE_URL"
LLM_API_KEY_ENV = "SPATIAL_LLM_API_KEY"
_ON = ("on", "true", "1", "yes")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    database_url: str
    endpoint: str
    api_key: str
    api_version: str
    planning_deployment: str
    composition_deployment: str
    timeout_seconds: float
    tree_node_cap: int = NODE_CAP
    semantic_ranker: bool = False
    # An OpenAI-compatible /v1 server for planning and composition, in place of Azure.
    llm_base_url: str | None = None
    llm_api_key: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        llm_base_url = env.get(LLM_BASE_URL_ENV) or None
        required = [
            "SPATIAL_SERVICE_DATABASE_URL",
            "SPATIAL_PLANNING_DEPLOYMENT",
            "SPATIAL_COMPOSITION_DEPLOYMENT",
        ]
        required += (
            [LLM_API_KEY_ENV] if llm_base_url else ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
        )
        missing = [name for name in required if not env.get(name)]
        if missing:
            raise ConfigError(f"set {', '.join(missing)}")
        settings = cls(
            database_url=env["SPATIAL_SERVICE_DATABASE_URL"],
            endpoint=env.get("AZURE_OPENAI_ENDPOINT", ""),
            api_key=env.get("AZURE_OPENAI_API_KEY", ""),
            api_version=env.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            planning_deployment=env["SPATIAL_PLANNING_DEPLOYMENT"],
            composition_deployment=env["SPATIAL_COMPOSITION_DEPLOYMENT"],
            timeout_seconds=float(env.get("SPATIAL_LLM_TIMEOUT_SECONDS", "30")),
            tree_node_cap=int(env.get("SPATIAL_TREE_NODE_CAP", str(NODE_CAP))),
            semantic_ranker=env.get(SEMANTIC_RANKER_ENV, "off").strip().lower() in _ON,
            llm_base_url=llm_base_url,
            llm_api_key=env.get(LLM_API_KEY_ENV, ""),
        )
        # On Azure each is a deployment with its own quota. A /v1 server hosts models, not
        # roles, so one model there may do both.
        same = settings.planning_deployment == settings.composition_deployment
        if same and not settings.llm_base_url:
            raise ConfigError("planning and composition need separate deployments on Azure")
        return settings


def chat_llm(settings: Settings) -> Llm:
    """Planning and composition on the /v1 server when one is named, on Azure otherwise."""
    # Imported here so tests that stub the model never load the OpenAI stack.
    from agent_framework.openai import OpenAIChatCompletionClient
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    def deployment(name: str) -> Deployment:
        client: AsyncOpenAI
        if settings.llm_base_url:
            client = AsyncOpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                timeout=settings.timeout_seconds,
                max_retries=1,
            )
        else:
            client = AsyncAzureOpenAI(
                azure_endpoint=settings.endpoint,
                api_key=settings.api_key,
                api_version=settings.api_version,
                timeout=settings.timeout_seconds,
                max_retries=1,
            )
        return Deployment(name, OpenAIChatCompletionClient(model=name, async_client=client))

    return Llm(
        planning=deployment(settings.planning_deployment),
        composition=deployment(settings.composition_deployment),
    )


def documents_from_env(settings: Settings, env: Mapping[str, str] = os.environ) -> Documents:
    """Document retrieval as the environment sets it up."""
    try:
        index = search.from_env(env, settings.database_url)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    try:
        embedder = search.embedder_from_env(
            env,
            endpoint=settings.endpoint,
            api_key=settings.api_key,
            api_version=settings.api_version,
            timeout_seconds=settings.timeout_seconds,
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error
    return Retriever(index, embedder, semantic_ranker=settings.semantic_ranker)


def create_app(
    store: Store,
    llm: Llm,
    recorder: Metrics | None = None,
    *,
    tree_node_cap: int = NODE_CAP,
    documents: Documents | None = None,
    evaluation: Path | None = None,
) -> FastAPI:
    service = TurnService(store, llm, tree_node_cap, documents=documents)
    recorder = recorder or Metrics()
    app = FastAPI(title="spatial-knowledge-layer")

    @app.post("/turn", response_model=Turn)
    async def turn(request: TurnRequest) -> Turn | JSONResponse:
        try:
            taken = await service.take(request)
        except TurnRefused as refused:
            # The code, where there is one, is what the caller acts on. The detail is for reading.
            content = {"detail": str(refused), "code": refused.code}
            return JSONResponse(status_code=refused.status, content=content)
        # Every turn that comes out, of any kind, is recorded here and nowhere else.
        recorder.record(taken)
        if taken.kind is TurnKind.UNAVAILABLE:
            return JSONResponse(status_code=503, content=taken.model_dump(mode="json"))
        return taken

    @app.get("/venue", response_model=VenueFacts)
    def venue() -> VenueFacts | JSONResponse:
        try:
            return venue_facts.read(store)
        except venue_facts.VenueUnavailable as unavailable:
            content = {"detail": str(unavailable), "code": unavailable.code}
            return JSONResponse(status_code=503, content=content)

    @app.get("/places", response_model=Places)
    def places(persona: Persona, snapshot: str = Query(min_length=1)) -> Places | JSONResponse:
        try:
            return listing.places(listing.served_venue(store, snapshot, persona))
        except PlacesRefused as refused:
            return _refusal(refused)

    # :path, since a building's id has a slash in it ("osm:way/100").
    @app.get("/places/{building_id:path}", response_model=BuildingPlaces)
    def building_places(
        building_id: str, persona: Persona, snapshot: str = Query(min_length=1)
    ) -> BuildingPlaces | JSONResponse:
        try:
            venue = listing.served_venue(store, snapshot, persona)
            return listing.building_places(venue, building_id)
        except PlacesRefused as refused:
            return _refusal(refused)

    @app.get("/eval")
    def eval_snapshot() -> Any:
        # Read on every request: a run writes the file while the service is up.
        try:
            return eval_contract.read(evaluation)
        except eval_contract.EvalSnapshotError as error:
            content = {"detail": str(error), "code": "eval-snapshot-unreadable"}
            return JSONResponse(status_code=500, content=content)

    @app.get("/health/live")
    def live() -> JSONResponse:
        return JSONResponse(content=health.UP)

    @app.get("/health/ready")
    def ready() -> JSONResponse:
        status, content = health.ready(store)
        return JSONResponse(status_code=status, content=content)

    return app


def _refusal(refused: PlacesRefused) -> JSONResponse:
    content = {"detail": str(refused), "code": refused.code}
    return JSONResponse(status_code=refused.status, content=content)


def _eval_snapshot(env: Mapping[str, str]) -> Path | None:
    named = env.get(eval_contract.SNAPSHOT_ENV)
    return Path(named) if named else None


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    store = PostgresStore(settings.database_url)
    store.check_role()
    recorder = metrics.from_env()
    try:
        uvicorn.run(
            create_app(
                store,
                chat_llm(settings),
                recorder,
                tree_node_cap=settings.tree_node_cap,
                documents=documents_from_env(settings),
                evaluation=_eval_snapshot(os.environ),
            ),
            host=os.environ.get("SPATIAL_SERVICE_HOST", "127.0.0.1"),
            port=int(os.environ.get("SPATIAL_SERVICE_PORT", "8000")),
        )
    finally:
        recorder.close()
