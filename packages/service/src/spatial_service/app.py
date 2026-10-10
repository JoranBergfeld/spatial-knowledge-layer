"""The service: one turn operation over HTTP.

    POST /turn   a persona, a snapshot (or none, for the current one), prior state
                 and a question; returns the turn with its trace

Configuration is from the environment:

    SPATIAL_SERVICE_DATABASE_URL     a login that is a member of spatial_reader, and no more
    AZURE_OPENAI_ENDPOINT            the Azure OpenAI resource
    AZURE_OPENAI_API_KEY             its key
    AZURE_OPENAI_API_VERSION         optional, 2024-10-21 by default
    SPATIAL_PLANNING_DEPLOYMENT      the deployment that plans
    SPATIAL_COMPOSITION_DEPLOYMENT   the deployment that composes; a different one
    SPATIAL_LLM_TIMEOUT_SECONDS      optional, 30 by default
    SPATIAL_TREE_NODE_CAP            optional: how many spaces of the collapsed tree the
                                     planner is shown, 200 by default
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from spatial_service.llm import Deployment, Llm
from spatial_service.store import PostgresStore, Store
from spatial_service.tree import NODE_CAP
from spatial_service.turn import Turn, TurnKind, TurnRefused, TurnRequest, TurnService


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

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        missing = [
            name
            for name in (
                "SPATIAL_SERVICE_DATABASE_URL",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY",
                "SPATIAL_PLANNING_DEPLOYMENT",
                "SPATIAL_COMPOSITION_DEPLOYMENT",
            )
            if not env.get(name)
        ]
        if missing:
            raise ConfigError(f"set {', '.join(missing)}")
        settings = cls(
            database_url=env["SPATIAL_SERVICE_DATABASE_URL"],
            endpoint=env["AZURE_OPENAI_ENDPOINT"],
            api_key=env["AZURE_OPENAI_API_KEY"],
            api_version=env.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            planning_deployment=env["SPATIAL_PLANNING_DEPLOYMENT"],
            composition_deployment=env["SPATIAL_COMPOSITION_DEPLOYMENT"],
            timeout_seconds=float(env.get("SPATIAL_LLM_TIMEOUT_SECONDS", "30")),
            tree_node_cap=int(env.get("SPATIAL_TREE_NODE_CAP", str(NODE_CAP))),
        )
        if settings.planning_deployment == settings.composition_deployment:
            raise ConfigError("planning and composition need separate deployments")
        return settings


def azure_llm(settings: Settings) -> Llm:
    # Imported here so tests that stub the model never load the OpenAI stack.
    from agent_framework.openai import OpenAIChatCompletionClient
    from openai import AsyncAzureOpenAI

    def deployment(name: str) -> Deployment:
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


def create_app(store: Store, llm: Llm, tree_node_cap: int = NODE_CAP) -> FastAPI:
    service = TurnService(store, llm, tree_node_cap)
    app = FastAPI(title="spatial-knowledge-layer")

    @app.post("/turn", response_model=Turn)
    async def turn(request: TurnRequest) -> Turn | JSONResponse:
        try:
            taken = await service.take(request)
        except TurnRefused as refused:
            return JSONResponse(status_code=refused.status, content={"detail": str(refused)})
        if taken.kind is TurnKind.UNAVAILABLE:
            return JSONResponse(status_code=503, content=taken.model_dump(mode="json"))
        return taken

    return app


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    store = PostgresStore(settings.database_url)
    store.check_role()
    uvicorn.run(
        create_app(store, azure_llm(settings), settings.tree_node_cap),
        host=os.environ.get("SPATIAL_SERVICE_HOST", "127.0.0.1"),
        port=int(os.environ.get("SPATIAL_SERVICE_PORT", "8000")),
    )
