"""The model calls of a turn, through Microsoft Agent Framework.

Planning and composition each have a deployment of their own, set by
configuration; recovery plans again, so the planning deployment serves it. Each
call is one agent run with a structured response format, and reports its token
usage under the deployment that served it. Any failure to get a
response is LlmUnavailable: the turn then says the model could not be reached and
answers nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agent_framework import Agent, SupportsChatGetResponse
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)


class Purpose(StrEnum):
    PLANNING = "planning"
    # A second plan after a trigger, from the deployment that plans.
    RECOVERY = "recovery"
    COMPOSITION = "composition"


class LlmCall(BaseModel):
    """One model call's token usage, and the deployment that served it."""

    model_config = ConfigDict(frozen=True)

    purpose: Purpose
    deployment: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class Deployment:
    name: str
    client: SupportsChatGetResponse[Any]


class LlmUnavailable(Exception):
    def __init__(self, purpose: Purpose, deployment: str, cause: BaseException) -> None:
        super().__init__(f"{purpose} on {deployment}: {cause!r}")
        self.purpose = purpose
        self.deployment = deployment


@dataclass(frozen=True)
class Llm:
    planning: Deployment
    composition: Deployment

    async def call(
        self,
        purpose: Purpose,
        *,
        instructions: str,
        payload: dict[str, Any],
        response_format: type[BaseModel],
    ) -> tuple[str, LlmCall]:
        """The model's raw text, unparsed: the caller checks it against its own schema."""
        deployment = self.composition if purpose is Purpose.COMPOSITION else self.planning
        agent = Agent(deployment.client, instructions, name=f"spatial-{purpose}")
        message = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        try:
            response = await agent.run(message, options={"response_format": response_format})
        except Exception as error:
            log.warning("%s call to %s failed: %r", purpose, deployment.name, error)
            raise LlmUnavailable(purpose, deployment.name, error) from error
        usage = response.usage_details or {}
        return response.text, LlmCall(
            purpose=purpose,
            deployment=deployment.name,
            input_tokens=usage.get("input_token_count"),
            output_tokens=usage.get("output_token_count"),
            total_tokens=usage.get("total_token_count"),
        )
