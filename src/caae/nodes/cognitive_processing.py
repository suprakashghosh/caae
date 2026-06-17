"""Cognitive processing node — LLM reasoning constrained to a schema contract."""

import json
import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from caae.models.config import IntentRoute
from caae.nodes.deps import Dependencies

logger = logging.getLogger(__name__)


def _get_deps(config: RunnableConfig) -> Dependencies:
    """Extract the Dependencies container from the LangGraph config."""
    return config["configurable"]["deps"]


async def cognitive_processing_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Analyze retrieved MCP resources via LLM and extract structured data.

    Steps:
        1. Check ``resolved_intent`` — if ``"unknown"``, return an error dict.
        2. Look up the ``IntentRoute`` and its ``runtime_schema_contract``.
        3. Resolve the schema class via ``schema_registry.resolve(contract_path)``.
        4. Call the LLM with ``.with_structured_output(schema_class)`` to constrain output
           to the registered Pydantic schema.
        5. The prompt instructs the LLM to analyze retrieved resources and the original payload.
        6. Serialize the LLM's structured output to a dict.

    Returns:
        ``{"extracted_quantitative_data": dict}`` — structured data matching the schema contract,
        or ``{"extracted_quantitative_data": {"error": "..."}}`` on failure.
    """
    # LangGraph may pass state as a Pydantic model; normalize to dict
    if not isinstance(state, dict):
        state = state.model_dump()

    deps = _get_deps(config)
    resolved_intent: str | None = state.get("resolved_intent")

    # ── Early exit: unknown intent ──────────────────────────────────────────
    if not resolved_intent or resolved_intent == "unknown":
        logger.warning("No resolved intent; skipping cognitive processing")
        return {"extracted_quantitative_data": {"error": "no intent resolved"}}

    route: IntentRoute | None = deps.workflow_policy.intent_routing_matrix.get(resolved_intent)
    if route is None:
        logger.warning("No route for intent '%s'; skipping cognitive processing", resolved_intent)
        return {"extracted_quantitative_data": {"error": f"no route for intent '{resolved_intent}'"}}

    # ── Resolve schema contract ─────────────────────────────────────────────
    try:
        schema_class: type[BaseModel] = deps.schema_registry.resolve(route.runtime_schema_contract)
    except Exception:
        logger.exception("Failed to resolve schema contract '%s'", route.runtime_schema_contract)
        return {
            "extracted_quantitative_data": {
                "error": f"schema contract '{route.runtime_schema_contract}' not found",
            }
        }

    # ── Build LLM prompt ────────────────────────────────────────────────────
    payload: dict[str, Any] = state.get("inbound_event_payload", {})
    resources: list[dict[str, Any]] = state.get("mcp_retrieved_resources", [])

    messages = [
        SystemMessage(
            content=(
                "You are a data extraction engine. Analyze the provided MCP resources "
                "and event payload to extract structured information.\n"
                f"Your output MUST conform to the following schema contract: {route.runtime_schema_contract}\n"
                "Only include fields defined in that schema. Do not invent extra fields."
            )
        ),
        HumanMessage(
            content=(
                f"Event payload:\n{json.dumps(payload, indent=2, default=str)}\n\n"
                f"MCP resources:\n{json.dumps(resources, indent=2, default=str)}\n\n"
                "Extract the required structured data from the above."
            )
        ),
    ]

    # ── LLM structured inference ────────────────────────────────────────────
    try:
        llm: BaseChatModel = init_chat_model(
            model=deps.llm_model_name,
            model_provider=deps.llm_provider,
        )
        structured_llm = llm.with_structured_output(schema_class)

        result: BaseModel = await structured_llm.ainvoke(messages)

        extracted: dict[str, Any] = result.model_dump()
        logger.info(
            "Cognitive processing succeeded for intent '%s' with schema '%s'",
            resolved_intent,
            route.runtime_schema_contract,
        )
        return {"extracted_quantitative_data": extracted}

    except Exception:
        logger.exception("LLM cognitive processing failed for intent '%s'", resolved_intent)
        return {
            "extracted_quantitative_data": {
                "error": "cognitive processing LLM call failed",
                "intent": resolved_intent,
            }
        }
