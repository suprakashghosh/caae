"""Context assessor node — interprets client/user context, resolves intent."""

import json
import logging
import uuid
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from caae.nodes.deps import Dependencies

logger = logging.getLogger(__name__)


class IntentClassification(BaseModel):
    """Structured output from the LLM intent classifier."""

    intent: str = Field(description="The identified intent label")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)
    reasoning: str = Field(description="Brief reasoning for the classification")


def _get_deps(config: RunnableConfig) -> Dependencies:
    """Extract the Dependencies container from the LangGraph config."""
    return config["configurable"]["deps"]


def _build_intent_prompt(available_intents: list[str], payload: str) -> list:
    """Build the LLM message list for intent classification."""
    intents_str = "\n".join(f"  - {i}" for i in available_intents)
    return [
        SystemMessage(
            content=(
                "You are an intent classifier for a workflow automation engine.\n"
                "Given a user event payload, classify it into one of the supported intents.\n"
                "Respond with the intent label, a confidence score (0-1), and reasoning.\n\n"
                f"Supported intents:\n{intents_str}\n\n"
                "If the payload does not clearly match any intent, return intent 'unknown' "
                "with a low confidence score."
            )
        ),
        HumanMessage(content=f"Classify this event payload:\n{payload}"),
    ]


async def context_assessor_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Ingest the inbound event payload, generate a session ID, and resolve intent via LLM.

    Steps:
        1. Extract ``inbound_event_payload`` from state.
        2. Generate or reuse a ``session_id`` (UUID if not present in payload).
        3. Query the LLM (``init_chat_model`` with ``.with_structured_output(IntentClassification)``)
           to classify intent from the list of available intents in ``workflow_policy.intent_routing_matrix``.
        4. If the LLM call fails or confidence < 0.5, fall back to ``"unknown"``.

    Returns:
        ``{"session_id": str, "resolved_intent": str}``
    """
    # LangGraph may pass state as a Pydantic model; normalize to dict
    if not isinstance(state, dict):
        state = state.model_dump()

    deps = _get_deps(config)
    payload: dict[str, Any] = state.get("inbound_event_payload", {})

    # ── Session ID ──────────────────────────────────────────────────────────
    session_id: str = state.get("session_id") or payload.get("session_id") or str(uuid.uuid4())

    # ── Intent classification via LLM ───────────────────────────────────────
    available_intents = list(deps.workflow_policy.intent_routing_matrix.keys())
    resolved_intent: str = "unknown"

    try:
        llm: BaseChatModel = init_chat_model(
            model=deps.llm_model_name,
            model_provider=deps.llm_provider,
        )
        structured_llm = llm.with_structured_output(IntentClassification)

        messages = _build_intent_prompt(available_intents, json.dumps(payload, default=str))
        result: IntentClassification = await structured_llm.ainvoke(messages)

        if result.confidence >= 0.5 and result.intent in available_intents:
            resolved_intent = result.intent
            logger.info(
                "Intent classified: '%s' (confidence=%.3f, reasoning=%s)",
                result.intent,
                result.confidence,
                result.reasoning,
            )
        else:
            logger.warning(
                "LLM classification rejected — intent=%s confidence=%.3f",
                result.intent,
                result.confidence,
            )
    except Exception:
        logger.exception("LLM intent classification failed; falling back to 'unknown'")

    return {"session_id": session_id, "resolved_intent": resolved_intent}
