"""Action execution node — invokes MCP tools to perform business actions."""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from caae.models.config import IntentRoute
from caae.nodes.deps import Dependencies

logger = logging.getLogger(__name__)


def _get_deps(config: RunnableConfig) -> Dependencies:
    """Extract the Dependencies container from the LangGraph config."""
    return config["configurable"]["deps"]


async def action_execution_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Execute required MCP tools using the extracted quantitative data.

    Steps:
        1. Check ``resolved_intent`` — if ``"unknown"``, return a skipped result.
        2. Look up the ``IntentRoute`` from the workflow policy.
        3. For each tool in ``route.required_tools``, construct arguments from
           ``extracted_quantitative_data`` and call ``mcp_client.call_tool()``.
        4. Aggregate all tool results into ``execution_mutation_result``.

    Returns:
        ``{"execution_mutation_result": dict}`` — top-level keys include
        ``status`` and optionally tool-specific result entries.
    """
    # LangGraph may pass state as a Pydantic model; normalize to dict
    if not isinstance(state, dict):
        state = state.model_dump()

    deps = _get_deps(config)
    resolved_intent: str | None = state.get("resolved_intent")

    # ── Early exit: unknown intent ──────────────────────────────────────────
    if not resolved_intent or resolved_intent == "unknown":
        logger.info("Intent is '%s'; skipping action execution", resolved_intent)
        return {"execution_mutation_result": {"status": "skipped", "reason": "no intent resolved"}}

    route: IntentRoute | None = deps.workflow_policy.intent_routing_matrix.get(resolved_intent)
    if route is None:
        logger.warning("No route for intent '%s'; skipping action execution", resolved_intent)
        return {
            "execution_mutation_result": {
                "status": "skipped",
                "reason": f"no route for intent '{resolved_intent}'",
            }
        }

    extracted_data: dict[str, Any] = state.get("extracted_quantitative_data", {})
    server_name: str = route.primary_mcp_server
    results: dict[str, Any] = {"status": "completed", "tool_results": {}}

    # ── Iterate required tools ──────────────────────────────────────────────
    for tool_name in route.required_tools:
        try:
            logger.info("Calling tool '%s' on server '%s'", tool_name, server_name)
            tool_result = await deps.mcp_client.call_tool(
                server_name=server_name,
                tool_name=tool_name,
                arguments=extracted_data,
            )
            results["tool_results"][tool_name] = tool_result
            logger.debug("Tool '%s' returned: %s", tool_name, tool_result)
        except Exception:
            logger.exception("Tool '%s' on server '%s' failed", tool_name, server_name)
            results["tool_results"][tool_name] = {
                "error": f"Tool '{tool_name}' execution failed",
                "is_error": True,
            }
            results["status"] = "partial_failure"

    return {"execution_mutation_result": results}
