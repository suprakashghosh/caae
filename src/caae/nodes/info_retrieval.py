"""Info retrieval node — gathers external data via MCP tools and resources."""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from caae.models.config import IntentRoute
from caae.nodes.deps import Dependencies

logger = logging.getLogger(__name__)


def _get_deps(config: RunnableConfig) -> Dependencies:
    """Extract the Dependencies container from the LangGraph config."""
    return config["configurable"]["deps"]


async def info_retrieval_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Discover MCP tools and read resources for the resolved intent.

    Steps:
        1. Look up ``resolved_intent`` in ``workflow_policy.intent_routing_matrix``.
        2. If intent is ``"unknown"`` or not found, return empty resources.
        3. Get the ``IntentRoute`` and discover available tools on the primary MCP server.
        4. For each required tool, fetch its schema to warm the metadata cache.
        5. Optionally read resources if ``inbound_event_payload`` contains a ``resource_uris`` list.
        6. Assemble and return the resources list.

    Returns:
        ``{"mcp_retrieved_resources": list[dict]}`` — each entry contains
        ``{"server": str, "tools": list[dict], "resources": list[dict]}``.
    """
    # LangGraph may pass state as a Pydantic model; normalize to dict
    if not isinstance(state, dict):
        state = state.model_dump()

    deps = _get_deps(config)
    resolved_intent: str | None = state.get("resolved_intent")

    # ── Early exit for unknown / missing intent ─────────────────────────────
    if not resolved_intent or resolved_intent == "unknown":
        logger.info("Intent is '%s'; skipping MCP resource retrieval", resolved_intent)
        return {"mcp_retrieved_resources": []}

    route: IntentRoute | None = deps.workflow_policy.intent_routing_matrix.get(resolved_intent)
    if route is None:
        logger.warning("No route found for intent '%s'; skipping retrieval", resolved_intent)
        return {"mcp_retrieved_resources": []}

    server_name: str = route.primary_mcp_server
    payload: dict[str, Any] = state.get("inbound_event_payload", {})

    # ── Tool discovery — list all tools on the primary server ───────────────
    try:
        available_tools = await deps.mcp_client.list_tools(server_name)
        logger.info("Discovered %d tools on server '%s'", len(available_tools), server_name)
    except Exception:
        logger.exception("Failed to list tools on server '%s'", server_name)
        return {"mcp_retrieved_resources": []}

    # ── Schema caching — iterate required tools ─────────────────────────────
    tool_schemas: list[dict[str, Any]] = []
    for tool_name in route.required_tools:
        try:
            schema = deps.mcp_client.get_tool_schema(server_name, tool_name)
            tool_schemas.append({"name": tool_name, "input_schema": schema})
        except Exception:
            logger.warning("Tool '%s' not found on server '%s'; skipping", tool_name, server_name)

    # ── Resource reading (optional) ─────────────────────────────────────────
    resource_data: list[dict[str, Any]] = []
    resource_uris: list[dict[str, str]] = payload.get("resource_uris", [])
    for item in resource_uris:
        uri: str = item.get("uri", "")
        res_server: str = item.get("server", server_name)
        if not uri:
            continue
        try:
            result = await deps.mcp_client.read_resource(res_server, uri)
            resource_data.append({"server": res_server, "uri": uri, "data": result})
        except Exception:
            logger.warning("Failed to read resource '%s' on server '%s'", uri, res_server)

    resources_entry: dict[str, Any] = {
        "server": server_name,
        "tools": tool_schemas,
        "resources": resource_data,
    }

    logger.info(
        "Retrieved %d tool schemas and %d resources from '%s'",
        len(tool_schemas),
        len(resource_data),
        server_name,
    )

    return {"mcp_retrieved_resources": [resources_entry]}
