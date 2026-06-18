"""CAAEEngine orchestrator — central coordination hub for the automation engine."""

from __future__ import annotations

import logging
import uuid
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from caae.config.loader import load_mcp_config, load_workflow_policy
from caae.graph import build_caae_graph
from caae.mcp.client_manager import MCPClientManager
from caae.models.config import MCPConfig, WorkflowPolicy
from caae.models.schemas import SchemaRegistry, get_default_registry
from caae.models.state import UnifiedContextState
from caae.nodes.deps import Dependencies
from caae.observability.langfuse_handler import LangfuseHandler

logger = logging.getLogger(__name__)


class CAAEEngine:
    """Central orchestrator for the CAAE workflow.

    Responsible for:
    - Loading MCP client configurations
    - Building and compiling the LangGraph workflow
    - Managing session state across the pipeline
    """

    def __init__(
        self,
        mcp_config_path: str,
        workflow_policy_path: str,
        llm_model_name: str = "gpt-4o",
        llm_provider: str = "openai",
    ) -> None:
        """Initialize the engine with configuration paths.

        Args:
            mcp_config_path: Path to the MCP configuration JSON file.
            workflow_policy_path: Path to the workflow policy JSON file.
            llm_model_name: LLM model identifier (default "gpt-4o").
            llm_provider: LLM provider name (default "openai").
        """
        self._mcp_config_path = mcp_config_path
        self._workflow_policy_path = workflow_policy_path
        self._llm_model_name = llm_model_name
        self._llm_provider = llm_provider

        # Populated during start()
        self._mcp_config: MCPConfig | None = None
        self._mcp_client: MCPClientManager | None = None
        self._workflow_policy: WorkflowPolicy | None = None
        self._schema_registry: SchemaRegistry | None = None
        self._graph: CompiledStateGraph | None = None
        self._langfuse_handler: LangfuseHandler | None = None

        # In-memory session store (V1)
        self._sessions: dict[str, dict[str, Any]] = {}

    def initialize_session(self, session_id: str, payload: dict[str, Any]) -> None:
        """Initialize a session placeholder in the in-memory store.

        Args:
            session_id: Unique session identifier.
            payload: Inbound event payload dict.
        """
        self._sessions[session_id] = UnifiedContextState(
            session_id=session_id,
            inbound_event_payload=payload,
        ).model_dump()

    async def start(self) -> None:
        """Load configuration, start MCP connections, and compile the LangGraph.

        Raises:
            FileNotFoundError: If a config file does not exist.
            ConfigLoadError: If a config file is invalid.
        """
        mcp_config = load_mcp_config(self._mcp_config_path)
        self._mcp_config = mcp_config
        self._workflow_policy = load_workflow_policy(self._workflow_policy_path)
        self._schema_registry = get_default_registry()

        mcp_client = MCPClientManager()
        await mcp_client.start(mcp_config)
        self._mcp_client = mcp_client

        # Initialize Langfuse observability handler
        self._langfuse_handler = LangfuseHandler()
        if self._langfuse_handler.enabled:
            mcp_client.set_langfuse_handler(self._langfuse_handler)

        checkpointer = MemorySaver()
        self._graph = build_caae_graph(checkpointer=checkpointer)

        logger.info("CAAEEngine started successfully")

    async def stop(self) -> None:
        """Gracefully shut down MCP client connections and Langfuse handler."""
        if self._mcp_client is not None:
            await self._mcp_client.stop()
            self._mcp_client = None

        if self._langfuse_handler is not None:
            self._langfuse_handler.shutdown()
            self._langfuse_handler = None

        logger.info("CAAEEngine stopped")

    async def run_session(self, event: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
        """Execute a full CAAE workflow session for the given event.

        Creates an initial ``UnifiedContextState`` from the event, injects
        runtime dependencies, invokes the compiled LangGraph, and stores the
        completed session state in memory.

        Args:
            event: The inbound event payload dict.
            session_id: Optional session ID. If not provided, the graph generates one.

        Returns:
            The final state dict produced by graph execution.

        Raises:
            RuntimeError: If the engine has not been started.
        """
        if self._graph is None or self._mcp_client is None:
            raise RuntimeError("CAAEEngine not started. Call start() first.")

        # Build initial state from the inbound event
        initial_state = UnifiedContextState(
            session_id=session_id or "",
            inbound_event_payload=event,
        )

        # ── Session trace root ─────────────────────────────────────────────
        trace_span = None
        trace_context: dict[str, Any] = {}
        if self._langfuse_handler and self._langfuse_handler.enabled:
            trace_span = self._langfuse_handler.start_span(
                name="caae-session",
                input_data=event,
            )
            if trace_span is not None:
                trace_context["trace_root_id"] = str(id(trace_span))
            self._mcp_client.set_langfuse_handler(self._langfuse_handler, trace_context)

        # Ensure dependencies are initialized (narrow from Optional)
        if self._workflow_policy is None:
            raise RuntimeError("Engine not properly initialized: missing workflow_policy")
        if self._schema_registry is None:
            raise RuntimeError("Engine not properly initialized: missing schema_registry")

        # Assemble runtime dependencies
        deps = Dependencies(
            mcp_client=self._mcp_client,
            workflow_policy=self._workflow_policy,
            schema_registry=self._schema_registry,
            llm_model_name=self._llm_model_name,
            llm_provider=self._llm_provider,
            langfuse_handler=self._langfuse_handler,
        )

        # Build LangGraph config with optional Langfuse callbacks
        from langfuse.langchain import CallbackHandler as LangchainCallbackHandler

        callbacks: list[Any] = []
        if self._langfuse_handler and self._langfuse_handler.enabled:
            callbacks.append(LangchainCallbackHandler())

        thread_id = session_id or initial_state.session_id or str(uuid.uuid4())
        config: dict[str, Any] = {"configurable": {"deps": deps, "thread_id": thread_id}}
        if callbacks:
            config["callbacks"] = callbacks

        # Execute the graph
        raw_state: UnifiedContextState | dict[str, Any] = await self._graph.ainvoke(
            initial_state,
            config=cast(RunnableConfig, config),
        )

        # Normalise: ainvoke may return dict when checkpointer is used
        if isinstance(raw_state, dict):
            final_state = UnifiedContextState.model_validate(raw_state)
        else:
            final_state = raw_state

        # Serialise and cache the result
        result: dict[str, Any] = final_state.model_dump()
        self._sessions[final_state.session_id] = result

        # ── End session trace ──────────────────────────────────────────────
        if trace_span is not None and self._langfuse_handler is not None:
            self._langfuse_handler.end_span(trace_span, output_data=result)

        return result

    async def _run_session_and_store(self, payload: dict[str, Any], session_id: str) -> None:
        """Run a session in background and store result, catching failures.

        Args:
            payload: The inbound event payload dict.
            session_id: The session identifier.
        """
        try:
            await self.run_session(payload, session_id)
        except Exception:
            logger.exception("Session %s failed", session_id)
            self._sessions[session_id] = {
                **self._sessions.get(session_id, {}),
                "error": "session execution failed",
                "evaluation_passed": False,
            }

    def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a previously completed session's state from the in-memory store.

        Args:
            session_id: The UUID of the session to look up.

        Returns:
            The session state dict, or ``None`` if no session with that ID exists.
        """
        return self._sessions.get(session_id)

    def get_mcp_server_statuses(self) -> dict[str, dict[str, Any]]:
        """Get the connection status of all configured MCP servers.

        Returns:
            A dict mapping server names to ``{"status": "connected"|"disconnected", ...}``.
        """
        statuses: dict[str, dict[str, Any]] = {}
        if self._mcp_config is None or self._mcp_client is None:
            return statuses

        connected = set(self._mcp_client.connected_server_names)
        configured = set(self._mcp_config.mcp_servers.keys())

        for server_name in configured:
            if server_name in connected:
                statuses[server_name] = {"status": "connected"}
            else:
                statuses[server_name] = {
                    "status": "disconnected",
                    "error": "Server not connected during startup",
                }
        return statuses
