"""Tests for Langfuse observability handler, budget enforcement, and MCP tracing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from caae.mcp.client_manager import MCPClientManager, MCPToolError
from caae.models.config import (
    GlobalConstraints,
    IntentRoute,
    MCPConfig,
    StdioMCPServerConfig,
    WorkflowPolicy,
)
from caae.models.schemas.base import SchemaRegistry
from caae.models.schemas.clinical import AppointmentBookingPayload
from caae.nodes.cognitive_processing import cognitive_processing_node
from caae.nodes.deps import Dependencies
from caae.observability.langfuse_handler import LangfuseHandler

# =============================================================================
# Helpers
# =============================================================================


def _make_handler_disabled() -> LangfuseHandler:
    """Create a LangfuseHandler bypassing __init__ in disabled state."""
    handler = LangfuseHandler.__new__(LangfuseHandler)
    handler._enabled = False
    handler._session_costs = {}
    handler._client = MagicMock()
    return handler


def _make_handler_enabled() -> LangfuseHandler:
    """Create a LangfuseHandler bypassing __init__ in enabled state."""
    handler = LangfuseHandler.__new__(LangfuseHandler)
    handler._enabled = True
    handler._session_costs = {}
    handler._client = MagicMock()
    return handler


def _default_workflow_policy() -> WorkflowPolicy:
    """WorkflowPolicy with a single route and default (5.0) budget."""
    route = IntentRoute(
        primary_mcp_server="scheduling_engine",
        required_tools=["check_availability"],
        runtime_schema_contract="schemas.clinical.AppointmentBookingPayload",
        post_execution_state="trigger_sms_nurture",
    )
    return WorkflowPolicy(
        client_profile_id="test",
        intent_routing_matrix={"appointment_booking_request": route},
        global_constraints=GlobalConstraints(max_session_cost_usd=5.0),
    )


def _make_deps(handler: LangfuseHandler | None = None) -> Dependencies:
    """Build Dependencies for cognitive-processing node tests."""
    registry = SchemaRegistry()
    registry.register(
        "schemas.clinical.AppointmentBookingPayload",
        AppointmentBookingPayload,
    )
    return Dependencies(
        mcp_client=MagicMock(),
        workflow_policy=_default_workflow_policy(),
        schema_registry=registry,
        langfuse_handler=handler,
    )


def _make_config(deps: Dependencies) -> dict:
    return {"configurable": {"deps": deps}}


# =============================================================================
# TestLangfuseHandler
# =============================================================================


class TestLangfuseHandler:
    """Tests for LangfuseHandler creation, tracing, cost tracking, lifecycle."""

    # ── Creation ─────────────────────────────────────────────────────────

    def test_handler_creation_with_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LANGFUSE env vars set → handler.enabled is True."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        with patch("caae.observability.langfuse_handler.Langfuse") as mock_cls:
            mock_cls.return_value = MagicMock()
            handler = LangfuseHandler()
            assert handler.enabled is True

    def test_handler_creation_without_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No LANGFUSE env vars → handler.enabled is False."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        with patch("caae.observability.langfuse_handler.Langfuse") as mock_cls:
            mock_cls.return_value = MagicMock()
            handler = LangfuseHandler()
            assert handler.enabled is False

    # ── Disabled — no-ops ────────────────────────────────────────────────

    def test_start_trace_when_disabled(self) -> None:
        """start_trace() returns None when disabled."""
        handler = _make_handler_disabled()
        assert handler.start_trace("session-1") is None

    def test_start_span_when_disabled(self) -> None:
        """start_span() returns None when disabled."""
        handler = _make_handler_disabled()
        assert handler.start_span("my_span") is None

    def test_end_span_when_disabled(self) -> None:
        """end_span() is a no-op when disabled."""
        handler = _make_handler_disabled()
        span = MagicMock()
        # Should not raise
        handler.end_span(span, output_data={"result": "ok"})
        span.update.assert_not_called()
        span.end.assert_not_called()

    def test_record_tool_call_when_disabled(self) -> None:
        """record_tool_call() is a no-op when disabled."""
        handler = _make_handler_disabled()
        handler.record_tool_call(
            trace=None,
            server_name="server_a",
            tool_name="echo",
            arguments={"msg": "hi"},
        )
        # No exception means success

    def test_flush_when_disabled(self) -> None:
        """flush() is a no-op when disabled."""
        handler = _make_handler_disabled()
        handler.flush()
        handler._client.flush.assert_not_called()

    def test_shutdown_when_disabled(self) -> None:
        """shutdown() is a no-op when disabled."""
        handler = _make_handler_disabled()
        handler.shutdown()
        handler._client.flush.assert_not_called()
        handler._client.shutdown.assert_not_called()

    # ── Enabled — tracing ────────────────────────────────────────────────

    def test_start_trace_when_enabled(self) -> None:
        """start_trace() calls client.start_observation() when enabled."""
        handler = _make_handler_enabled()
        mock_obs = MagicMock()
        handler._client.start_observation.return_value = mock_obs

        result = handler.start_trace("session-1", metadata={"env": "test"})

        handler._client.start_observation.assert_called_once_with(
            name="session-session-1",
            as_type="span",
            input={"session_id": "session-1"},
            metadata={"env": "test"},
        )
        assert result is mock_obs

    def test_start_span_when_enabled(self) -> None:
        """start_span() calls client.start_observation() when enabled."""
        handler = _make_handler_enabled()
        mock_obs = MagicMock()
        handler._client.start_observation.return_value = mock_obs

        result = handler.start_span("context_assessor", input_data={"key": "val"})

        handler._client.start_observation.assert_called_once_with(
            name="context_assessor",
            as_type="span",
            input={"key": "val"},
            metadata={},
        )
        assert result is mock_obs

    def test_end_span_when_enabled(self) -> None:
        """end_span() calls span.update() then span.end() when enabled."""
        handler = _make_handler_enabled()
        span = MagicMock()

        handler.end_span(span, output_data={"result": "done"})

        span.update.assert_called_once_with(output={"result": "done"})
        span.end.assert_called_once()

    def test_end_span_exception_is_caught(self) -> None:
        """If span.update() or span.end() raises, the exception is caught and logged."""
        handler = _make_handler_enabled()
        span = MagicMock()
        span.update.side_effect = ValueError("boom")

        # Should not propagate
        handler.end_span(span, output_data={"x": 1})
        # Verify update was called and end was not (since update raised)
        span.update.assert_called_once()
        span.end.assert_not_called()

    def test_record_tool_call_when_enabled(self) -> None:
        """record_tool_call() creates a tool observation with all params."""
        handler = _make_handler_enabled()
        mock_tool_span = MagicMock()
        handler._client.start_observation.return_value = mock_tool_span

        handler.record_tool_call(
            trace=None,
            server_name="srv_a",
            tool_name="echo",
            arguments={"msg": "hello"},
            result={"content": "echoed"},
            latency_ms=42.5,
            is_error=False,
        )

        handler._client.start_observation.assert_called_once_with(
            name="tool:srv_a/echo",
            as_type="tool",
            input={"msg": "hello"},
            metadata={
                "server_name": "srv_a",
                "tool_name": "echo",
                "latency_ms": 42.5,
                "is_error": False,
            },
        )
        mock_tool_span.update.assert_called_once_with(output={"content": "echoed"})
        mock_tool_span.end.assert_called_once()

    # ── Cost tracking ────────────────────────────────────────────────────

    def test_track_cost(self) -> None:
        """track_cost() accumulates costs for a session."""
        handler = _make_handler_enabled()
        handler.track_cost("session-1", prompt_tokens=1000, completion_tokens=500, model="gpt-4o")
        # gpt-4o: 1000*0.0025/1000 + 500*0.01/1000 = 0.0025 + 0.005 = 0.0075
        assert handler.get_session_cost("session-1") == pytest.approx(0.0075)

        # Second call accumulates
        handler.track_cost("session-1", prompt_tokens=500, completion_tokens=500, model="gpt-4o")
        # incremental: 500*0.0025/1000 + 500*0.01/1000 = 0.00125 + 0.005 = 0.00625
        # total: 0.0075 + 0.00625 = 0.01375
        assert handler.get_session_cost("session-1") == pytest.approx(0.01375)

    def test_is_within_budget(self) -> None:
        """is_within_budget() returns True when cost is under the limit."""
        handler = _make_handler_enabled()
        handler.track_cost("session-1", prompt_tokens=100, completion_tokens=100, model="gpt-4o")
        assert handler.is_within_budget("session-1", max_cost_usd=1.0) is True

    def test_is_within_budget_exceeded(self) -> None:
        """is_within_budget() returns False when cost exceeds the limit."""
        handler = _make_handler_enabled()
        handler.track_cost("session-1", prompt_tokens=1000000, completion_tokens=1000000, model="gpt-4o")
        # gpt-4o: 1000000*0.0025/1000 + 1000000*0.01/1000 = 2.50 + 10.0 = 12.50
        assert handler.is_within_budget("session-1", max_cost_usd=5.0) is False

    def test_track_cost_different_models(self) -> None:
        """Cost tracking differs per model pricing."""
        handler = _make_handler_enabled()
        handler.track_cost("s-gpt4o", prompt_tokens=1000, completion_tokens=1000, model="gpt-4o")
        handler.track_cost("s-mini", prompt_tokens=1000, completion_tokens=1000, model="gpt-4o-mini")
        handler.track_cost("s-claude", prompt_tokens=1000, completion_tokens=1000, model="claude-3-5-sonnet-20241022")
        handler.track_cost("s-unknown", prompt_tokens=1000, completion_tokens=1000, model="unknown-model")

        # gpt-4o: 1000*0.0025/1000 + 1000*0.01/1000 = 0.0125
        assert handler.get_session_cost("s-gpt4o") == pytest.approx(0.0125)
        # gpt-4o-mini: 1000*0.00015/1000 + 1000*0.0006/1000 = 0.00075
        assert handler.get_session_cost("s-mini") == pytest.approx(0.00075)
        # claude-3-5-sonnet: 1000*0.003/1000 + 1000*0.015/1000 = 0.018
        assert handler.get_session_cost("s-claude") == pytest.approx(0.018)
        # unknown model uses default: 1000*0.005/1000 + 1000*0.02/1000 = 0.025
        assert handler.get_session_cost("s-unknown") == pytest.approx(0.025)

        # gpt-4o should cost more than gpt-4o-mini
        assert handler.get_session_cost("s-gpt4o") > handler.get_session_cost("s-mini")


# =============================================================================
# TestBudgetEnforcement
# =============================================================================


class TestBudgetEnforcement:
    """Budget enforcement in cognitive_processing_node via LangfuseHandler."""

    async def test_cognitive_processing_budget_exceeded(self) -> None:
        """cognitive_processing_node returns budget error when cost exceeds limit."""
        handler = MagicMock(spec=LangfuseHandler)
        handler.enabled = True
        handler.is_within_budget.return_value = False
        handler.get_session_cost.return_value = 10.0

        deps = _make_deps(handler=handler)
        state = {
            "session_id": "budget-session",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
        }
        result = await cognitive_processing_node(state, _make_config(deps))

        assert "error" in result["extracted_quantitative_data"]
        assert result["extracted_quantitative_data"]["error"] == "session cost exceeds budget"
        handler.is_within_budget.assert_called_once_with("budget-session", 5.0)

    async def test_cognitive_processing_budget_within_limits(self) -> None:
        """cognitive_processing_node proceeds normally when within budget."""
        handler = MagicMock(spec=LangfuseHandler)
        handler.enabled = True
        handler.is_within_budget.return_value = True

        deps = _make_deps(handler=handler)
        state = {
            "session_id": "ok-session",
            "inbound_event_payload": {"lead_id": "L123"},
            "resolved_intent": "appointment_booking_request",
            "mcp_retrieved_resources": [],
        }

        llm_result = AppointmentBookingPayload(
            lead_id="L123",
            practitioner_id="dr-smith",
            preferred_date="2024-01-15",
            preferred_time="10:00",
            appointment_type="initial_consultation",
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=llm_result)

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        patcher = patch("caae.nodes.cognitive_processing.init_chat_model", return_value=mock_llm)
        patcher.start()
        try:
            result = await cognitive_processing_node(state, _make_config(deps))
        finally:
            patcher.stop()

        assert "error" not in result["extracted_quantitative_data"]
        assert result["extracted_quantitative_data"]["lead_id"] == "L123"
        handler.is_within_budget.assert_called_once_with("ok-session", 5.0)


# =============================================================================
# TestMCPClientManagerTracing
# =============================================================================


class TestMCPClientManagerTracing:
    """MCP call_tool integration with LangfuseHandler tracing."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        """Mock ClientSession that returns a successful tool result."""
        session = AsyncMock()
        session.__aenter__.return_value = session
        session.initialize = AsyncMock()

        tools_result = MagicMock()
        tools_result.tools = []
        session.list_tools = AsyncMock(return_value=tools_result)

        call_result = MagicMock()
        call_result.content = [MagicMock()]
        call_result.isError = False
        call_result.structuredContent = {"result": "ok"}
        session.call_tool = AsyncMock(return_value=call_result)
        return session

    @pytest.fixture
    def mock_transport_cm(self) -> AsyncMock:
        cm = AsyncMock()
        cm.__aenter__.return_value = (MagicMock(), MagicMock())
        return cm

    @pytest.fixture
    def config(self) -> MCPConfig:
        return MCPConfig(
            system_mode="testing",
            active_environment="test",
            mcp_servers={
                "test_server": StdioMCPServerConfig(command="python", args=["-c", "pass"]),
            },
        )

    async def _start_manager(
        self,
        manager: MCPClientManager,
        config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        with (
            patch("caae.mcp.client_manager.create_stdio_connection", return_value=mock_transport_cm),
            patch("caae.mcp.client_manager.ClientSession", return_value=mock_session),
        ):
            await manager.start(config)

    async def test_call_tool_records_latency(
        self,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
        config: MCPConfig,
    ) -> None:
        """call_tool invokes record_tool_call when handler is set and enabled."""
        manager = MCPClientManager()
        handler = MagicMock(spec=LangfuseHandler)
        handler.enabled = True
        manager.set_langfuse_handler(handler)

        await self._start_manager(manager, config, mock_session, mock_transport_cm)

        await manager.call_tool("test_server", "echo", {"msg": "hi"})

        handler.record_tool_call.assert_called_once()
        _, kwargs = handler.record_tool_call.call_args
        assert kwargs["server_name"] == "test_server"
        assert kwargs["tool_name"] == "echo"
        assert kwargs["arguments"] == {"msg": "hi"}
        assert kwargs["is_error"] is False
        assert kwargs["latency_ms"] >= 0
        # result should contain structured_content
        assert kwargs["result"]["structured_content"] == {"result": "ok"}

        await manager.stop()

    async def test_call_tool_timeout_records_error(
        self,
        mock_transport_cm: AsyncMock,
        config: MCPConfig,
    ) -> None:
        """Timed-out tool call is recorded as an error in Langfuse."""
        manager = MCPClientManager()
        handler = MagicMock(spec=LangfuseHandler)
        handler.enabled = True
        manager.set_langfuse_handler(handler)

        session = AsyncMock()
        session.__aenter__.return_value = session
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        session.call_tool = AsyncMock(side_effect=TimeoutError())

        with (
            patch("caae.mcp.client_manager.create_stdio_connection", return_value=mock_transport_cm),
            patch("caae.mcp.client_manager.ClientSession", return_value=session),
        ):
            await manager.start(config)

        with pytest.raises(MCPToolError, match="timed out"):
            await manager.call_tool("test_server", "echo", {"msg": "hi"})

        handler.record_tool_call.assert_called_once()
        _, kwargs = handler.record_tool_call.call_args
        assert kwargs["server_name"] == "test_server"
        assert kwargs["tool_name"] == "echo"
        assert kwargs["arguments"] == {"msg": "hi"}
        assert kwargs["is_error"] is True
        assert kwargs["result"] is None
        assert kwargs["latency_ms"] >= 0

        await manager.stop()

    async def test_call_tool_no_handler(
        self,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
        config: MCPConfig,
    ) -> None:
        """call_tool works normally when no handler is set."""
        manager = MCPClientManager()
        await self._start_manager(manager, config, mock_session, mock_transport_cm)

        result = await manager.call_tool("test_server", "echo", {"msg": "hi"})

        assert result["is_error"] is False
        assert result["structured_content"] == {"result": "ok"}

        await manager.stop()
