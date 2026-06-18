"""Integration tests for the full CAAE LangGraph pipeline.

Wires the compiled graph and invokes with mocked dependencies (LLM, MCP).
No real external services required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from caae.graph import build_caae_graph
from caae.mcp.models import ToolInfo
from caae.models.config import GlobalConstraints, IntentRoute, WorkflowPolicy
from caae.models.schemas.base import SchemaRegistry
from caae.models.schemas.clinical import AppointmentBookingPayload
from caae.models.state import UnifiedContextState
from caae.nodes.context_assessor import IntentClassification
from caae.nodes.deps import Dependencies
from caae.nodes.evaluation_gate import verify_output_compliance

# ── Test Helpers ─────────────────────────────────────────────────────────────


def _appointment_policy() -> WorkflowPolicy:
    """WorkflowPolicy with a single route: appointment_booking_request."""
    route = IntentRoute(
        primary_mcp_server="scheduling_engine",
        required_tools=["check_availability", "book_slot"],
        runtime_schema_contract="schemas.clinical.AppointmentBookingPayload",
        post_execution_state="trigger_sms_nurture",
    )
    return WorkflowPolicy(
        client_profile_id="test_profile",
        intent_routing_matrix={"appointment_booking_request": route},
        global_constraints=GlobalConstraints(),
    )


def _schema_registry_with_appointment() -> SchemaRegistry:
    """SchemaRegistry with AppointmentBookingPayload registered."""
    reg = SchemaRegistry()
    reg.register(
        "schemas.clinical.AppointmentBookingPayload",
        AppointmentBookingPayload,
    )
    return reg


def _mock_mcp_client(**kwargs) -> MagicMock:
    """Build a mocked MCPClientManager.

    Accepted keyword-only overrides:
        tools       — return value for list_tools (default [])
        tool_schema — return value for get_tool_schema (default {"type": "object"})
        call_result — return value for call_tool (default {"content": [], "is_error": False})
    """
    client = MagicMock()
    client.list_tools = AsyncMock(return_value=kwargs.get("tools", []))
    client.get_tool_schema = MagicMock(return_value=kwargs.get("tool_schema", {"type": "object"}))
    client.call_tool = AsyncMock(return_value=kwargs.get("call_result", {"content": [], "is_error": False}))
    client.read_resource = AsyncMock(return_value={"contents": []})
    return client


# ── Test Cases ───────────────────────────────────────────────────────────────


class TestGraphPipeline:
    """Integration tests for the full CAAE graph with mocked dependencies."""

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _patch_context_assessor_llm(return_value) -> patch:
        """Patch init_chat_model in context_assessor module.

        Returns a started patcher (call ``.stop()`` to clean up).
        """
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=return_value)

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

        patcher = patch("caae.nodes.context_assessor.init_chat_model")
        mock_init = patcher.start()
        mock_init.return_value = mock_llm
        return patcher

    @staticmethod
    def _patch_cognitive_llm(return_value) -> patch:
        """Patch init_chat_model in cognitive_processing module.

        Returns a started patcher (call ``.stop()`` to clean up).
        """
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=return_value)

        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

        patcher = patch("caae.nodes.cognitive_processing.init_chat_model")
        mock_init = patcher.start()
        mock_init.return_value = mock_llm
        return patcher

    # ── Happy path ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_happy_path_appointment_booking(self):
        """Complete happy path — full pipeline succeeds end-to-end."""
        # ── Arrange ──────────────────────────────────────────────────────
        mcp_client = _mock_mcp_client(
            tools=[
                ToolInfo(name="check_availability", input_schema={"type": "object", "properties": {}}),
                ToolInfo(name="book_slot", input_schema={"type": "object", "properties": {}}),
            ],
            tool_schema={"type": "object", "properties": {}},
            call_result={"content": [{"text": "ok"}], "is_error": False},
        )
        deps = Dependencies(
            mcp_client=mcp_client,
            workflow_policy=_appointment_policy(),
            schema_registry=_schema_registry_with_appointment(),
        )
        graph = build_caae_graph()
        initial_state = UnifiedContextState(
            session_id="",
            inbound_event_payload={"message": "Book appointment for Dr. Smith"},
        )

        intent_result = IntentClassification(
            intent="appointment_booking_request",
            confidence=0.95,
            reasoning="matches appointment pattern",
        )
        cognitive_result = AppointmentBookingPayload(
            lead_id="L123",
            practitioner_id="dr-smith",
            preferred_date="2024-01-15",
            preferred_time="10:00",
            appointment_type="initial_consultation",
        )

        patcher_ctx = self._patch_context_assessor_llm(intent_result)
        patcher_cog = self._patch_cognitive_llm(cognitive_result)
        try:
            # ── Act ──────────────────────────────────────────────────────
            result = await graph.ainvoke(
                initial_state,
                config={"configurable": {"deps": deps}},
            )
        finally:
            patcher_cog.stop()
            patcher_ctx.stop()

        # ── Assert ───────────────────────────────────────────────────────
        assert result["resolved_intent"] == "appointment_booking_request"
        assert result["evaluation_passed"] is True

        # MCP resources retrieved
        assert len(result["mcp_retrieved_resources"]) == 1
        entry = result["mcp_retrieved_resources"][0]
        assert entry["server"] == "scheduling_engine"
        assert len(entry["tools"]) == 2

        # Execution completed
        assert result["execution_mutation_result"]["status"] == "completed"
        assert "check_availability" in result["execution_mutation_result"]["tool_results"]
        assert "book_slot" in result["execution_mutation_result"]["tool_results"]

        # MCP interactions happened
        mcp_client.list_tools.assert_awaited_once()
        assert mcp_client.get_tool_schema.call_count == 2
        assert mcp_client.call_tool.await_count == 2

    # ── Unknown intent ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_unknown_intent_path(self):
        """Unknown intent — graph exhausts retries, state reflects skip/error."""
        # ── Arrange ──────────────────────────────────────────────────────
        mcp_client = _mock_mcp_client()
        deps = Dependencies(
            mcp_client=mcp_client,
            workflow_policy=_appointment_policy(),
            schema_registry=_schema_registry_with_appointment(),
        )
        graph = build_caae_graph()
        initial_state = UnifiedContextState(
            session_id="",
            inbound_event_payload={"message": "gibberish input"},
        )

        unknown_intent = IntentClassification(
            intent="unknown",
            confidence=0.1,
            reasoning="no clear intent",
        )

        # Will be called on every loop iteration; always returns "unknown"
        patcher_ctx = self._patch_context_assessor_llm(unknown_intent)
        # cognitive_processing will not reach its LLM (early exit on unknown),
        # but patch anyway so no real init_chat_model is called.
        dummy_cognitive = AppointmentBookingPayload(
            lead_id="", practitioner_id="", preferred_date="", preferred_time="", appointment_type=""
        )
        patcher_cog = self._patch_cognitive_llm(dummy_cognitive)
        try:
            # ── Act ──────────────────────────────────────────────────────
            result = await graph.ainvoke(
                initial_state,
                config={"configurable": {"deps": deps}},
            )
        finally:
            patcher_cog.stop()
            patcher_ctx.stop()

        # ── Assert ───────────────────────────────────────────────────────
        assert result["resolved_intent"] == "unknown"
        assert result["evaluation_passed"] is False
        assert result["validation_retry_count"] == 3  # exhausted

        # MCP was never called (early exit for unknown)
        assert result["mcp_retrieved_resources"] == []

        # Cognitive processing returned an error for unknown
        assert "error" in result["extracted_quantitative_data"]

        # Action execution was skipped
        assert result["execution_mutation_result"]["status"] == "skipped"

    # ── Retry-loop back (full graph) ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_retry_loop_back(self):
        """Full graph: validation fails twice, then passes on third attempt (loop-back path)."""
        # ── Arrange ──────────────────────────────────────────────────────
        mcp_client = _mock_mcp_client(
            tools=[
                ToolInfo(name="check_availability", input_schema={"type": "object", "properties": {}}),
                ToolInfo(name="book_slot", input_schema={"type": "object", "properties": {}}),
            ],
            tool_schema={"type": "object", "properties": {}},
            call_result={"content": [{"text": "ok"}], "is_error": False},
        )
        deps = Dependencies(
            mcp_client=mcp_client,
            workflow_policy=_appointment_policy(),
            schema_registry=_schema_registry_with_appointment(),
        )
        graph = build_caae_graph()
        initial_state = UnifiedContextState(
            session_id="",
            inbound_event_payload={"message": "Book appointment for Dr. Smith"},
        )

        # Context assessor LLM: always returns appointment_booking_request
        intent_result = IntentClassification(
            intent="appointment_booking_request",
            confidence=0.95,
            reasoning="matches appointment pattern",
        )
        patcher_ctx = self._patch_context_assessor_llm(intent_result)

        # Cognitive processing LLM: raise exception (fails) twice, then return valid data
        valid_cognitive = AppointmentBookingPayload(
            lead_id="L123",
            practitioner_id="dr-smith",
            preferred_date="2024-01-15",
            preferred_time="10:00",
            appointment_type="initial_consultation",
        )

        mock_cog_structured = AsyncMock()
        mock_cog_structured.ainvoke = AsyncMock(
            side_effect=[Exception("LLM failed"), Exception("LLM failed"), valid_cognitive]
        )
        mock_cog_llm = MagicMock()
        mock_cog_llm.with_structured_output = MagicMock(return_value=mock_cog_structured)

        patcher_cog = patch("caae.nodes.cognitive_processing.init_chat_model")
        mock_cog_init = patcher_cog.start()
        mock_cog_init.return_value = mock_cog_llm
        try:
            # ── Act ──────────────────────────────────────────────────────
            result = await graph.ainvoke(
                initial_state,
                config={"configurable": {"deps": deps}},
            )
        finally:
            patcher_cog.stop()
            patcher_ctx.stop()

        # ── Assert ───────────────────────────────────────────────────────
        assert result["resolved_intent"] == "appointment_booking_request"
        assert result["evaluation_passed"] is True
        # Failed twice (retries 1 & 2), passed on third attempt (retry stays at 2)
        assert result["validation_retry_count"] == 2

        # Final execution completed with valid data
        assert result["execution_mutation_result"]["status"] == "completed"
        assert "check_availability" in result["execution_mutation_result"]["tool_results"]
        assert "book_slot" in result["execution_mutation_result"]["tool_results"]

        # Session ID was generated and preserved across loop iterations
        assert result["session_id"] != ""

    # ── Human handoff via full graph ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_human_handoff_via_graph(self):
        """Full graph: validation fails 3 times, escalates to human handoff."""
        # ── Arrange ──────────────────────────────────────────────────────
        mcp_client = _mock_mcp_client(
            tools=[
                ToolInfo(name="check_availability", input_schema={"type": "object", "properties": {}}),
                ToolInfo(name="book_slot", input_schema={"type": "object", "properties": {}}),
            ],
            tool_schema={"type": "object", "properties": {}},
            call_result={"content": [{"text": "ok"}], "is_error": False},
        )
        deps = Dependencies(
            mcp_client=mcp_client,
            workflow_policy=_appointment_policy(),
            schema_registry=_schema_registry_with_appointment(),
        )
        graph = build_caae_graph()
        initial_state = UnifiedContextState(
            session_id="",
            inbound_event_payload={"message": "Book appointment for Dr. Smith"},
        )

        # Context assessor LLM: always returns appointment_booking_request
        intent_result = IntentClassification(
            intent="appointment_booking_request",
            confidence=0.95,
            reasoning="matches appointment pattern",
        )
        patcher_ctx = self._patch_context_assessor_llm(intent_result)

        # Cognitive processing LLM: always raises exception (always fails)
        mock_cog_structured = AsyncMock()
        mock_cog_structured.ainvoke = AsyncMock(side_effect=Exception("LLM always fails"))
        mock_cog_llm = MagicMock()
        mock_cog_llm.with_structured_output = MagicMock(return_value=mock_cog_structured)

        patcher_cog = patch("caae.nodes.cognitive_processing.init_chat_model")
        mock_cog_init = patcher_cog.start()
        mock_cog_init.return_value = mock_cog_llm
        try:
            # ── Act ──────────────────────────────────────────────────────
            result = await graph.ainvoke(
                initial_state,
                config={"configurable": {"deps": deps}},
            )
        finally:
            patcher_cog.stop()
            patcher_ctx.stop()

        # ── Assert ───────────────────────────────────────────────────────
        assert result["validation_retry_count"] >= 3
        assert result["evaluation_passed"] is False
        assert result["resolved_intent"] == "appointment_booking_request"
        assert result["session_id"] != ""

    # ── Human handoff escalation (pure routing test) ─────────────────────

    def test_human_handoff_escalation(self):
        """verify_output_compliance routes to human_handoff when retries exhausted."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
            validation_retry_count=3,
            evaluation_passed=False,
        )
        route = verify_output_compliance(state)
        assert route == "human_handoff_escalation"

    def test_human_handoff_escalation_even_if_passed(self):
        """Retries exhausted takes priority over evaluation_passed."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
            validation_retry_count=3,
            evaluation_passed=True,
        )
        route = verify_output_compliance(state)
        assert route == "human_handoff_escalation"

    # ── Graph structure ──────────────────────────────────────────────────

    def test_graph_compiles_and_has_correct_nodes(self):
        """Compiled graph contains all expected node names."""
        graph = build_caae_graph()

        assert graph is not None
        assert graph.name == "LangGraph"

        node_names = set(graph.nodes.keys())
        expected = {
            "context_assessor",
            "info_retrieval",
            "cognitive_processing",
            "action_execution",
            "evaluation_gate",
        }
        assert expected.issubset(node_names), f"Missing nodes. Expected subset {expected}, got {node_names}"

    # ── Checkpointer integration ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_graph_with_checkpointer(self):
        """Graph compiled with MemorySaver checkpointer runs and returns state correctly."""
        mcp_client = _mock_mcp_client(
            tools=[
                ToolInfo(name="check_availability", input_schema={"type": "object", "properties": {}}),
                ToolInfo(name="book_slot", input_schema={"type": "object", "properties": {}}),
            ],
            tool_schema={"type": "object", "properties": {}},
            call_result={"content": [{"text": "ok"}], "is_error": False},
        )
        deps = Dependencies(
            mcp_client=mcp_client,
            workflow_policy=_appointment_policy(),
            schema_registry=_schema_registry_with_appointment(),
        )
        graph = build_caae_graph(checkpointer=MemorySaver())
        initial_state = UnifiedContextState(
            session_id="checkpointer-test",
            inbound_event_payload={"message": "Book appointment"},
        )

        intent_result = IntentClassification(
            intent="appointment_booking_request",
            confidence=0.95,
            reasoning="matches appointment pattern",
        )
        cognitive_result = AppointmentBookingPayload(
            lead_id="L123",
            practitioner_id="dr-smith",
            preferred_date="2024-01-15",
            preferred_time="10:00",
            appointment_type="initial_consultation",
        )

        patcher_ctx = self._patch_context_assessor_llm(intent_result)
        patcher_cog = self._patch_cognitive_llm(cognitive_result)
        try:
            result = await graph.ainvoke(
                initial_state,
                config={"configurable": {"deps": deps, "thread_id": "test-thread"}},
            )
        finally:
            patcher_cog.stop()
            patcher_ctx.stop()

        assert result["resolved_intent"] == "appointment_booking_request"
        assert result["evaluation_passed"] is True
        assert result["session_id"] == "checkpointer-test"
        assert len(result["mcp_retrieved_resources"]) == 1

    # ── Retry-loop routing ───────────────────────────────────────────────

    def test_retry_re_evaluate_routing(self):
        """verify_output_compliance routes to re_evaluate for retry_count < 3."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
            validation_retry_count=1,
            evaluation_passed=False,
        )
        assert verify_output_compliance(state) == "re_evaluate_context_node"

    def test_commit_state_on_success(self):
        """verify_output_compliance routes to commit_state_and_exit when passed."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
            validation_retry_count=0,
            evaluation_passed=True,
        )
        assert verify_output_compliance(state) == "commit_state_and_exit"
