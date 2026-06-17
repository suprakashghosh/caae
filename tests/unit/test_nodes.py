"""Unit tests for CAAE LangGraph nodes."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from pydantic import BaseModel

from caae.mcp.models import ToolInfo
from caae.models.config import GlobalConstraints, IntentRoute, WorkflowPolicy
from caae.models.schemas.base import SchemaRegistry
from caae.models.schemas.clinical import AppointmentBookingPayload
from caae.models.state import UnifiedContextState
from caae.nodes.action_execution import action_execution_node
from caae.nodes.cognitive_processing import cognitive_processing_node
from caae.nodes.context_assessor import IntentClassification, context_assessor_node
from caae.nodes.deps import Dependencies
from caae.nodes.evaluation_gate import evaluation_gate_node, verify_output_compliance
from caae.nodes.info_retrieval import info_retrieval_node

# ── Helpers ─────────────────────────────────────────────────────────────────


def _default_workflow_policy() -> WorkflowPolicy:
    """WorkflowPolicy matching configs/workflow_policy.json."""
    route_appt = IntentRoute(
        primary_mcp_server="scheduling_engine",
        required_tools=["check_availability", "book_slot"],
        runtime_schema_contract="schemas.clinical.AppointmentBookingPayload",
        post_execution_state="trigger_sms_nurture",
    )
    route_competitor = IntentRoute(
        primary_mcp_server="youtube_analytics_crawler",
        required_tools=["fetch_channel_metrics", "scrape_transcript"],
        runtime_schema_contract="schemas.media.QuantitativeIntelPayload",
        post_execution_state="compile_script_outline",
    )
    return WorkflowPolicy(
        client_profile_id="med_spa_conversion_hub",
        intent_routing_matrix={
            "appointment_booking_request": route_appt,
            "competitor_intel_deep_dive": route_competitor,
        },
        global_constraints=GlobalConstraints(),
    )


def _default_schema_registry() -> SchemaRegistry:
    """SchemaRegistry with AppointmentBookingPayload registered."""
    reg = SchemaRegistry()
    reg.register(
        "schemas.clinical.AppointmentBookingPayload",
        AppointmentBookingPayload,
    )
    return reg


def _make_deps(
    mcp_client: MagicMock | None = None,
    schema_registry: SchemaRegistry | None = None,
    policy: WorkflowPolicy | None = None,
) -> Dependencies:
    """Build Dependencies; defaults for unspecified args."""
    return Dependencies(
        mcp_client=mcp_client or MagicMock(),
        workflow_policy=policy or _default_workflow_policy(),
        schema_registry=schema_registry or _default_schema_registry(),
    )


def _make_config(deps: Dependencies) -> dict:
    return {"configurable": {"deps": deps}}


def _patch_llm(module_path: str, return_value=None, *, raise_on_ainvoke=False):
    """Patch init_chat_model in *module_path* and return the patcher.

    Usage::

        with _patch_llm("caae.nodes.context_assessor", return_value=...) as patcher:
            ...
    """
    mock_structured = AsyncMock()
    if raise_on_ainvoke:
        mock_structured.ainvoke = AsyncMock(side_effect=Exception("LLM invocation failed"))
    else:
        mock_structured.ainvoke = AsyncMock(return_value=return_value)

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

    patcher = patch(f"{module_path}.init_chat_model")
    mock_init = patcher.start()
    mock_init.return_value = mock_llm
    return patcher


# ── TestContextAssessorNode ─────────────────────────────────────────────────


class TestContextAssessorNode:
    """context_assessor_node — intent classification via LLM."""

    async def test_successful_intent_classification(self):
        """LLM returns high-confidence known intent → that intent is used."""
        deps = _make_deps()
        config = _make_config(deps)
        state = {"inbound_event_payload": {"foo": "bar"}}

        mock_result = IntentClassification(
            intent="appointment_booking_request",
            confidence=0.9,
            reasoning="matches appointment pattern",
        )

        patcher = _patch_llm("caae.nodes.context_assessor", return_value=mock_result)
        try:
            result = await context_assessor_node(state, config)
        finally:
            patcher.stop()

        assert result["resolved_intent"] == "appointment_booking_request"
        # session_id should be a valid UUID (generated since payload lacks one)
        UUID(result["session_id"])

    async def test_llm_failure_falls_back_to_unknown(self):
        """LLM exception → fallback to 'unknown' intent."""
        deps = _make_deps()
        config = _make_config(deps)
        state = {"inbound_event_payload": {"foo": "bar"}}

        patcher = _patch_llm("caae.nodes.context_assessor", raise_on_ainvoke=True)
        try:
            result = await context_assessor_node(state, config)
        finally:
            patcher.stop()

        assert result["resolved_intent"] == "unknown"
        UUID(result["session_id"])

    async def test_low_confidence_falls_back_to_unknown(self):
        """Confidence < 0.5 → fallback to 'unknown' even if intent is known."""
        deps = _make_deps()
        config = _make_config(deps)
        state = {"inbound_event_payload": {"foo": "bar"}}

        mock_result = IntentClassification(
            intent="appointment_booking_request",
            confidence=0.3,
            reasoning="low confidence",
        )

        patcher = _patch_llm("caae.nodes.context_assessor", return_value=mock_result)
        try:
            result = await context_assessor_node(state, config)
        finally:
            patcher.stop()

        assert result["resolved_intent"] == "unknown"

    async def test_high_confidence_intent_not_in_routing_matrix_falls_back(self):
        """High confidence for intent not in routing matrix → unknown."""
        deps = _make_deps()
        config = _make_config(deps)
        state = {"inbound_event_payload": {}}

        mock_result = IntentClassification(
            intent="nonexistent_intent",
            confidence=0.95,
            reasoning="not in routing matrix",
        )

        patcher = _patch_llm("caae.nodes.context_assessor", return_value=mock_result)
        try:
            result = await context_assessor_node(state, config)
        finally:
            patcher.stop()

        assert result["resolved_intent"] == "unknown"

    async def test_session_id_generated_when_not_in_payload(self):
        """No session_id in payload → new UUID generated."""
        deps = _make_deps()
        config = _make_config(deps)
        state = {"inbound_event_payload": {"foo": "bar"}}

        mock_result = IntentClassification(
            intent="appointment_booking_request",
            confidence=0.9,
            reasoning="test",
        )

        patcher = _patch_llm("caae.nodes.context_assessor", return_value=mock_result)
        try:
            result = await context_assessor_node(state, config)
        finally:
            patcher.stop()

        UUID(result["session_id"])  # raises if invalid

    async def test_session_id_preserved_when_in_payload(self):
        """session_id in payload → reused, not regenerated."""
        deps = _make_deps()
        config = _make_config(deps)
        state = {
            "inbound_event_payload": {
                "session_id": "existing-session-abc",
                "foo": "bar",
            }
        }

        mock_result = IntentClassification(
            intent="appointment_booking_request",
            confidence=0.9,
            reasoning="test",
        )

        patcher = _patch_llm("caae.nodes.context_assessor", return_value=mock_result)
        try:
            result = await context_assessor_node(state, config)
        finally:
            patcher.stop()

        assert result["session_id"] == "existing-session-abc"


# ── TestInfoRetrievalNode ───────────────────────────────────────────────────


class TestInfoRetrievalNode:
    """info_retrieval_node — MCP tool discovery & resource reading."""

    async def test_successful_tool_discovery_and_schema_caching(self):
        """Known intent → tools listed, schemas cached, resources empty."""
        mcp_client = MagicMock()
        mcp_client.list_tools = AsyncMock(
            return_value=[
                ToolInfo(name="check_availability", input_schema={"type": "object"}),
                ToolInfo(name="book_slot", input_schema={"type": "object"}),
            ]
        )
        mcp_client.get_tool_schema = MagicMock(side_effect=lambda s, tool: {"type": "object", "properties": {}})

        deps = _make_deps(mcp_client=mcp_client)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
        }
        result = await info_retrieval_node(state, _make_config(deps))

        assert len(result["mcp_retrieved_resources"]) == 1
        entry = result["mcp_retrieved_resources"][0]
        assert entry["server"] == "scheduling_engine"
        assert len(entry["tools"]) == 2
        assert entry["tools"][0]["name"] == "check_availability"
        assert entry["tools"][1]["name"] == "book_slot"
        assert entry["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
        assert entry["resources"] == []

    async def test_unknown_intent_returns_empty_resources(self):
        """resolved_intent == 'unknown' → empty list."""
        deps = _make_deps()
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "unknown",
        }
        result = await info_retrieval_node(state, _make_config(deps))
        assert result["mcp_retrieved_resources"] == []

    async def test_missing_route_returns_empty_resources(self):
        """Intent not in routing matrix → empty list."""
        deps = _make_deps()
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "nonexistent_intent",
        }
        result = await info_retrieval_node(state, _make_config(deps))
        assert result["mcp_retrieved_resources"] == []

    async def test_mcp_list_tools_failure_returns_empty_resources(self):
        """MCP list_tools raises → empty list."""
        mcp_client = MagicMock()
        mcp_client.list_tools = AsyncMock(side_effect=Exception("Connection refused"))

        deps = _make_deps(mcp_client=mcp_client)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
        }
        result = await info_retrieval_node(state, _make_config(deps))
        assert result["mcp_retrieved_resources"] == []

    async def test_resource_uris_in_payload_get_read(self):
        """resource_uris in payload → read_resource called, data included."""
        mcp_client = MagicMock()
        mcp_client.list_tools = AsyncMock(return_value=[])
        mcp_client.get_tool_schema = MagicMock(return_value={})
        mcp_client.read_resource = AsyncMock(return_value={"contents": [{"text": "patient data"}]})

        deps = _make_deps(mcp_client=mcp_client)
        state = {
            "session_id": "test",
            "inbound_event_payload": {
                "resource_uris": [
                    {"uri": "resource://patient/123", "server": "scheduling_engine"},
                ]
            },
            "resolved_intent": "appointment_booking_request",
        }
        result = await info_retrieval_node(state, _make_config(deps))

        entry = result["mcp_retrieved_resources"][0]
        assert len(entry["resources"]) == 1
        assert entry["resources"][0]["uri"] == "resource://patient/123"
        assert entry["resources"][0]["data"] == {"contents": [{"text": "patient data"}]}

    async def test_tool_not_found_on_server_handled_gracefully(self):
        """get_tool_schema raises for a tool → other tools still included."""
        mcp_client = MagicMock()
        mcp_client.list_tools = AsyncMock(
            return_value=[
                ToolInfo(name="check_availability", input_schema={}),
                ToolInfo(name="book_slot", input_schema={}),
            ]
        )
        mcp_client.get_tool_schema = MagicMock(
            side_effect=lambda s, tool: (
                {"type": "object"}
                if tool == "check_availability"
                else (_ for _ in ()).throw(KeyError("Tool not found"))
            )
        )

        deps = _make_deps(mcp_client=mcp_client)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
        }
        result = await info_retrieval_node(state, _make_config(deps))

        entry = result["mcp_retrieved_resources"][0]
        assert len(entry["tools"]) == 1  # book_slot skipped
        assert entry["tools"][0]["name"] == "check_availability"

    async def test_empty_resource_uri_skipped(self):
        """resource_uris entry with empty uri → skipped."""
        mcp_client = MagicMock()
        mcp_client.list_tools = AsyncMock(return_value=[])
        mcp_client.get_tool_schema = MagicMock(return_value={})

        deps = _make_deps(mcp_client=mcp_client)
        state = {
            "session_id": "test",
            "inbound_event_payload": {
                "resource_uris": [
                    {"uri": "", "server": "scheduling_engine"},
                ]
            },
            "resolved_intent": "appointment_booking_request",
        }
        result = await info_retrieval_node(state, _make_config(deps))

        entry = result["mcp_retrieved_resources"][0]
        assert entry["resources"] == []

    async def test_read_resource_failure_logged_and_skipped(self):
        """read_resource raises → entry omitted, others included."""
        mcp_client = MagicMock()
        mcp_client.list_tools = AsyncMock(return_value=[])
        mcp_client.get_tool_schema = MagicMock(return_value={})
        mcp_client.read_resource = AsyncMock(side_effect=Exception("Resource not found"))

        deps = _make_deps(mcp_client=mcp_client)
        state = {
            "session_id": "test",
            "inbound_event_payload": {
                "resource_uris": [
                    {"uri": "resource://patient/999"},
                ]
            },
            "resolved_intent": "appointment_booking_request",
        }
        result = await info_retrieval_node(state, _make_config(deps))

        entry = result["mcp_retrieved_resources"][0]
        assert entry["resources"] == []


# ── TestCognitiveProcessingNode ─────────────────────────────────────────────


class _FakeAppointmentSchema(BaseModel):
    """Minimal in-test schema matching AppointmentBookingPayload shape."""

    lead_id: str
    practitioner_id: str
    preferred_date: str
    preferred_time: str
    appointment_type: str
    notes: str | None = None


class TestCognitiveProcessingNode:
    """cognitive_processing_node — LLM structured extraction."""

    async def test_successful_structured_output(self):
        """LLM returns structured data → extracted_quantitative_data populated."""
        registry = SchemaRegistry()
        registry.register(
            "schemas.clinical.AppointmentBookingPayload",
            AppointmentBookingPayload,
        )
        deps = _make_deps(schema_registry=registry)
        state = {
            "session_id": "test",
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

        patcher = _patch_llm("caae.nodes.cognitive_processing", return_value=llm_result)
        try:
            result = await cognitive_processing_node(state, _make_config(deps))
        finally:
            patcher.stop()

        assert "error" not in result["extracted_quantitative_data"]
        assert result["extracted_quantitative_data"]["lead_id"] == "L123"
        assert result["extracted_quantitative_data"]["practitioner_id"] == "dr-smith"
        assert result["extracted_quantitative_data"]["appointment_type"] == "initial_consultation"

    async def test_unknown_intent_returns_error(self):
        """resolved_intent == 'unknown' → error dict."""
        deps = _make_deps()
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "unknown",
        }
        result = await cognitive_processing_node(state, _make_config(deps))
        assert "error" in result["extracted_quantitative_data"]
        assert result["extracted_quantitative_data"]["error"] == "no intent resolved"

    async def test_no_resolved_intent_returns_error(self):
        """resolved_intent is None → error dict."""
        deps = _make_deps()
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": None,
        }
        result = await cognitive_processing_node(state, _make_config(deps))
        assert "error" in result["extracted_quantitative_data"]

    async def test_missing_route_returns_error(self):
        """Intent not in routing matrix → error dict."""
        deps = _make_deps()
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "nonexistent_intent",
        }
        result = await cognitive_processing_node(state, _make_config(deps))
        assert "error" in result["extracted_quantitative_data"]

    async def test_schema_contract_not_found_returns_error(self):
        """schema_registry.resolve raises → error dict."""
        registry = SchemaRegistry()  # nothing registered
        deps = _make_deps(schema_registry=registry)

        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
        }
        result = await cognitive_processing_node(state, _make_config(deps))
        assert "error" in result["extracted_quantitative_data"]

    async def test_llm_failure_returns_error_dict(self):
        """LLM ainvoke raises → error dict."""
        registry = SchemaRegistry()
        registry.register(
            "schemas.clinical.AppointmentBookingPayload",
            AppointmentBookingPayload,
        )
        deps = _make_deps(schema_registry=registry)
        state = {
            "session_id": "test",
            "inbound_event_payload": {"lead_id": "L123"},
            "resolved_intent": "appointment_booking_request",
            "mcp_retrieved_resources": [],
        }

        patcher = _patch_llm("caae.nodes.cognitive_processing", raise_on_ainvoke=True)
        try:
            result = await cognitive_processing_node(state, _make_config(deps))
        finally:
            patcher.stop()

        assert "error" in result["extracted_quantitative_data"]
        assert result["extracted_quantitative_data"]["error"] == "cognitive processing LLM call failed"


# ── TestActionExecutionNode ─────────────────────────────────────────────────


class TestActionExecutionNode:
    """action_execution_node — MCP tool invocation."""

    async def test_successful_tool_execution(self):
        """All required tools succeed → status 'completed'."""
        mcp_client = MagicMock()
        mcp_client.call_tool = AsyncMock(return_value={"content": [{"text": "ok"}], "is_error": False})

        deps = _make_deps(mcp_client=mcp_client)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
            "extracted_quantitative_data": {
                "lead_id": "L123",
                "practitioner_id": "dr-smith",
                "preferred_date": "2024-01-15",
                "preferred_time": "10:00",
                "appointment_type": "initial_consultation",
            },
        }
        result = await action_execution_node(state, _make_config(deps))

        assert result["execution_mutation_result"]["status"] == "completed"
        assert "check_availability" in result["execution_mutation_result"]["tool_results"]
        assert "book_slot" in result["execution_mutation_result"]["tool_results"]

    async def test_unknown_intent_returns_skipped(self):
        """resolved_intent == 'unknown' → status 'skipped'."""
        deps = _make_deps()
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "unknown",
        }
        result = await action_execution_node(state, _make_config(deps))
        assert result["execution_mutation_result"]["status"] == "skipped"

    async def test_no_resolved_intent_returns_skipped(self):
        """resolved_intent is None → status 'skipped'."""
        deps = _make_deps()
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": None,
        }
        result = await action_execution_node(state, _make_config(deps))
        assert result["execution_mutation_result"]["status"] == "skipped"

    async def test_no_route_returns_skipped(self):
        """Intent not in routing matrix → status 'skipped'."""
        deps = _make_deps()
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "nonexistent_intent",
        }
        result = await action_execution_node(state, _make_config(deps))
        assert result["execution_mutation_result"]["status"] == "skipped"

    async def test_tool_call_failure_partial_failure(self):
        """One tool fails → status 'partial_failure', error in that tool entry."""
        mcp_client = MagicMock()

        async def call_tool_side(server_name, tool_name, arguments):
            if tool_name == "check_availability":
                return {"content": [{"text": "slots available"}], "is_error": False}
            raise Exception("Slot booking failed")

        mcp_client.call_tool = AsyncMock(side_effect=call_tool_side)

        deps = _make_deps(mcp_client=mcp_client)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
            "extracted_quantitative_data": {
                "lead_id": "L123",
                "practitioner_id": "dr-smith",
                "preferred_date": "2024-01-15",
                "preferred_time": "10:00",
                "appointment_type": "initial_consultation",
            },
        }
        result = await action_execution_node(state, _make_config(deps))

        assert result["execution_mutation_result"]["status"] == "partial_failure"
        assert result["execution_mutation_result"]["tool_results"]["book_slot"]["is_error"] is True

    async def test_multiple_tools_all_called(self):
        """Each required tool is called with extracted data as arguments."""
        mcp_client = MagicMock()
        mcp_client.call_tool = AsyncMock(return_value={"content": [{"text": "ok"}], "is_error": False})

        deps = _make_deps(mcp_client=mcp_client)
        extracted = {
            "lead_id": "L123",
            "practitioner_id": "dr-smith",
            "preferred_date": "2024-01-15",
            "preferred_time": "10:00",
            "appointment_type": "initial_consultation",
        }
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
            "extracted_quantitative_data": extracted,
        }
        result = await action_execution_node(state, _make_config(deps))

        assert result["execution_mutation_result"]["status"] == "completed"
        assert mcp_client.call_tool.call_count == 2
        # Verify both tools called with extracted data
        mcp_client.call_tool.assert_any_call(
            server_name="scheduling_engine",
            tool_name="check_availability",
            arguments=extracted,
        )
        mcp_client.call_tool.assert_any_call(
            server_name="scheduling_engine",
            tool_name="book_slot",
            arguments=extracted,
        )


# ── TestEvaluationGateNode ──────────────────────────────────────────────────


class TestEvaluationGateNode:
    """evaluation_gate_node — validate outputs & manage retry count."""

    async def test_validation_passes_for_valid_data(self):
        """Data matches schema contract → passed, retries unchanged."""
        registry = SchemaRegistry()
        registry.register(
            "schemas.clinical.AppointmentBookingPayload",
            AppointmentBookingPayload,
        )
        deps = _make_deps(schema_registry=registry)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
            "execution_mutation_result": None,  # trigger fallback to extracted
            "extracted_quantitative_data": {
                "lead_id": "lead-123",
                "practitioner_id": "dr-smith",
                "preferred_date": "2024-01-15",
                "preferred_time": "10:00",
                "appointment_type": "initial_consultation",
            },
        }
        result = await evaluation_gate_node(state, _make_config(deps))
        assert result["evaluation_passed"] is True
        assert result["validation_retry_count"] == 0

    async def test_validation_fails_for_unknown_intent(self):
        """resolved_intent == 'unknown' → not passed, retry incremented."""
        deps = _make_deps()
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "unknown",
            "validation_retry_count": 0,
        }
        result = await evaluation_gate_node(state, _make_config(deps))
        assert result["evaluation_passed"] is False
        assert result["validation_retry_count"] == 1

    async def test_validation_fails_for_invalid_data_against_schema(self):
        """Data missing required fields → not passed, retry incremented."""
        registry = SchemaRegistry()
        registry.register(
            "schemas.clinical.AppointmentBookingPayload",
            AppointmentBookingPayload,
        )
        deps = _make_deps(schema_registry=registry)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
            "execution_mutation_result": None,
            "extracted_quantitative_data": {
                "lead_id": "lead-123",  # missing required fields
            },
        }
        result = await evaluation_gate_node(state, _make_config(deps))
        assert result["evaluation_passed"] is False
        assert result["validation_retry_count"] == 1

    async def test_retry_count_increments_on_failure(self):
        """Validation failure increments retry count."""
        registry = SchemaRegistry()
        registry.register(
            "schemas.clinical.AppointmentBookingPayload",
            AppointmentBookingPayload,
        )
        deps = _make_deps(schema_registry=registry)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
            "validation_retry_count": 1,
            "execution_mutation_result": None,
            "extracted_quantitative_data": {
                "lead_id": "lead-999",  # incomplete
            },
        }
        result = await evaluation_gate_node(state, _make_config(deps))
        assert result["evaluation_passed"] is False
        assert result["validation_retry_count"] == 2

    async def test_retry_count_capped_at_3(self):
        """Retry capped at 3, does not exceed."""
        registry = SchemaRegistry()
        registry.register(
            "schemas.clinical.AppointmentBookingPayload",
            AppointmentBookingPayload,
        )
        deps = _make_deps(schema_registry=registry)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
            "validation_retry_count": 2,
            "execution_mutation_result": None,
            "extracted_quantitative_data": {
                "lead_id": "lead-999",  # incomplete
            },
        }
        result = await evaluation_gate_node(state, _make_config(deps))
        assert result["evaluation_passed"] is False
        assert result["validation_retry_count"] == 3

    async def test_retry_count_unchanged_on_success(self):
        """Validation success does NOT increment retry count."""
        registry = SchemaRegistry()
        registry.register(
            "schemas.clinical.AppointmentBookingPayload",
            AppointmentBookingPayload,
        )
        deps = _make_deps(schema_registry=registry)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
            "validation_retry_count": 2,
            "execution_mutation_result": None,
            "extracted_quantitative_data": {
                "lead_id": "lead-123",
                "practitioner_id": "dr-smith",
                "preferred_date": "2024-01-15",
                "preferred_time": "10:00",
                "appointment_type": "initial_consultation",
            },
        }
        result = await evaluation_gate_node(state, _make_config(deps))
        assert result["evaluation_passed"] is True
        # retry count unchanged on success
        assert result["validation_retry_count"] == 2

    async def test_already_at_3_retries_returns_false_unchanged(self):
        """retry_count >= 3 → early return with evaluation_passed=False, count unchanged."""
        deps = _make_deps()
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
            "validation_retry_count": 3,
        }
        result = await evaluation_gate_node(state, _make_config(deps))
        assert result["evaluation_passed"] is False
        assert result["validation_retry_count"] == 3  # unchanged

    async def test_error_in_data_triggers_failure(self):
        """execution_mutation_result with error marker → validation fails."""
        registry = SchemaRegistry()
        registry.register(
            "schemas.clinical.AppointmentBookingPayload",
            AppointmentBookingPayload,
        )
        deps = _make_deps(schema_registry=registry)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
            "execution_mutation_result": {
                "error": "something went wrong",
            },
        }
        result = await evaluation_gate_node(state, _make_config(deps))
        assert result["evaluation_passed"] is False

    async def test_skipped_status_in_data_triggers_failure(self):
        """execution_mutation_result with status=skipped → validation fails."""
        registry = SchemaRegistry()
        registry.register(
            "schemas.clinical.AppointmentBookingPayload",
            AppointmentBookingPayload,
        )
        deps = _make_deps(schema_registry=registry)
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "appointment_booking_request",
            "execution_mutation_result": {
                "status": "skipped",
                "reason": "no intent resolved",
            },
        }
        result = await evaluation_gate_node(state, _make_config(deps))
        assert result["evaluation_passed"] is False

    async def test_missing_route_fails_validation(self):
        """resolved_intent not in routing matrix → validation fails."""
        deps = _make_deps()
        state = {
            "session_id": "test",
            "inbound_event_payload": {},
            "resolved_intent": "nonexistent_intent",
            "validation_retry_count": 0,
        }
        result = await evaluation_gate_node(state, _make_config(deps))
        assert result["evaluation_passed"] is False
        assert result["validation_retry_count"] == 1


# ── TestVerifyOutputCompliance ──────────────────────────────────────────────


class TestVerifyOutputCompliance:
    """verify_output_compliance — routing function for conditional edges."""

    def test_evaluation_passed_true_commits_and_exits(self):
        """evaluation_passed=True → 'commit_state_and_exit'."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
            evaluation_passed=True,
            validation_retry_count=0,
        )
        assert verify_output_compliance(state) == "commit_state_and_exit"

    def test_evaluation_false_retries_below_3_re_evaluates(self):
        """evaluation_passed=False, retries < 3 → 're_evaluate_context_node'."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
            evaluation_passed=False,
            validation_retry_count=1,
        )
        assert verify_output_compliance(state) == "re_evaluate_context_node"

    def test_retry_count_3_escalates(self):
        """validation_retry_count >= 3 → 'human_handoff_escalation' regardless of evaluation."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
            evaluation_passed=False,
            validation_retry_count=3,
        )
        assert verify_output_compliance(state) == "human_handoff_escalation"

    def test_retry_count_3_escalates_even_if_passed(self):
        """validation_retry_count >= 3 → escalation, even if evaluation_passed."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
            evaluation_passed=True,
            validation_retry_count=3,
        )
        # retry exhaustion takes priority
        assert verify_output_compliance(state) == "human_handoff_escalation"

    def test_evaluation_none_retries_below_3_re_evaluates(self):
        """evaluation_passed=None, retries < 3 → re_evaluate."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
            evaluation_passed=None,
            validation_retry_count=2,
        )
        assert verify_output_compliance(state) == "re_evaluate_context_node"
