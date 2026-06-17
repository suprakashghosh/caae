"""Tests for the UnifiedContextState model."""

import pytest
from pydantic import ValidationError

from caae.models.state import UnifiedContextState


class TestUnifiedContextState:
    """Tests for UnifiedContextState creation, validation, and serialization."""

    def test_create_minimal_state(self) -> None:
        """Create state with only required fields (session_id and inbound_event_payload)."""
        state = UnifiedContextState(
            session_id="test-session-1",
            inbound_event_payload={"message": "hello"},
        )
        assert state.session_id == "test-session-1"
        assert state.resolved_intent is None
        assert state.mcp_retrieved_resources == []
        assert state.extracted_quantitative_data == {}
        assert state.execution_mutation_result == {}
        assert state.validation_retry_count == 0

    def test_create_full_state(self) -> None:
        """Create state with all fields populated."""
        state = UnifiedContextState(
            session_id="test-session-2",
            inbound_event_payload={"message": "book appointment"},
            resolved_intent="appointment_booking_request",
            mcp_retrieved_resources=[{"slot": "10:00"}],
            extracted_quantitative_data={"score": 0.95},
            execution_mutation_result={"appointment_id": "abc123"},
            validation_retry_count=2,
        )
        assert state.resolved_intent == "appointment_booking_request"
        assert len(state.mcp_retrieved_resources) == 1
        assert state.validation_retry_count == 2

    def test_validation_retry_count_max(self) -> None:
        """validation_retry_count allows up to 3."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
            validation_retry_count=3,
        )
        assert state.validation_retry_count == 3

    def test_validation_retry_count_exceeds_max(self) -> None:
        """validation_retry_count > 3 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UnifiedContextState(
                session_id="test",
                inbound_event_payload={},
                validation_retry_count=4,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("validation_retry_count",) and e["type"] == "less_than_equal" for e in errors)

    def test_validation_retry_count_negative(self) -> None:
        """Negative validation_retry_count should be accepted (no min constraint in spec)."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
            validation_retry_count=-1,
        )
        assert state.validation_retry_count == -1

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            UnifiedContextState(
                session_id="test",
                inbound_event_payload={},
                unknown_field="should_fail",
            )
        errors = exc_info.value.errors()
        assert any(e["type"] == "extra_forbidden" for e in errors)

    def test_json_round_trip(self) -> None:
        """State should round-trip through JSON serialization."""
        original = UnifiedContextState(
            session_id="round-trip-test",
            inbound_event_payload={"user_id": "u123", "action": "book"},
            resolved_intent="appointment_booking_request",
            mcp_retrieved_resources=[{"date": "2024-01-15", "slots": ["10:00", "14:00"]}],
            extracted_quantitative_data={"relevance_score": 0.92},
            execution_mutation_result={"booked": True, "appointment_id": "apt-456"},
            validation_retry_count=1,
        )
        json_str = original.model_dump_json()
        reconstructed = UnifiedContextState.model_validate_json(json_str)
        assert reconstructed == original

    def test_dict_round_trip(self) -> None:
        """State should round-trip through dict serialization."""
        original = UnifiedContextState(
            session_id="dict-test",
            inbound_event_payload={"message": "test"},
            resolved_intent="competitor_intel_deep_dive",
        )
        data = original.model_dump()
        reconstructed = UnifiedContextState.model_validate(data)
        assert reconstructed == original

    def test_model_dump_only_set_fields(self) -> None:
        """model_dump should include all fields including defaults."""
        state = UnifiedContextState(
            session_id="test",
            inbound_event_payload={},
        )
        data = state.model_dump()
        assert "session_id" in data
        assert "resolved_intent" in data
        assert data["resolved_intent"] is None
        assert data["validation_retry_count"] == 0
