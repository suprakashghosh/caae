"""Tests for the SchemaRegistry and built-in schemas."""

import pytest
from pydantic import BaseModel, ValidationError

from caae.models.schemas import get_default_registry
from caae.models.schemas.base import SchemaRegistry, SchemaRegistryError
from caae.models.schemas.clinical import AppointmentBookingPayload
from caae.models.schemas.media import QuantitativeIntelPayload


class TestSchemaRegistry:
    """Tests for SchemaRegistry registration and resolution."""

    def test_register_and_resolve(self) -> None:
        """Register a schema and resolve it."""
        registry = SchemaRegistry()
        registry.register("schemas.test.MySchema", AppointmentBookingPayload)
        result = registry.resolve("schemas.test.MySchema")
        assert result is AppointmentBookingPayload

    def test_resolve_nonexistent_raises_error(self) -> None:
        """Resolving an unregistered path raises SchemaRegistryError."""
        registry = SchemaRegistry()
        with pytest.raises(SchemaRegistryError, match="No schema registered"):
            registry.resolve("schemas.nonexistent.Foo")

    def test_resolve_error_lists_available(self) -> None:
        """Error message includes available schema paths."""
        registry = SchemaRegistry()
        registry.register("schemas.clinical.AppointmentBookingPayload", AppointmentBookingPayload)
        with pytest.raises(SchemaRegistryError, match="schemas.clinical.AppointmentBookingPayload"):
            registry.resolve("schemas.nonexistent.Foo")

    def test_duplicate_registration_raises_error(self) -> None:
        """Registering the same path twice raises ValueError."""
        registry = SchemaRegistry()
        registry.register("schemas.test.MySchema", AppointmentBookingPayload)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("schemas.test.MySchema", AppointmentBookingPayload)

    def test_list_schemas_empty(self) -> None:
        """list_schemas returns empty list for fresh registry."""
        registry = SchemaRegistry()
        assert registry.list_schemas() == []

    def test_list_schemas_sorted(self) -> None:
        """list_schemas returns paths in sorted order."""
        registry = SchemaRegistry()
        registry.register("schemas.media.QuantitativeIntelPayload", QuantitativeIntelPayload)
        registry.register("schemas.clinical.AppointmentBookingPayload", AppointmentBookingPayload)
        assert registry.list_schemas() == [
            "schemas.clinical.AppointmentBookingPayload",
            "schemas.media.QuantitativeIntelPayload",
        ]

    def test_has_schema(self) -> None:
        """has_schema returns True for registered, False for unregistered."""
        registry = SchemaRegistry()
        registry.register("schemas.test.MySchema", AppointmentBookingPayload)
        assert registry.has_schema("schemas.test.MySchema") is True
        assert registry.has_schema("schemas.nonexistent") is False

    def test_register_custom_model(self) -> None:
        """Register a custom Pydantic model and resolve it."""

        class MyCustomModel(BaseModel):
            name: str
            value: int

        registry = SchemaRegistry()
        registry.register("schemas.custom.MyCustomModel", MyCustomModel)
        resolved = registry.resolve("schemas.custom.MyCustomModel")
        assert resolved is MyCustomModel
        # Verify the resolved model can validate data
        instance = resolved.model_validate({"name": "test", "value": 42})
        assert instance.name == "test"
        assert instance.value == 42

    def test_schema_registry_error_is_key_error(self) -> None:
        """SchemaRegistryError is a proper subclass of KeyError."""
        assert issubclass(SchemaRegistryError, KeyError)


class TestDefaultRegistry:
    """Tests for the default registry with built-in schemas."""

    def test_default_registry_has_clinical_schema(self) -> None:
        """Default registry resolves AppointmentBookingPayload."""
        registry = get_default_registry()
        result = registry.resolve("schemas.clinical.AppointmentBookingPayload")
        assert result is AppointmentBookingPayload

    def test_default_registry_has_media_schema(self) -> None:
        """Default registry resolves QuantitativeIntelPayload."""
        registry = get_default_registry()
        result = registry.resolve("schemas.media.QuantitativeIntelPayload")
        assert result is QuantitativeIntelPayload

    def test_default_registry_singleton(self) -> None:
        """get_default_registry returns the same instance on repeated calls."""
        registry1 = get_default_registry()
        registry2 = get_default_registry()
        assert registry1 is registry2

    def test_default_registry_lists_both_schemas(self) -> None:
        """Default registry lists exactly the 2 built-in schemas."""
        registry = get_default_registry()
        schemas = registry.list_schemas()
        assert schemas == [
            "schemas.clinical.AppointmentBookingPayload",
            "schemas.media.QuantitativeIntelPayload",
        ]


class TestAppointmentBookingPayload:
    """Tests for the AppointmentBookingPayload schema."""

    def test_create_valid_payload(self) -> None:
        """Create a valid appointment booking payload."""
        payload = AppointmentBookingPayload(
            lead_id="lead-123",
            practitioner_id="dr-smith",
            preferred_date="2024-01-15",
            preferred_time="10:00",
            appointment_type="initial_consultation",
        )
        assert payload.lead_id == "lead-123"
        assert payload.notes is None

    def test_create_with_notes(self) -> None:
        """Create payload with optional notes field."""
        payload = AppointmentBookingPayload(
            lead_id="lead-456",
            practitioner_id="dr-jones",
            preferred_date="2024-02-01",
            preferred_time="14:30",
            appointment_type="follow_up",
            notes="Patient prefers afternoon slots",
        )
        assert payload.notes == "Patient prefers afternoon slots"

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AppointmentBookingPayload(
                lead_id="lead-789",
                practitioner_id="dr-who",
                preferred_date="2024-03-01",
                preferred_time="09:00",
                appointment_type="treatment",
                unknown_field="bad",
            )
        errors = exc_info.value.errors()
        assert any(e["type"] == "extra_forbidden" for e in errors)

    def test_json_round_trip(self) -> None:
        """Payload should round-trip through JSON."""
        payload = AppointmentBookingPayload(
            lead_id="lead-rt",
            practitioner_id="dr-rt",
            preferred_date="2024-01-20",
            preferred_time="11:00",
            appointment_type="initial_consultation",
            notes="Round trip test",
        )
        json_str = payload.model_dump_json()
        reconstructed = AppointmentBookingPayload.model_validate_json(json_str)
        assert reconstructed == payload

    def test_schema_contract_path_matches_workflow_policy(self) -> None:
        """The dotted path 'schemas.clinical.AppointmentBookingPayload' matches
        the runtime_schema_contract in workflow_policy.json."""
        registry = get_default_registry()
        schema_class = registry.resolve("schemas.clinical.AppointmentBookingPayload")
        assert schema_class is AppointmentBookingPayload


class TestQuantitativeIntelPayload:
    """Tests for the QuantitativeIntelPayload schema."""

    def test_create_valid_payload(self) -> None:
        """Create a valid quantitative intel payload."""
        payload = QuantitativeIntelPayload(
            topic="YouTube competitor analysis",
            competitor_names=["ChannelA", "ChannelB"],
            metrics_requested=["views", "engagement_rate"],
        )
        assert payload.topic == "YouTube competitor analysis"
        assert payload.depth == "summary"  # default

    def test_create_minimal_payload(self) -> None:
        """Create payload with only required fields."""
        payload = QuantitativeIntelPayload(topic="content research")
        assert payload.competitor_names == []
        assert payload.metrics_requested == []
        assert payload.date_range_start is None
        assert payload.date_range_end is None
        assert payload.depth == "summary"

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should be rejected."""
        with pytest.raises(ValidationError):
            QuantitativeIntelPayload(
                topic="test",
                unknown_field="bad",
            )

    def test_json_round_trip(self) -> None:
        """Payload should round-trip through JSON."""
        payload = QuantitativeIntelPayload(
            topic="trend analysis",
            competitor_names=["CompX"],
            metrics_requested=["subscriber_count"],
            date_range_start="2024-01-01",
            date_range_end="2024-01-31",
            depth="detailed",
        )
        json_str = payload.model_dump_json()
        reconstructed = QuantitativeIntelPayload.model_validate_json(json_str)
        assert reconstructed == payload

    def test_schema_contract_path_matches_workflow_policy(self) -> None:
        """The dotted path 'schemas.media.QuantitativeIntelPayload' matches
        the runtime_schema_contract in workflow_policy.json."""
        registry = get_default_registry()
        schema_class = registry.resolve("schemas.media.QuantitativeIntelPayload")
        assert schema_class is QuantitativeIntelPayload
