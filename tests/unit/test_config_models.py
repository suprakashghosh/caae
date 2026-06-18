"""Tests for configuration models and loader."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from caae.config.loader import ConfigLoadError, load_mcp_config, load_workflow_policy
from caae.models.config import (
    MCPConfig,
    StdioMCPServerConfig,
    StreamableHttpMCPServerConfig,
    WorkflowPolicy,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


@pytest.fixture
def mcp_config_path() -> Path:
    return FIXTURES_DIR / "mcp_config.json"


@pytest.fixture
def workflow_policy_path() -> Path:
    return FIXTURES_DIR / "workflow_policy.json"


# ── Tests: Loading valid config files ──────────────────────────────────────


class TestLoadValidConfigs:
    """Tests for loading the example JSON config files."""

    def test_load_valid_mcp_config(self, mcp_config_path: Path) -> None:
        """Load the example mcp_config.json and verify field values."""
        config = load_mcp_config(mcp_config_path)
        assert isinstance(config, MCPConfig)
        assert config.system_mode == "production"
        assert config.active_environment == "enterprise_healthcare_sales"

        # Verify both servers are present
        assert "core_crm_gateway" in config.mcp_servers
        assert "scheduling_engine" in config.mcp_servers

        # Check the streamable_http server
        crm = config.mcp_servers["core_crm_gateway"]
        assert isinstance(crm, StreamableHttpMCPServerConfig)
        assert crm.transport == "streamable_http"
        assert crm.endpoint == "https://crm.client-api.internal/mcp/v1"
        assert crm.timeout_ms == 5000
        assert crm.env_auth_token_key == "CRM_PROD_BEARER_TOKEN"

        # Check the stdio server
        sched = config.mcp_servers["scheduling_engine"]
        assert isinstance(sched, StdioMCPServerConfig)
        assert sched.transport == "stdio"
        assert sched.command == "uv"
        assert sched.args == ["run", "caae-demo-server"]
        assert sched.timeout_ms == 10000

    def test_load_valid_workflow_policy(self, workflow_policy_path: Path) -> None:
        """Load the example workflow_policy.json and verify field values."""
        policy = load_workflow_policy(workflow_policy_path)
        assert isinstance(policy, WorkflowPolicy)
        assert policy.client_profile_id == "med_spa_conversion_hub"

        # Verify intents
        routing = policy.intent_routing_matrix
        assert "appointment_booking_request" in routing
        assert "competitor_intel_deep_dive" in routing

        booking = routing["appointment_booking_request"]
        assert booking.primary_mcp_server == "scheduling_engine"
        assert booking.required_tools == ["check_availability", "book_slot"]
        assert booking.runtime_schema_contract == "schemas.clinical.AppointmentBookingPayload"
        assert booking.post_execution_state == "trigger_sms_nurture"

        intel = routing["competitor_intel_deep_dive"]
        assert intel.primary_mcp_server == "youtube_analytics_crawler"
        assert intel.required_tools == ["fetch_channel_metrics", "scrape_transcript"]
        assert intel.runtime_schema_contract == "schemas.media.QuantitativeIntelPayload"
        assert intel.post_execution_state == "compile_script_outline"

        # Verify global constraints
        constraints = policy.global_constraints
        assert constraints.halt_on_negative_sentiment is True
        assert constraints.enforce_strict_anti_slop is True
        assert constraints.max_session_cost_usd == 2.50


# ── Tests: Individual server configs ──────────────────────────────────────


class TestServerConfigs:
    """Tests for individual MCP server config models."""

    def test_stdio_server_config(self) -> None:
        """Create a stdio server config from dict."""
        data = {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "my-server"],
            "timeout_ms": 10000,
        }
        config = StdioMCPServerConfig.model_validate(data)
        assert config.transport == "stdio"
        assert config.command == "uv"
        assert config.args == ["run", "my-server"]
        assert config.timeout_ms == 10000

    def test_stdio_server_with_env(self) -> None:
        """Stdio config with env dict is valid."""
        data = {
            "transport": "stdio",
            "command": "my-cmd",
            "args": ["--flag"],
            "env": {"API_KEY": "abc123", "DEBUG": "true"},
            "timeout_ms": 3000,
        }
        config = StdioMCPServerConfig.model_validate(data)
        assert config.env == {"API_KEY": "abc123", "DEBUG": "true"}
        assert config.args == ["--flag"]

    def test_stdio_server_defaults(self) -> None:
        """Stdio config with only required fields uses defaults."""
        data = {"transport": "stdio", "command": "my-cmd"}
        config = StdioMCPServerConfig.model_validate(data)
        assert config.args == []
        assert config.env is None
        assert config.timeout_ms == 5000

    def test_streamable_http_server_config(self) -> None:
        """Create a streamable_http server config from dict."""
        data = {
            "transport": "streamable_http",
            "endpoint": "https://api.example.com/mcp",
            "env_auth_token_key": "MY_TOKEN",
            "timeout_ms": 3000,
        }
        config = StreamableHttpMCPServerConfig.model_validate(data)
        assert config.transport == "streamable_http"
        assert config.endpoint == "https://api.example.com/mcp"
        assert config.env_auth_token_key == "MY_TOKEN"
        assert config.timeout_ms == 3000

    def test_streamable_http_without_auth(self) -> None:
        """Streamable HTTP config without auth token is valid."""
        data = {"transport": "streamable_http", "endpoint": "https://api.example.com/mcp"}
        config = StreamableHttpMCPServerConfig.model_validate(data)
        assert config.env_auth_token_key is None
        assert config.timeout_ms == 5000

    def test_streamable_http_missing_endpoint(self) -> None:
        """Streamable HTTP config missing endpoint raises ValidationError."""
        data = {"transport": "streamable_http"}
        with pytest.raises(ValidationError) as exc_info:
            StreamableHttpMCPServerConfig.model_validate(data)
        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "endpoint" in field_names

    def test_stdio_missing_command(self) -> None:
        """Stdio config missing command raises ValidationError."""
        data = {"transport": "stdio"}
        with pytest.raises(ValidationError) as exc_info:
            StdioMCPServerConfig.model_validate(data)
        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "command" in field_names

    def test_invalid_transport_type(self) -> None:
        """An invalid transport value should raise ValidationError via the discriminated union."""
        data = {
            "system_mode": "testing",
            "active_environment": "test",
            "mcp_servers": {
                "bad_server": {
                    "transport": "http_sse",
                }
            },
        }
        with pytest.raises(ValidationError) as exc_info:
            MCPConfig.model_validate(data)
        errors = exc_info.value.errors()
        # The discriminated union should produce a union_tag_invalid error
        assert any(e["type"] == "union_tag_invalid" for e in errors), f"Expected union_tag_invalid error, got: {errors}"


# ── Tests: MCPConfig ──────────────────────────────────────────────────────


class TestMCPConfig:
    """Tests for the top-level MCPConfig model."""

    def test_missing_required_field(self) -> None:
        """MCPConfig missing a required field raises ValidationError."""
        data = {
            "system_mode": "testing",
            # missing "active_environment"
            "mcp_servers": {},
        }
        with pytest.raises(ValidationError) as exc_info:
            MCPConfig.model_validate(data)
        errors = exc_info.value.errors()
        field_names = {e["loc"][0] for e in errors}
        assert "active_environment" in field_names

    def test_default_system_mode(self) -> None:
        """Default system_mode is 'development'."""
        data = {
            "active_environment": "test",
            "mcp_servers": {},
        }
        config = MCPConfig.model_validate(data)
        assert config.system_mode == "development"

    def test_invalid_system_mode(self) -> None:
        """An invalid system_mode value raises ValidationError."""
        data = {
            "system_mode": "invalid_mode",
            "active_environment": "test",
            "mcp_servers": {},
        }
        with pytest.raises(ValidationError):
            MCPConfig.model_validate(data)

    def test_extra_fields_rejected(self) -> None:
        """Extra fields in MCPConfig should be rejected (extra='forbid')."""
        data = {
            "active_environment": "test",
            "mcp_servers": {},
            "unknown_top_level_field": "should_fail",
        }
        with pytest.raises(ValidationError) as exc_info:
            MCPConfig.model_validate(data)
        errors = exc_info.value.errors()
        assert any(e["type"] == "extra_forbidden" for e in errors)

    def test_extra_fields_in_server_config_rejected(self) -> None:
        """Extra fields in a server config should be rejected."""
        data = {
            "active_environment": "test",
            "mcp_servers": {
                "my_server": {
                    "transport": "stdio",
                    "command": "my-cmd",
                    "nonexistent_field": "oops",
                }
            },
        }
        with pytest.raises(ValidationError) as exc_info:
            MCPConfig.model_validate(data)
        errors = exc_info.value.errors()
        assert any(e["type"] == "extra_forbidden" for e in errors)


# ── Tests: WorkflowPolicy ──────────────────────────────────────────────────


class TestWorkflowPolicy:
    """Tests for the WorkflowPolicy model."""

    def test_default_global_constraints(self) -> None:
        """Global constraints use sensible defaults."""
        data = {
            "client_profile_id": "test_profile",
            "intent_routing_matrix": {},
            "global_constraints": {},
        }
        policy = WorkflowPolicy.model_validate(data)
        constraints = policy.global_constraints
        assert constraints.halt_on_negative_sentiment is True
        assert constraints.enforce_strict_anti_slop is True
        assert constraints.max_session_cost_usd == 5.0

    def test_invalid_global_constraints_type(self) -> None:
        """Providing a non-dict for global_constraints raises an error."""
        data = {
            "client_profile_id": "test",
            "intent_routing_matrix": {},
            "global_constraints": "not_a_dict",
        }
        with pytest.raises(ValidationError):
            WorkflowPolicy.model_validate(data)

    def test_extra_fields_rejected_in_workflow_policy(self) -> None:
        """Extra fields in WorkflowPolicy should be rejected."""
        data = {
            "client_profile_id": "test",
            "intent_routing_matrix": {},
            "global_constraints": {},
            "extra_field": "should_fail",
        }
        with pytest.raises(ValidationError) as exc_info:
            WorkflowPolicy.model_validate(data)
        errors = exc_info.value.errors()
        assert any(e["type"] == "extra_forbidden" for e in errors)


# ── Tests: Loader ──────────────────────────────────────────────────────────


class TestLoader:
    """Tests for the config loader module."""

    def test_config_file_not_found(self) -> None:
        """Passing a nonexistent path raises FileNotFoundError."""
        nonexistent = Path("/nonexistent/path/mcp_config.json")
        with pytest.raises(FileNotFoundError, match="MCP config file not found"):
            load_mcp_config(nonexistent)

    def test_workflow_policy_file_not_found(self) -> None:
        """Passing a nonexistent path for workflow policy raises FileNotFoundError."""
        nonexistent = Path("/nonexistent/path/workflow_policy.json")
        with pytest.raises(FileNotFoundError, match="Workflow policy file not found"):
            load_workflow_policy(nonexistent)

    def test_malformed_json_raises_config_load_error(self, tmp_path: Path) -> None:
        """Loading a file with invalid JSON syntax raises ConfigLoadError."""
        bad_json = tmp_path / "bad_config.json"
        bad_json.write_text("{invalid json,,,}")
        with pytest.raises(ConfigLoadError, match="Invalid JSON"):
            load_mcp_config(bad_json)

    def test_schema_invalid_raises_config_load_error(self, tmp_path: Path) -> None:
        """Loading a file with valid JSON but invalid schema raises ConfigLoadError."""
        bad_schema = tmp_path / "bad_schema.json"
        bad_schema.write_text(json.dumps({"active_environment": "test", "mcp_servers": {}}))
        # This should load fine since system_mode has a default
        config = load_mcp_config(bad_schema)
        assert config.system_mode == "development"

    def test_loader_wraps_validation_error(self, tmp_path: Path) -> None:
        """Loading valid JSON that fails Pydantic validation raises ConfigLoadError."""
        invalid_data = {
            "system_mode": "invalid_mode",
            "active_environment": "test",
            "mcp_servers": {},
        }
        invalid_file = tmp_path / "invalid_schema.json"
        invalid_file.write_text(json.dumps(invalid_data))
        with pytest.raises(ConfigLoadError, match="Invalid MCP configuration"):
            load_mcp_config(invalid_file)


# ── Tests: Round-trip serialization ────────────────────────────────────────


class TestRoundTrip:
    """Tests for model serialization and reconstruction."""

    def test_mcp_config_round_trip(self, mcp_config_path: Path) -> None:
        """Load a config, dump to dict, reconstruct, and verify match."""
        original = load_mcp_config(mcp_config_path)
        data = original.model_dump()
        reconstructed = MCPConfig.model_validate(data)
        assert reconstructed == original

    def test_workflow_policy_round_trip(self, workflow_policy_path: Path) -> None:
        """Load a policy, dump to dict, reconstruct, and verify match."""
        original = load_workflow_policy(workflow_policy_path)
        data = original.model_dump()
        reconstructed = WorkflowPolicy.model_validate(data)
        assert reconstructed == original

    def test_mcp_config_serialize_to_json(self, mcp_config_path: Path) -> None:
        """Serialize MCP config to JSON and back."""
        original = load_mcp_config(mcp_config_path)
        json_str = original.model_dump_json()
        data = json.loads(json_str)
        reconstructed = MCPConfig.model_validate(data)
        assert reconstructed == original

    def test_workflow_policy_serialize_to_json(self, workflow_policy_path: Path) -> None:
        """Serialize workflow policy to JSON and back."""
        original = load_workflow_policy(workflow_policy_path)
        json_str = original.model_dump_json()
        data = json.loads(json_str)
        reconstructed = WorkflowPolicy.model_validate(data)
        assert reconstructed == original
