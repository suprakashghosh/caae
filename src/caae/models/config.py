"""Configuration models for MCP servers and workflow policies."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── MCP Server Models ──────────────────────────────────────────────────────


class StdioMCPServerConfig(BaseModel):
    """Configuration for an MCP server using stdio transport."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    timeout_ms: int = 5000


class StreamableHttpMCPServerConfig(BaseModel):
    """Configuration for an MCP server using streamable HTTP transport."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["streamable_http"] = "streamable_http"
    endpoint: str  # HTTP endpoint URL
    env_auth_token_key: str | None = None
    timeout_ms: int = 5000


MCPServerConfig = Annotated[
    StdioMCPServerConfig | StreamableHttpMCPServerConfig,
    Field(discriminator="transport"),
]


class MCPConfig(BaseModel):
    """Top-level MCP configuration loaded from mcp_config.json."""

    model_config = ConfigDict(extra="forbid")

    system_mode: Literal["production", "development", "testing"] = "development"
    active_environment: str
    mcp_servers: dict[str, MCPServerConfig]


# ── Workflow Policy Models ─────────────────────────────────────────────────


class IntentRoute(BaseModel):
    """A single intent routing entry within a workflow policy."""

    model_config = ConfigDict(extra="forbid")

    primary_mcp_server: str
    required_tools: list[str] = Field(default_factory=list)
    runtime_schema_contract: str
    post_execution_state: str


class GlobalConstraints(BaseModel):
    """Global workflow constraints."""

    model_config = ConfigDict(extra="forbid")

    halt_on_negative_sentiment: bool = True
    enforce_strict_anti_slop: bool = True
    max_session_cost_usd: float = 5.0


class WorkflowPolicy(BaseModel):
    """Top-level workflow policy loaded from workflow_policy.json."""

    model_config = ConfigDict(extra="forbid")

    client_profile_id: str
    intent_routing_matrix: dict[str, IntentRoute]
    global_constraints: GlobalConstraints
