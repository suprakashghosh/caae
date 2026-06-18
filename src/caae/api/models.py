"""API request/response models."""

from typing import Any

from pydantic import BaseModel, Field


class InboundEventPayload(BaseModel):
    """Request model for injecting an event into a session."""

    payload: dict[str, Any]
    session_id: str | None = None


class SessionCreateResponse(BaseModel):
    """Response model for session creation."""

    session_id: str


class HealthResponse(BaseModel):
    """Response model for basic health check."""

    status: str


class MCPServerHealth(BaseModel):
    """Status of a single MCP server."""

    status: str = Field(description='"connected", "disconnected", or "error"')
    error: str | None = Field(default=None, description="Error message if status is not connected")


class MCPServersHealthResponse(BaseModel):
    """Response model for MCP server health check."""

    servers: dict[str, MCPServerHealth]
