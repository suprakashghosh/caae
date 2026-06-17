"""MCP data models — ToolInfo and related dataclasses."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolInfo:
    """Describes an MCP tool available on a server.

    Attributes:
        name: The tool's unique name on the server.
        description: A human-readable description of what the tool does.
        input_schema: JSON Schema describing the expected arguments.
    """

    name: str
    description: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
