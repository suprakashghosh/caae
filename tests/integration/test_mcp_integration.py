"""Integration tests for MCP client infrastructure using a real FastMCP server."""

import os
import sys
import tempfile

import pytest

from caae.mcp import (
    MCPClientManager,
    MCPConnectionError,
)
from caae.models.config import (
    MCPConfig,
    StdioMCPServerConfig,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_server_script() -> str:
    """Create and return the path to a temporary FastMCP server script."""
    script = '''
"""Test MCP server for integration testing."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("TestIntegrationServer")

@mcp.tool()
def echo(message: str) -> str:
    """Echo back the message."""
    return message

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b

@mcp.tool()
def get_info() -> dict:
    """Return server info."""
    return {"name": "TestIntegrationServer", "version": "1.0.0"}

mcp.run()
'''
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(script)
        return f.name


# ── Integration Tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.integration
class TestStdioConnectionIntegration:
    """Integration tests using a real FastMCP server via stdio transport."""

    @pytest.fixture(autouse=True)
    def _setup_server(self) -> None:
        """Create a temporary server script and set up config."""
        self._script_path = _create_server_script()
        self._config = MCPConfig(
            system_mode="testing",
            active_environment="test",
            mcp_servers={
                "test_server": StdioMCPServerConfig(
                    command=sys.executable,
                    args=[self._script_path],
                    timeout_ms=30000,
                ),
            },
        )

    def teardown_method(self) -> None:
        """Clean up the temporary server script."""
        if hasattr(self, "_script_path") and os.path.exists(self._script_path):
            os.unlink(self._script_path)

    async def test_tool_discovery_returns_correct_tools(self) -> None:
        """MCPClientManager discovers tools and returns correct names and schemas."""
        manager = MCPClientManager()
        try:
            await manager.start(self._config)

            tools = await manager.list_tools("test_server")
            tool_names = [t.name for t in tools]

            assert "echo" in tool_names
            assert "add" in tool_names
            assert "get_info" in tool_names
            assert len(tools) == 3

            # Check tool descriptions
            echo_tool = next(t for t in tools if t.name == "echo")
            assert echo_tool.description == "Echo back the message."

            # Check input schemas
            add_tool = next(t for t in tools if t.name == "add")
            assert add_tool.input_schema["type"] == "object"
            assert "a" in add_tool.input_schema["properties"]
            assert "b" in add_tool.input_schema["properties"]
        finally:
            await manager.stop()

    async def test_call_tool_echo(self) -> None:
        """Call the echo tool and verify the response."""
        manager = MCPClientManager()
        try:
            await manager.start(self._config)

            result = await manager.call_tool(
                "test_server",
                "echo",
                {"message": "Hello, MCP!"},
            )

            assert result["is_error"] is False
            # Content should contain text with the echoed message
            assert len(result["content"]) > 0
        finally:
            await manager.stop()

    async def test_call_tool_add(self) -> None:
        """Call the add tool and verify correct arithmetic."""
        manager = MCPClientManager()
        try:
            await manager.start(self._config)

            result = await manager.call_tool(
                "test_server",
                "add",
                {"a": 3, "b": 7},
            )

            assert result["is_error"] is False
            assert len(result["content"]) > 0
        finally:
            await manager.stop()

    async def test_call_tool_get_info(self) -> None:
        """Call the get_info tool and inspect the result."""
        manager = MCPClientManager()
        try:
            await manager.start(self._config)

            result = await manager.call_tool(
                "test_server",
                "get_info",
                {},
            )

            assert result["is_error"] is False
        finally:
            await manager.stop()

    async def test_get_tool_schema(self) -> None:
        """get_tool_schema() returns the cached JSON schema synchronously."""
        manager = MCPClientManager()
        try:
            await manager.start(self._config)

            schema = manager.get_tool_schema("test_server", "echo")
            assert schema["type"] == "object"
            assert "message" in schema["properties"]

            # Unknown tool raises KeyError
            with pytest.raises(KeyError, match="nonexistent_tool"):
                manager.get_tool_schema("test_server", "nonexistent_tool")
        finally:
            await manager.stop()

    async def test_call_tool_with_invalid_arguments(self) -> None:
        """Calling a tool with wrong arguments returns isError=True."""
        manager = MCPClientManager()
        try:
            await manager.start(self._config)

            result = await manager.call_tool(
                "test_server",
                "add",
                {"a": "not_a_number"},
            )
            # FastMCP returns an error result rather than raising an exception
            assert result["is_error"] is True
        finally:
            await manager.stop()

    async def test_call_tool_on_nonexistent_server(self) -> None:
        """Calling a tool on an unknown server raises MCPConnectionError."""
        manager = MCPClientManager()
        try:
            await manager.start(self._config)

            with pytest.raises(MCPConnectionError, match="not connected"):
                await manager.call_tool("nonexistent_server", "echo", {})
        finally:
            await manager.stop()

    async def test_list_tools_on_nonexistent_server(self) -> None:
        """Listing tools on an unknown server raises MCPConnectionError."""
        manager = MCPClientManager()
        try:
            await manager.start(self._config)

            with pytest.raises(MCPConnectionError, match="not connected"):
                await manager.list_tools("nonexistent_server")
        finally:
            await manager.stop()

    async def test_full_lifecycle(self) -> None:
        """Test the full startup → discovery → invocation → shutdown cycle."""
        manager = MCPClientManager()
        try:
            await manager.start(self._config)

            # 1. Discover tools
            tools = await manager.list_tools("test_server")
            assert len(tools) == 3

            # 2. Verify schema via synchronous lookup
            schema = manager.get_tool_schema("test_server", "add")
            assert "a" in schema["properties"]
            assert "b" in schema["properties"]

            # 3. Invoke a tool
            result = await manager.call_tool(
                "test_server",
                "add",
                {"a": 10, "b": 20},
            )
            assert result["is_error"] is False

            # 4. Invoke another tool
            result2 = await manager.call_tool(
                "test_server",
                "echo",
                {"message": "done"},
            )
            assert result2["is_error"] is False
        finally:
            await manager.stop()

        # After stop, the server is no longer reachable
        assert manager._started is False
