"""Tests for MCP client manager, transport layer, and models."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from caae.mcp import (
    MCPClientManager,
    MCPConnectionError,
    MCPToolError,
    ToolInfo,
    create_stdio_connection,
    create_streamable_http_connection,
)
from caae.models.config import (
    MCPConfig,
    StdioMCPServerConfig,
    StreamableHttpMCPServerConfig,
)

# ── Tests: ToolInfo model ───────────────────────────────────────────────────


class TestToolInfo:
    """Tests for the ToolInfo dataclass."""

    def test_defaults(self) -> None:
        """Basic ToolInfo with only name works."""
        info = ToolInfo(name="my_tool")
        assert info.name == "my_tool"
        assert info.description is None
        assert info.input_schema == {}

    def test_full(self) -> None:
        """ToolInfo with all fields."""
        info = ToolInfo(
            name="add",
            description="Adds two numbers",
            input_schema={"type": "object", "properties": {"a": {"type": "integer"}}},
        )
        assert info.name == "add"
        assert info.description == "Adds two numbers"
        assert info.input_schema["type"] == "object"

    def test_description_none(self) -> None:
        """ToolInfo allows None description."""
        info = ToolInfo(name="t", description=None)
        assert info.description is None


# ── Tests: Transport — Stdio ────────────────────────────────────────────────


class TestCreateStdioConnection:
    """Tests for create_stdio_connection."""

    async def test_creates_correct_params(self) -> None:
        """Verify StdioServerParameters is built from config."""
        config = StdioMCPServerConfig(
            command="uv",
            args=["run", "my-server"],
            env={"KEY": "val"},
        )

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = (MagicMock(), MagicMock())

        with patch("caae.mcp.transport.stdio_client", return_value=mock_cm) as mock_stdio:
            async with create_stdio_connection(config) as (read, write):
                assert read is not None
                assert write is not None

            # Check that stdio_client was called with correct params
            mock_stdio.assert_called_once()
            call_args = mock_stdio.call_args[0][0]
            assert call_args.command == "uv"
            assert call_args.args == ["run", "my-server"]
            assert call_args.env == {"KEY": "val"}

    async def test_default_env_is_none(self) -> None:
        """When config has no env, StdioServerParameters gets env=None."""
        config = StdioMCPServerConfig(command="python", args=["-c", "pass"])

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = (MagicMock(), MagicMock())

        with patch("caae.mcp.transport.stdio_client", return_value=mock_cm) as mock_stdio:
            async with create_stdio_connection(config):
                pass

            call_args = mock_stdio.call_args[0][0]
            assert call_args.env is None

    async def test_wraps_exception(self) -> None:
        """If stdio_client raises, MCPConnectionError is raised."""
        config = StdioMCPServerConfig(command="nonexistent")

        with patch(
            "caae.mcp.transport.stdio_client",
            side_effect=Exception("boom"),
        ):
            with pytest.raises(MCPConnectionError, match="Failed to create stdio"):
                async with create_stdio_connection(config):
                    pass  # pragma: no cover


# ── Tests: Transport — Streamable HTTP ──────────────────────────────────────


class TestCreateStreamableHttpConnection:
    """Tests for create_streamable_http_connection."""

    async def test_passes_correct_url(self) -> None:
        """Verify the URL is passed to streamable_http_client."""
        config = StreamableHttpMCPServerConfig(
            endpoint="https://example.com/mcp",
        )

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = (MagicMock(), MagicMock(), MagicMock())

        with patch(
            "caae.mcp.transport.streamable_http_client",
            return_value=mock_cm,
        ) as mock_client:
            async with create_streamable_http_connection(config):
                pass

            mock_client.assert_called_once_with("https://example.com/mcp")

    async def test_with_auth_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When env_auth_token_key is set, reads env var and sends auth header."""
        monkeypatch.setenv("MY_SERVICE_TOKEN", "secret123")
        config = StreamableHttpMCPServerConfig(
            endpoint="https://api.example.com/mcp",
            env_auth_token_key="MY_SERVICE_TOKEN",
        )

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = (MagicMock(), MagicMock(), MagicMock())

        real_async_client = httpx.AsyncClient

        with patch(
            "caae.mcp.transport.streamable_http_client",
            return_value=mock_cm,
        ) as mock_client:
            with patch(
                "caae.mcp.transport.httpx.AsyncClient",
                wraps=real_async_client,
            ) as mock_httpx_cls:
                async with create_streamable_http_connection(config):
                    pass

                # Verify httpx.AsyncClient was created with auth header
                mock_httpx_cls.assert_called_once()
                call_kwargs = mock_httpx_cls.call_args.kwargs
                assert call_kwargs["headers"] == {"Authorization": "Bearer secret123"}
                assert call_kwargs["follow_redirects"] is True

                # Verify streamable_http_client was called with the client
                mock_client.assert_called_once()
                client_arg = mock_client.call_args.kwargs.get("http_client")
                assert client_arg is not None

    async def test_auth_token_missing_raises_error(self) -> None:
        """If env_auth_token_key is set but env var is missing, raises MCPConnectionError."""
        config = StreamableHttpMCPServerConfig(
            endpoint="https://api.example.com/mcp",
            env_auth_token_key="MISSING_TOKEN_VAR",
        )

        with pytest.raises(
            MCPConnectionError,
            match="Auth token env var 'MISSING_TOKEN_VAR' not set",
        ):
            async with create_streamable_http_connection(config):
                pass  # pragma: no cover

    async def test_without_auth_token(self) -> None:
        """Without auth token, streamable_http_client is called with just the URL."""
        config = StreamableHttpMCPServerConfig(
            endpoint="https://api.example.com/mcp",
        )

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = (MagicMock(), MagicMock(), MagicMock())

        with patch(
            "caae.mcp.transport.streamable_http_client",
            return_value=mock_cm,
        ) as mock_client:
            async with create_streamable_http_connection(config):
                pass

            # Should be called with url positional, no http_client
            mock_client.assert_called_once_with("https://api.example.com/mcp")

    async def test_wraps_exception(self) -> None:
        """If streamable_http_client raises, MCPConnectionError is raised."""
        config = StreamableHttpMCPServerConfig(endpoint="https://example.com/mcp")

        with patch(
            "caae.mcp.transport.streamable_http_client",
            side_effect=Exception("network error"),
        ):
            with pytest.raises(
                MCPConnectionError,
                match="Failed to create streamable HTTP",
            ):
                async with create_streamable_http_connection(config):
                    pass  # pragma: no cover


# ── Fixtures: Mocked MCP SDK ────────────────────────────────────────────────


@pytest.fixture
def mock_tool() -> MagicMock:
    """Create a mock MCP Tool object."""
    tool = MagicMock()
    tool.name = "echo"
    tool.description = "Echoes a message"
    tool.inputSchema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }
    return tool


@pytest.fixture
def mock_tool_add() -> MagicMock:
    """Create a second mock MCP Tool object."""
    tool = MagicMock()
    tool.name = "add"
    tool.description = "Adds two numbers"
    tool.inputSchema = {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    }
    return tool


@pytest.fixture
def mock_session(mock_tool: MagicMock, mock_tool_add: MagicMock) -> AsyncMock:
    """Create a mock ClientSession.

    The mock supports the async context manager protocol: __aenter__ returns
    *self* so that ``enter_async_context(session_mock)`` yields the same mock.
    """
    session = AsyncMock()

    # Async context manager protocol — return self on enter
    session.__aenter__.return_value = session

    # Mock list_tools result
    tools_result = MagicMock()
    tools_result.tools = [mock_tool, mock_tool_add]
    session.list_tools = AsyncMock(return_value=tools_result)

    # Mock call_tool result
    call_result = MagicMock()
    call_result.content = [MagicMock()]
    call_result.isError = False
    call_result.structuredContent = {"result": "hello"}
    session.call_tool = AsyncMock(return_value=call_result)

    # Mock read_resource result
    resource_result = MagicMock()
    resource_result.contents = [MagicMock()]
    session.read_resource = AsyncMock(return_value=resource_result)

    return session


@pytest.fixture
def mock_transport_cm() -> AsyncMock:
    """Create a mock async context manager for transport that yields (read, write)."""
    cm = AsyncMock()
    cm.__aenter__.return_value = (MagicMock(), MagicMock())
    return cm


@pytest.fixture
def mcp_config() -> MCPConfig:
    """Create a test MCPConfig with two servers."""
    return MCPConfig(
        system_mode="testing",
        active_environment="test",
        mcp_servers={
            "server_a": StdioMCPServerConfig(
                command="python",
                args=["-m", "server_a"],
            ),
            "server_b": StreamableHttpMCPServerConfig(
                endpoint="https://server-b.example.com/mcp",
            ),
        },
    )


@pytest.fixture
def manager() -> MCPClientManager:
    """Create a fresh MCPClientManager."""
    return MCPClientManager()


# ── Tests: MCPClientManager ─────────────────────────────────────────────────


class TestMCPClientManagerStart:
    """Tests for MCPClientManager.start()."""

    async def test_connects_to_all_servers(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """start() connects to all configured servers."""
        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        assert manager._started is True
        assert len(manager._servers) == 2
        assert "server_a" in manager._servers
        assert "server_b" in manager._servers

        # Each session should have been initialized and listed tools
        assert mock_session.initialize.call_count == 2
        assert mock_session.list_tools.call_count == 2

    async def test_continues_on_failure(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """If one server fails to connect, start() continues with remaining servers."""

        # Make the second transport fail
        def failing_transport():
            raise Exception("Connection refused")
            yield  # pragma: no cover

        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                side_effect=failing_transport(),
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        # Only server_a should be connected
        assert len(manager._servers) == 1
        assert "server_a" in manager._servers
        assert "server_b" not in manager._servers

    async def test_raises_if_already_started(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """Starting an already-started manager raises RuntimeError."""
        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        with pytest.raises(RuntimeError, match="already started"):
            await manager.start(mcp_config)

    async def test_caches_tools(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """start() caches the tool list for each server."""
        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        conn = manager._servers["server_a"]
        assert len(conn.tools) == 2
        assert conn.tools[0].name == "echo"
        assert conn.tools[1].name == "add"


class TestMCPClientManagerStop:
    """Tests for MCPClientManager.stop()."""

    async def test_stop_clears_servers(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """stop() clears all server connections."""
        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        await manager.stop()

        assert manager._started is False
        assert len(manager._servers) == 0

    async def test_stop_graceful_on_error(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """stop() does not raise even if shutdown fails."""
        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        # Force the exit stack to fail on close
        if manager._exit_stack is not None:
            manager._exit_stack.aclose = AsyncMock(side_effect=Exception("shutdown error"))

        # Should not raise
        await manager.stop()
        assert manager._started is False


class TestMCPClientManagerListTools:
    """Tests for MCPClientManager.list_tools()."""

    async def test_returns_cached_tools(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """list_tools() returns cached tool list."""
        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        tools = await manager.list_tools("server_a")
        assert len(tools) == 2
        assert tools[0].name == "echo"
        assert tools[1].name == "add"

    async def test_raises_for_unknown_server(
        self,
        manager: MCPClientManager,
    ) -> None:
        """list_tools() raises MCPConnectionError for unknown server."""
        with pytest.raises(MCPConnectionError, match="not connected"):
            await manager.list_tools("nonexistent")


class TestMCPClientManagerCallTool:
    """Tests for MCPClientManager.call_tool()."""

    async def test_delegates_to_session(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """call_tool() delegates to the session and returns the result."""
        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        result = await manager.call_tool("server_a", "echo", {"message": "hello"})

        mock_session.call_tool.assert_called_once_with(
            "echo",
            {"message": "hello"},
        )
        assert result["is_error"] is False
        assert result["structured_content"] == {"result": "hello"}
        assert len(result["content"]) == 1

    async def test_raises_for_unknown_server(
        self,
        manager: MCPClientManager,
    ) -> None:
        """call_tool() raises MCPConnectionError for unknown server."""
        with pytest.raises(MCPConnectionError, match="not connected"):
            await manager.call_tool("nonexistent", "tool", {})

    async def test_wraps_session_error(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """call_tool() wraps session errors in MCPToolError."""
        mock_session.call_tool = AsyncMock(side_effect=ValueError("internal error"))

        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        with pytest.raises(MCPToolError, match="Failed to call tool"):
            await manager.call_tool("server_a", "echo", {"message": "hi"})

    async def test_timeout_raises_mcp_tool_error(
        self,
        manager: MCPClientManager,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """call_tool() raises MCPToolError when the tool call times out."""
        mock_session.call_tool = AsyncMock(side_effect=TimeoutError())

        # Create a config with a small timeout so the test is fast
        config = MCPConfig(
            system_mode="testing",
            active_environment="test",
            mcp_servers={
                "server_a": StdioMCPServerConfig(
                    command="python",
                    args=["-m", "server_a"],
                    timeout_ms=100,
                ),
            },
        )

        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(config)

        with pytest.raises(
            MCPToolError,
            match="timed out",
        ):
            await manager.call_tool("server_a", "echo", {"message": "hi"})


class TestMCPClientManagerReadResource:
    """Tests for MCPClientManager.read_resource()."""

    async def test_delegates_to_session(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """read_resource() delegates to session and returns contents."""
        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        result = await manager.read_resource("server_a", "resource://path")

        mock_session.read_resource.assert_called_once_with("resource://path")
        assert "contents" in result

    async def test_raises_for_unknown_server(
        self,
        manager: MCPClientManager,
    ) -> None:
        """read_resource() raises MCPConnectionError for unknown server."""
        with pytest.raises(MCPConnectionError, match="not connected"):
            await manager.read_resource("nonexistent", "resource://path")


class TestMCPClientManagerGetToolSchema:
    """Tests for MCPClientManager.get_tool_schema()."""

    async def test_returns_schema_for_known_tool(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """get_tool_schema() returns input_schema for a known tool."""
        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        schema = manager.get_tool_schema("server_a", "echo")
        assert schema["type"] == "object"
        assert "message" in schema["properties"]
        assert schema["required"] == ["message"]

    async def test_raises_keyerror_for_unknown_tool(
        self,
        manager: MCPClientManager,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """get_tool_schema() raises KeyError for unknown tool."""
        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            await manager.start(mcp_config)

        with pytest.raises(KeyError, match="unknown_tool"):
            manager.get_tool_schema("server_a", "unknown_tool")

    async def test_raises_for_unknown_server(
        self,
        manager: MCPClientManager,
    ) -> None:
        """get_tool_schema() raises MCPConnectionError for unknown server."""
        with pytest.raises(MCPConnectionError, match="not connected"):
            manager.get_tool_schema("nonexistent", "tool")

    def test_synchronous(self) -> None:
        """get_tool_schema() is a synchronous method."""
        # Just verifying the method is not a coroutine
        manager = MCPClientManager()
        assert not hasattr(manager.get_tool_schema, "__await__")


# ── Tests: Async context manager protocol ───────────────────────────────────


class TestMCPClientManagerAsyncContextManager:
    """Tests for MCPClientManager as an async context manager."""

    async def test_aenter_aexit(
        self,
        mcp_config: MCPConfig,
        mock_session: AsyncMock,
        mock_transport_cm: AsyncMock,
    ) -> None:
        """__aenter__ returns self, __aexit__ calls stop()."""
        manager = MCPClientManager()

        with (
            patch(
                "caae.mcp.client_manager.create_stdio_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.create_streamable_http_connection",
                return_value=mock_transport_cm,
            ),
            patch(
                "caae.mcp.client_manager.ClientSession",
                return_value=mock_session,
            ),
        ):
            async with manager as mgr:
                assert mgr is manager
                assert manager._started is False  # start not called yet
                await manager.start(mcp_config)
                assert manager._started is True

        # After exiting context, manager should be stopped
        assert manager._started is False
        assert len(manager._servers) == 0
