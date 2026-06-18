"""MCP client manager — manages connections to MCP servers."""

import asyncio
import logging
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession

from caae.mcp.models import ToolInfo
from caae.mcp.transport import (
    MCPConnectionError,
    create_stdio_connection,
    create_streamable_http_connection,
)
from caae.models.config import MCPConfig

logger = logging.getLogger(__name__)


class MCPToolError(Exception):
    """Raised when an MCP tool call fails."""


@dataclass
class _ServerConnection:
    """Internal state for a connected MCP server."""

    session: ClientSession
    tools: list[ToolInfo] = field(default_factory=list)
    timeout_ms: int = 5000


class MCPClientManager:
    """Manages MCP client connections, tool discovery, and tool invocation.

    Usage::

        manager = MCPClientManager()
        await manager.start(config)
        try:
            tools = await manager.list_tools("my_server")
            result = await manager.call_tool("my_server", "echo", {"message": "hi"})
        finally:
            await manager.stop()

    Or as an async context manager::

        async with MCPClientManager() as manager:
            await manager.start(config)
            ...
    """

    def __init__(self) -> None:
        self._servers: dict[str, _ServerConnection] = {}
        self._exit_stack: AsyncExitStack | None = None
        self._started: bool = False
        self._langfuse_handler: Any = None  # Optional LangfuseHandler
        self._trace_context: dict[str, Any] | None = None

    @property
    def connected_server_names(self) -> list[str]:
        """Return names of currently connected MCP servers (synchronous)."""
        return list(self._servers.keys())

    def set_langfuse_handler(self, handler: Any, trace_context: dict[str, Any] | None = None) -> None:
        """Set the Langfuse handler for tool call tracing.

        Args:
            handler: A LangfuseHandler instance (or None to clear).
            trace_context: Optional trace identifiers to attach to tool observations.
        """
        self._langfuse_handler = handler
        self._trace_context = trace_context

    async def start(self, config: MCPConfig) -> None:
        """Initialize connections to all configured MCP servers.

        For each server in *config.mcp_servers*:
        1. Create the appropriate transport (stdio or streamable_http)
        2. Open a ClientSession
        3. Call ``session.initialize()``
        4. Call ``session.list_tools()`` and cache the results

        If a particular server fails to connect, a warning is logged and
        startup continues with the remaining servers.

        Args:
            config: The MCP configuration describing which servers to connect to.

        Raises:
            RuntimeError: If the manager is already started.
        """
        if self._started:
            raise RuntimeError("MCPClientManager is already started")

        self._exit_stack = AsyncExitStack()
        self._servers = {}

        for server_name, server_config in config.mcp_servers.items():
            try:
                read: object
                write: object

                if server_config.transport == "stdio":
                    read, write = await self._exit_stack.enter_async_context(
                        create_stdio_connection(server_config)  # type: ignore[arg-type]
                    )
                elif server_config.transport == "streamable_http":
                    read, write = await self._exit_stack.enter_async_context(
                        create_streamable_http_connection(server_config)  # type: ignore[arg-type]
                    )
                else:
                    logger.warning(
                        "Unknown transport '%s' for server '%s', skipping",
                        server_config.transport,
                        server_name,
                    )
                    continue

                session = await self._exit_stack.enter_async_context(
                    ClientSession(read, write)  # type: ignore[arg-type]
                )
                await session.initialize()

                tools_result = await session.list_tools()
                tools = [
                    ToolInfo(
                        name=tool.name,
                        description=tool.description,
                        input_schema=tool.inputSchema,
                    )
                    for tool in tools_result.tools
                ]

                self._servers[server_name] = _ServerConnection(
                    session=session,
                    tools=tools,
                    timeout_ms=server_config.timeout_ms,
                )

                logger.info(
                    "Connected to MCP server '%s' (%s, %d tools)",
                    server_name,
                    server_config.transport,
                    len(tools),
                )
            except Exception:
                logger.warning(
                    "Failed to connect to MCP server '%s'",
                    server_name,
                    exc_info=True,
                )

        self._started = True

    async def stop(self) -> None:
        """Gracefully shut down all client sessions and transport connections."""
        self._started = False
        self._servers = {}

        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:
                logger.warning("Error during MCP shutdown", exc_info=True)
            finally:
                self._exit_stack = None

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        """Invoke a tool on a specific MCP server.

        Args:
            server_name: The name of the server (as configured in MCPConfig).
            tool_name: The name of the tool to invoke.
            arguments: The arguments to pass to the tool.

        Returns:
            A dict with keys:
            - ``'content'``: list of content blocks
            - ``'is_error'``: bool
            - ``'structured_content'``: any (optional)

        Raises:
            MCPConnectionError: If the server is not connected.
            MCPToolError: If the tool call fails.
        """
        connection = self._servers.get(server_name)
        if connection is None:
            raise MCPConnectionError(f"Server '{server_name}' is not connected")

        start_time = time.monotonic()

        try:
            result = await asyncio.wait_for(
                connection.session.call_tool(tool_name, arguments),
                timeout=connection.timeout_ms / 1000.0,  # Convert ms to seconds
            )
            latency_ms = (time.monotonic() - start_time) * 1000

            return_dict = {
                "content": result.content,
                "is_error": result.isError,
                "structured_content": result.structuredContent,
            }

            # Record in Langfuse if handler is set
            if self._langfuse_handler is not None:
                lh = self._langfuse_handler
                if lh.enabled:
                    lh.record_tool_call(
                        trace=None,
                        server_name=server_name,
                        tool_name=tool_name,
                        arguments=arguments,
                        result=return_dict,
                        latency_ms=latency_ms,
                        is_error=result.isError,
                        trace_context=self._trace_context,
                    )

            return return_dict
        except TimeoutError:
            latency_ms = (time.monotonic() - start_time) * 1000
            if self._langfuse_handler is not None and self._langfuse_handler.enabled:
                self._langfuse_handler.record_tool_call(
                    trace=None,
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=None,
                    latency_ms=latency_ms,
                    is_error=True,
                    trace_context=self._trace_context,
                )
            raise MCPToolError(
                f"Tool call '{tool_name}' on server '{server_name}' timed out after {connection.timeout_ms}ms"
            ) from None
        except MCPToolError:
            raise
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            if self._langfuse_handler is not None and self._langfuse_handler.enabled:
                self._langfuse_handler.record_tool_call(
                    trace=None,
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=None,
                    latency_ms=latency_ms,
                    is_error=True,
                    trace_context=self._trace_context,
                )
            raise MCPToolError(f"Failed to call tool '{tool_name}' on server '{server_name}': {e}") from e

    async def list_tools(self, server_name: str) -> list[ToolInfo]:
        """List available tools for a specific MCP server.

        Returns cached tool info if the server was connected during ``start()``.

        Args:
            server_name: The name of the server.

        Returns:
            A list of ToolInfo objects describing the available tools.

        Raises:
            MCPConnectionError: If the server is not connected.
        """
        connection = self._servers.get(server_name)
        if connection is None:
            raise MCPConnectionError(f"Server '{server_name}' is not connected")
        return list(connection.tools)

    async def read_resource(self, server_name: str, uri: str) -> dict:
        """Read a resource from a specific MCP server.

        Args:
            server_name: The name of the server.
            uri: The resource URI to read.

        Returns:
            A dict with key ``'contents'`` containing the resource contents.

        Raises:
            MCPConnectionError: If the server is not connected.
        """
        connection = self._servers.get(server_name)
        if connection is None:
            raise MCPConnectionError(f"Server '{server_name}' is not connected")

        result = await connection.session.read_resource(uri)
        return {"contents": result.contents}

    def get_tool_schema(self, server_name: str, tool_name: str) -> dict:
        """Get the JSON Schema for a specific tool (synchronous, cached lookup).

        Args:
            server_name: The name of the server.
            tool_name: The name of the tool.

        Returns:
            The JSON Schema dict describing the tool's expected arguments.

        Raises:
            MCPConnectionError: If the server is not connected.
            KeyError: If the tool is not found on the server.
        """
        connection = self._servers.get(server_name)
        if connection is None:
            raise MCPConnectionError(f"Server '{server_name}' is not connected")

        for tool in connection.tools:
            if tool.name == tool_name:
                return tool.input_schema

        raise KeyError(f"Tool '{tool_name}' not found on server '{server_name}'")

    async def __aenter__(self) -> "MCPClientManager":
        """Enter async context manager.

        The manager must be started separately via ``start()``.
        """
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context manager — calls ``stop()``."""
        await self.stop()
