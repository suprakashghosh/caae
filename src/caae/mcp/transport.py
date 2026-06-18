"""MCP transport layer — stdio and streamable HTTP adapters."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from caae.models.config import StdioMCPServerConfig, StreamableHttpMCPServerConfig


class MCPConnectionError(Exception):
    """Raised when an MCP connection cannot be established or maintained."""


@asynccontextmanager
async def create_stdio_connection(
    config: StdioMCPServerConfig,
) -> AsyncGenerator[tuple[object, object], None]:
    """Create a stdio transport connection to an MCP server.

    Yields (read_stream, write_stream) from mcp.client.stdio.stdio_client.

    Args:
        config: The stdio server configuration.

    Yields:
        A tuple of (read_stream, write_stream) for bidirectional communication.

    Raises:
        MCPConnectionError: If the connection cannot be established.
    """
    params = StdioServerParameters(
        command=config.command,
        args=config.args,
        env=config.env,
    )
    try:
        async with stdio_client(params) as (read, write):
            yield (read, write)
    except Exception as e:
        raise MCPConnectionError(f"Failed to create stdio connection to '{config.command}': {e}") from e


@asynccontextmanager
async def create_streamable_http_connection(
    config: StreamableHttpMCPServerConfig,
) -> AsyncGenerator[tuple[object, object], None]:
    """Create a Streamable HTTP transport connection to an MCP server.

    Yields (read_stream, write_stream) from
    mcp.client.streamable_http.streamable_http_client.
    Handles auth token injection via httpx.AsyncClient if env_auth_token_key is set.

    Args:
        config: The streamable HTTP server configuration.

    Yields:
        A tuple of (read_stream, write_stream) for bidirectional communication.

    Raises:
        MCPConnectionError: If the connection cannot be established, or if the
            auth token env var is set but missing.
    """
    url = config.endpoint

    if config.env_auth_token_key:
        token = os.environ.get(config.env_auth_token_key)
        if not token:
            raise MCPConnectionError(f"Auth token env var '{config.env_auth_token_key}' not set")
        http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(config.timeout_ms / 1000.0),
            follow_redirects=True,
        )
        try:
            async with http_client:
                async with streamable_http_client(url, http_client=http_client) as (read, write, _):
                    yield (read, write)
        except Exception as e:
            raise MCPConnectionError(f"Failed to create streamable HTTP connection to '{url}': {e}") from e
    else:
        try:
            async with streamable_http_client(url) as (read, write, _):
                yield (read, write)
        except Exception as e:
            raise MCPConnectionError(f"Failed to create streamable HTTP connection to '{url}': {e}") from e
