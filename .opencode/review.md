# CAAE Sub-Task 4 Review — MCP Client Manager & Transport Layer

## Summary

The implementation is solid overall and all 97 tests pass (38 unit + integration tests for this sub-task, plus the existing suite). The transport factory correctly adapts both `stdio` and Streamable HTTP transports, auth token injection works, and `MCPClientManager` handles per-server startup failures gracefully. Tool discovery, invocation, resource reading, and schema lookup are all implemented with clean `AsyncExitStack` lifecycle management.

The blocking issue is that the required **tool-call timeout handling** is missing: `timeout_ms` from each server config is never passed to `session.call_tool`, so a hanging tool call will block indefinitely. Because this is an explicit acceptance criterion, the sub-task needs rework before it can be considered complete.

## Issues Found

### Major

1. **Tool-call timeout is not enforced (missing acceptance criterion)**
   - **Location:** `src/caae/mcp/client_manager.py:150-187`
   - **Why it matters:** Both `StdioMCPServerConfig` and `StreamableHttpMCPServerConfig` define `timeout_ms`, and the sub-task acceptance criteria require that "if a tool call exceeds `timeout_ms`, it should be caught and reported." Right now `call_tool` ignores the value entirely.
   - **Evidence:**
     ```python
     result = await connection.session.call_tool(tool_name, arguments)
     ```
     No `read_timeout_seconds`, no `asyncio.wait_for`, and `connection` does not even store the configured `timeout_ms`.
   - **Fix:**
     1. Add `timeout_ms: int` to `_ServerConnection` and capture it during `start()`.
     2. Pass it to the SDK's timeout parameter:
        ```python
        from datetime import timedelta
        result = await connection.session.call_tool(
            tool_name,
            arguments,
            read_timeout_seconds=timedelta(milliseconds=connection.timeout_ms),
        )
        ```
     3. Catch the resulting timeout exception and surface it as an `MCPToolError` or as a result dict with `is_error=True`.

### Minor

1. **Streamable HTTP transport hides the third tuple element from the SDK**
   - **Location:** `src/caae/mcp/transport.py:47-50`
   - **Why it matters:** `mcp.client.streamable_http.streamable_http_client` yields `(read_stream, write_stream, get_session_id)`. The implementation unpacks the third value but only yields `(read, write)`, and the return type annotation says `tuple[object, object]`. The current manager only needs two values, so this is not a runtime bug, but it diverges from the sub-task spec that says the context manager should yield `(read, write, _)`.
   - **Fix:** Either yield `(read, write, _get_session_id)` and update the return type to `AsyncGenerator[tuple[object, object, object], None]`, or document that the wrapper intentionally discards the session-id callback.

2. **Hard-coded HTTP client timeout ignores `config.timeout_ms`**
   - **Location:** `src/caae/mcp/transport.py:73-77`
   - **Why it matters:** The `httpx.AsyncClient` is created with `timeout=httpx.Timeout(30, read=300)`, which has no relationship to the server configuration. This makes the configured timeout misleading for HTTP transports.
   - **Fix:** Derive the httpx timeout from `config.timeout_ms` (or at least document why these specific defaults were chosen).

3. **No test coverage for the required timeout behavior**
   - **Location:** `tests/unit/test_mcp_client_manager.py`, `tests/integration/test_mcp_integration.py`
   - **Why it matters:** Because the timeout feature is unimplemented, there is naturally no test for it. Once the feature is added, a unit test should verify that a long-running `call_tool` is cancelled/timeout-wrapped at the configured `timeout_ms`.
   - **Fix:** Add a unit test where `mock_session.call_tool` blocks longer than the configured timeout, and assert the manager raises `MCPToolError` (or returns the agreed error representation).

### Suggestions

1. **Consider raising `MCPToolError` for SDK `isError=True` results**
   - **Location:** `src/caae/mcp/client_manager.py:178-183`
   - **Why it matters:** Currently a tool that returns `isError=True` is returned as a normal dict. This is a valid design choice, but aligning with the exception type (`MCPToolError`) may be cleaner for the LangGraph node that consumes this layer. Not blocking.

2. **Tighten stream type annotations**
   - **Location:** `src/caae/mcp/transport.py:21, 50`
   - **Why it matters:** Using `tuple[object, object]` works but sacrifices static-analysis value. The MCP SDK exposes concrete `MemoryObjectReceiveStream` / `MemoryObjectSendStream` types; using them would make the contract clearer.

3. **Log successful server connection at `INFO` level is already done; keep it.**
   - This is just a note that the existing logging at `src/caae/mcp/client_manager.py:122-127` is helpful and should be retained.

## Specific File Comments

| File | Observation |
|------|-------------|
| `src/caae/mcp/models.py` | Clean `ToolInfo` dataclass. Correctly maps `tool.name`, `tool.description`, and `tool.inputSchema` from the MCP SDK `Tool` object. |
| `src/caae/mcp/transport.py` | `create_stdio_connection` builds `StdioServerParameters` correctly and wraps exceptions in `MCPConnectionError`. `create_streamable_http_connection` injects the auth token via a dedicated `httpx.AsyncClient`, and missing-token cases raise a clear error. Note the two minor issues above (return signature and hard-coded timeout). |
| `src/caae/mcp/client_manager.py` | Lifecycle is well managed with `AsyncExitStack`; per-server failures are caught and logged so startup continues. `list_tools`, `read_resource`, `get_tool_schema`, and `call_tool` are all present and correctly delegate to the cached `ClientSession`. The major gap is the missing `timeout_ms` enforcement in `call_tool`. |
| `src/caae/mcp/__init__.py` | Correctly re-exports `MCPClientManager`, `MCPToolError`, `ToolInfo`, `MCPConnectionError`, and the two transport factory functions. |
| `pyproject.toml` | The new `integration` pytest marker is registered correctly. |
| `tests/unit/test_mcp_client_manager.py` | Thorough unit coverage with mocked sessions: stdio params, auth header injection, missing auth token, unknown servers/tools, connection failures, async context manager protocol, and schema lookup. All 29 unit tests pass. Add a timeout test once the feature is implemented. |
| `tests/integration/test_mcp_integration.py` | Spawns a real FastMCP server via stdio and verifies discovery, invocation, schema lookup, and shutdown. All 9 integration tests pass. Does not cover HTTP transport auth end-to-end, but that is appropriately tested at the unit level. |

## Verdict

**Needs rework.**

The code is well structured and the test suite is green, but the required tool-call timeout handling is not implemented. Once `call_tool` respects `config.timeout_ms` and a corresponding test is added, this sub-task will be ready to merge.
