# Code Review Summary

**Scope**: Sub-Task 7 — Langfuse Observability Integration: `src/caae/observability/langfuse_handler.py`, `src/caae/observability/__init__.py`, `src/caae/nodes/deps.py`, `src/caae/engine.py`, `src/caae/mcp/client_manager.py`, `src/caae/nodes/cognitive_processing.py`, `tests/unit/test_observability.py`.

**Overall risk**: High

**Verdict**: Request changes

## Findings

### [P0] Blocking

- **Budget enforcement cannot trigger on real LLM usage because token usage is read from the structured-output Pydantic model**
  - **Location**: `src/caae/nodes/cognitive_processing.py:109-122`
  - **Why it matters**: The node calls `structured_llm = llm.with_structured_output(schema_class)` and awaits it. The returned value is an instance of `schema_class` (a Pydantic `BaseModel`), not a LangChain `AIMessage`. Pydantic models do not carry `response_metadata`, so `hasattr(result, "response_metadata")` is `False` and `track_cost()` is always called with zero tokens. The internal session cost stays at `0.0`, so `is_within_budget()` will never detect an over-budget session in production.
  - **Evidence**: Code checks `result.response_metadata` after `with_structured_output`; LangChain's `with_structured_output` strips message metadata and returns the parsed schema object. Tests bypass this by mocking `init_chat_model` and the budget check itself.
  - **Fix**: Either (a) invoke the base LLM (`await llm.ainvoke(messages)`) to obtain the `AIMessage` with `response_metadata`, then parse it through the schema, or (b) attach a callback that captures token usage and feed the totals into `track_cost()`.

- **Budget enforcement is node-local and does not halt the session**
  - **Location**: `src/caae/nodes/cognitive_processing.py:91-101`, `src/caae/graph.py:32-37`
  - **Why it matters**: The acceptance criterion says "if cost exceeds max_session_cost_usd, the session is halted". The current check only runs immediately before the cognitive-processing LLM call. It returns `{"extracted_quantitative_data": {"error": "session cost exceeds budget"}}`, but LangGraph's linear edges still route to `action_execution`, which then runs with the error payload. The graph is not stopped.
  - **Evidence**: `graph.add_edge("cognitive_processing", "action_execution")` is unconditional. `action_execution_node` receives `extracted_quantitative_data.error` and calls tools with it.
  - **Fix**: Centralize budget tracking across all LLM calls (including `context_assessor_node`) and halt the graph by raising a dedicated `BudgetExceededError` from the node, then catch it in `CAAEEngine.run_session` and return/record a final budget-exceeded state. Alternatively, add a conditional edge after the budget check that routes to `END`.

### [P1] High

- **Manual tool-call observations are not linked to the CallbackHandler trace**
  - **Location**: `src/caae/observability/langfuse_handler.py:111-149`, `src/caae/engine.py:133-141`
  - **Why it matters**: The acceptance criterion requires "a Langfuse trace with 5 node spans + tool call spans + cost records". The engine never creates a session trace root, and `record_tool_call()` calls `self._client.start_observation()` without a `trace_context`. Each MCP tool call therefore becomes a separate orphan trace (or a trace with no parent), not a child span of the session trace produced by `LangchainCallbackHandler`.
  - **Evidence**: `LangfuseHandler.start_trace()` is defined but never invoked. `record_tool_call()` ignores its `trace` argument and does not pass `trace_context={"trace_id": ..., "parent_observation_id": ...}` to `start_observation()`.
  - **Fix**: In `CAAEEngine.run_session`, create a session trace via `self._langfuse_handler.start_trace(session_id)` and pass its `trace_id` (and optionally the root span `id`) as `trace_context` to both `LangchainCallbackHandler(trace_context=...)` and `record_tool_call()`. Update `record_tool_call()` to accept and forward `trace_context`.

- **Cost tracking only covers the cognitive-processing LLM call**
  - **Location**: `src/caae/nodes/context_assessor.py:78-85`, `src/caae/nodes/cognitive_processing.py:109-122`
  - **Why it matters**: `context_assessor_node` also invokes an LLM (`init_chat_model().with_structured_output(IntentClassification)`), but it never calls `track_cost()` or `is_within_budget()`. A session can exceed its budget during intent classification before the cognitive-processing check ever runs.
  - **Evidence**: No references to `langfuse_handler` in `context_assessor.py`.
  - **Fix**: Track token usage in `context_assessor_node` the same way as in `cognitive_processing_node` (after fixing the metadata extraction issue), and perform the budget check before any billable LLM call.

### [P2] Medium

- **`LangfuseHandler._is_configured()` ignores constructor arguments**
  - **Location**: `src/caae/observability/langfuse_handler.py:17-35,38-42`
  - **Why it matters**: If a caller passes `public_key`/`secret_key` explicitly but does not set environment variables, `_enabled` is `False` even though the underlying `Langfuse` client is configured and would emit traces.
  - **Evidence**: `_is_configured()` only reads `os.environ.get("LANGFUSE_PUBLIC_KEY")` and `LANGFUSE_SECRET_KEY`; it never inspects `self._client`'s init args.
  - **Fix**: Check the supplied `public_key`/`secret_key` arguments in addition to the environment variables, e.g. `return bool(public_key or secret_key or os.environ.get(...))`.

- **`record_tool_call()` accepts an unused `trace` parameter**
  - **Location**: `src/caae/observability/langfuse_handler.py:111-120`
  - **Why it matters**: The parameter suggests parent-trace linking is supported, but it is never used. Callers in `MCPClientManager` always pass `trace=None`, and the method creates unparented observations.
  - **Evidence**: `trace` is not referenced inside the method body.
  - **Fix**: Either remove the parameter or convert it into a `trace_context` dict and pass it to `start_observation()`.

- **Two Langfuse client instances may flush independently**
  - **Location**: `src/caae/engine.py:77,137`, `src/caae/observability/langfuse_handler.py:30`
  - **Why it matters**: `LangfuseHandler` creates a `Langfuse()` client, while `LangchainCallbackHandler()` internally calls `get_client()` which may return a singleton in simple setups. In multi-project or explicitly-constructed-client scenarios the two can diverge, leading to partial flushes or traces split across clients.
  - **Evidence**: `CallbackHandler` uses `get_client(public_key=None)`, not the `LangfuseHandler._client` instance.
  - **Fix**: Reuse a single client instance. Construct `CallbackHandler(public_key=...)` or, better, expose the handler's client to the callback if the SDK supports it; at minimum, ensure both use identical credentials and flush both on shutdown.

### [P3] Low

- **Test file has unused imports and an unused local variable**
  - **Location**: `tests/unit/test_observability.py:3,8,423`
  - **Why it matters**: `import os`, `MCPConnectionError`, and `result = await manager.call_tool(...)` are unused. `ruff check` flags them.
  - **Evidence**: `ruff check tests/unit/test_observability.py` reports `F401` and `F841`.
  - **Fix**: Remove the unused imports and variable assignment.

## Positive Notes

- The Langfuse v4 API usage in `langfuse_handler.py` (`start_observation`, `update`, `end`, `as_type="tool"`) matches the SDK signatures.
- `from langfuse.langchain import CallbackHandler` is the correct v4 import path.
- Disabled-mode no-ops are consistently guarded by `_enabled` checks in `LangfuseHandler`, and the engine only adds the LangGraph callback when `handler.enabled` is `True`.
- MCP tool-call timeout detection and latency recording are implemented correctly in `client_manager.call_tool()`.
- Pricing math in `track_cost()` matches the stated per-1k-token table, and the 22 unit tests pass (`pytest tests/unit/test_observability.py -v`).

## Suggested Next Steps

- [ ] Fix token-usage extraction so `track_cost()` receives real prompt/completion counts (P0).
- [ ] Move the budget check to a central point and halt the graph on overrun, covering both `context_assessor_node` and `cognitive_processing_node` (P0).
- [ ] Create a session trace root in `CAAEEngine.run_session` and wire `trace_context` through `LangchainCallbackHandler` and `record_tool_call()` so tool spans belong to the same trace (P1).
- [ ] Update `_is_configured()` to honor explicit constructor credentials (P2).
- [ ] Remove or use the unused `trace` parameter in `record_tool_call()` (P2).
- [ ] Clean up unused imports/variables in `tests/unit/test_observability.py` (P3).
- [ ] Re-run `pytest tests/unit/test_observability.py`, `ruff check`, and a real Langfuse smoke test to verify end-to-end traces, tool spans, and cost records.
