# Plan: Central Adaptable Automation Engine (CAAE) — Full Implementation

## Objective

Build the CAAE runtime from the ground up: an enterprise-grade, data-agnostic multi-agent orchestration engine that decouples cognitive orchestration logic from data ingestion and operational mutation, using MCP (Model Context Protocol) for tool discovery/invocation, LangGraph for the state machine, FastAPI as the host process, Pydantic for validation, Langfuse for observability, and DeepEval for CI regression gates.

## Requirements Snapshot

- **R1:** Data-agnostic core runtime — zero industry-specific knowledge in engine code; capabilities discovered dynamically via MCP JSON-RPC 2.0 negotiations
- **R2:** 5-node LangGraph cyclic state machine (Context Assessor → Info Retrieval → Cognitive Processing → Action Execution → Evaluation Gate) with loop-back on validation failure
- **R3:** MCP Host application that manages MCP Client lifecycle (stdio + Streamable HTTP transports), tool discovery, and tool invocation
- **R4:** Declarative configuration via `mcp_config.json` (server connections, transport, auth) and `workflow_policy.json` (intent routing, schema contracts, constraints)
- **R5:** Strict Pydantic data contracts for all state transit (UnifiedContextState, tool invocation payloads)
- **R6:** Deterministic guardrails — reject open-ended LLM loops; use bounded Pydantic validation schemas and automated regression gates
- **R7:** Bi-directional modality — both Information Synthesis and Transactional Operational workflows
- **R8:** Langfuse telemetry (per-session traces, span monitoring, cost tracking)
- **R9:** DeepEval CI/CD regression gate (Groundedness ≥ 0.95, Conversational Relevancy ≥ 0.90, Schema Adherence 100%)
- **R10:** Provider-agnostic LLM integration via LangChain ChatModel abstraction (swappable via config)
- **R11:** FastAPI host process as supervisor (lifecycle management, health checks, API surface)
- **R12:** Demo MCP server as a reference implementation (in-memory scheduling/CRM mock)

## Scope

- Core engine: LangGraph state machine, MCP client manager, config loader, state models
- FastAPI host application with REST API for session management
- MCP Client mapping layer (stdio + Streamable HTTP transports)
- One functional demo MCP server (scheduling mock with tools: `check_availability`, `book_slot`, `list_appointments`)
- Declarative configuration system (`mcp_config.json`, `workflow_policy.json`)
- Pydantic validation pipeline
- Langfuse tracing integration
- DeepEval regression test suite
- Comprehensive unit + integration tests
- Project scaffolding (uv, pyproject.toml, src layout, CI)

## Assumptions and Constraints

- Python 3.12+ target
- `uv` as package manager with `pyproject.toml` (src layout: `src/caae/`)
- LangGraph for graph orchestration; LangChain-core for LLM abstraction
- MCP Python SDK (`mcp` package) for client/server implementations
- LLM provider selection via `mcp_config.json` or env var (default: OpenAI, but swappable)
- Demo server runs locally via stdio transport for development
- First pass does not include a web UI — API-only
- `workflow_policy.json` intent routing is loaded at startup and not hot-reloaded in V1
- Human-handoff escalation is a state exit (logged, not a full handoff UI in V1)

## Risks and Areas Requiring Care

1. **MCP SDK transport stability** — The Streamable HTTP transport is the spec's recommended production transport; stdio is more battle-tested. Start with stdio for the demo server.
2. **LangGraph state mutation patterns** — Each node must return a dict of state updates, not the full state object. Pydantic models used as schema but nodes return partial dicts.
3. **LLM tool-call format mapping** — LangChain's tool-calling abstraction must align with MCP tool schemas discovered at runtime. Need a mapping layer.
4. **Cyclic graph guardrails** — The evaluation gate loop-back must have a hard ceiling (max 3 retries per spec) to prevent infinite loops.
5. **Async coordination** — FastAPI is async, MCP clients are async, LangGraph supports async. All three must coordinate in the same event loop.
6. **Schema contract resolution** — `workflow_policy.json` references schemas like `schemas.clinical.AppointmentBookingPayload`. Need a schema registry that resolves these dotted paths at runtime.

## Decisions Log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Demo MCP server as **separate package** (`src/caae_demo_server/`) | Mirrors spec's "isolated micro-service" architecture; keeps core engine industry-agnostic; serves as a template for real MCP servers |
| D2 | Anti-slop handled via **prompts + structured output + schema validation** in V1 | No custom anti-slop pipeline; Node 3 enforces quality through prompt engineering and output constraints |
| D3 | HTTP transport uses **Streamable HTTP** (`streamable_http_client`), not SSE | MCP spec deprecated HTTP+SSE (2024-11-05); SDK recommends `streamable-http` for production; config uses `"streamable_http"` transport type |

## Core Concepts

### LangGraph StateGraph with Cyclic Flow

```python
from langgraph.graph import StateGraph, START, END
from caae.models.state import UnifiedContextState

builder = StateGraph(UnifiedContextState)

# 5 nodes from the spec
builder.add_node("context_assessor", context_assessor_node)
builder.add_node("info_retrieval", info_retrieval_node)
builder.add_node("cognitive_processing", cognitive_processing_node)
builder.add_node("action_execution", action_execution_node)
builder.add_node("evaluation_gate", evaluation_gate_node)

# Linear flow
builder.add_edge(START, "context_assessor")
builder.add_edge("context_assessor", "info_retrieval")
builder.add_edge("info_retrieval", "cognitive_processing")
builder.add_edge("cognitive_processing", "action_execution")
builder.add_edge("action_execution", "evaluation_gate")

# Conditional: pass → exit, fail → loop back
builder.add_conditional_edges(
    "evaluation_gate",
    verify_output_compliance,
    {
        "commit_state_and_exit": END,
        "re_evaluate_context_node": "context_assessor",
        "human_handoff_escalation": END,
    },
)
```

### MCP Client Lifecycle

```python
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

async def connect_to_server(config: MCPServerConfig):
    if config.transport == "stdio":
        params = StdioServerParameters(command=config.command, args=config.args, env=config.env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                # ... use session
```

### Provider-Agnostic LLM

```python
from langchain.chat_models import init_chat_model

# Config-driven: model name maps to provider automatically
llm = init_chat_model("gpt-4o", model_provider="openai")
# OR
llm = init_chat_model("claude-sonnet-4-20250514", model_provider="anthropic")
```

## Sub-Tasks

### Sub-Task 1: Project Scaffolding & Dependencies

- **Status:** Completed
- **Objective:** Set up the project structure, package manager, dependencies, and CI skeleton so all subsequent work has a runnable foundation.
- **Related Requirements:** R1, R10, R11
- **Dependencies and Preconditions:** None (first task)
- **In Scope for This Sub-Task:**
  - Initialize uv project with `pyproject.toml` and src layout (`src/caae/`)
  - Define all dependencies: `fastapi`, `uvicorn`, `langgraph`, `langchain-core`, `langchain-openai`, `langchain-anthropic`, `mcp`, `pydantic`, `langfuse`, `deepeval`, `httpx`, `python-dotenv`
  - Define dev dependencies: `pytest`, `pytest-asyncio`, `ruff`, `mypy`
  - Create package structure:
    ```
    src/caae/
        __init__.py
        main.py              # FastAPI app entry
        models/
            __init__.py
            state.py         # UnifiedContextState
            config.py        # mcp_config / workflow_policy Pydantic models
            schemas/         # Schema registry (placeholder)
                __init__.py
        nodes/
            __init__.py
            context_assessor.py
            info_retrieval.py
            cognitive_processing.py
            action_execution.py
            evaluation_gate.py
        mcp/
            __init__.py
            client_manager.py    # MCP client lifecycle manager
            transport.py         # Transport factory (stdio/HTTP)
        config/
            __init__.py
            loader.py            # JSON config loader
        observability/
            __init__.py
            langfuse_handler.py  # Langfuse tracing
        validation/
            __init__.py
            schema_checker.py    # Pydantic schema validation
            deepeval_gates.py    # DeepEval regression gates
        api/
            __init__.py
            routes/
                __init__.py
                sessions.py      # Session management endpoints
                health.py        # Health check
    configs/
        mcp_config.json
        workflow_policy.json
    tests/
        conftest.py
        unit/
        integration/
    ```
  - Create `configs/mcp_config.json` and `configs/workflow_policy.json` with the examples from the spec
  - Create `.env.example` with required env vars (LLM API keys, Langfuse keys)
  - Wire up `ruff` config in `pyproject.toml`
  - Create a minimal FastAPI app in `main.py` with a health endpoint
  - Verify `uv run python -m caae.main` starts and `/health` returns 200
- **Out of Scope for This Sub-Task:** Any business logic, graph wiring, MCP connections
- **Instructions:**
  1. Run `uv init --package` in project root, then restructure to src layout
  2. Add all deps to `pyproject.toml` under `[project.dependencies]` and `[project.optional-dependencies]`
  3. Create all directories and `__init__.py` files listed above
  4. Write placeholder `main.py` with FastAPI app + `/health` endpoint
  5. Create config JSON files from spec examples
  6. Run `uv sync` to verify dependency resolution
- **Acceptance Criteria:**
  - `uv sync` completes without errors
  - `uv run python -c "import caae"` works
  - `uv run uvicorn caae.main:app --port 8000` starts, and `GET /health` returns `{"status": "ok"}`
- **Cautionary Points:**
  - Use compatible version ranges: `langgraph>=0.2`, `mcp>=1.0`, `fastapi>=0.115`
  - `langchain-openai` and `langchain-anthropic` are optional extras (user may only have one API key)
- **Testing Suggestions:** `uv run pytest tests/ -v` (empty test suite passes); manual curl to `/health`
- **Done When:** Project scaffolding is complete, dependencies resolve, FastAPI health endpoint works

---

### Sub-Task 2: Configuration Models & Loader

- **Status:** Completed
- **Objective:** Implement Pydantic models for `mcp_config.json` and `workflow_policy.json`, and a loader that reads them from disk into validated objects.
- **Related Requirements:** R4
- **Dependencies and Preconditions:** Sub-Task 1 (project structure exists)
- **In Scope for This Sub-Task:**
  - `src/caae/models/config.py` — Pydantic models:
    - `MCPServerConfig` (transport, endpoint/command/args, timeout_ms, env_auth_token_key)
    - `MCPConfig` (system_mode, active_environment, mcp_servers: dict[str, MCPServerConfig])
    - `IntentRoute` (primary_mcp_server, required_tools, runtime_schema_contract, post_execution_state)
    - `GlobalConstraints` (halt_on_negative_sentiment, enforce_strict_anti_slop, max_session_cost_usd)
    - `WorkflowPolicy` (client_profile_id, intent_routing_matrix: dict[str, IntentRoute], global_constraints: GlobalConstraints)
  - `src/caae/config/loader.py` — functions:
    - `load_mcp_config(path: str) -> MCPConfig`
    - `load_workflow_policy(path: str) -> WorkflowPolicy`
    - Both read JSON, validate with Pydantic, raise clear errors on invalid config
  - Unit tests for both loaders (valid config, missing fields, invalid transport type, etc.)
- **Out of Scope for This Sub-Task:** Runtime use of config (wiring into graph/MCP client)
- **Instructions:**
  1. Define Pydantic models matching the JSON structures in Section 4 of CAAE.md
  2. Support both `stdio` and `streamable_http` transport types with a discriminated union or Literal field
  3. Implement `load_*` functions using `json.load()` + model validation
  4. Write tests against the example JSON in `configs/`
- **Acceptance Criteria:**
  - Loading the spec's example JSON produces valid Pydantic model instances
  - Loading a malformed JSON raises `ValidationError` with clear field-level messages
  - Both `stdio` and `streamable_http` transport configs parse correctly
- **Cautionary Points:** The `stdio` config has `command`/`args` fields while `streamable_http` has `endpoint`/`env_auth_token_key`; need conditional validation per transport type
- **Testing Suggestions:** `uv run pytest tests/unit/test_config_models.py -v`
- **Done When:** Config models validate spec examples; loader functions tested with valid + invalid inputs

---

### Sub-Task 3: Unified Context State & Schema Registry

- **Status:** Completed
- **Objective:** Implement the `UnifiedContextState` Pydantic model and a schema registry that resolves dotted schema contract paths from `workflow_policy.json`.
- **Related Requirements:** R5, R6
- **Dependencies and Preconditions:** Sub-Task 2 (config models exist, schema contracts are defined)
- **In Scope for This Sub-Task:**
  - `src/caae/models/state.py` — `UnifiedContextState` exactly matching spec Section 5.1:
    - `session_id: str`
    - `inbound_event_payload: Dict[str, Any]`
    - `resolved_intent: Optional[str]`
    - `mcp_retrieved_resources: List[Dict[str, Any]]`
    - `extracted_quantitative_data: Dict[str, Any]`
    - `execution_mutation_result: Dict[str, Any]`
    - `validation_retry_count: int = 0`
  - `src/caae/models/schemas/` — Schema registry:
    - `base.py` — `SchemaRegistry` class with `register(path: str, schema: Type[BaseModel])` and `resolve(path: str) -> Type[BaseModel]`
    - `clinical.py` — `AppointmentBookingPayload` (demo schema for scheduling)
    - `media.py` — `QuantitativeIntelPayload` (demo schema for content research)
    - `__init__.py` — Auto-registers all built-in schemas
  - Unit tests: state model serialization, schema registry resolution, missing schema error
- **Out of Scope for This Sub-Task:** Using the state in the graph (that's Sub-Task 5)
- **Instructions:**
  1. Implement `UnifiedContextState` with Pydantic v2 `BaseModel`, exact field names and types from spec
  2. `validation_retry_count` needs a `le=3` or `max_value=3` constraint (spec says max 3 retries)
  3. Build `SchemaRegistry` as a singleton dict mapping dotted paths → Pydantic model classes
  4. Define 2 demo schemas matching the `runtime_schema_contract` values in the example `workflow_policy.json`
  5. Write unit tests for round-trip serialization and registry lookup
- **Acceptance Criteria:**
  - `UnifiedContextState` round-trips through JSON serialization
  - `SchemaRegistry.resolve("schemas.clinical.AppointmentBookingPayload")` returns the correct model class
  - `SchemaRegistry.resolve("nonexistent")` raises a clear `KeyError` or custom exception
- **Cautionary Points:** LangGraph nodes return `dict` updates, not full state objects. The state model is used as the graph schema but nodes return partial dicts. Ensure the model has `model_config = ConfigDict(extra="forbid")` or similar to catch invalid state updates.
- **Testing Suggestions:** `uv run pytest tests/unit/test_state_model.py tests/unit/test_schema_registry.py -v`
- **Done When:** State model and schema registry are implemented and tested

---

### Sub-Task 4: MCP Client Manager & Transport Layer

- **Status:** Completed
- **Objective:** Implement the MCP client mapping layer — manages connection lifecycle, tool discovery, and tool invocation for both stdio and Streamable HTTP transports.
- **Related Requirements:** R3, R1
- **Dependencies and Preconditions:** Sub-Task 2 (config models for server connection info)
- **In Scope for This Sub-Task:**
  - `src/caae/mcp/transport.py` — Transport factory:
    - `create_stdio_connection(config: MCPServerConfig)` → async context manager yielding `(read, write)` streams
    - `create_streamable_http_connection(config: MCPServerConfig)` → async context manager yielding `(read, write, _)` streams
  - `src/caae/mcp/client_manager.py` — `MCPClientManager`:
    - `async start(config: MCPConfig)` — spin up all configured MCP clients, initialize sessions, discover tools
    - `async stop()` — gracefully shut down all client sessions
    - `async call_tool(server_name: str, tool_name: str, arguments: dict) -> dict` — invoke a tool on a specific server
    - `async list_tools(server_name: str) -> list[ToolInfo]` — get available tools for a server
    - `async read_resource(server_name: str, uri: str) -> dict` — read an MCP resource
    - `get_tool_schema(server_name: str, tool_name: str) -> dict` — get the JSON Schema for a specific tool
    - Internal: maintains `dict[str, ClientSession]` mapping server names to active sessions
  - `src/caae/mcp/models.py` — `ToolInfo` dataclass (name, description, input_schema)
  - Integration tests with a simple echo MCP server (can use `mcp` SDK's built-in examples or create a minimal one)
- **Out of Scope for This Sub-Task:** The demo scheduling server (Sub-Task 8), graph wiring (Sub-Task 5)
- **Instructions:**
  1. Implement transport factory using `mcp.client.stdio.stdio_client` and `mcp.client.streamable_http.streamable_http_client`
  2. `MCPClientManager` wraps each configured server's session lifecycle
  3. On `start()`, iterate `config.mcp_servers`, create connections, call `session.initialize()`, call `session.list_tools()`, cache results
  4. `call_tool()` delegates to `session.call_tool(tool_name, arguments)` and returns structured result
  5. For integration testing, create a tiny MCP server inline using `mcp` SDK's `FastMCP` that exposes one tool
- **Acceptance Criteria:**
  - `MCPClientManager` can start, discover tools from a stdio MCP server, call a tool, and shut down cleanly
  - Tool discovery returns name + JSON Schema for each tool
  - Timeout handling: if a tool call exceeds `timeout_ms`, it's caught and reported
- **Cautionary Points:**
  - MCP sessions are async context managers — the manager must hold them open for the app lifetime
  - Use `streamable_http_client` as the primary HTTP transport (MCP spec deprecated the old SSE transport)
  - Auth token injection: for `streamable_http`, read from `os.environ[env_auth_token_key]` and set as header
- **Testing Suggestions:** `uv run pytest tests/integration/test_mcp_client_manager.py -v` (requires a running MCP server; use inline `FastMCP` for test)
- **Done When:** MCP client manager can discover and invoke tools via stdio transport in an integration test

---

### Sub-Task 5: LangGraph State Machine — 5-Node Pipeline

- **Status:** Completed
- **Objective:** Implement the core 5-node LangGraph cyclic state machine with all nodes, edges, conditional routing, and the verification loop.
- **Related Requirements:** R2, R6
- **Dependencies and Preconditions:** Sub-Task 3 (state model), Sub-Task 4 (MCP client manager)
- **In Scope for This Sub-Task:**
  - `src/caae/nodes/context_assessor.py` — Node 1:
    - Ingests inbound event payload
    - Parses/instantiates `session_id` (generate UUID if not present)
    - Determines initial intent via LLM structured output (using `Route` schema from `workflow_policy.intent_routing_matrix`)
    - Returns state updates: `session_id`, `resolved_intent`
  - `src/caae/nodes/info_retrieval.py` — Node 2:
    - Uses `resolved_intent` to look up `workflow_policy.intent_routing_matrix[intent]`
    - Gets `primary_mcp_server` and `required_tools`
    - Calls `MCPClientManager.list_tools()` and `read_resource()` as needed
    - Populates `mcp_retrieved_resources` with raw data
    - Returns state update: `mcp_retrieved_resources`
  - `src/caae/nodes/cognitive_processing.py` — Node 3:
    - For content tasks: enforces structural requirements via prompt engineering and structured output + schema validation (anti-slop handled through prompts and output constraints, not a custom pipeline in V1)
    - For transactional tasks: runs multi-dimensional recommendation scoring
    - Uses LLM with structured output constrained by the `runtime_schema_contract` from policy
    - Returns state update: `extracted_quantitative_data`
  - `src/caae/nodes/action_execution.py` — Node 4:
    - If intent resolves to mutation: calls `MCPClientManager.call_tool()` with validated args
    - If intent resolves to content generation: constructs compiled document object
    - Returns state update: `execution_mutation_result`
  - `src/caae/nodes/evaluation_gate.py` — Node 5:
    - Implements `verify_output_compliance()` logic from spec Section 6.3
    - Checks retry count (≥ 3 → `human_handoff_escalation`)
    - Runs Pydantic schema validation
    - Returns routing decision: `"commit_state_and_exit"` | `"re_evaluate_context_node"` | `"human_handoff_escalation"`
  - `src/caae/graph.py` — `build_caae_graph()`:
    - Creates `StateGraph(UnifiedContextState)`
    - Adds all 5 nodes
    - Adds linear edges: START → context_assessor → info_retrieval → cognitive_processing → action_execution → evaluation_gate
    - Adds conditional edges from evaluation_gate using `verify_output_compliance`
    - Compiles and returns the graph
  - Unit tests for each node (mocked dependencies), integration test for full graph execution
- **Out of Scope for This Sub-Task:** LLM provider specifics (use mocks), MCP server implementation, observability wiring
- **Instructions:**
  1. Each node function takes `state: UnifiedContextState` (or dict) and returns a `dict` of state updates
  2. Context Assessor uses `init_chat_model()` from LangChain with structured output for intent classification
  3. Info Retrieval and Action Execution need `MCPClientManager` injected (pass via graph config or closure)
  4. Cognitive Processing uses LLM with structured output constrained to the schema from the registry
  5. Evaluation Gate is purely programmatic — no LLM call
  6. For testing: mock `MCPClientManager` and LLM calls; test node logic in isolation
  7. Integration test: wire the full graph, inject mocks, invoke with a sample event, assert final state
- **Acceptance Criteria:**
  - Each node unit test passes with mocked dependencies
  - Full graph integration test: input event → resolved intent → tool call → validation → exit
  - Loop-back test: validation fails twice, succeeds on third attempt
  - Human handoff test: 3 validation failures → `human_handoff_escalation` exit
- **Cautionary Points:**
  - LangGraph nodes receive state as a `dict` or the schema type; return partial dict updates
  - The `MCPClientManager` is not serializable — pass it via LangGraph's `config` parameter or a closure/factory pattern
  - LLM calls in nodes must be async; use `@task` decorator or ensure all nodes are async functions
  - `validation_retry_count` increment happens in the evaluation gate; ensure it's part of the returned state update
- **Testing Suggestions:** `uv run pytest tests/unit/test_nodes.py tests/integration/test_graph.py -v`
- **Done When:** All 5 nodes implemented; graph compiles; happy path + loop-back + handoff paths tested

---

### Sub-Task 6: Validation Pipeline (Pydantic + DeepEval)

- **Status:** Pending
- **Objective:** Implement the deterministic validation pipeline — Pydantic schema checking and DeepEval regression gates.
- **Related Requirements:** R6, R9
- **Dependencies and Preconditions:** Sub-Task 3 (state model, schema registry)
- **In Scope for This Sub-Task:**
  - `src/caae/validation/schema_checker.py`:
    - `execute_pydantic_schema_check(result: dict, schema_contract: str) -> bool`
    - Resolves schema contract path via `SchemaRegistry`, validates result dict against it
    - Returns `True` if valid, `False` otherwise (logs validation errors)
  - `src/caae/validation/deepeval_gates.py`:
    - `run_groundedness_assertion(output, context) -> bool` — DeepEval `GroundednessMetric` threshold ≥ 0.95
    - `run_relevancy_assertion(output, input_) -> bool` — DeepEval `ConversationalRelevancyMetric` threshold ≥ 0.90
    - `run_schema_adherence_assertion(output, schema) -> bool` — DeepEval `SchemaAdherenceMetric` threshold 100%
    - Each returns bool pass/fail with detailed metric logging
  - `src/caae/validation/__init__.py` — `ValidationPipeline` class that orchestrates all three checks
  - Unit tests with mock LLM-based evaluations (DeepEval requires an LLM; mock it for unit tests)
  - DeepEval test case examples as fixtures
- **Out of Scope for This Sub-Task:** Wiring into the evaluation gate node (Sub-Task 5 already references these)
- **Instructions:**
  1. `execute_pydantic_schema_check` — use `SchemaRegistry.resolve(contract_path)` to get the model class, then `model_class.model_validate(result)`
  2. DeepEval metrics need an evaluation LLM — configure via env var `DEEPEVAL_MODEL` (default to same LLM as engine)
  3. For unit testing, use DeepEval's mock mode or test-only fixtures
  4. `ValidationPipeline` runs schema check first (cheap), then groundedness, then relevancy
- **Acceptance Criteria:**
  - Pydantic schema check correctly validates/invalidates payloads against registered schemas
  - DeepEval gates return pass/fail with metric scores
  - `ValidationPipeline` returns a clear pass/fail verdict with detailed breakdown
- **Cautionary Points:**
  - DeepEval metrics are LLM-based evaluations — they're inherently non-deterministic; use generous thresholds in tests
  - DeepEval requires an OpenAI key by default; ensure the provider-agnostic setup works here too
- **Testing Suggestions:** `uv run pytest tests/unit/test_validation.py -v`
- **Done When:** Validation pipeline implemented; schema checking and DeepEval gates tested

---

### Sub-Task 7: Langfuse Observability Integration

- **Status:** Pending
- **Objective:** Integrate Langfuse tracing into every node execution, tool call, and state transition for full observability.
- **Related Requirements:** R8
- **Dependencies and Preconditions:** Sub-Task 5 (graph with nodes), Sub-Task 4 (MCP client manager with tool calls)
- **In Scope for This Sub-Task:**
  - `src/caae/observability/langfuse_handler.py`:
    - `LangfuseHandler` class wrapping Langfuse SDK
    - `start_trace(session_id: str) -> Trace` — create parent trace for a session
    - `start_span(trace, name: str, input_data: dict) -> Span` — create child spans for each node
    - `end_span(span, output_data: dict)` — close span with output
    - `record_tool_call(trace, server_name, tool_name, arguments, result, latency_ms)` — record tool invocation
    - `track_cost(trace, prompt_tokens, completion_tokens, model)` — accumulate cost per session
    - `is_within_budget(session_id, max_cost_usd) -> bool` — check against `workflow_policy.global_constraints.max_session_cost_usd`
  - Integration: wrap each LangGraph node with tracing (use LangGraph's callback system or manual span creation)
  - Integration: wrap `MCPClientManager.call_tool()` with latency tracking
  - Unit tests with mocked Langfuse client
- **Out of Scope for This Sub-Task:** DeepEval integration (Sub-Task 6), UI dashboard
- **Instructions:**
  1. Use `langfuse` Python SDK: `from langfuse import Langfuse`
  2. LangGraph supports callbacks — implement a `LangfuseCallbackHandler` that hooks into `on_chain_start`, `on_chain_end`, `on_tool_start`, `on_tool_end`
  3. For MCP tool calls that happen outside LangGraph's callback scope, manually create spans
  4. Cost tracking: parse `response_metadata` from LLM responses for token counts
  5. Budget enforcement: before each LLM call in `cognitive_processing`, check `is_within_budget`
- **Acceptance Criteria:**
  - Running a session produces a Langfuse trace with 5 node spans + tool call spans + cost records
  - Budget enforcement: if cost exceeds `max_session_cost_usd`, the session is halted
  - Timeout detection: MCP tool calls past `timeout_ms` are logged to Langfuse
- **Cautionary Points:**
  - Langfuse SDK needs `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` env vars
  - LangGraph's callback system vs manual tracing — prefer callbacks for consistency
  - If Langfuse is not configured (no keys), tracing should be a no-op, not an error
- **Testing Suggestions:** `uv run pytest tests/unit/test_observability.py -v`
- **Done When:** Langfuse traces appear for full sessions; cost tracking works; budget enforcement halts over-budget sessions

---

### Sub-Task 8: Demo MCP Server — Scheduling Engine

- **Status:** Pending
- **Objective:** Build a functional in-memory scheduling MCP server as a reference implementation and demo, exposing tools that the core engine can invoke.
- **Related Requirements:** R12
- **Dependencies and Preconditions:** Sub-Task 4 (MCP client manager can connect to servers)
- **In Scope for This Sub-Task:**
  - `src/caae_demo_server/` — separate package for the demo MCP server:
    - `__init__.py`
    - `server.py` — FastMCP server with tools:
      - `check_availability(date: str, practitioner_id: str) -> list[dict]` — returns available time slots
      - `book_slot(lead_id: str, timestamp_iso: str, practitioner_id: str) -> dict` — books an appointment
      - `list_appointments(practitioner_id: str) -> list[dict]` — lists booked appointments
      - `cancel_appointment(appointment_id: str) -> dict` — cancels an appointment
    - `data.py` — in-memory data store (dict-based, pre-populated with sample practitioners and slots)
    - `__main__.py` — entry point for `uv run caae-demo-server`
  - Update `configs/mcp_config.json` to reference the demo server via stdio transport
  - Integration test: `MCPClientManager` connects to demo server, discovers tools, calls tools
- **Out of Scope for This Sub-Task:** Production persistence, multi-tenancy, auth
- **Instructions:**
  1. Use `mcp.server.fastmcp.FastMCP` to create the server
  2. Define each tool with `@mcp.tool()` decorator, include type-annotated params and docstrings
  3. In-memory store: `dict[str, list[Appointment]]` keyed by practitioner_id
  4. Pre-populate with 3 sample practitioners and a week of available slots
  5. Register as a `scripts` entry point in `pyproject.toml`: `caae-demo-server = "caae_demo_server.__main__:main"`
  6. Ensure the demo server can be launched via `uv run caae-demo-server`
- **Acceptance Criteria:**
  - Demo server starts via `uv run caae-demo-server`
  - `MCPClientManager` connects, discovers 4 tools with correct schemas
  - `check_availability` returns slots; `book_slot` creates appointment; `list_appointments` shows it
  - Tool schemas include proper JSON Schema for validation
- **Cautionary Points:**
  - FastMCP auto-generates schemas from function signatures — ensure type annotations are precise
  - In-memory store is not thread-safe — fine for demo, but document this limitation
- **Testing Suggestions:** `uv run pytest tests/integration/test_demo_server.py -v`
- **Done When:** Demo server runs; all 4 tools are discoverable and callable via MCP client

---

### Sub-Task 9: FastAPI Host Application — Session Management API

- **Status:** Pending
- **Objective:** Build the FastAPI host application with REST API endpoints for session lifecycle, event ingestion, and status monitoring.
- **Related Requirements:** R11
- **Dependencies and Preconditions:** Sub-Task 5 (graph), Sub-Task 4 (MCP client manager), Sub-Task 7 (observability)
- **In Scope for This Sub-Task:**
  - `src/caae/main.py` — FastAPI app with lifespan:
    - On startup: load configs, start `MCPClientManager`, build graph
    - On shutdown: stop `MCPClientManager`
  - `src/caae/api/routes/health.py`:
    - `GET /health` — basic health check
    - `GET /health/mcp-servers` — status of all connected MCP servers
  - `src/caae/api/routes/sessions.py`:
    - `POST /sessions` — create a new session, accepts `InboundEventPayload`, returns `session_id`
    - `GET /sessions/{session_id}` — get current session state
    - `POST /sessions/{session_id}/events` — inject a new event into an existing session (re-runs from context assessor)
    - `POST /sessions/{session_id}/cancel` — cancel a running session
  - `src/caae/api/models.py` — request/response Pydantic models for API endpoints
  - `src/caae/engine.py` — `CAAEEngine` orchestrator:
    - Holds references to: graph, MCPClientManager, WorkflowPolicy, LangfuseHandler
    - `async run_session(event: dict) -> UnifiedContextState` — invoke graph with event, return final state
    - `async get_session_state(session_id: str) -> UnifiedContextState` — retrieve state from checkpointer
  - Wire LangGraph checkpointer (in-memory `MemorySaver` for V1) for session state persistence
  - Integration tests: full API flow (create session → check status → get result)
- **Out of Scope for This Sub-Task:** WebSocket streaming, multi-user auth, production persistence (use SQLite checkpointer later)
- **Instructions:**
  1. Use FastAPI's `lifespan` context manager for startup/shutdown
  2. `CAAEEngine` is the single orchestrator — the API routes delegate to it
  3. LangGraph `MemorySaver` checkpointer for in-memory state persistence
  4. Session execution is async — `asyncio.create_task()` for background execution, store result in a dict
  5. For V1, sessions run to completion (no streaming intermediate results via API)
- **Acceptance Criteria:**
  - `POST /sessions` with an event payload returns a `session_id`
  - `GET /sessions/{id}` returns the current/final state
  - Health endpoint shows MCP server connectivity status
  - Full flow: create session → engine runs graph → session state available via GET
- **Cautionary Points:**
  - Long-running graph executions shouldn't block the API — use background tasks
  - `MemorySaver` is not persistent across restarts — acceptable for V1
  - Session state dict is not thread-safe — use `asyncio.Lock` if needed
- **Testing Suggestions:** `uv run pytest tests/integration/test_api.py -v`
- **Done When:** FastAPI app starts, sessions can be created and queried end-to-end through the API

---

### Sub-Task 10: End-to-End Integration & CI Pipeline

- **Status:** Pending
- **Objective:** Wire everything together, create an end-to-end integration test, set up CI with DeepEval regression gates.
- **Related Requirements:** R9, R2, R7
- **Dependencies and Preconditions:** All previous sub-tasks complete
- **In Scope for This Sub-Task:**
  - End-to-end test scenario:
    - Start demo MCP server
    - Start FastAPI app
    - POST a scheduling request event
    - Assert: intent resolved to `appointment_booking_request`
    - Assert: MCP tools called (`check_availability`, `book_slot`)
    - Assert: output validated and session committed
  - End-to-end test for content/research scenario:
    - POST a competitor intel request
    - Assert: intent resolved, tools called, output structured
  - GitHub Actions CI workflow (`.github/workflows/ci.yml`):
    - Lint: `ruff check`
    - Type check: `mypy`
    - Unit tests: `pytest tests/unit`
    - Integration tests: `pytest tests/integration`
    - DeepEval regression gate: `pytest tests/regression` (runs as separate step)
  - DeepEval regression test suite:
    - `tests/regression/test_groundedness.py` — ≥ 0.95 threshold
    - `tests/regression/test_relevancy.py` — ≥ 0.90 threshold
    - `tests/regression/test_schema_adherence.py` — 100% match
  - README update with setup/run instructions
- **Out of Scope for This Sub-Task:** Production deployment, Docker (can add later), load testing
- **Instructions:**
  1. Write E2E tests using `httpx.AsyncClient` against the FastAPI app
  2. For DeepEval regression, create test cases with known inputs/expected outputs
  3. CI workflow: use `uv` in GitHub Actions (astral-sh/setup-uv action)
  4. Mark DeepEval tests as slow/optional for local dev (use `pytest.mark.regression`)
- **Acceptance Criteria:**
  - Full E2E test passes: event in → scheduled appointment out
  - CI pipeline runs lint, type-check, unit, integration tests
  - DeepEval regression tests evaluate against defined thresholds
  - README documents: how to install, configure, run, and test
- **Cautionary Points:**
  - DeepEval tests require an LLM API key and are non-deterministic — they may flake
  - CI should allow DeepEval failures to be non-blocking initially (warn, don't fail the build)
  - Integration tests need the demo server running — use test fixtures to start/stop it
- **Testing Suggestions:** `uv run pytest tests/ -v --timeout=120`
- **Done When:** Full E2E test passes; CI pipeline is green; DeepEval gates evaluated

---

## Final Integration & Verification

- **System-Wide Test:** Run the full CAAE stack (FastAPI + demo MCP server + LangGraph graph + Langfuse tracing) and execute a complete appointment booking flow via the REST API. Verify:
  1. Event ingestion triggers the 5-node pipeline
  2. MCP tool discovery finds `check_availability` and `book_slot`
  3. LLM resolves intent correctly
  4. Tool calls are executed and results returned
  5. Validation gate passes (schema + groundedness)
  6. Final state is queryable via API
  7. Langfuse trace contains all spans with latency + cost data
  8. DeepEval regression suite passes thresholds

- **Completion Checklist:**
  - [ ] `uv sync` resolves all dependencies
  - [ ] `uv run ruff check src/` passes
  - [ ] `uv run mypy src/` passes
  - [ ] `uv run pytest tests/unit/ -v` passes
  - [ ] `uv run pytest tests/integration/ -v` passes
  - [ ] Demo MCP server starts and responds to tool calls
  - [ ] FastAPI app starts with all MCP connections established
  - [ ] Full E2E appointment booking flow succeeds via API
  - [ ] Langfuse traces appear for executed sessions
  - [ ] DeepEval regression gates evaluated
  - [ ] CI pipeline configured and passing
  - [ ] README documents setup, configuration, and usage

## Resolved Questions

- **Q1 (Resolved → D1):** Demo MCP server is a **separate package** (`src/caae_demo_server/`) — mirrors the spec's isolated micro-service architecture, keeps the core engine industry-agnostic, and serves as a reference template for building real MCP servers.
- **Q2 (Resolved → D2):** Anti-slop is handled via **prompts + structured output + schema validation** in V1. No custom anti-slop pipeline (e.g., banned-phrase filters, structural enforcement modules). Node 3 uses prompt engineering and output constraints to enforce content quality.
- **Q3 (Resolved → D3):** HTTP transport uses **Streamable HTTP** (`streamable_http_client`). The MCP spec has deprecated the old HTTP+SSE transport. The Python SDK recommends `streamable-http` for production deployments. Config uses `"streamable_http"` as the transport type instead of `"http_sse"`.
