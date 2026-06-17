# CAAE — Central Adaptable Automation Engine

A data-agnostic multi-agent orchestration runtime that decouples cognitive orchestration from data ingestion and operational mutation. Built on **MCP** (Model Context Protocol) for tool discovery/invocation, **LangGraph** for state-machine orchestration, **FastAPI** as the host process, and **Pydantic v2** for strict data contracts.

## Architecture

```
Inbound Event
     │
     ▼
┌──────────────────┐
│ Context Assessor  │  ← Classify intent via LLM structured output
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Info Retrieval    │  ← MCP tool discovery + resource reading
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Cognitive Process │  ← LLM reasoning constrained by schema contracts
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Action Execution  │  ← MCP tool invocation (mutations/queries)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────────────────┐
│ Evaluation Gate   │──→  │ Pass → commit & exit         │
└────────┬─────────┘     │ Retry → loop back (max 3)     │
         │               │ Fail → human handoff          │
         └───────────────┴──────────────────────────────┘
```

A single `UnifiedContextState` flows through all five nodes. The evaluation gate enforces deterministic guardrails — Pydantic schema validation, structured output compliance, and max 3 retry loops.

## Key Design Principles

- **Data-agnostic core** — zero industry-specific knowledge in engine code; capabilities discovered dynamically via MCP JSON-RPC 2.0 negotiations
- **Declarative configuration** — `mcp_config.json` (server connections) and `workflow_policy.json` (intent routing, schema contracts, constraints)
- **Deterministic guardrails** — bounded retry loops, Pydantic schema validation, mandatory output compliance checks
- **Provider-agnostic LLM** — LangChain ChatModel abstraction; swap OpenAI for Anthropic via config
- **Full observability** — Langfuse per-session traces, span monitoring, cost tracking

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
git clone <repo-url> caae && cd caae
uv sync
```

### Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM provider (or `ANTHROPIC_API_KEY`) |
| `LANGFUSE_PUBLIC_KEY` | Observability tracing |
| `LANGFUSE_SECRET_KEY` | Observability tracing |
| `CRM_PROD_BEARER_TOKEN` | Auth token for HTTP MCP server (optional) |

### Run

```bash
# Start the CAAE engine (FastAPI + LangGraph)
uv run caae-server

# Start the demo scheduling MCP server (separate process)
uv run caae-demo-server
```

The engine exposes a REST API at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## Project Structure

```
src/caae/
├── main.py                  # FastAPI entry point with lifespan
├── engine.py                # CAAEEngine orchestrator (stub)
├── graph.py                 # LangGraph StateGraph builder (stub)
├── models/
│   ├── config.py            # MCPConfig, WorkflowPolicy Pydantic models
│   ├── state.py             # UnifiedContextState graph schema
│   └── schemas/             # Schema registry + demo contracts
│       ├── base.py           # SchemaRegistry
│       ├── clinical.py       # AppointmentBookingPayload
│       └── media.py          # QuantitativeIntelPayload
├── mcp/
│   ├── client_manager.py    # MCPClientManager lifecycle + tool invocation
│   ├── transport.py         # stdio / streamable-http transport factory
│   └── models.py            # ToolInfo dataclass
├── config/
│   └── loader.py            # JSON config loader + ConfigLoadError
├── nodes/                   # 5-node LangGraph pipeline (stubs)
│   ├── context_assessor.py
│   ├── info_retrieval.py
│   ├── cognitive_processing.py
│   ├── action_execution.py
│   └── evaluation_gate.py
├── observability/            # Langfuse tracing (stub)
├── validation/               # Pydantic + DeepEval gates (stub)
└── api/routes/               # FastAPI endpoints (stubs)
    ├── health.py
    └── sessions.py

src/caae_demo_server/          # Demo scheduling MCP server (separate package)

configs/
├── mcp_config.json           # MCP server connections
└── workflow_policy.json       # Intent routing matrix + constraints

tests/
├── unit/                     # 89 unit tests
└── integration/              # 9 integration tests (real MCP + stdio)
```

## Configuration

### MCP Servers (`configs/mcp_config.json`)

```json
{
  "system_mode": "production",
  "active_environment": "enterprise_healthcare_sales",
  "mcp_servers": {
    "core_crm_gateway": {
      "transport": "streamable_http",
      "endpoint": "https://crm.client-api.internal/mcp/v1",
      "timeout_ms": 5000,
      "env_auth_token_key": "CRM_PROD_BEARER_TOKEN"
    },
    "scheduling_engine": {
      "transport": "stdio",
      "command": "uv",
      "args": ["run", "caae-demo-server"],
      "timeout_ms": 5000
    }
  }
}
```

Supports two transports:
- **`stdio`** — spawns a subprocess (`command` + `args`)
- **`streamable_http`** — connects to an HTTP endpoint (MCP spec's recommended production transport)

### Workflow Policy (`configs/workflow_policy.json`)

```json
{
  "client_profile_id": "med_spa_conversion_hub",
  "intent_routing_matrix": {
    "appointment_booking_request": {
      "primary_mcp_server": "scheduling_engine",
      "required_tools": ["check_availability", "book_slot"],
      "runtime_schema_contract": "schemas.clinical.AppointmentBookingPayload",
      "post_execution_state": "trigger_sms_nurture"
    }
  },
  "global_constraints": {
    "halt_on_negative_sentiment": true,
    "enforce_strict_anti_slop": true,
    "max_session_cost_usd": 2.50
  }
}
```

Each intent route maps to an MCP server, required tools, a Pydantic schema contract (resolved at runtime), and a post-execution state.

## Core Components

### MCP Client Manager

```python
from caae.mcp import MCPClientManager

async with MCPClientManager() as manager:
    await manager.start(mcp_config)

    # Discover tools
    tools = await manager.list_tools("scheduling_engine")

    # Invoke a tool (with timeout enforcement)
    result = await manager.call_tool("scheduling_engine", "book_slot", {
        "lead_id": "123",
        "timestamp_iso": "2025-01-15T10:00:00Z",
        "practitioner_id": "dr-smith",
    })

    # Get a tool's JSON Schema
    schema = manager.get_tool_schema("scheduling_engine", "book_slot")

    await manager.stop()
```

- Per-server timeout enforcement via `timeout_ms` from config
- Graceful per-server failure tolerance (startup continues if one server is down)
- Auth token injection for `streamable_http` via environment variables

### Schema Registry

```python
from caae.models import SchemaRegistry, get_default_registry

registry = get_default_registry()

# Resolve a schema contract path from workflow_policy.json
AppointmentBookingPayload = registry.resolve("schemas.clinical.AppointmentBookingPayload")

# Validate data against a schema
AppointmentBookingPayload.model_validate(event_data)
```

### UnifiedContextState

```python
from caae.models import UnifiedContextState

state = UnifiedContextState(
    session_id="abc-123",
    inbound_event_payload={"message": "Book appointment for Dr. Smith"},
    resolved_intent="appointment_booking_request",
)
# validation_retry_count is bounded to max 3
```

## Testing

```bash
# Unit tests (89 tests)
uv run pytest tests/unit/ -v

# Integration tests — real MCP server via stdio (9 tests)
uv run pytest tests/integration/ -v

# Lint
uv run ruff check src/ tests/

# Full suite
uv run pytest -v
```

## Tech Stack

| Layer | Technology |
|---|---|
| Host Process | FastAPI + Uvicorn |
| State Machine | LangGraph |
| Tool Protocol | MCP (Model Context Protocol) |
| Data Contracts | Pydantic v2 |
| LLM Integration | LangChain ChatModel (OpenAI / Anthropic) |
| Observability | Langfuse |
| CI Regression | DeepEval |
| Package Manager | uv |

## License

See [LICENSE](LICENSE).