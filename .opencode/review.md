# Code Review Summary

**Scope**: Sub-Task 8 — Demo MCP Server (Scheduling Engine)
**Files reviewed**: `data.py`, `server.py`, `__main__.py`, `__init__.py`, `test_demo_server.py`, `mcp_config.json`, `pyproject.toml`, `test_config_models.py`
**Overall risk**: Low (for demo server itself) / Medium (one cross-cutting bug found)
**Verdict**: Approve with comments — one P1 fix and one P0 cross-cutting bug found

---

## Positive Notes

- **Clean separation of concerns**: `data.py` (pure logic) / `server.py` (FastMCP binding) / `__main__.py` (entry point). Each layer is minimal and testable independently.
- **Double-booking prevention correct**: `book_appointment` checks `status == "confirmed"` before booking, so cancelled slots are re-bookable. Correct.
- **Idempotent cancel**: Cancelling an already-cancelled appointment returns "cancelled" (not an error). Acceptable for demo.
- **Integration tests thorough**: Tests cover tool discovery, schema validation, full lifecycle (check→book→verify unavailable→list→cancel→verify), double-booking rejection, not-found cases, unknown practitioners.
- **Ruff/mypy clean**: Zero ruff violations, zero mypy errors across all 4 demo server source files.
- **Pre-populated store correct**: 3 practitioners (`dr-smith`, `dr-jones`, `dr-wilson`), slot generation produces 16 × 30-min slots (09:00–17:00 UTC).
- **Tool schemas valid JSON Schema**: All 4 tools verified to have `type: "object"` with `properties` dict. Parameter names match expectations.
- **Config integration works**: `mcp_config.json` parsed correctly by `MCPConfig` model; discriminated union discriminates `stdio` vs `streamable_http`.
- **`pyproject.toml` entry correct**: `caae-demo-server = "caae_demo_server.__main__:main"` wired correctly.

---

## Findings

### [P0] Blocking — `create_streamable_http_connection` yields 3-tuple but caller unpacks 2

- **Location**: `src/caae/mcp/transport.py:82-83,88-89` (yields 3-tuple) → `src/caae/mcp/client_manager.py:107-108` (unpacks 2)
- **Why it matters**: Any MCP server configured with `transport: "streamable_http"` (including `core_crm_gateway` in `mcp_config.json`) will **crash at startup** with `ValueError: too many values to unpack`. The demo server itself uses `stdio` and is unaffected, but this is a production blocker for the `core_crm_gateway` integration.
- **Evidence**:
  - `transport.py:82`: `yield (read, write, get_session_id)` — 3 values
  - `transport.py:88`: `yield (read, write, get_session_id)` — 3 values
  - `client_manager.py:107-108`: `read, write = await self._exit_stack.enter_async_context(create_streamable_http_connection(...))` — destructures into 2 variables
  - Unit test `mock_transport_cm` fixture (line 286) returns a 2-tuple for both transports, masking the bug
- **Fix**: Either (a) unpack all 3: `read, write, _session_id = await ...` or (b) if `get_session_id` is not needed, change the transport to not yield it: `yield (read, write)`. Option (b) is simpler since `ClientSession` only takes `(read, write)`.

### [P1] High — `cancel_appointment` returns inconsistent shape for "not_found"

- **Location**: `src/caae_demo_server/data.py:147`
- **Why it matters**: The docstring promises `{"appointment_id": ..., "status": ...}`, but the "not_found" path returns `{"status": "not_found"}` without `appointment_id`. Any LLM or client destructuring `result["appointment_id"]` will get a `KeyError`.
- **Evidence**:
  - `data.py:133-146`: Docstring says "Dict with `appointment_id` and `status`"
  - `data.py:147`: `return {"status": "not_found"}` — missing `appointment_id`
  - The successful path (line 146) returns both keys
  - The integration test at line 223 only checks `cancel_data2["status"] == "not_found"`, not missing `appointment_id`
- **Fix**: Change line 147 to `return {"appointment_id": appointment_id, "status": "not_found"}`

### [P2] Medium — `_generate_slots_for_date` crashes on invalid date strings

- **Location**: `src/caae_demo_server/data.py:41`
- **Why it matters**: `datetime.strptime(date_str, "%Y-%m-%d")` raises `ValueError` for malformed dates (e.g., "2024-13-01", "not-a-date"). FastMCP will propagate this as an unhandled exception to the MCP client. No graceful error message.
- **Evidence**: No try/except in the call chain from `check_availability` → `get_available_slots` → `_generate_slots_for_date`.
- **Fix**: Wrap `_generate_slots_for_date` in try/except `ValueError`, return `[]` or raise a descriptive error. Alternatively, add Pydantic validation on the tool parameter or document that the caller must pre-validate.

### [P2] Medium — `list_appointments` returns cancelled appointments without filter

- **Location**: `src/caae_demo_server/data.py:109-130`
- **Why it matters**: Cancelled appointments interleave with confirmed ones. No parameter to filter by status. LLM consumers may misinterpret cancelled appointments as active. The integration test (line 207-214) verifies cancelled appointments still appear — this is intentional but limits the tool's usefulness.
- **Evidence**: `list_appointments_for_practitioner` returns all appointments regardless of status. No `status` filter parameter.
- **Fix**: Add an optional `status: str | None = None` parameter. If provided, filter the returned list.

### [P2] Medium — `mcp_config.json` contains duplicate `scheduling_engine` / `demo_scheduling` entries

- **Location**: `configs/mcp_config.json:11-22`
- **Why it matters**: Both entries launch separate `caae-demo-server` processes with independent in-memory state. A booking made via `scheduling_engine` won't appear in `demo_scheduling` and vice versa. The `workflow_policy.json` references `scheduling_engine`, but tests use `demo_scheduling`. This split is fragile and could confuse operators.
- **Evidence**: Lines 11-16 (`scheduling_engine`) and 17-22 (`demo_scheduling`) are identical except for the server name.
- **Fix**: Either consolidate into a single entry (update tests and policy to use the same name) or clearly document in the config that they are independent demo instances.

### [P3] Low — `book_appointment` returns `"unavailable"` for unknown practitioners (semantically misleading)

- **Location**: `src/caae_demo_server/data.py:84-85`
- **Why it matters**: When `practitioner_id` is not in `PRACTITIONERS`, the function returns `{"status": "unavailable"}`. But the slot isn't "unavailable" — the practitioner doesn't exist. The same return value is used for genuinely booked slots (line 89), conflating two different error conditions.
- **Fix**: Return `{"status": "error", "detail": "Unknown practitioner"}` or a separate error code.

### [P3] Low — No `__all__` or explicit exports in `caae_demo_server/__init__.py`

- **Location**: `src/caae_demo_server/__init__.py:1`
- **Why it matters**: The package has a single docstring line. No explicit public API exported. Low impact since this is a runnable entry point, not an importable library.
- **Fix**: Add `__all__ = []` or export the `mcp` instance for programmatic use.

---

## Suggested Next Steps

- [ ] **[P0]** Fix the streamable_http 3-tuple → 2-tuple unpacking bug in `client_manager.py` (also fix `mock_transport_cm` to use 3-tuple for streamable_http tests)
- [ ] **[P1]** Add `appointment_id` to the `cancel_appointment` "not_found" return dict
- [ ] **[P2]** Wrap `_generate_slots_for_date` in try/except for invalid date strings
- [ ] **[P2]** Add optional `status` filter to `list_appointments_for_practitioner`
- [ ] **[P2]** Consider consolidating `scheduling_engine` / `demo_scheduling` config entries
- [ ] **[P3]** Differentiate return statuses for unknown-practitioner vs genuinely-booked slots
- [ ] **Test**: Add unit test for `data.py` functions directly (faster feedback than full integration tests)
- [ ] **Test**: After P0 fix, add integration test that exercises streamable_http transport (mock HTTP server)
