"""Integration tests for the FastAPI REST API.

Tests the session lifecycle, health checks, and event ingestion endpoints.
Mocks LLM calls, MCP config, and MCP client to avoid real external services.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from caae.api.models import InboundEventPayload
from caae.engine import CAAEEngine


@pytest.fixture
def mock_engine():
    """Create a fully mocked CAAEEngine for API testing."""
    engine = MagicMock(spec=CAAEEngine)
    engine.start = AsyncMock()
    engine.stop = AsyncMock()

    # MCP server statuses
    engine.get_mcp_server_statuses.return_value = {
        "server_a": {"status": "connected"},
        "server_b": {"status": "disconnected", "error": "Server not connected during startup"},
    }

    # Session state — stored in a real dict so get/set works
    fake_sessions: dict[str, dict] = {}

    async def fake_run_session(event, session_id=None):
        sid = session_id or "test-auto-id"
        result = {
            "session_id": sid,
            "inbound_event_payload": event,
            "resolved_intent": "appointment_booking_request",
            "mcp_retrieved_resources": [
                {"server": "scheduling_engine", "tools": ["check_availability", "book_slot"]},
            ],
            "extracted_quantitative_data": {
                "lead_id": "L123",
                "practitioner_id": "dr-smith",
            },
            "execution_mutation_result": {"status": "completed", "tool_results": {"check_availability": "ok"}},
            "validation_retry_count": 0,
            "evaluation_passed": True,
        }
        fake_sessions[sid] = result
        return result

    engine.run_session = AsyncMock(side_effect=fake_run_session)
    engine._run_session_and_store = AsyncMock(side_effect=fake_run_session)

    def fake_get_session_state(sid):
        return fake_sessions.get(sid)

    engine.get_session_state = MagicMock(side_effect=fake_get_session_state)

    def fake_initialize_session(session_id, payload):
        from caae.models.state import UnifiedContextState

        fake_sessions[session_id] = UnifiedContextState(
            session_id=session_id,
            inbound_event_payload=payload,
        ).model_dump()

    engine.initialize_session = MagicMock(side_effect=fake_initialize_session)

    return engine


@pytest.fixture
async def async_client(mock_engine):
    """Create an httpx async client against the real FastAPI app with mocked engine.

    Note: ``httpx.ASGITransport`` does not trigger the ASGI lifespan scope,
    so we set ``app.state.engine`` directly here.
    """
    from caae.main import app

    # Inject mock engine into app state (lifespan is not triggered by httpx)
    app.state.engine = mock_engine

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestHealth:
    """Health and monitoring endpoints."""

    async def test_get_health_returns_ok(self, async_client):
        """GET /health returns 200 with status ok."""
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "ok"}

    async def test_get_health_mcp_servers(self, async_client):
        """GET /health/mcp-servers returns status for all servers."""
        resp = await async_client.get("/health/mcp-servers")
        assert resp.status_code == 200
        data = resp.json()
        assert "servers" in data
        servers = data["servers"]
        assert "server_a" in servers
        assert servers["server_a"]["status"] == "connected"
        assert "server_b" in servers
        assert servers["server_b"]["status"] == "disconnected"
        assert servers["server_b"]["error"] is not None


class TestSessions:
    """Session lifecycle endpoints."""

    async def test_create_session_returns_session_id(self, async_client):
        """POST /sessions returns a session_id."""
        payload = InboundEventPayload(payload={"message": "Book appointment"}).model_dump()
        resp = await async_client.post("/sessions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    async def test_get_session_returns_state(self, async_client):
        """GET /sessions/{id} returns the session state after creation."""
        # Create a session first
        payload = InboundEventPayload(payload={"message": "Book appointment"}).model_dump()
        create_resp = await async_client.post("/sessions", json=payload)
        assert create_resp.status_code == 200
        session_id = create_resp.json()["session_id"]

        # Yield to let background task complete
        await asyncio.sleep(0)

        # Retrieve it
        get_resp = await async_client.get(f"/sessions/{session_id}")
        assert get_resp.status_code == 200
        state = get_resp.json()
        assert state["session_id"] == session_id
        assert state["resolved_intent"] == "appointment_booking_request"
        assert state["evaluation_passed"] is True

    async def test_get_session_returns_404_for_unknown(self, async_client):
        """GET /sessions/{id} returns 404 for unknown session."""
        resp = await async_client.get("/sessions/unknown-session-id")
        assert resp.status_code == 404

    async def test_inject_event_returns_new_session(self, async_client):
        """POST /sessions/{id}/events returns a new session_id."""
        # Create initial session
        payload = InboundEventPayload(payload={"message": "initial"}).model_dump()
        create_resp = await async_client.post("/sessions", json=payload)
        session_id = create_resp.json()["session_id"]

        # Inject new event
        new_payload = InboundEventPayload(payload={"message": "follow-up"}).model_dump()
        inject_resp = await async_client.post(
            f"/sessions/{session_id}/events",
            json=new_payload,
        )
        assert inject_resp.status_code == 200
        data = inject_resp.json()
        assert "session_id" in data
        assert data["session_id"] != session_id

    async def test_cancel_session_returns_stub(self, async_client):
        """POST /sessions/{id}/cancel returns not_implemented."""
        # Create a session first
        payload = InboundEventPayload(payload={"message": "test"}).model_dump()
        create_resp = await async_client.post("/sessions", json=payload)
        session_id = create_resp.json()["session_id"]

        cancel_resp = await async_client.post(f"/sessions/{session_id}/cancel")
        assert cancel_resp.status_code == 501

    async def test_cancel_session_returns_404_for_unknown(self, async_client):
        """POST /sessions/{id}/cancel returns 404 for unknown session."""
        resp = await async_client.post("/sessions/unknown-session-id/cancel")
        assert resp.status_code == 404


class TestFullFlow:
    """End-to-end flow: create session -> state available via GET."""

    async def test_full_flow_create_then_get(self, async_client):
        """Full flow: create session, verify state accessible via GET."""
        payload = InboundEventPayload(payload={"event": "test_full_flow"}).model_dump()
        create_resp = await async_client.post("/sessions", json=payload)
        assert create_resp.status_code == 200
        session_id = create_resp.json()["session_id"]

        # Yield to let background task complete
        await asyncio.sleep(0)

        get_resp = await async_client.get(f"/sessions/{session_id}")
        assert get_resp.status_code == 200
        state = get_resp.json()
        assert state["session_id"] == session_id
        assert state["inbound_event_payload"] == {"event": "test_full_flow"}

    async def test_get_mcp_servers_before_create(self, async_client):
        """Health MCP check works without creating a session first."""
        resp = await async_client.get("/health/mcp-servers")
        assert resp.status_code == 200
        assert "servers" in resp.json()
