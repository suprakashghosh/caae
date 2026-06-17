"""Unit tests for CAAEEngine orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from caae.engine import CAAEEngine


@pytest.fixture
def engine() -> CAAEEngine:
    """Fixture: CAAEEngine with test config paths."""
    return CAAEEngine(
        mcp_config_path="/fake/mcp.json",
        workflow_policy_path="/fake/policy.json",
        llm_model_name="test-model",
        llm_provider="test-provider",
    )


class TestCAAEEngine:
    """CAAEEngine — start / run_session / stop lifecycle."""

    # ── start() ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_start_loads_configs_starts_mcp_builds_graph(self, engine):
        """start() loads configs, starts MCP client, and builds the graph."""
        mock_mcp_config = MagicMock()
        mock_policy = MagicMock()
        mock_registry = MagicMock()
        mock_mcp_client = MagicMock()
        mock_graph = MagicMock()

        mock_mcp_client.start = AsyncMock()
        mock_mcp_client.stop = AsyncMock()

        with (
            patch("caae.engine.load_mcp_config", return_value=mock_mcp_config) as mock_load_mcp,
            patch("caae.engine.load_workflow_policy", return_value=mock_policy) as mock_load_policy,
            patch("caae.engine.get_default_registry", return_value=mock_registry) as mock_get_reg,
            patch("caae.engine.MCPClientManager", return_value=mock_mcp_client) as mock_mcp_cls,
            patch("caae.engine.build_caae_graph", return_value=mock_graph) as mock_build,
        ):
            await engine.start()

        mock_load_mcp.assert_called_once_with("/fake/mcp.json")
        mock_load_policy.assert_called_once_with("/fake/policy.json")
        mock_get_reg.assert_called_once()
        mock_mcp_cls.assert_called_once()
        mock_mcp_client.start.assert_awaited_once_with(mock_mcp_config)
        mock_build.assert_called_once()

        assert engine._graph is mock_graph
        assert engine._mcp_client is mock_mcp_client
        assert engine._workflow_policy is mock_policy
        assert engine._schema_registry is mock_registry

    # ── run_session() ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_run_session_raises_if_not_started(self, engine):
        """run_session() without start() raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not started"):
            await engine.run_session({"event": "data"})

    @pytest.mark.asyncio
    async def test_run_session_invokes_graph_and_stores_result(self, engine):
        """run_session() invokes compiled graph and caches result."""
        mock_policy = MagicMock()
        mock_registry = MagicMock()
        mock_mcp_client = MagicMock()

        # Mock the compiled graph
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock()

        final_state = MagicMock()
        final_state.model_dump.return_value = {"session_id": "abc-123", "evaluation_passed": True}
        final_state.session_id = "abc-123"
        mock_graph.ainvoke.return_value = final_state

        # Attach dependencies to engine
        engine._graph = mock_graph
        engine._mcp_client = mock_mcp_client
        engine._workflow_policy = mock_policy
        engine._schema_registry = mock_registry

        event = {"message": "hello"}
        result = await engine.run_session(event)

        # Graph invoked
        mock_graph.ainvoke.assert_awaited_once()

        # State serialized and cached
        assert result == {"session_id": "abc-123", "evaluation_passed": True}
        assert engine._sessions["abc-123"] == result

    # ── stop() ───────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stop_shuts_down_mcp_client(self, engine):
        """stop() calls mcp_client.stop() and clears the reference."""
        mock_mcp_client = MagicMock()
        mock_mcp_client.stop = AsyncMock()
        engine._mcp_client = mock_mcp_client

        await engine.stop()

        mock_mcp_client.stop.assert_awaited_once()
        assert engine._mcp_client is None

    @pytest.mark.asyncio
    async def test_stop_no_mcp_client_does_not_raise(self, engine):
        """stop() with no MCP client does not raise."""
        engine._mcp_client = None
        await engine.stop()  # should not raise

    # ── get_session_state() ──────────────────────────────────────────────

    def test_get_session_state_returns_none_for_missing(self, engine):
        """get_session_state() returns None for unknown session."""
        assert engine.get_session_state("nonexistent") is None

    def test_get_session_state_returns_stored_state(self, engine):
        """get_session_state() returns previously stored state."""
        engine._sessions["known-id"] = {"result": "ok"}
        assert engine.get_session_state("known-id") == {"result": "ok"}
