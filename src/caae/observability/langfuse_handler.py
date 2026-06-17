"""Langfuse observability handler — tracing, cost tracking, and budget enforcement."""

import logging
from typing import Any

from langfuse import Langfuse

logger = logging.getLogger(__name__)


class LangfuseHandler:
    """Manages Langfuse tracing for CAAE sessions.

    If Langfuse is not configured (no API keys), all operations are safe no-ops.
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the Langfuse client.

        Args:
            public_key: Langfuse public key. Falls back to LANGFUSE_PUBLIC_KEY env var.
            secret_key: Langfuse secret key. Falls back to LANGFUSE_SECRET_KEY env var.
            base_url: Langfuse base URL. Falls back to LANGFUSE_BASE_URL env var.
        """
        self._public_key = public_key
        self._secret_key = secret_key
        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
        )
        self._enabled: bool = self._is_configured()
        self._session_costs: dict[str, float] = {}

    def _is_configured(self) -> bool:
        """Check if Langfuse is properly configured (has API keys).

        Checks constructor args first, then falls back to env vars.
        """
        import os

        pk = self._public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        sk = self._secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        return bool(pk and sk)

    @property
    def client(self) -> Langfuse:
        """Get the underlying Langfuse client."""
        return self._client

    @property
    def enabled(self) -> bool:
        """Whether Langfuse tracing is active."""
        return self._enabled

    def start_trace(self, session_id: str, metadata: dict[str, Any] | None = None) -> Any:
        """Create a parent trace observation for a session.

        Args:
            session_id: The session ID for this trace.
            metadata: Optional metadata dict.

        Returns:
            A Langfuse observation (span) representing the trace root.
            Can be used as a context manager.
        """
        if not self._enabled:
            return None
        return self._client.start_observation(
            name=f"session-{session_id}",
            as_type="span",
            input={"session_id": session_id},
            metadata=metadata or {},
        )

    def start_span(
        self, name: str, input_data: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None
    ) -> Any:
        """Create a child span observation.

        Args:
            name: Span name (e.g., "context_assessor", "tool_call").
            input_data: Optional input data for the span.
            metadata: Optional metadata dict.

        Returns:
            A Langfuse observation (span).
        """
        if not self._enabled:
            return None
        return self._client.start_observation(
            name=name,
            as_type="span",
            input=input_data,
            metadata=metadata or {},
        )

    def end_span(self, span: Any, output_data: dict[str, Any] | None = None) -> None:
        """Close a span with output data.

        Args:
            span: The Langfuse observation to close.
            output_data: Optional output data for the span.
        """
        if span is None or not self._enabled:
            return
        try:
            span.update(output=output_data)
            span.end()
        except Exception:
            logger.warning("Failed to end Langfuse span", exc_info=True)

    def record_tool_call(
        self,
        trace: Any,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        is_error: bool = False,
        trace_context: dict[str, Any] | None = None,
    ) -> None:
        """Record an MCP tool invocation as a Langfuse observation.

        Args:
            trace: The parent trace observation (may be None).
            server_name: Name of the MCP server.
            tool_name: Name of the tool invoked.
            arguments: Arguments passed to the tool.
            result: Tool call result (if available).
            latency_ms: Latency of the tool call in milliseconds.
            is_error: Whether the tool call resulted in an error.
            trace_context: Optional dict with trace identifiers (e.g. trace_root_id)
                           merged into the observation metadata.
        """
        if not self._enabled:
            return
        try:
            tool_span = self._client.start_observation(
                name=f"tool:{server_name}/{tool_name}",
                as_type="tool",
                input=arguments,
                metadata={
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "latency_ms": latency_ms,
                    "is_error": is_error,
                    **(trace_context or {}),
                },
            )
            tool_span.update(output=result)
            tool_span.end()
        except Exception:
            logger.warning("Failed to record tool call in Langfuse", exc_info=True)

    def track_cost(self, session_id: str, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        """Track LLM token usage and estimate cost for a session.

        Args:
            session_id: The session ID to track costs for.
            prompt_tokens: Number of prompt/input tokens used.
            completion_tokens: Number of completion/output tokens used.
            model: The LLM model name.
        """
        if not self._enabled:
            return

        cost_per_1k_tokens = {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        }

        default_cost = {"input": 0.005, "output": 0.02}
        cost_table = cost_per_1k_tokens.get(model, default_cost)

        incremental_cost = (
            prompt_tokens * cost_table["input"] / 1000.0 + completion_tokens * cost_table["output"] / 1000.0
        )

        self._session_costs[session_id] = self._session_costs.get(session_id, 0.0) + incremental_cost
        logger.debug(
            "Cost tracked for session %s: $%.6f (total: $%.6f)",
            session_id,
            incremental_cost,
            self._session_costs[session_id],
        )

    def is_within_budget(self, session_id: str, max_cost_usd: float) -> bool:
        """Check if a session is within its cost budget.

        Args:
            session_id: The session ID to check.
            max_cost_usd: Maximum allowed cost in USD.

        Returns:
            True if the session cost is within budget, False otherwise.
        """
        current_cost = self._session_costs.get(session_id, 0.0)
        return current_cost <= max_cost_usd

    def get_session_cost(self, session_id: str) -> float:
        """Get the total cost tracked for a session.

        Args:
            session_id: The session ID.

        Returns:
            Total cost in USD for the session.
        """
        return self._session_costs.get(session_id, 0.0)

    def flush(self) -> None:
        """Flush all pending traces to Langfuse."""
        if not self._enabled:
            return
        try:
            self._client.flush()
        except Exception:
            logger.warning("Failed to flush Langfuse traces", exc_info=True)

    def shutdown(self) -> None:
        """Flush and shut down the Langfuse client."""
        if not self._enabled:
            return
        try:
            self._client.flush()
            self._client.shutdown()
        except Exception:
            logger.warning("Failed to shut down Langfuse client", exc_info=True)
