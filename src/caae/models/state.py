"""UnifiedContextState — central state model for the CAAE LangGraph state machine."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UnifiedContextState(BaseModel):
    """Central state model for the CAAE LangGraph state machine.

    This model serves as the graph schema. LangGraph nodes receive this state
    and return partial dict updates (not full state objects).
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    inbound_event_payload: dict[str, Any]
    resolved_intent: str | None = None
    mcp_retrieved_resources: list[dict[str, Any]] = Field(default_factory=list)
    extracted_quantitative_data: dict[str, Any] = Field(default_factory=dict)
    execution_mutation_result: dict[str, Any] = Field(default_factory=dict)
    validation_retry_count: int = Field(default=0, le=3)
