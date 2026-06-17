"""Media domain schemas — e.g. QuantitativeIntelPayload."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QuantitativeIntelPayload(BaseModel):
    """Schema for quantitative content intelligence payloads.

    Used as the runtime_schema_contract for the 'competitor_intel_deep_dive'
    intent in workflow_policy.json.
    """

    model_config = ConfigDict(extra="forbid")

    topic: str
    competitor_names: list[str] = Field(default_factory=list)
    metrics_requested: list[str] = Field(default_factory=list)  # e.g., ["views", "engagement_rate", "subscriber_count"]
    date_range_start: str | None = None  # ISO 8601 date
    date_range_end: str | None = None  # ISO 8601 date
    depth: Literal["summary", "detailed", "comprehensive"] = "summary"
