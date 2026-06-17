"""Clinical domain schemas — e.g. AppointmentBookingPayload."""


from pydantic import BaseModel, ConfigDict


class AppointmentBookingPayload(BaseModel):
    """Schema for appointment booking requests.

    Used as the runtime_schema_contract for the 'appointment_booking_request'
    intent in workflow_policy.json.
    """

    model_config = ConfigDict(extra="forbid")

    lead_id: str
    practitioner_id: str
    preferred_date: str  # ISO 8601 date
    preferred_time: str  # HH:MM format
    appointment_type: str  # e.g., "initial_consultation", "follow_up", "treatment"
    notes: str | None = None
