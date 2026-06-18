"""In-memory data store for the demo MCP server."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any


@dataclass
class Appointment:
    """Represents a booked appointment."""

    appointment_id: str
    lead_id: str
    practitioner_id: str
    timestamp_iso: str
    status: str = "confirmed"


# Pre-populated sample practitioners
PRACTITIONERS = ["dr-smith", "dr-jones", "dr-wilson"]

# In-memory store: appointments keyed by practitioner_id
_appointments: dict[str, list[Appointment]] = {p: [] for p in PRACTITIONERS}

# Business hours
_START_HOUR = 9
_END_HOUR = 17
_SLOT_MINUTES = 30


def _generate_slots_for_date(date_str: str) -> list[str]:
    """Generate all 30-min ISO datetime strings for a given date (09:00-17:00 UTC).

    Args:
        date_str: ISO date string (e.g. "2024-01-15").

    Returns:
        List of ISO 8601 datetime strings for each slot.
    """
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: '{date_str}'. Expected ISO date YYYY-MM-DD.")
    slots: list[str] = []
    current = datetime(date.year, date.month, date.day, _START_HOUR, 0, 0, tzinfo=UTC)
    end = datetime(date.year, date.month, date.day, _END_HOUR, 0, 0, tzinfo=UTC)
    while current < end:
        slots.append(current.isoformat())
        current += timedelta(minutes=_SLOT_MINUTES)
    return slots


def get_available_slots(date_str: str, practitioner_id: str) -> list[dict[str, Any]]:
    """Return available 30-min slots for a practitioner on a given date.

    Only slots that are not already booked are marked available.

    Args:
        date_str: ISO date string (e.g. "2024-01-15").
        practitioner_id: Practitioner identifier.

    Returns:
        List of slot dicts with ``timestamp_iso`` and ``available`` keys.
    """
    if practitioner_id not in PRACTITIONERS:
        return []

    all_slots = _generate_slots_for_date(date_str)
    booked_times = {a.timestamp_iso for a in _appointments.get(practitioner_id, []) if a.status == "confirmed"}

    return [{"timestamp_iso": slot, "available": slot not in booked_times} for slot in all_slots]


def book_appointment(lead_id: str, timestamp_iso: str, practitioner_id: str) -> dict[str, Any]:
    """Book a slot for the given lead at the exact timestamp.

    Args:
        lead_id: Lead/customer identifier.
        timestamp_iso: ISO 8601 datetime string of the slot.
        practitioner_id: Practitioner identifier.

    Returns:
        Dict with appointment details on success, or ``{"status": "unavailable"}``
        if the slot is already booked or the practitioner is unknown.
    """
    if practitioner_id not in PRACTITIONERS:
        return {"status": "unknown_practitioner"}

    for appt in _appointments.get(practitioner_id, []):
        if appt.timestamp_iso == timestamp_iso and appt.status == "confirmed":
            return {"status": "unavailable"}

    appt = Appointment(
        appointment_id=str(uuid.uuid4()),
        lead_id=lead_id,
        practitioner_id=practitioner_id,
        timestamp_iso=timestamp_iso,
        status="confirmed",
    )
    _appointments[practitioner_id].append(appt)

    return {
        "appointment_id": appt.appointment_id,
        "lead_id": appt.lead_id,
        "practitioner_id": appt.practitioner_id,
        "timestamp_iso": appt.timestamp_iso,
        "status": appt.status,
    }


def list_appointments_for_practitioner(
    practitioner_id: str,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List all booked appointments for a practitioner, optionally filtered by status.

    Args:
        practitioner_id: Practitioner identifier.
        status: Optional status filter (e.g. "confirmed", "cancelled").

    Returns:
        List of appointment dicts.
    """
    if practitioner_id not in PRACTITIONERS:
        return []

    appointments = _appointments.get(practitioner_id, [])
    if status is not None:
        appointments = [a for a in appointments if a.status == status]

    return [
        {
            "appointment_id": a.appointment_id,
            "lead_id": a.lead_id,
            "practitioner_id": a.practitioner_id,
            "timestamp_iso": a.timestamp_iso,
            "status": a.status,
        }
        for a in appointments
    ]


def cancel_appointment(appointment_id: str) -> dict[str, Any]:
    """Cancel an appointment by ID.

    Args:
        appointment_id: UUID of the appointment to cancel.

    Returns:
        Dict with ``appointment_id`` and ``status`` ("cancelled" or "not_found").
    """
    for practitioner_id in PRACTITIONERS:
        for appt in _appointments[practitioner_id]:
            if appt.appointment_id == appointment_id:
                appt.status = "cancelled"
                return {"appointment_id": appt.appointment_id, "status": "cancelled"}
    return {"appointment_id": appointment_id, "status": "not_found"}
