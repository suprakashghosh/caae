"""Demo MCP server implementation — scheduling engine."""

from mcp.server.fastmcp import FastMCP

from caae_demo_server import data

mcp = FastMCP("CAAE Demo Scheduling")


@mcp.tool()
def check_availability(date: str, practitioner_id: str) -> list[dict]:
    """Check available time slots for a practitioner on a given date.

    Args:
        date: ISO date string (e.g. "2024-01-15").
        practitioner_id: The practitioner's identifier (e.g. "dr-smith").

    Returns:
        A list of 30-minute slot dicts with "timestamp_iso" and "available" keys.
    """
    return data.get_available_slots(date, practitioner_id)


@mcp.tool()
def book_slot(lead_id: str, timestamp_iso: str, practitioner_id: str) -> dict:
    """Book an appointment slot for a lead.

    Args:
        lead_id: The lead/customer identifier.
        timestamp_iso: ISO 8601 datetime string for the slot.
        practitioner_id: The practitioner's identifier.

    Returns:
        A dict with appointment details (appointment_id, lead_id, practitioner_id,
        timestamp_iso, status). If the slot is unavailable, status is "unavailable".
    """
    return data.book_appointment(lead_id, timestamp_iso, practitioner_id)


@mcp.tool()
def list_appointments(practitioner_id: str, status: str | None = None) -> list[dict]:
    """List all booked appointments for a practitioner, optionally filtered by status.

    Args:
        practitioner_id: The practitioner's identifier.
        status: Optional status filter (e.g. "confirmed", "cancelled").

    Returns:
        A list of appointment dicts.
    """
    return data.list_appointments_for_practitioner(practitioner_id, status=status)


@mcp.tool()
def cancel_appointment(appointment_id: str) -> dict:
    """Cancel an appointment by ID.

    Args:
        appointment_id: The UUID of the appointment to cancel.

    Returns:
        A dict with "appointment_id" and "status" ("cancelled" or "not_found").
    """
    return data.cancel_appointment(appointment_id)
