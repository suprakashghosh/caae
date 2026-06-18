"""Integration tests for the demo MCP scheduling server.

Connects to the running demo server via stdio transport, discovers tools,
and exercises the full lifecycle (check -> book -> list -> cancel).
"""

import json
from typing import Any

import pytest

from caae.mcp import MCPClientManager
from caae.models.config import MCPConfig, StdioMCPServerConfig

_DEMO_SERVER_CONFIG = StdioMCPServerConfig(
    command="uv",
    args=["run", "caae-demo-server"],
    timeout_ms=30000,
)


# FastMCP splits list<T> returns into one TextContent per element.
# Helper: collect all content items into a Python list.
def _collect_items(content: list[Any]) -> list[Any]:
    """Collect all text content items into a list of deserialized values."""
    return [json.loads(item.text) for item in content]


def _first_item(content: list[Any]) -> Any:
    """Deserialize the first text content item."""
    return json.loads(content[0].text)


@pytest.mark.asyncio
@pytest.mark.integration
class TestDemoSchedulingServer:
    """Integration tests for the demo scheduling MCP server."""

    @pytest.fixture
    def manager(self) -> MCPClientManager:
        """Return a fresh MCPClientManager (caller must start/stop)."""
        return MCPClientManager()

    @pytest.fixture
    def config(self) -> MCPConfig:
        """Return an MCPConfig pointing only at the demo server."""
        return MCPConfig(
            system_mode="testing",
            active_environment="test",
            mcp_servers={"scheduling_engine": _DEMO_SERVER_CONFIG},
        )

    async def test_tool_discovery(self, manager: MCPClientManager, config: MCPConfig) -> None:
        """Discover tools and verify all 4 exist with correct schemas."""
        await manager.start(config)
        try:
            tools = await manager.list_tools("scheduling_engine")
            tool_names = [t.name for t in tools]

            assert "check_availability" in tool_names
            assert "book_slot" in tool_names
            assert "list_appointments" in tool_names
            assert "cancel_appointment" in tool_names
            assert len(tools) == 4

            # Verify every tool has a valid JSON schema
            for tool in tools:
                assert tool.input_schema["type"] == "object"
                assert "properties" in tool.input_schema
                assert isinstance(tool.input_schema["properties"], dict)
        finally:
            await manager.stop()

    async def test_tool_schemas_have_correct_params(self, manager: MCPClientManager, config: MCPConfig) -> None:
        """Each tool's schema lists the expected parameter names."""
        await manager.start(config)
        try:
            schema = manager.get_tool_schema("scheduling_engine", "check_availability")
            assert "date" in schema["properties"]
            assert "practitioner_id" in schema["properties"]

            schema = manager.get_tool_schema("scheduling_engine", "book_slot")
            assert "lead_id" in schema["properties"]
            assert "timestamp_iso" in schema["properties"]
            assert "practitioner_id" in schema["properties"]

            schema = manager.get_tool_schema("scheduling_engine", "list_appointments")
            assert "practitioner_id" in schema["properties"]

            schema = manager.get_tool_schema("scheduling_engine", "cancel_appointment")
            assert "appointment_id" in schema["properties"]
        finally:
            await manager.stop()

    async def test_check_availability_empty_for_unknown_practitioner(
        self, manager: MCPClientManager, config: MCPConfig
    ) -> None:
        """Unknown practitioner returns empty list."""
        await manager.start(config)
        try:
            result = await manager.call_tool(
                "scheduling_engine",
                "check_availability",
                {"date": "2024-01-15", "practitioner_id": "unknown-doc"},
            )
            assert not result["is_error"]
            slots = _collect_items(result["content"])
            assert slots == []
        finally:
            await manager.stop()

    async def test_check_availability_returns_slots(self, manager: MCPClientManager, config: MCPConfig) -> None:
        """check_availability returns 30-min slots for a valid practitioner and date."""
        await manager.start(config)
        try:
            result = await manager.call_tool(
                "scheduling_engine",
                "check_availability",
                {"date": "2024-01-15", "practitioner_id": "dr-smith"},
            )
            assert not result["is_error"]
            slots = _collect_items(result["content"])
            assert isinstance(slots, list)
            assert len(slots) > 0

            # Each slot has required keys
            for slot in slots:
                assert "timestamp_iso" in slot
                assert "available" in slot

            # All slots available initially
            assert all(s["available"] for s in slots)

            # 09:00-17:00 in 30-min increments = 16 slots
            assert len(slots) == 16
        finally:
            await manager.stop()

    async def test_full_lifecycle(self, manager: MCPClientManager, config: MCPConfig) -> None:
        """Full lifecycle: check -> book -> verify unavailable -> list -> cancel -> verify."""
        await manager.start(config)
        try:
            practitioner = "dr-jones"
            the_date = "2024-06-18"

            # 1. Check availability
            avail_result = await manager.call_tool(
                "scheduling_engine",
                "check_availability",
                {"date": the_date, "practitioner_id": practitioner},
            )
            slots = _collect_items(avail_result["content"])
            assert len(slots) >= 2
            first_slot = slots[0]["timestamp_iso"]
            second_slot = slots[1]["timestamp_iso"]

            # 2. Book the first slot
            book_result = await manager.call_tool(
                "scheduling_engine",
                "book_slot",
                {
                    "lead_id": "lead-001",
                    "timestamp_iso": first_slot,
                    "practitioner_id": practitioner,
                },
            )
            booking = _first_item(book_result["content"])
            assert booking["status"] == "confirmed"
            assert booking["lead_id"] == "lead-001"
            assert booking["practitioner_id"] == practitioner
            assert booking["timestamp_iso"] == first_slot
            appointment_id = booking["appointment_id"]
            assert appointment_id is not None

            # 3. Verify that booked slot is now unavailable
            avail_result2 = await manager.call_tool(
                "scheduling_engine",
                "check_availability",
                {"date": the_date, "practitioner_id": practitioner},
            )
            slots2 = _collect_items(avail_result2["content"])
            booked_slot = next(s for s in slots2 if s["timestamp_iso"] == first_slot)
            assert booked_slot["available"] is False

            # 4. List appointments
            list_result = await manager.call_tool(
                "scheduling_engine",
                "list_appointments",
                {"practitioner_id": practitioner},
            )
            appointments = _collect_items(list_result["content"])
            assert len(appointments) == 1
            assert appointments[0]["appointment_id"] == appointment_id
            assert appointments[0]["status"] == "confirmed"

            # 5. Cancel the appointment
            cancel_result = await manager.call_tool(
                "scheduling_engine",
                "cancel_appointment",
                {"appointment_id": appointment_id},
            )
            cancel_data = _first_item(cancel_result["content"])
            assert cancel_data["status"] == "cancelled"
            assert cancel_data["appointment_id"] == appointment_id

            # 6. List again -- still shows but status is now cancelled
            list_result2 = await manager.call_tool(
                "scheduling_engine",
                "list_appointments",
                {"practitioner_id": practitioner},
            )
            appointments2 = _collect_items(list_result2["content"])
            assert len(appointments2) == 1
            assert appointments2[0]["status"] == "cancelled"

            # 7. Cancel non-existent appointment returns not_found
            cancel_result2 = await manager.call_tool(
                "scheduling_engine",
                "cancel_appointment",
                {"appointment_id": "non-existent-uuid"},
            )
            cancel_data2 = _first_item(cancel_result2["content"])
            assert cancel_data2["status"] == "not_found"

            # 8. Double-booking returns unavailable
            second_slot_iso = second_slot
            book_a = await manager.call_tool(
                "scheduling_engine",
                "book_slot",
                {
                    "lead_id": "lead-002",
                    "timestamp_iso": second_slot_iso,
                    "practitioner_id": practitioner,
                },
            )
            assert _first_item(book_a["content"])["status"] == "confirmed"

            book_b = await manager.call_tool(
                "scheduling_engine",
                "book_slot",
                {
                    "lead_id": "lead-003",
                    "timestamp_iso": second_slot_iso,
                    "practitioner_id": practitioner,
                },
            )
            assert _first_item(book_b["content"])["status"] == "unavailable"
        finally:
            await manager.stop()

    async def test_list_appointments_empty_for_unknown_practitioner(
        self, manager: MCPClientManager, config: MCPConfig
    ) -> None:
        """list_appointments for unknown practitioner returns empty list."""
        await manager.start(config)
        try:
            result = await manager.call_tool(
                "scheduling_engine",
                "list_appointments",
                {"practitioner_id": "nonexistent"},
            )
            assert not result["is_error"]
            appointments = _collect_items(result["content"])
            assert appointments == []
        finally:
            await manager.stop()
