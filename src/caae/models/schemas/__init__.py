"""Schema registry for runtime payload validation."""

from caae.models.schemas.base import SchemaRegistry, SchemaRegistryError
from caae.models.schemas.clinical import AppointmentBookingPayload
from caae.models.schemas.media import QuantitativeIntelPayload

__all__ = [
    "SchemaRegistry",
    "SchemaRegistryError",
    "AppointmentBookingPayload",
    "QuantitativeIntelPayload",
]

# Global schema registry with built-in schemas pre-registered
_default_registry: SchemaRegistry | None = None


def get_default_registry() -> SchemaRegistry:
    """Get the default schema registry with all built-in schemas registered."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SchemaRegistry()
        _default_registry.register("schemas.clinical.AppointmentBookingPayload", AppointmentBookingPayload)
        _default_registry.register("schemas.media.QuantitativeIntelPayload", QuantitativeIntelPayload)
    return _default_registry
