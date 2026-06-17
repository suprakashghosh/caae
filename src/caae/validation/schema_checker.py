"""Schema checker — validates runtime payloads against registered schemas."""

import logging
from typing import Any

from pydantic import ValidationError

from caae.models.schemas import get_default_registry
from caae.models.schemas.base import SchemaRegistry, SchemaRegistryError

logger = logging.getLogger(__name__)


def execute_pydantic_schema_check(
    result: dict[str, Any],
    schema_contract: str,
    registry: SchemaRegistry | None = None,
) -> tuple[bool, str | None]:
    """Validate a result dict against a registered Pydantic schema contract.

    Args:
        result: The data dict to validate.
        schema_contract: Dotted path like 'schemas.clinical.AppointmentBookingPayload'.
        registry: SchemaRegistry instance. If None, uses the default registry.

    Returns:
        Tuple of (passed: bool, error: str | None).
        On success, returns (True, None).
        On failure, returns (False, error_description).
    """
    if registry is None:
        registry = get_default_registry()

    try:
        schema_class = registry.resolve(schema_contract)
    except SchemaRegistryError:
        msg = f"Schema contract '{schema_contract}' not found in registry"
        logger.warning(msg)
        return (False, msg)

    try:
        schema_class.model_validate(result)
        return (True, None)
    except ValidationError as exc:
        msg = f"Schema validation failed for '{schema_contract}': {exc.errors()}"
        logger.warning(msg)
        return (False, msg)
    except Exception as exc:
        msg = f"Unexpected error during schema validation for '{schema_contract}': {exc}"
        logger.error(msg)
        return (False, msg)
