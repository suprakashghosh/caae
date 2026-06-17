"""Schema registry — maps schema contract names to Pydantic models."""


from pydantic import BaseModel


class SchemaRegistryError(KeyError):
    """Raised when a schema contract path cannot be resolved."""


class SchemaRegistry:
    """Registry mapping dotted schema contract paths to Pydantic model classes.

    Used by the evaluation gate and schema checker to validate runtime payloads
    against the schema contracts defined in workflow_policy.json.
    """

    _schemas: dict[str, type[BaseModel]]

    def __init__(self) -> None:
        self._schemas: dict[str, type[BaseModel]] = {}

    def register(self, path: str, schema: type[BaseModel]) -> None:
        """Register a Pydantic model under a dotted path.

        Args:
            path: Dotted path like 'schemas.clinical.AppointmentBookingPayload'
            schema: The Pydantic model class to register

        Raises:
            ValueError: If a schema is already registered at the given path
        """
        if path in self._schemas:
            raise ValueError(f"Schema already registered at path '{path}'")
        self._schemas[path] = schema

    def resolve(self, path: str) -> type[BaseModel]:
        """Resolve a dotted path to its registered Pydantic model class.

        Args:
            path: Dotted path like 'schemas.clinical.AppointmentBookingPayload'

        Returns:
            The registered Pydantic model class

        Raises:
            SchemaRegistryError: If no schema is registered at the given path
        """
        if path not in self._schemas:
            raise SchemaRegistryError(f"No schema registered at path '{path}'. Available: {list(self._schemas.keys())}")
        return self._schemas[path]

    def list_schemas(self) -> list[str]:
        """Return a sorted list of all registered schema paths."""
        return sorted(self._schemas.keys())

    def has_schema(self, path: str) -> bool:
        """Check if a schema is registered at the given path."""
        return path in self._schemas
