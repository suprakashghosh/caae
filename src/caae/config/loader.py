"""Configuration loader — reads JSON config files from the configs/ directory."""

import json
from pathlib import Path

from pydantic import ValidationError

from caae.models.config import MCPConfig, WorkflowPolicy


class ConfigLoadError(Exception):
    """Raised when a configuration file cannot be loaded or validated."""


def load_mcp_config(path: str | Path) -> MCPConfig:
    """Load and validate MCP configuration from a JSON file.

    Args:
        path: Path to the MCP configuration JSON file.

    Returns:
        Validated MCPConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigLoadError: If the file contains invalid JSON or fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MCP config file not found: {path}")
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigLoadError(f"Invalid JSON in MCP config file {path}: {e}") from e
    try:
        return MCPConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigLoadError(f"Invalid MCP configuration in {path}:\n{e}") from e


def load_workflow_policy(path: str | Path) -> WorkflowPolicy:
    """Load and validate workflow policy from a JSON file.

    Args:
        path: Path to the workflow policy JSON file.

    Returns:
        Validated WorkflowPolicy instance.

    Raises:
        FileNotFoundError: If the policy file does not exist.
        ConfigLoadError: If the file contains invalid JSON or fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workflow policy file not found: {path}")
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigLoadError(f"Invalid JSON in workflow policy file {path}: {e}") from e
    try:
        return WorkflowPolicy.model_validate(data)
    except ValidationError as e:
        raise ConfigLoadError(f"Invalid workflow policy in {path}:\n{e}") from e
