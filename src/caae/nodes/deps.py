"""Dependencies dataclass — injected via LangGraph config into every node."""

from dataclasses import dataclass

from caae.mcp.client_manager import MCPClientManager
from caae.models.config import WorkflowPolicy
from caae.models.schemas.base import SchemaRegistry
from caae.observability.langfuse_handler import LangfuseHandler


@dataclass
class Dependencies:
    """Runtime dependencies injected into LangGraph nodes via ``config["configurable"]["deps"]``.

    Every node extracts this object from the LangGraph RunnableConfig
    to access shared resources (MCP, policy, schema registry, LLM).
    """

    mcp_client: MCPClientManager
    workflow_policy: WorkflowPolicy
    schema_registry: SchemaRegistry
    llm_model_name: str = "gpt-4o"
    llm_provider: str = "openai"
    langfuse_handler: LangfuseHandler | None = None
