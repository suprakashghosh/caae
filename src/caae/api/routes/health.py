"""Health check endpoint."""

from fastapi import APIRouter, Request

from caae.api.models import HealthResponse, MCPServerHealth, MCPServersHealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Return health status of the CAAE server."""
    return {"status": "ok"}


@router.get("/health/mcp-servers", response_model=MCPServersHealthResponse)
def mcp_servers_health(request: Request):
    """Return connection status of all configured MCP servers."""
    engine = request.app.state.engine
    raw_statuses = engine.get_mcp_server_statuses()

    servers: dict[str, MCPServerHealth] = {}
    for name, status_data in raw_statuses.items():
        servers[name] = MCPServerHealth(
            status=status_data["status"],
            error=status_data.get("error"),
        )
    return MCPServersHealthResponse(servers=servers)
