"""Entry point for the demo MCP server."""

from caae_demo_server.server import mcp


def main() -> None:
    """Start the demo MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
