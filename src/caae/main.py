"""FastAPI application entry point for CAAE."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from caae.api.routes.health import router as health_router
from caae.api.routes.sessions import router as sessions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Placeholder lifespan handler for startup/shutdown events."""
    # TODO: Initialize engine, MCP clients, observability in subsequent sub-tasks
    yield
    # TODO: Cleanup resources in subsequent sub-tasks


app = FastAPI(
    title="CAAE",
    description="Central Adaptable Automation Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router, prefix="", tags=["health"])
app.include_router(sessions_router, prefix="", tags=["sessions"])


def run() -> None:
    """Entry point for `caae-server` CLI command."""
    uvicorn.run("caae.main:app", host="0.0.0.0", port=8000, reload=False)
