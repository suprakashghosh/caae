"""FastAPI application entry point for CAAE."""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from caae.api.routes.health import router as health_router
from caae.api.routes.sessions import router as sessions_router
from caae.engine import CAAEEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler — initialises engine and cleans up on shutdown."""
    mcp_config_path = os.environ.get(
        "CAAE_MCP_CONFIG",
        "configs/mcp_config.json",
    )
    workflow_policy_path = os.environ.get(
        "CAAE_WORKFLOW_POLICY",
        "configs/workflow_policy.json",
    )

    engine = CAAEEngine(
        mcp_config_path=mcp_config_path,
        workflow_policy_path=workflow_policy_path,
    )
    await engine.start()
    app.state.engine = engine
    yield
    await engine.stop()


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
