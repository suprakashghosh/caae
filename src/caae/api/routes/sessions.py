"""Session management endpoints."""

import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, status

from caae.api.models import (
    InboundEventPayload,
    SessionCreateResponse,
)
from caae.models.state import UnifiedContextState

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session(payload: InboundEventPayload, request: Request):
    """Create a new CAAE session and start graph execution.

    Returns a ``session_id`` immediately. The graph runs in background;
    poll ``GET /sessions/{id}`` for the final state.
    """
    engine = request.app.state.engine
    session_id = str(uuid.uuid4())

    # Placeholder while graph runs
    engine.initialize_session(session_id, payload.payload)

    # Start background execution
    asyncio.create_task(engine._run_session_and_store(payload.payload, session_id))

    return SessionCreateResponse(session_id=session_id)


@router.get("/sessions/{session_id}", response_model=UnifiedContextState)
async def get_session(session_id: str, request: Request):
    """Return the current / final state for a session."""
    engine = request.app.state.engine
    state = engine.get_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return UnifiedContextState(**state)


@router.post("/sessions/{session_id}/events", response_model=SessionCreateResponse)
async def inject_event(session_id: str, payload: InboundEventPayload, request: Request):
    """Inject a new event into an existing session (V1: creates a new session)."""
    engine = request.app.state.engine

    # Validate the original session exists
    existing = engine.get_session_state(session_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Original session not found")

    new_session_id = str(uuid.uuid4())
    logger.info("Injecting event from session %s into new session %s", session_id, new_session_id)

    engine.initialize_session(new_session_id, payload.payload)

    asyncio.create_task(engine._run_session_and_store(payload.payload, new_session_id))

    return SessionCreateResponse(session_id=new_session_id)


@router.post("/sessions/{session_id}/cancel")
async def cancel_session(session_id: str, request: Request):
    """Cancel a running session (V1 stub: not implemented)."""
    engine = request.app.state.engine
    state = engine.get_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="cancel not implemented",
    )
