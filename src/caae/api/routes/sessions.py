"""Session management endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/sessions")
def create_session():
    """Create a new CAAE session and return a stub session ID."""
    # TODO: Implement session creation in subsequent sub-tasks
    return {"session_id": "stub"}
