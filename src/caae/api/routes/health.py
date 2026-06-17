"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    """Return health status of the CAAE server."""
    return {"status": "ok"}
