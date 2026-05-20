"""FastAPI application entry point.

Start locally (outside Docker):
    uvicorn app.main:app --reload --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.routers.chat import router as chat_router
from app.routers.uploads import router as uploads_router
from app.routers.sam_med2d import router as sam_med2d_router
from app.routers.retfound import router as retfound_router

# ---------------------------------------------------------------------------
# Logging — structured, stdout, suitable for Docker log collection
# ---------------------------------------------------------------------------

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the ASGI server while the application is running.
    """
    logger.info("Starting Multi-Modal AI Agent API")
    yield
    logger.info("Shutting down Multi-Modal AI Agent API")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Modal AI Agent",
    description=(
        "GPT-4o orchestrated multi-modal agent with specialised tools for "
        "medical QA (MedGemma), legal QA (Pinecone + Qwen/Bedrock), "
        "vision analysis (Qwen3-VL-2B), and object detection (YOLOv12-S)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the Streamlit UI (and any other origin during development) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose Prometheus-compatible metrics at /metrics. Prometheus scrapes this
# endpoint through the Docker network; do not expose it publicly in EC2
# security groups.
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)

app.include_router(chat_router, prefix="/api")
app.include_router(uploads_router, prefix="/api", tags=["uploads"])
app.include_router(sam_med2d_router, prefix="/api/sam-med2d", tags=["sam-med2d"])
app.include_router(retfound_router, prefix="/api/retfound", tags=["retfound"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe used by Docker Compose healthcheck.

    Returns:
        JSON body ``{"status": "ok"}``.
    """
    return {"status": "ok"}
