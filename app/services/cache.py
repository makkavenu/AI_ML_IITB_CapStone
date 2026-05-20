"""Redis-backed response cache for the AI agent API.

The cache is intentionally placed around the final chat response, not inside the
LangGraph nodes. This keeps tool/orchestrator code simple and allows repeated
questions to skip expensive LLM/tool calls while still preserving guardrails,
S3 verification, DynamoDB persistence, and SSE semantics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from app.services.metrics import CACHE_EVENTS_TOTAL, CACHE_OPERATION_DURATION_SECONDS

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised in Docker/Redis integration environments
    import redis.asyncio as redis_asyncio
except Exception:  # pragma: no cover - keeps local unit tests importable without redis installed
    redis_asyncio = None  # type: ignore[assignment]

REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
CACHE_KEY_PREFIX: str = os.getenv("CACHE_KEY_PREFIX", "ai_agent:response")

_redis_client: Any | None = None


def is_cache_enabled() -> bool:
    """Return whether response caching should be attempted."""
    return CACHE_ENABLED and CACHE_TTL_SECONDS > 0 and redis_asyncio is not None


def _normalise_for_key(value: Any) -> Any:
    """Reduce request payloads to deterministic, JSON-serialisable cache-key data."""
    if isinstance(value, dict):
        return {str(k): _normalise_for_key(v) for k, v in sorted(value.items()) if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [_normalise_for_key(v) for v in value]
    return value


def build_response_cache_key(
    *,
    message: str,
    orchestrator_model: str,
    history: list[dict[str, Any]] | None = None,
    files: list[dict[str, Any]] | None = None,
    version: str = "v1",
) -> str:
    """Build a stable Redis key for an agent response.

    The key excludes request_id/session_id so the same semantic request can be
    reused across sessions. Uploaded files are represented by stable S3 metadata
    after backend verification; this prevents accidental cross-file reuse.
    """
    file_fingerprints: list[dict[str, Any]] = []
    for f in files or []:
        file_fingerprints.append({
            "filename": f.get("filename"),
            "content_type": f.get("content_type"),
            "size_bytes": f.get("size_bytes"),
            "s3_bucket": f.get("s3_bucket"),
            "s3_key": f.get("s3_key"),
        })

    canonical = json.dumps(
        _normalise_for_key({
            "version": version,
            "message": (message or "").strip(),
            "orchestrator_model": orchestrator_model,
            "history": history or [],
            "files": file_fingerprints,
        }),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{CACHE_KEY_PREFIX}:{digest}"


def _get_redis_client() -> Any | None:
    """Create or reuse one Redis asyncio client."""
    global _redis_client
    if not is_cache_enabled():
        return None
    if _redis_client is None:
        _redis_client = redis_asyncio.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    return _redis_client


async def get_cached_response(cache_key: str, *, redis_client: Any | None = None) -> dict[str, Any] | None:
    """Return a cached final-event payload, or None on miss/disabled/error."""
    if not is_cache_enabled() and redis_client is None:
        CACHE_EVENTS_TOTAL.labels(event="disabled").inc()
        return None

    client = redis_client or _get_redis_client()
    if client is None:
        CACHE_EVENTS_TOTAL.labels(event="disabled").inc()
        return None

    with CACHE_OPERATION_DURATION_SECONDS.labels(operation="get").time():
        try:
            raw = await client.get(cache_key)
        except Exception:
            logger.exception("Redis cache get failed | key=%s", cache_key)
            CACHE_EVENTS_TOTAL.labels(event="error").inc()
            return None

    if not raw:
        CACHE_EVENTS_TOTAL.labels(event="miss").inc()
        return None

    try:
        cached = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring invalid cached JSON | key=%s", cache_key)
        CACHE_EVENTS_TOTAL.labels(event="error").inc()
        return None

    CACHE_EVENTS_TOTAL.labels(event="hit").inc()
    return cached


async def set_cached_response(
    cache_key: str,
    payload: dict[str, Any],
    *,
    ttl_seconds: int | None = None,
    redis_client: Any | None = None,
) -> None:
    """Store a successful final response in Redis."""
    if not is_cache_enabled() and redis_client is None:
        CACHE_EVENTS_TOTAL.labels(event="disabled").inc()
        return

    client = redis_client or _get_redis_client()
    if client is None:
        CACHE_EVENTS_TOTAL.labels(event="disabled").inc()
        return

    ttl = int(ttl_seconds or CACHE_TTL_SECONDS)
    if ttl <= 0:
        CACHE_EVENTS_TOTAL.labels(event="disabled").inc()
        return

    value = json.dumps(payload, ensure_ascii=False, default=str)
    with CACHE_OPERATION_DURATION_SECONDS.labels(operation="set").time():
        try:
            await client.setex(cache_key, ttl, value)
            CACHE_EVENTS_TOTAL.labels(event="set").inc()
        except Exception:
            logger.exception("Redis cache set failed | key=%s", cache_key)
            CACHE_EVENTS_TOTAL.labels(event="error").inc()
