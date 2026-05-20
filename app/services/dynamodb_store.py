"""Persistence helpers for the ai-agent-requests DynamoDB table."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from app.services.aws_clients import request_table

logger = logging.getLogger(__name__)

SESSION_INDEX_NAME: str = os.getenv("DDB_SESSION_INDEX_NAME", "session_id-created_at-index")

# Keep DynamoDB records small. Large binaries/base64 strings remain in S3,
# Redis/SSE, or in memory for model calls; DynamoDB stores only metadata.
MAX_DDB_STRING_CHARS: int = int(os.getenv("MAX_DDB_STRING_CHARS", "20000"))
_LARGE_PAYLOAD_KEY_TERMS: tuple[str, ...] = (
    "base64",
    "data_url",
    "embedding",
    "mask",
    "overlay",
    "image_bytes",
    "raw",
)


def _looks_like_large_payload_key(key_name: str | None) -> bool:
    lower = (key_name or "").lower()
    return any(term in lower for term in _LARGE_PAYLOAD_KEY_TERMS)


def _ddb_safe_write_value(value: Any, *, key_name: str | None = None) -> Any:
    """Return a DynamoDB-safe value by omitting large binary/model payloads.

    DynamoDB items are limited to 400 KB. The app already stores uploaded
    images in S3 and streams UI-only images over SSE, so DynamoDB should keep
    metadata, pointers, statuses, and final text rather than base64 payloads.
    """
    if isinstance(value, dict):
        return {k: _ddb_safe_write_value(v, key_name=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_ddb_safe_write_value(v, key_name=key_name) for v in value]
    if isinstance(value, str):
        if _looks_like_large_payload_key(key_name):
            return {
                "omitted_from_dynamodb": True,
                "reason": "large payload is stored outside DynamoDB",
                "original_chars": len(value),
            }
        if len(value) > MAX_DDB_STRING_CHARS:
            omitted = len(value) - MAX_DDB_STRING_CHARS
            return f"{value[:MAX_DDB_STRING_CHARS]}\n...[truncated {omitted} chars for DynamoDB item-size safety]"
    return value


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert DynamoDB Decimal/list/dict values into JSON-safe Python types."""
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def put_request_item(item: dict[str, Any]) -> None:
    """Insert a new request item.

    Table primary key: ``request_id`` (string). For session history queries the
    table should also have GSI ``session_id-created_at-index`` with
    ``session_id`` as partition key and ``created_at`` as sort key.
    """
    try:
        request_table().put_item(
            Item=_ddb_safe_write_value(item),
            ConditionExpression="attribute_not_exists(request_id)",
        )
    except ClientError:
        logger.exception("put_request_item failed | request_id=%s", item.get("request_id"))
        raise


def update_request_status(
    request_id: str,
    status: str,
    *,
    stage: str | None = None,
    error_message: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Update request status/stage/error fields."""
    now = utc_now_iso()
    names = {"#status": "status", "#updated_at": "updated_at"}
    values: dict[str, Any] = {":status": status, ":updated_at": now}
    set_parts = ["#status = :status", "#updated_at = :updated_at"]

    if stage is not None:
        names["#stage"] = "stage"
        values[":stage"] = stage
        set_parts.append("#stage = :stage")
    if error_message is not None:
        names["#error_message"] = "error_message"
        values[":error_message"] = error_message
        set_parts.append("#error_message = :error_message")
    if extra:
        for idx, (key, value) in enumerate(extra.items()):
            n = f"#extra_{idx}"
            v = f":extra_{idx}"
            names[n] = key
            values[v] = _ddb_safe_write_value(value, key_name=key)
            set_parts.append(f"{n} = {v}")

    try:
        request_table().update_item(
            Key={"request_id": request_id},
            UpdateExpression="SET " + ", ".join(set_parts),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )
    except ClientError:
        logger.exception("update_request_status failed | request_id=%s", request_id)
        raise


def get_request_item(request_id: str) -> dict[str, Any] | None:
    """Return one request item from DynamoDB, or None when not found."""
    resp = request_table().get_item(Key={"request_id": request_id})
    item = resp.get("Item")
    return _json_safe(item) if item else None


def list_requests_by_session(
    session_id: str,
    *,
    limit: int = 50,
    newest_first: bool = True,
) -> list[dict[str, Any]]:
    """List request/chat history items for one chat session via DynamoDB GSI.

    Requires a GSI named by ``DDB_SESSION_INDEX_NAME`` (default
    ``session_id-created_at-index``) with:

    * partition key: ``session_id`` (String)
    * sort key: ``created_at`` (String)
    """
    safe_limit = max(1, min(int(limit), 100))
    try:
        resp = request_table().query(
            IndexName=SESSION_INDEX_NAME,
            KeyConditionExpression=Key("session_id").eq(session_id),
            ScanIndexForward=not newest_first,
            Limit=safe_limit,
        )
    except ClientError:
        logger.exception("list_requests_by_session failed | session_id=%s", session_id)
        raise
    return [_json_safe(item) for item in resp.get("Items", [])]
