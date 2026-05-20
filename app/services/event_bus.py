"""In-process request event bus used by the SSE endpoint.

This is intentionally lightweight for a single-process Docker/EC2 demo. For a
multi-replica production deployment, replace this with Redis Streams, SNS/SQS,
or another shared event backend so clients can reconnect to any API instance.
"""

import asyncio
from collections import defaultdict
from typing import Any

_EVENT_HISTORY: dict[str, list[dict[str, Any]]] = defaultdict(list)
_SUBSCRIBERS: dict[str, set[asyncio.Queue]] = defaultdict(set)

_TERMINAL_EVENT_TYPES = {"final", "error"}


def publish(request_id: str, event: dict[str, Any]) -> None:
    """Publish an event for a request.

    Args:
        request_id: Unique chat request id.
        event: JSON-serialisable event payload. A ``type`` key is expected.
    """
    event_with_id = {"request_id": request_id, **event}
    _EVENT_HISTORY[request_id].append(event_with_id)
    for queue in list(_SUBSCRIBERS.get(request_id, set())):
        queue.put_nowait(event_with_id)


async def subscribe(request_id: str):
    """Yield historical and future events for one request id."""
    queue: asyncio.Queue = asyncio.Queue()
    _SUBSCRIBERS[request_id].add(queue)
    try:
        for event in _EVENT_HISTORY.get(request_id, []):
            yield event
            if event.get("type") in _TERMINAL_EVENT_TYPES:
                return

        while True:
            event = await queue.get()
            yield event
            if event.get("type") in _TERMINAL_EVENT_TYPES:
                return
    finally:
        _SUBSCRIBERS.get(request_id, set()).discard(queue)
