"""Prometheus application-level metrics for the AI agent API.

These metrics complement the generic FastAPI request metrics exposed by
``prometheus-fastapi-instrumentator`` in ``app.main``. They are intentionally
low-cardinality: never use request_id, session_id, filename, or S3 key as labels.
"""

from prometheus_client import Counter, Histogram

CHAT_REQUESTS_TOTAL = Counter(
    "ai_agent_chat_requests_total",
    "Total chat requests accepted by the API.",
    ["endpoint", "orchestrator_model"],
)

CHAT_REQUEST_STATUS_TOTAL = Counter(
    "ai_agent_chat_request_status_total",
    "Total chat requests by terminal or important status.",
    ["status"],
)

CHAT_REQUEST_DURATION_SECONDS = Histogram(
    "ai_agent_chat_request_duration_seconds",
    "End-to-end async worker duration for a chat request.",
    buckets=(1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)

FILE_PRESIGN_REQUESTS_TOTAL = Counter(
    "ai_agent_file_presign_requests_total",
    "Total S3 presign requests by status.",
    ["status"],
)

FILE_PRESIGN_FILES_TOTAL = Counter(
    "ai_agent_file_presign_files_total",
    "Total number of file upload URLs generated.",
)

GUARDRAIL_BLOCKED_TOTAL = Counter(
    "ai_agent_guardrail_blocked_total",
    "Total requests blocked by guardrails by phase.",
    ["phase"],
)

TOOL_SELECTION_TOTAL = Counter(
    "ai_agent_tool_selection_total",
    "Total tool routing selections emitted by the orchestrator.",
    ["tool"],
)


MEDICAL_OUTPUT_GUARDRAIL_TOTAL = Counter(
    "ai_agent_medical_output_guardrail_total",
    "Medical-domain output guardrail actions applied to final responses.",
    ["action", "risk_category"],
)

CACHE_EVENTS_TOTAL = Counter(
    "ai_agent_cache_events_total",
    "Redis response-cache events by type.",
    ["event"],
)

CACHE_OPERATION_DURATION_SECONDS = Histogram(
    "ai_agent_cache_operation_duration_seconds",
    "Redis response-cache operation latency in seconds.",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)

