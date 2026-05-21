"""Streamlit chat UI for the Multi-Modal AI Agent.

Updated flow:
- Text only: POST /api/chat/messages, then GET /api/chat/messages/{request_id}/events
- Text + files: POST /api/uploads/presign, PUT files to S3, POST /api/chat/messages,
  then GET /api/chat/messages/{request_id}/events
"""

import base64
import json
import logging
import os
import sys
import uuid
from typing import Iterator, Optional

import requests
import streamlit as st

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_BASE_URL: str = os.getenv("API_URL", "http://api:8000")
PRESIGN_ENDPOINT: str = f"{API_BASE_URL}/api/uploads/presign"
CHAT_MESSAGES_ENDPOINT: str = f"{API_BASE_URL}/api/chat/messages"
CHAT_SESSIONS_ENDPOINT: str = f"{API_BASE_URL}/api/chat/sessions"
REQUEST_TIMEOUT_SECONDS: int = 300
MAX_FILES_PER_CHAT: int = 10

_TOOL_ICONS: dict[str, str] = {
    "medical_qa": "🏥",
    "retfound_analyze": "👁️",
    "endofm_analyze": "🩺",
    "sam_med2d_segment": "🧩",
    "totalsegmentator_segment": "🧠",
    "legal_qa": "⚖️",
    "vision_llm": "👁️",
    "object_detection": "🔍",
}

_ORCHESTRATOR_MODEL_OPTIONS: list[dict[str, str]] = [
    {"key": "qwen3-vl-235b", "label": "Qwen3-VL-235B"},
    {"key": "gpt-4o", "label": "GPT-4o"},
]
_DEFAULT_ORCHESTRATOR_MODEL_KEY: str = _ORCHESTRATOR_MODEL_OPTIONS[0]["key"]


def _uploaded_file_size(uploaded_file) -> int:
    """Return uploaded file size without consuming the stream when possible."""
    if hasattr(uploaded_file, "size") and uploaded_file.size is not None:
        return int(uploaded_file.size)
    pos = uploaded_file.tell()
    uploaded_file.seek(0, 2)
    size = uploaded_file.tell()
    uploaded_file.seek(pos)
    return int(size)


def presign_and_upload_files(uploaded_files: list, session_id: str) -> list[dict]:
    """Get S3 presigned URLs from FastAPI and upload files directly to S3."""
    file_descriptors: list[dict] = []
    file_payloads: list[dict] = []

    for uploaded_file in uploaded_files:
        raw = uploaded_file.getvalue()
        content_type = uploaded_file.type or "application/octet-stream"
        file_payloads.append({
            "original_filename": uploaded_file.name,
            "content_type": content_type,
            "raw": raw,
        })
        file_descriptors.append({
            "filename": uploaded_file.name,
            "content_type": content_type,
            "size_bytes": len(raw),
        })

    presign_response = requests.post(
        PRESIGN_ENDPOINT,
        json={"session_id": session_id, "files": file_descriptors},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    presign_response.raise_for_status()
    presigned = presign_response.json()

    uploaded_refs: list[dict] = []
    for payload, target in zip(file_payloads, presigned["files"]):
        filename = target["filename"]
        upload_headers = target.get("upload_headers") or {}
        content_type = upload_headers.get(
            "Content-Type",
            target.get("content_type") or payload["content_type"] or "application/octet-stream",
        )
        upload_headers = {**upload_headers, "Content-Type": content_type}
        put_response = requests.put(
            target["presigned_put_url"],
            data=payload["raw"],
            headers=upload_headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not put_response.ok:
            error_preview = put_response.text[:1200] if put_response.text else put_response.reason
            raise RuntimeError(
                f"S3 upload failed for {filename!r} with HTTP {put_response.status_code}: "
                f"{error_preview}"
            )
        
        uploaded_refs.append({
            "upload_id": target["upload_id"],
            "filename": filename,
            "content_type": content_type,
            "size_bytes": target.get("size_bytes"),
            "s3_bucket": target["s3_bucket"],
            "s3_key": target["s3_key"],
            "s3_uri": target["s3_uri"],
            "public_url": target.get("public_url"),
        })

    return uploaded_refs


def create_chat_message(
    message: str,
    session_id: str,
    history: list[dict],
    files: list[dict],
    orchestrator_model: Optional[str],
) -> dict:
    """Create a chat request in the backend and return request metadata."""
    payload = {
        "message": message,
        "session_id": session_id,
        "history": history,
        "files": files,
    }
    if orchestrator_model:
        payload["orchestrator_model"] = orchestrator_model

    response = requests.post(
        CHAT_MESSAGES_ENDPOINT,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def stream_request_events(request_id: str) -> Iterator[dict]:
    """Open the backend SSE endpoint and yield parsed JSON events."""
    events_url = f"{CHAT_MESSAGES_ENDPOINT}/{request_id}/events"
    response = requests.get(
        events_url,
        stream=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"Accept": "text/event-stream"},
    )
    try:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("data: "):
                data = raw_line[len("data: "):]
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("Bad SSE payload: %r", data[:120])
    finally:
        response.close()


def list_session_history(session_id: str, limit: int = 50) -> list[dict]:
    """Load request history for a session from DynamoDB via FastAPI."""
    response = requests.get(
        f"{CHAT_SESSIONS_ENDPOINT}/{session_id}/messages",
        params={"limit": limit, "newest_first": False},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("items", [])


def _messages_from_history_items(items: list[dict]) -> list[dict]:
    """Convert DynamoDB request records into Streamlit chat bubbles."""
    messages: list[dict] = []
    for item in items:
        user_text = item.get("message") or "(file-only request)"
        messages.append({
            "role": "user",
            "content": user_text,
            "files": item.get("files") or [],
        })
        if item.get("status") == "COMPLETED" and item.get("response"):
            messages.append({
                "role": "assistant",
                "content": item.get("response", ""),
                "tool_used": item.get("tool_used", ""),
                "tools_chain": item.get("tools_chain", []) or [],
                "answer_model": item.get("answer_model", ""),
                "answer_model_chain": item.get("answer_model_chain", []) or [],
                "cache_hit": bool(item.get("cache_hit", False)),
                "guardrail_flagged": bool(item.get("guardrail_flagged", False)),
            })
        elif item.get("status") in {"FAILED", "BLOCKED"}:
            messages.append({
                "role": "assistant",
                "content": f"Request {item.get('status')}: {item.get('error_message', '')}",
                "guardrail_flagged": item.get("status") == "BLOCKED",
            })
    return messages


def _render_message(msg: dict) -> None:
    """Render a single chat message bubble."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        files = msg.get("files") or []
        if files:
            st.caption("Attached files: " + ", ".join(f.get("filename", "file") for f in files))
        
        answer_model_chain: list[str] = msg.get("answer_model_chain") or []
        answer_model = msg.get("answer_model") or (answer_model_chain[-1] if answer_model_chain else "")

        if answer_model:
            st.caption(f"🤖 Answer model: `{answer_model}`")

        if answer_model_chain and len(answer_model_chain) > 1:
            st.caption("🤖 Model chain: " + " → ".join(f"`{m}`" for m in answer_model_chain))

        if msg.get("cache_hit"):
            st.caption("⚡ Served from Redis cache")

        tools_chain: list[str] = msg.get("tools_chain") or []
        if tools_chain:
            chain_str = " → ".join(f"{_TOOL_ICONS.get(t, '🔧')} `{t}`" for t in tools_chain)
            st.caption(f"🛠️ Tool chain: {chain_str}")
        elif msg.get("tool_used"):
            tool = msg.get("tool_used", "")
            st.caption(f"{_TOOL_ICONS.get(tool, '🔧')} Tool used: `{tool}`")
        if msg.get("guardrail_flagged"):
            st.warning("⚠️ Guardrail flagged this response.")
        annotated = msg.get("annotated_image_base64")
        if annotated:
            st.image(base64.b64decode(annotated), caption="Detected Objects", use_container_width=True)


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(page_title="Multi-Modal AI Agent", page_icon="🤖", layout="wide")

    if "orchestrator_model" not in st.session_state:
        st.session_state.orchestrator_model = _DEFAULT_ORCHESTRATOR_MODEL_KEY
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict] = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    st.title("🤖 Multi-Modal AI Agent")
    st.caption(
        "Direct-to-S3 uploads · DynamoDB request tracking · FastAPI SSE progress · "
        "GPT-4o / Qwen orchestrated routing"
    )

    with st.sidebar:
        st.header("⚙️ Orchestrator model")
        model_labels = [opt["label"] for opt in _ORCHESTRATOR_MODEL_OPTIONS]
        model_keys = [opt["key"] for opt in _ORCHESTRATOR_MODEL_OPTIONS]
        default_idx = model_keys.index(st.session_state.orchestrator_model) if st.session_state.orchestrator_model in model_keys else 0
        selected_label = st.radio(
            "Choose which LLM routes & answers",
            options=model_labels,
            index=default_idx,
            label_visibility="collapsed",
            key="orchestrator_model_radio",
        )
        st.session_state.orchestrator_model = model_keys[model_labels.index(selected_label)]
        st.caption(f"Active: `{st.session_state.orchestrator_model}`")

        st.markdown("---")
        st.markdown("**Session**")
        st.write(f"**ID:** `{st.session_state.session_id}`")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        if st.button("↩️ Load this session from DynamoDB", use_container_width=True):
            try:
                items = list_session_history(st.session_state.session_id)
                st.session_state.messages = _messages_from_history_items(items)
                st.success(f"Loaded {len(items)} persisted request(s).")
            except Exception as exc:
                st.error(f"Could not load session history: {exc}")

        st.markdown("---")
        st.markdown("**Backend APIs**")
        with st.expander("Core API endpoints", expanded=True):
            st.caption("Health")
            st.code(f"{API_BASE_URL}/health", language=None)
            st.caption("Upload presign")
            st.code(PRESIGN_ENDPOINT, language=None)
            st.caption("Create chat request")
            st.code(CHAT_MESSAGES_ENDPOINT, language=None)
            st.caption("Stream request events")
            st.code(f"{CHAT_MESSAGES_ENDPOINT}/{{request_id}}/events", language=None)
            st.caption("Session history")
            st.code(f"{CHAT_SESSIONS_ENDPOINT}/{{session_id}}/messages", language=None)
            st.caption("Prometheus metrics")
            st.code(f"{API_BASE_URL}/metrics", language=None)

        with st.expander("Model wrapper endpoints", expanded=False):
            st.caption("SAM-Med2D")
            st.code(f"{API_BASE_URL}/api/sam-med2d/predict", language=None)
            st.caption("RETFound")
            st.code(f"{API_BASE_URL}/api/retfound/infer", language=None)
            st.caption("RETFound health")
            st.code(f"{API_BASE_URL}/api/retfound/health", language=None)

        with st.expander("Legacy compatibility", expanded=False):
            st.code(f"{API_BASE_URL}/api/chat", language=None)
            st.code(f"{API_BASE_URL}/api/chat/stream", language=None)

    for msg in st.session_state.messages:
        _render_message(msg)

    chat_value = st.chat_input(
        "Ask anything… attach up to 10 files of any format",
        accept_file="multiple",
        file_type=None,
    )
    if not chat_value:
        return

    user_input = (chat_value.text or "").strip()
    uploaded_files = list(chat_value.files or [])

    if len(uploaded_files) > MAX_FILES_PER_CHAT:
        st.error(f"Please attach at most {MAX_FILES_PER_CHAT} files per chat message.")
        return
    if not user_input and not uploaded_files:
        st.error("Enter a message or attach at least one file.")
        return

    user_msg = {
        "role": "user",
        "content": user_input or "(file-only request)",
        "files": [{"filename": f.name, "content_type": f.type, "size_bytes": _uploaded_file_size(f)} for f in uploaded_files],
    }
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(user_msg["content"])
        if uploaded_files:
            st.caption("Attached files: " + ", ".join(f.name for f in uploaded_files))
            for f in uploaded_files:
                if (f.type or "").startswith("image/"):
                    st.image(f.getvalue(), caption=f.name, width=260)

    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        status = st.status("📨 Submitting request…", expanded=True)
        final_event: Optional[dict] = None
        try:
            uploaded_refs: list[dict] = []
            if uploaded_files:
                status.markdown("Requesting S3 presigned upload URLs…")
                uploaded_refs = presign_and_upload_files(uploaded_files, st.session_state.session_id)
                status.markdown(f"Uploaded {len(uploaded_refs)} file(s) to S3. Creating chat request…")

            created = create_chat_message(
                user_input,
                st.session_state.session_id,
                history_payload,
                uploaded_refs,
                st.session_state.orchestrator_model,
            )
            request_id = created["request_id"]
            status.markdown(f"Request `{request_id}` accepted. Waiting for processing events…")

            for event in stream_request_events(request_id):
                etype = event.get("type")
                if etype in {"accepted", "status"}:
                    stage = event.get("stage", "processing")
                    status.markdown(f"Stage: `{stage}`")
                elif etype == "routing":
                    status.markdown(event.get("message", "Routing…"))
                elif etype == "tool_done":
                    tool = event.get("tool", "")
                    status.markdown(f"{_TOOL_ICONS.get(tool, '🔧')} `{tool}` finished. Continuing…")
                elif etype == "final":
                    final_event = event
                    status.update(label="✅ Done", state="complete", expanded=False)
                    break
                elif etype == "error":
                    status.update(label="❌ Error", state="error", expanded=True)
                    st.error(event.get("message", "Unknown processing error."))
                    break
        except requests.HTTPError as exc:
            status.update(label="❌ HTTP error", state="error")
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            st.error(f"API error {exc.response.status_code}: {detail}")
        except requests.ConnectionError:
            status.update(label="❌ Connection error", state="error")
            st.error("Cannot reach the API. Is the backend running?")
        except requests.Timeout:
            status.update(label="❌ Timeout", state="error")
            st.error("Request timed out. The backend may still be processing.")
        except Exception as exc:
            status.update(label="❌ Unexpected error", state="error")
            logger.exception("Unexpected Streamlit error")
            st.error(f"Unexpected error: {exc}")

        if final_event is not None:
            response_text = final_event.get("response", "No response received.")
            tool_used = final_event.get("tool_used", "")
            tools_chain = final_event.get("tools_chain", []) or []
            answer_model = final_event.get("answer_model", "")
            answer_model_chain = final_event.get("answer_model_chain", []) or []
            cache_hit = bool(final_event.get("cache_hit", False))
            guardrail_flagged = final_event.get("guardrail_flagged", False)
            annotated_b64 = final_event.get("annotated_image_base64")

            st.markdown(response_text)
            if answer_model:
                st.caption(f"🤖 Answer model: `{answer_model}`")

            if answer_model_chain and len(answer_model_chain) > 1:
                st.caption("🤖 Model chain: " + " → ".join(f"`{m}`" for m in answer_model_chain))

            if cache_hit:
                st.caption("⚡ Served from Redis cache")
            if tools_chain:
                chain_str = " → ".join(f"{_TOOL_ICONS.get(t, '🔧')} `{t}`" for t in tools_chain)
                st.caption(f"🛠️ Tool chain: {chain_str}")
            elif tool_used:
                st.caption(f"{_TOOL_ICONS.get(tool_used, '🔧')} Tool used: `{tool_used}`")
            if guardrail_flagged:
                st.warning("⚠️ Guardrail flagged this response.")
            if annotated_b64:
                st.image(base64.b64decode(annotated_b64), caption="Detected Objects", use_container_width=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "tool_used": tool_used,
                "tools_chain": tools_chain,
                "answer_model": answer_model,
                "answer_model_chain": answer_model_chain,
                "cache_hit": cache_hit,
                "guardrail_flagged": guardrail_flagged,
                "annotated_image_base64": annotated_b64,
            })


if __name__ == "__main__":
    main()
