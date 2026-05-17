"""Streamlit chat UI for the Multi-Modal AI Agent.

Run locally (outside Docker):
    streamlit run ui/streamlit_app.py
"""

import base64
import json
import logging
import sys
import uuid
from typing import Iterator, Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Inside Docker Compose the API service is reachable by its service name.
# Override via the API_URL env var when running outside Docker.
import os

API_BASE_URL: str = os.getenv("API_URL", "http://api:8000")
CHAT_ENDPOINT: str = f"{API_BASE_URL}/api/chat"
CHAT_STREAM_ENDPOINT: str = f"{API_BASE_URL}/api/chat/stream"
REQUEST_TIMEOUT_SECONDS: int = 300

_TOOL_ICONS: dict[str, str] = {
    "medical_qa": "🏥",
    "legal_qa": "⚖️",
    "vision_llm": "👁️",
    "object_detection": "🔍",
}

# Orchestrator model options shown in the sidebar.
# The first entry is the default selection.
_ORCHESTRATOR_MODEL_OPTIONS: list[dict[str, str]] = [
    {"key": "qwen3-32b", "label": "Qwen3-32B (AWS Bedrock)"},
    {"key": "gpt-4o",    "label": "GPT-4o (OpenAI)"},
]
_DEFAULT_ORCHESTRATOR_MODEL_KEY: str = _ORCHESTRATOR_MODEL_OPTIONS[0]["key"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode_image(file_bytes: bytes) -> str:
    """Base64-encode raw image bytes.

    Args:
        file_bytes: Raw bytes read from an uploaded image file.

    Returns:
        UTF-8 base64 string suitable for embedding in a data-URI.
    """
    return base64.b64encode(file_bytes).decode("utf-8")


def call_chat_api(
    message: str,
    image_base64: Optional[str],
    session_id: str,
    history: list[dict],
    orchestrator_model: Optional[str] = None,
) -> dict:
    """POST a chat message to the FastAPI backend (non-streaming).

    Kept for completeness; the UI primarily uses ``stream_chat_api``.
    """
    payload: dict = {
        "message": message,
        "session_id": session_id,
        "history": history,
    }
    if image_base64:
        payload["image_base64"] = image_base64
    if orchestrator_model:
        payload["orchestrator_model"] = orchestrator_model

    logger.info(
        "call_chat_api | session=%s image=%s history_turns=%d",
        session_id, bool(image_base64), len(history),
    )
    response = requests.post(
        CHAT_ENDPOINT,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def stream_chat_api(
    message: str,
    image_base64: Optional[str],
    session_id: str,
    history: list[dict],
    orchestrator_model: Optional[str] = None,
) -> Iterator[dict]:
    """POST to /api/chat/stream and yield parsed Server-Sent Event payloads.

    Args:
        message: User question.
        image_base64: Optional base64-encoded image data.
        session_id: Opaque session identifier.
        history: Prior conversation turns.
        orchestrator_model: Optional model key (e.g. ``"qwen3-32b"`` or
            ``"gpt-4o"``) selecting which LLM orchestrates the agent.

    Yields:
        Parsed JSON dicts. Each has a ``type`` field:
        ``routing``, ``tool_done``, ``final`` or ``error``.

    Raises:
        requests.HTTPError / ConnectionError / Timeout on transport issues.
    """
    payload: dict = {
        "message": message,
        "session_id": session_id,
        "history": history,
    }
    if image_base64:
        payload["image_base64"] = image_base64
    if orchestrator_model:
        payload["orchestrator_model"] = orchestrator_model

    logger.info(
        "stream_chat_api | session=%s image=%s history_turns=%d",
        session_id, bool(image_base64), len(history),
    )
    # NOTE: do NOT use `with requests.post(...) as response:` here. When the
    # caller stops iterating (e.g. breaks after the `final` event) the context
    # manager closes the response which, combined with `iter_lines`, can raise
    # "The content for this response was already consumed" on next access.
    # Manage the lifecycle explicitly via try/finally instead.
    response = requests.post(
        CHAT_STREAM_ENDPOINT,
        json=payload,
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
                    logger.warning(
                        "stream_chat_api | bad SSE payload: %r", data[:120]
                    )
                    continue
    finally:
        response.close()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _render_message(msg: dict) -> None:
    """Render a single chat message bubble with optional metadata.

    Args:
        msg: Dict with keys ``role``, ``content``, and optionally
            ``tool_used``, ``tools_chain``, ``guardrail_flagged``, and
            ``annotated_image_base64``.
    """
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        tools_chain: list[str] = msg.get("tools_chain") or []
        if tools_chain:
            chain_str = " → ".join(
                f"{_TOOL_ICONS.get(t, '🔧')} `{t}`" for t in tools_chain
            )
            st.caption(f"🛠️ Tool chain: {chain_str}")
        else:
            tool = msg.get("tool_used", "")
            if tool:
                icon = _TOOL_ICONS.get(tool, "🔧")
                st.caption(f"{icon} Tool used: `{tool}`")
        if msg.get("guardrail_flagged"):
            st.warning("⚠️ Guardrail flagged this response.")
        annotated = msg.get("annotated_image_base64")
        if annotated:
            ann_bytes = base64.b64decode(annotated)
            st.image(ann_bytes, caption="Detected Objects", use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit chat application."""
    st.set_page_config(
        page_title="Multi-Modal AI Agent",
        page_icon="🤖",
        layout="wide",
    )

    if "orchestrator_model" not in st.session_state:
        st.session_state.orchestrator_model = _DEFAULT_ORCHESTRATOR_MODEL_KEY
    # ---- Session state initialisation ------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict] = []
    if "session_id" not in st.session_state:
        st.session_state.session_id: str = str(uuid.uuid4())

    # ---- Header ----------------------------------------------------------
    st.title("🤖 Multi-Modal AI Agent")
    st.caption(
        "Powered by **GPT-4o** · MedGemma · Legal RAG (Pinecone + Qwen/Bedrock) "
        "· Qwen3-VL-2B · YOLOv12-S"
    )

    # ---- Sidebar ---------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ S🧠 Orchestrator model**")
        model_labels = [opt["label"] for opt in _ORCHESTRATOR_MODEL_OPTIONS]
        model_keys = [opt["key"] for opt in _ORCHESTRATOR_MODEL_OPTIONS]
        try:
            default_idx = model_keys.index(st.session_state.orchestrator_model)
        except ValueError:
            default_idx = 0
        selected_label = st.radio(
            "Choose which LLM routes & answers",
            options=model_labels,
            index=default_idx,
            label_visibility="collapsed",
            key="orchestrator_model_radio",
        )
        st.session_state.orchestrator_model = model_keys[
            model_labels.index(selected_label)
        ]
        st.caption(
            f"Active: `{st.session_state.orchestrator_model}`"
        )

        st.markdown("---")
        st.markdown("**ession")
        st.write(f"**ID:** `{st.session_state.session_id[:8]}…`")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("**Available Tools**")
        for tool_name, icon in _TOOL_ICONS.items():
            st.markdown(f"{icon} `{tool_name}`")

        st.markdown("---")
        st.markdown("**API**")
        st.code(CHAT_ENDPOINT, language=None)

    # ---- Chat history ----------------------------------------------------
    for msg in st.session_state.messages:
        _render_message(msg)

    # ---- Chat input with built-in file upload ----------------------------
    # `accept_file=True` (Streamlit >= 1.43) renders an upload button inside
    # the chat input bar. The returned object exposes `.text` and `.files`.
    chat_value = st.chat_input(
        "Ask anything…  (attach an image to use vision / detection tools)",
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "webp"],
    )
    if not chat_value:
        return

    user_input: str = (chat_value.text or "").strip()
    uploaded_files = chat_value.files or []

    image_base64: Optional[str] = None
    image_bytes: Optional[bytes] = None
    if uploaded_files:
        # Only use the first attached image per turn.
        image_bytes = uploaded_files[0].read()
        image_base64 = encode_image(image_bytes)

    if not user_input and not image_base64:
        return

    # Append & render user message immediately
    user_msg: dict = {"role": "user", "content": user_input or "(image attached)"}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(user_msg["content"])
        if image_bytes:
            st.image(image_bytes, caption="Attached image", width=300)

    # Build prior history (exclude the just-appended user message) — text only.
    history_payload: list[dict] = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    # Call API & stream response
    with st.chat_message("assistant"):
        status = st.status("🤔 Analyzing your question…", expanded=True)
        final_event: Optional[dict] = None
        stream_error: Optional[str] = None
        try:
            for event in stream_chat_api(
                user_input or "Describe the attached image.",
                image_base64,
                st.session_state.session_id,
                history_payload,
                orchestrator_model=st.session_state.orchestrator_model,
            ):
                etype = event.get("type")
                if etype == "routing":
                    status.markdown(event.get("message", ""))
                elif etype == "tool_done":
                    tool = event.get("tool", "")
                    icon = _TOOL_ICONS.get(tool, "🔧")
                    status.markdown(
                        f"{icon} Results received from `{tool}`. Processing…"
                    )
                elif etype == "final":
                    final_event = event
                    status.update(
                        label="✅ Done", state="complete", expanded=False
                    )
                    # Stop iterating — the server has sent the last event and
                    # closed the stream. Continuing the loop would attempt to
                    # read from an already-consumed response and raise
                    # "The content for this response was already consumed".
                    break
                elif etype == "error":
                    stream_error = event.get("message", "Unknown agent error.")
                    status.update(
                        label="❌ Agent error", state="error", expanded=True
                    )
                    status.markdown(f"**Error:** {stream_error}")
                    break
        except requests.HTTPError as exc:
            status.update(label="❌ HTTP error", state="error")
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            stream_error = f"API error {exc.response.status_code}: {detail}"
            st.error(stream_error)
        except requests.ConnectionError:
            status.update(label="❌ Connection error", state="error")
            stream_error = "Cannot reach the API. Is the backend running?"
            st.error(stream_error)
        except requests.Timeout:
            status.update(label="❌ Timeout", state="error")
            stream_error = "Request timed out. The agent may still be processing."
            st.error(stream_error)
        except Exception as exc:
            status.update(label="❌ Unexpected error", state="error")
            logger.exception("Unexpected error in Streamlit streaming UI")
            stream_error = f"Unexpected error: {exc}"
            st.error(stream_error)

        if final_event is not None:
            response_text: str = final_event.get(
                "response", "No response received."
            )
            tool_used: str = final_event.get("tool_used", "")
            tools_chain: list[str] = final_event.get("tools_chain", []) or []
            guardrail_flagged: bool = final_event.get("guardrail_flagged", False)
            annotated_b64: Optional[str] = final_event.get("annotated_image_base64")

            st.markdown(response_text)
            if tools_chain:
                chain_str = " → ".join(
                    f"{_TOOL_ICONS.get(t, '🔧')} `{t}`" for t in tools_chain
                )
                st.caption(f"🛠️ Tool chain: {chain_str}")
            elif tool_used:
                icon = _TOOL_ICONS.get(tool_used, "🔧")
                st.caption(f"{icon} Tool used: `{tool_used}`")
            if guardrail_flagged:
                st.warning("⚠️ Guardrail flagged this response.")
            if annotated_b64:
                ann_bytes = base64.b64decode(annotated_b64)
                st.image(
                    ann_bytes,
                    caption="Detected Objects",
                    use_container_width=True,
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "tool_used": tool_used,
                    "tools_chain": tools_chain,
                    "guardrail_flagged": guardrail_flagged,
                    "annotated_image_base64": annotated_b64,
                }
            )


if __name__ == "__main__":
    main()
