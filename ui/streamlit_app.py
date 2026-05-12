"""Streamlit chat UI for the Multi-Modal AI Agent.

Run locally (outside Docker):
    streamlit run ui/streamlit_app.py
"""

import base64
import logging
import sys
import uuid
from typing import Optional

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
REQUEST_TIMEOUT_SECONDS: int = 120

_TOOL_ICONS: dict[str, str] = {
    "medical_qa": "🏥",
    "legal_qa": "⚖️",
    "vision_llm": "👁️",
    "object_detection": "🔍",
}


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
) -> dict:
    """POST a chat message to the FastAPI backend.

    Args:
        message: User's plain-text question.
        image_base64: Optional base64-encoded image data.
        session_id: Opaque session identifier.

    Returns:
        Parsed JSON response dict containing ``response``, ``tool_used``,
        and ``guardrail_flagged``.

    Raises:
        requests.HTTPError: On 4xx / 5xx responses.
        requests.ConnectionError: When the API is not reachable.
        requests.Timeout: When the request exceeds the timeout.
    """
    payload: dict = {"message": message, "session_id": session_id}
    if image_base64:
        payload["image_base64"] = image_base64

    logger.info(
        "call_chat_api | session=%s image=%s", session_id, bool(image_base64)
    )
    response = requests.post(
        CHAT_ENDPOINT,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _render_message(msg: dict) -> None:
    """Render a single chat message bubble with optional metadata.

    Args:
        msg: Dict with keys ``role``, ``content``, and optionally
            ``tool_used``, ``guardrail_flagged``, and
            ``annotated_image_base64``.
    """
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
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
        st.header("⚙️ Session")
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

    # ---- Image upload ----------------------------------------------------
    uploaded_file = st.file_uploader(
        "📎 Attach an image (optional — activates vision / detection tools)",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="visible",
    )
    image_base64: Optional[str] = None
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image_base64 = encode_image(image_bytes)
        st.image(uploaded_file, caption="Attached image", width=300)

    # ---- Chat input ------------------------------------------------------
    user_input: Optional[str] = st.chat_input("Ask anything…")
    if not user_input:
        return

    # Append & render user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    _render_message({"role": "user", "content": user_input})

    # Call API & stream response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = call_chat_api(
                    user_input,
                    image_base64,
                    st.session_state.session_id,
                )
                response_text: str = result.get("response", "No response received.")
                tool_used: str = result.get("tool_used", "")
                guardrail_flagged: bool = result.get("guardrail_flagged", False)

                annotated_b64: Optional[str] = result.get("annotated_image_base64")

                st.markdown(response_text)
                if tool_used:
                    icon = _TOOL_ICONS.get(tool_used, "🔧")
                    st.caption(f"{icon} Tool used: `{tool_used}`")
                if guardrail_flagged:
                    st.warning("⚠️ Guardrail flagged this response.")
                if annotated_b64:
                    ann_bytes = base64.b64decode(annotated_b64)
                    st.image(ann_bytes, caption="Detected Objects", use_container_width=True)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "tool_used": tool_used,
                        "guardrail_flagged": guardrail_flagged,
                        "annotated_image_base64": annotated_b64,
                    }
                )

            except requests.HTTPError as exc:
                status = exc.response.status_code
                try:
                    detail = exc.response.json().get("detail", exc.response.text)
                except Exception:
                    detail = exc.response.text
                err_msg = f"API error {status}: {detail}"
                logger.error("call_chat_api HTTP error | %s", err_msg)
                st.error(err_msg)

            except requests.ConnectionError:
                err_msg = "Cannot reach the API. Is the backend running?"
                logger.error(err_msg)
                st.error(err_msg)

            except requests.Timeout:
                err_msg = "Request timed out. The agent may still be processing."
                logger.error(err_msg)
                st.error(err_msg)

            except Exception as exc:
                err_msg = f"Unexpected error: {exc}"
                logger.exception("Unexpected error in Streamlit UI")
                st.error(err_msg)


if __name__ == "__main__":
    main()
