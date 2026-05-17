"""FastAPI router — /api/chat and /api/chat/stream endpoints.

Flow
----
1. Validate & scan user input with guardrail_scanner.scan_text_content().
2. Build initial LangGraph state (with optional image + prior history).
3. Invoke the compiled agent graph (ainvoke for /chat, astream for /chat/stream).
4. Scan agent output with guardrail_scanner.scan_text_content().
5. Return ChatResponse (/chat) or stream Server-Sent Events (/chat/stream).
"""

import base64
import io
import json
import logging
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from PIL import Image, ImageDraw
from pydantic import BaseModel

from agent.graph import agent_graph
from app.guardrails.guardrail_scanner import scan_text_content

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# System prompt — sent to GPT-4o as the first message in every conversation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a multi-modal AI assistant with access to four tools:\n"
    "  • medical_qa        — medical / health / clinical questions\n"
    "  • legal_qa          — legal / law / case-related questions\n"
    "  • vision_llm        — describing or extracting text from an image\n"
    "  • object_detection  — detecting, localising, counting, or highlighting "
    "objects in an image\n\n"
    "Rules:\n"
    "1. For NEW domain questions (medical, legal, or about an attached image), "
    "you MUST call the appropriate tool — do not answer from your own knowledge.\n"
    "2. For FOLLOW-UP questions that refer to a previous answer in the "
    "conversation (e.g. 'summarize that', 'explain it simpler', 'translate the "
    "above'), answer DIRECTLY from the conversation history without calling a tool.\n"
    "3. When the user mentions 'detect', 'find objects', 'highlight', "
    "'bounding box', or 'what objects are in this image', ALWAYS call "
    "`object_detection`.\n"
    "4. When the user explicitly names a tool, you MUST call that tool.\n"
    "5. CHAIN TOOLS when needed. Examples:\n"
    "   • If an image contains a LEGAL question or legal document and the user "
    "asks a legal question about it: FIRST call `vision_llm` to extract the "
    "text / question from the image, THEN call `legal_qa` with that extracted "
    "text as the query.\n"
    "   • If an image contains a MEDICAL question or report and the user asks "
    "a medical question about it: FIRST call `vision_llm` to extract the text, "
    "THEN call `medical_qa` with that extracted text as the query.\n"
    "   • Do NOT rely on your own multi-modal capabilities to read text from "
    "images — always use `vision_llm` for that.\n"
    "6. When the image question is purely descriptive ('what is this?', "
    "'describe this image'), call `vision_llm` alone — no chaining needed.\n"
    "7. After tools return results, decide if another tool call is needed. "
    "If you have enough information, produce the final answer for the user.\n"
    "8. For vision tools, do NOT include image data in the arguments — "
    "the system injects it automatically. Just call the tool with the query.\n"
    "9. You may call up to 4 tools per request."
)


# Friendly descriptions used for the live routing commentary in the UI.
_TOOL_FRIENDLY_NAME: dict[str, str] = {
    "medical_qa": "the **medical Q&A** tool (MedGemma)",
    "legal_qa": "the **legal Q&A** tool (Pinecone RAG + Qwen)",
    "vision_llm": "the **vision** tool (Qwen3-VL)",
    "object_detection": "the **object detection** tool (YOLOv8)",
}


def _routing_message(tool_name: str) -> str:
    """Build a user-facing one-liner describing the tool the agent just chose.

    Args:
        tool_name: The tool name emitted by the orchestrator.

    Returns:
        A short markdown string suitable for rendering above the spinner.
    """
    friendly = _TOOL_FRIENDLY_NAME.get(tool_name, f"`{tool_name}`")
    return (
        f"I've routed your question to {friendly}. "
        "Please hold on while I retrieve the answer…"
    )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ChatTurn(BaseModel):
    """A single prior turn in the conversation.

    Attributes:
        role: Either ``"user"`` or ``"assistant"``.
        content: Plain-text content of that turn.
    """

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /api/chat.

    Attributes:
        message: Plain-text user message.
        image_base64: Optional base64-encoded image (JPEG / PNG / WebP).
        session_id: Optional opaque session identifier for future multi-turn
            memory support.
        history: Prior conversation turns supplied by the client so the agent
            can answer follow-up questions. Only text content is sent — images
            from earlier turns are not replayed.
    """

    message: str
    image_base64: Optional[str] = None
    session_id: Optional[str] = None
    history: list[ChatTurn] = []


# Cap on how many prior turns we replay to keep token usage bounded.
_MAX_HISTORY_TURNS: int = 10


class ChatResponse(BaseModel):
    """Response body returned by POST /api/chat.

    Attributes:
        response: The agent's final textual answer.
        tool_used: Name of the last tool invoked by the agent (empty if none).
        tools_chain: Ordered list of every tool invoked for this request.
        guardrail_flagged: True when the output was blocked by guardrails.
        annotated_image_base64: Base64-encoded JPEG of the original image with
            bounding boxes drawn on it. Only populated when ``object_detection``
            was invoked and detection results are available.
    """

    response: str
    tool_used: str
    tools_chain: list[str] = []
    guardrail_flagged: bool
    annotated_image_base64: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BBOX_PALETTE: list[str] = [
    "#FF4B4B", "#44CC44", "#4B8BFF",
    "#FFB84B", "#FF44FF", "#00CCCC",
    "#FFFF44", "#BB44FF",
]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a CSS hex color string to an (R, G, B) tuple.

    Args:
        hex_color: Hex color string, e.g. ``"#FF4B4B"``.

    Returns:
        Tuple of three ints in the range 0–255.
    """
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _draw_detections(image_b64: str, tool_output_raw: str) -> Optional[str]:
    """Draw bounding boxes onto the source image using object detection results.

    Parses the JSON string returned by the ``object_detection`` tool, opens the
    original image, draws colour-coded boxes with confidence labels, and returns
    the annotated image as a base64-encoded JPEG.

    Args:
        image_b64: Base64-encoded original image (JPEG/PNG/WebP).
        tool_output_raw: JSON string returned by the object_detection tool.

    Returns:
        Base64-encoded JPEG string of the annotated image, or ``None`` if the
        tool output cannot be parsed or no detections are present.
    """
    try:
        data = json.loads(tool_output_raw)
        detections: list[dict] = data.get("detections", [])
        if not detections:
            return None

        img_b64_clean = image_b64.split(",", 1)[-1].strip()
        img_b64_clean += "=" * (-len(img_b64_clean) % 4)
        img_bytes = base64.b64decode(img_b64_clean)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["xyxy"]
            color = _BBOX_PALETTE[idx % len(_BBOX_PALETTE)]
            rgb = _hex_to_rgb(color)
            label = f"{det['class_name']} {det['confidence']:.0%}"

            # Bounding box (3-pixel border)
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)

            # Label background
            char_w, char_h = 7, 13
            label_w = len(label) * char_w + 6
            text_y = max(0.0, y1 - char_h - 4)
            draw.rectangle(
                [x1, text_y, x1 + label_w, text_y + char_h + 4],
                fill=rgb,
            )
            draw.text((x1 + 3, text_y + 2), label, fill=(255, 255, 255))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        annotated = base64.b64encode(buf.getvalue()).decode()
        logger.info("_draw_detections | annotated %d box(es)", len(detections))
        return annotated

    except Exception:
        logger.exception("_draw_detections | failed to annotate image")
        return None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_input_guardrail(message: str) -> None:
    """Run input guardrails; raise HTTPException on failure or block.

    Args:
        message: Raw user input.

    Raises:
        HTTPException 400 when blocked, 500 when scanner crashes.
    """
    try:
        scan = scan_text_content(message)
    except Exception:
        logger.exception("Input guardrail scan raised an unexpected exception")
        raise HTTPException(status_code=500, detail="Guardrail scan error on input.")
    if scan.flagged:
        logger.warning("Input blocked | category=%s reason=%s", scan.category, scan.reason)
        raise HTTPException(
            status_code=400,
            detail=f"Input blocked by safety guardrails: {scan.reason}",
        )


def _build_initial_state(request: ChatRequest) -> dict:
    """Construct the LangGraph initial state from a chat request.

    Args:
        request: Parsed ChatRequest body.

    Returns:
        State dict matching ``AgentState`` shape.
    """
    human_content: list = [{"type": "text", "text": request.message}]
    if request.image_base64:
        human_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{request.image_base64}",
                    "detail": "auto",
                },
            }
        )

    history_messages: list = []
    for turn in request.history[-_MAX_HISTORY_TURNS:]:
        text = (turn.content or "").strip()
        if not text:
            continue
        if turn.role == "user":
            history_messages.append(HumanMessage(content=text))
        elif turn.role == "assistant":
            history_messages.append(AIMessage(content=text))

    return {
        "messages": [
            SystemMessage(content=_SYSTEM_PROMPT),
            *history_messages,
            HumanMessage(content=human_content),
        ],
        "tool_used": "",
        "tools_chain": [],
        "image_base64": request.image_base64 or "",
        "tool_output": "",
        "tool_outputs_by_name": {},
        "iterations": 0,
    }


# ---------------------------------------------------------------------------
# Endpoint — POST /api/chat  (blocking, single JSON response)
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a user message through the multi-modal AI agent.

    Args:
        request: ChatRequest containing the user message and optional image.

    Returns:
        ChatResponse with the agent reply, the tool that was used, and a flag
        indicating whether guardrails blocked the output.

    Raises:
        HTTPException 400: Input blocked by safety guardrails.
        HTTPException 500: Internal agent or guardrail processing error.
    """
    logger.info(
        "POST /api/chat | session_id=%s image_provided=%s",
        request.session_id,
        bool(request.image_base64),
    )

    _check_input_guardrail(request.message)
    initial_state = _build_initial_state(request)

    try:
        final_state = await agent_graph.ainvoke(initial_state)
    except Exception:
        logger.exception("agent_graph.ainvoke raised an exception")
        raise HTTPException(status_code=500, detail="Agent processing error.")

    try:
        last_message = final_state["messages"][-1]
        response_text: str = (
            last_message.content
            if isinstance(last_message.content, str)
            else str(last_message.content)
        )
        tool_used: str = final_state.get("tool_used", "")
        tools_chain: list[str] = final_state.get("tools_chain", []) or []
        tool_outputs_by_name: dict = final_state.get("tool_outputs_by_name", {}) or {}
    except Exception:
        logger.exception("Failed to extract response from final agent state")
        raise HTTPException(status_code=500, detail="Response extraction error.")

    if not response_text.strip():
        logger.warning(
            "Empty model response | tool_used=%r image_provided=%s",
            tool_used, bool(request.image_base64),
        )
        response_text = (
            "I couldn't produce a response for that request. "
            "Please try rephrasing your question — for example: "
            "'Detect objects in this image' or 'Describe what is in this image'."
        )

    annotated_image_b64: Optional[str] = None
    od_output = tool_outputs_by_name.get("object_detection", "")
    if od_output and request.image_base64:
        annotated_image_b64 = _draw_detections(request.image_base64, od_output)

    try:
        output_scan = scan_text_content(response_text)
    except Exception:
        logger.exception("Output guardrail scan raised an unexpected exception")
        raise HTTPException(status_code=500, detail="Guardrail scan error on output.")

    if output_scan.flagged:
        logger.warning(
            "Output blocked | category=%s reason=%s",
            output_scan.category, output_scan.reason,
        )
        return ChatResponse(
            response="The agent's response was blocked by safety guardrails.",
            tool_used=tool_used,
            tools_chain=tools_chain,
            guardrail_flagged=True,
            annotated_image_base64=None,
        )

    logger.info(
        "POST /api/chat | success | tool_used=%r tools_chain=%s",
        tool_used, tools_chain,
    )
    return ChatResponse(
        response=response_text,
        tool_used=tool_used,
        tools_chain=tools_chain,
        guardrail_flagged=False,
        annotated_image_base64=annotated_image_b64,
    )


# ---------------------------------------------------------------------------
# Endpoint — POST /api/chat/stream  (Server-Sent Events)
# ---------------------------------------------------------------------------


def _sse(event: dict) -> str:
    """Format a dict as a single Server-Sent Event payload.

    Args:
        event: Arbitrary JSON-serialisable dict.

    Returns:
        SSE-formatted string ending in a blank line.
    """
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream agent execution as Server-Sent Events.

    Event types yielded to the client:

      * ``routing``     — orchestrator picked a tool; payload includes ``tool``
                          and ``message`` (user-facing commentary).
      * ``tool_done``   — tool finished; payload includes ``tool``.
      * ``final``       — full response ready; payload mirrors ChatResponse.
      * ``error``       — terminal error before final event.

    Args:
        request: Parsed ChatRequest body.

    Returns:
        StreamingResponse with ``text/event-stream`` media type.

    Raises:
        HTTPException 400: Input blocked by safety guardrails.
    """
    logger.info(
        "POST /api/chat/stream | session_id=%s image_provided=%s",
        request.session_id,
        bool(request.image_base64),
    )

    # Input guardrail runs synchronously before any streaming starts so we can
    # surface a clean 400 to the client.
    _check_input_guardrail(request.message)
    initial_state = _build_initial_state(request)

    async def event_generator() -> AsyncIterator[str]:
        emitted_call_ids: set[str] = set()
        emitted_done_for: set[str] = set()
        final_state: Optional[dict] = None

        try:
            async for state in agent_graph.astream(
                initial_state, stream_mode="values"
            ):
                final_state = state
                msgs = state.get("messages", []) or []
                if not msgs:
                    continue
                last = msgs[-1]

                # Orchestrator just emitted one or more tool calls.
                if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                    for tc in last.tool_calls:
                        tc_id = tc.get("id") or ""
                        if tc_id and tc_id in emitted_call_ids:
                            continue
                        emitted_call_ids.add(tc_id)
                        yield _sse({
                            "type": "routing",
                            "tool": tc["name"],
                            "message": _routing_message(tc["name"]),
                        })

                # Tool executor finished — emit a 'tool_done' for whichever
                # tool ran most recently (matched against tools_chain length).
                if isinstance(last, ToolMessage):
                    chain = state.get("tools_chain", []) or []
                    if chain:
                        latest_tool = chain[-1]
                        marker = f"{latest_tool}:{len(chain)}"
                        if marker not in emitted_done_for:
                            emitted_done_for.add(marker)
                            yield _sse({
                                "type": "tool_done",
                                "tool": latest_tool,
                            })
        except Exception as exc:
            logger.exception("chat_stream | graph execution failed")
            yield _sse({"type": "error", "message": f"Agent error: {exc}"})
            return

        if final_state is None:
            yield _sse({"type": "error", "message": "Agent produced no state."})
            return

        # ---- Extract final response from accumulated state ----------------
        try:
            last_message = final_state["messages"][-1]
            response_text: str = (
                last_message.content
                if isinstance(last_message.content, str)
                else str(last_message.content)
            )
        except Exception:
            logger.exception("chat_stream | could not extract final response")
            yield _sse({"type": "error", "message": "Response extraction error."})
            return

        tool_used: str = final_state.get("tool_used", "") or ""
        tools_chain: list[str] = final_state.get("tools_chain", []) or []
        tool_outputs_by_name: dict = final_state.get("tool_outputs_by_name", {}) or {}

        if not response_text.strip():
            response_text = (
                "I couldn't produce a response for that request. "
                "Please try rephrasing your question."
            )

        # ---- Output guardrail --------------------------------------------
        try:
            output_scan = scan_text_content(response_text)
        except Exception:
            logger.exception("chat_stream | output guardrail crashed")
            yield _sse({"type": "error", "message": "Guardrail scan error on output."})
            return

        if output_scan.flagged:
            yield _sse({
                "type": "final",
                "response": "The agent's response was blocked by safety guardrails.",
                "tool_used": tool_used,
                "tools_chain": tools_chain,
                "guardrail_flagged": True,
                "annotated_image_base64": None,
            })
            return

        # ---- Annotate detections if object_detection ran ------------------
        annotated_image_b64: Optional[str] = None
        od_output = tool_outputs_by_name.get("object_detection", "")
        if od_output and request.image_base64:
            annotated_image_b64 = _draw_detections(request.image_base64, od_output)

        yield _sse({
            "type": "final",
            "response": response_text,
            "tool_used": tool_used,
            "tools_chain": tools_chain,
            "guardrail_flagged": False,
            "annotated_image_base64": annotated_image_b64,
        })
        logger.info(
            "POST /api/chat/stream | success | tools_chain=%s",
            tools_chain,
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
