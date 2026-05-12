"""FastAPI router — /api/chat endpoint.

Flow
----
1. Validate & scan user input with guardrail_scanner.scan_text_content().
2. Build initial LangGraph state (with optional image).
3. Invoke the compiled agent graph asynchronously.
4. Scan agent output with guardrail_scanner.scan_text_content().
5. Return ChatResponse.
"""

import base64
import io
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
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
    "You are a multi-modal AI assistant. You MUST route every user request "
    "to exactly one of these four tools — do not answer directly:\n"
    "  • medical_qa        — medical / health / clinical questions\n"
    "  • legal_qa          — legal / law / case-related questions\n"
    "  • vision_llm        — describing, captioning, or answering open-ended "
    "questions about an image\n"
    "  • object_detection  — detecting, localising, counting, or highlighting "
    "objects in an image\n\n"
    "Rules:\n"
    "1. When the user mentions 'detect', 'find objects', 'highlight', 'bounding box', "
    "or asks 'what objects are in this image', ALWAYS call `object_detection`.\n"
    "2. When the user explicitly names a tool (e.g. 'use object detect tool'), "
    "you MUST call that tool.\n"
    "3. When an image is attached and the question is descriptive (e.g. 'what is this?', "
    "'describe this image'), call `vision_llm`.\n"
    "4. Never produce a final answer without first calling a tool when an image is attached.\n"
    "5. For vision tools, do NOT try to include the image data in the arguments — "
    "the system injects it automatically. Just call the tool with the user's query."
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for POST /api/chat.

    Attributes:
        message: Plain-text user message.
        image_base64: Optional base64-encoded image (JPEG / PNG / WebP).
        session_id: Optional opaque session identifier for future multi-turn
            memory support.
    """

    message: str
    image_base64: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response body returned by POST /api/chat.

    Attributes:
        response: The agent's final textual answer.
        tool_used: Name of the tool invoked by the agent (empty if none).
        guardrail_flagged: True when the output was blocked by guardrails.
        annotated_image_base64: Base64-encoded JPEG of the original image with
            bounding boxes drawn on it. Only populated when ``tool_used`` is
            ``object_detection`` and detection results are available.
    """

    response: str
    tool_used: str
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

    # ------------------------------------------------------------------
    # 1. Input guardrail scan
    # ------------------------------------------------------------------
    try:
        input_scan = scan_text_content(request.message)
    except Exception:
        logger.exception("Input guardrail scan raised an unexpected exception")
        raise HTTPException(status_code=500, detail="Guardrail scan error on input.")

    if input_scan.flagged:
        logger.warning("Input blocked | category=%s reason=%s", input_scan.category, input_scan.reason)
        raise HTTPException(
            status_code=400,
            detail=f"Input blocked by safety guardrails: {input_scan.reason}",
        )

    # ------------------------------------------------------------------
    # 2. Build initial agent state
    # ------------------------------------------------------------------
    # Compose message content — text always; image added when present.
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

    initial_state = {
        "messages": [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ],
        "tool_used": "",
        "image_base64": request.image_base64 or "",
        "tool_output": "",
    }

    # ------------------------------------------------------------------
    # 3. Run agent graph
    # ------------------------------------------------------------------
    try:
        final_state = await agent_graph.ainvoke(initial_state)
    except Exception:
        logger.exception("agent_graph.ainvoke raised an exception")
        raise HTTPException(status_code=500, detail="Agent processing error.")

    # ------------------------------------------------------------------
    # 4. Extract last response from state
    # ------------------------------------------------------------------
    try:
        last_message = final_state["messages"][-1]
        response_text: str = (
            last_message.content
            if isinstance(last_message.content, str)
            else str(last_message.content)
        )
        tool_used: str = final_state.get("tool_used", "")
    except Exception:
        logger.exception("Failed to extract response from final agent state")
        raise HTTPException(status_code=500, detail="Response extraction error.")

    # Guard against empty responses (e.g. when GPT-4o silently refuses or
    # produces no content and no tool call). Give the user a clear message
    # rather than rendering a blank assistant bubble in the UI.
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

    # ------------------------------------------------------------------
    # 4b. Draw bounding boxes when object_detection was used
    # ------------------------------------------------------------------
    annotated_image_b64: Optional[str] = None
    if tool_used == "object_detection" and request.image_base64:
        annotated_image_b64 = _draw_detections(
            request.image_base64,
            final_state.get("tool_output", ""),
        )

    # ------------------------------------------------------------------
    # 5. Output guardrail scan
    # ------------------------------------------------------------------
    try:
        output_scan = scan_text_content(response_text)
    except Exception:
        logger.exception("Output guardrail scan raised an unexpected exception")
        raise HTTPException(status_code=500, detail="Guardrail scan error on output.")

    if output_scan.flagged:
        logger.warning(
            "Output blocked | category=%s reason=%s", output_scan.category, output_scan.reason
        )
        return ChatResponse(
            response="The agent's response was blocked by safety guardrails.",
            tool_used=tool_used,
            guardrail_flagged=True,
            annotated_image_base64=None,
        )

    logger.info("POST /api/chat | success | tool_used=%r", tool_used)
    return ChatResponse(
        response=response_text,
        tool_used=tool_used,
        guardrail_flagged=False,
        annotated_image_base64=annotated_image_b64,
    )
