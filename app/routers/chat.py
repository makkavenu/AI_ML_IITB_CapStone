"""FastAPI router — /api/chat endpoint.

Flow
----
1. Validate & scan user input with guardrail_scanner.scan_text_content().
2. Build initial LangGraph state (with optional image).
3. Invoke the compiled agent graph asynchronously.
4. Scan agent output with guardrail_scanner.scan_text_content().
5. Return ChatResponse.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agent.graph import agent_graph
from app.guardrails.guardrail_scanner import scan_text_content

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# System prompt — sent to GPT-4o as the first message in every conversation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a multi-modal AI assistant with access to four specialised tools:\n"
    "  • medical_qa    — answers medical / health questions (MedGemma)\n"
    "  • legal_qa      — answers legal questions via Pinecone RAG + Qwen/Bedrock\n"
    "  • vision_llm    — analyses images and answers questions (Qwen3-VL-2B)\n"
    "  • object_detection — detects and localises objects in images (YOLOv12-S)\n\n"
    "Always select the most appropriate tool for the user's request.\n"
    "If an image is provided, prefer vision_llm for descriptive questions or "
    "object_detection when the user wants to know what objects are present."
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
    """

    response: str
    tool_used: str
    guardrail_flagged: bool


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
        )

    logger.info("POST /api/chat | success | tool_used=%r", tool_used)
    return ChatResponse(
        response=response_text,
        tool_used=tool_used,
        guardrail_flagged=False,
    )
