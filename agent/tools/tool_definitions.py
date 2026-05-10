"""Stub tool definitions for the multi-modal AI agent.

Each tool is decorated with ``@tool`` from ``langchain_core.tools``.
The function bodies are stubs — replace the return statements with real
endpoint calls when the backing services are available.
"""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Medical QA — MedGemma endpoint
# ---------------------------------------------------------------------------


@tool
async def medical_qa(query: str) -> str:
    """Answer medical and health-related questions using the MedGemma model.

    Args:
        query: The medical question or clinical text to analyse.

    Returns:
        A medically-informed response string from the MedGemma endpoint.
    """
    logger.info("medical_qa called | query[:120]=%r", query[:120])
    # TODO: Replace with actual HTTP call to MedGemma inference endpoint.
    # Example:
    #   async with httpx.AsyncClient() as client:
    #       resp = await client.post(MEDGEMMA_URL, json={"query": query}, timeout=30)
    #       return resp.json()["answer"]
    return (
        f"[STUB — MedGemma] Medical answer for: '{query}'. "
        "This endpoint will query the MedGemma model hosted at the configured "
        "MEDGEMMA_ENDPOINT_URL. Replace this stub with the real HTTP call."
    )


# ---------------------------------------------------------------------------
# Legal QA — Pinecone RAG + Qwen via AWS Bedrock
# ---------------------------------------------------------------------------


@tool
async def legal_qa(query: str) -> str:
    """Answer legal questions by retrieving relevant documents from Pinecone and
    synthesising an answer with Qwen via AWS Bedrock.

    Args:
        query: The legal question or case description to research.

    Returns:
        A RAG-augmented legal response string.
    """
    logger.info("legal_qa called | query[:120]=%r", query[:120])
    # TODO: Replace with actual Pinecone vector search + Bedrock Qwen invocation.
    # Example:
    #   docs = pinecone_index.query(vector=embed(query), top_k=5)
    #   context = "\n".join([d.metadata["text"] for d in docs.matches])
    #   response = bedrock_client.invoke_model(...)
    #   return response["answer"]
    return (
        f"[STUB — Legal RAG + Qwen/Bedrock] Legal answer for: '{query}'. "
        "This tool retrieves relevant legal documents from the Pinecone index and "
        "synthesises an answer using Qwen via AWS Bedrock. "
        "Replace this stub with the real Pinecone + Bedrock calls."
    )


# ---------------------------------------------------------------------------
# Vision LLM — Qwen3-VL-2B endpoint
# ---------------------------------------------------------------------------


@tool
async def vision_llm(query: str, image_base64: str = "") -> str:
    """Analyse an image and answer a natural-language question about it using
    the Qwen3-VL-2B vision-language model.

    Args:
        query: The question or instruction about the image.
        image_base64: Base64-encoded image data (JPEG/PNG).

    Returns:
        A vision-language response string from the Qwen3-VL-2B endpoint.
    """
    logger.info(
        "vision_llm called | query[:120]=%r, image_provided=%s",
        query[:120],
        bool(image_base64),
    )
    # TODO: Replace with actual HTTP call to Qwen3-VL-2B inference endpoint.
    # Example:
    #   async with httpx.AsyncClient() as client:
    #       resp = await client.post(
    #           QWEN_VL_URL,
    #           json={"query": query, "image": image_base64},
    #           timeout=60,
    #       )
    #       return resp.json()["response"]
    return (
        f"[STUB — Qwen3-VL-2B] Vision analysis for query: '{query}'. "
        "This tool sends the image and query to the Qwen3-VL-2B vision-language "
        "endpoint at QWEN_VL_ENDPOINT_URL. Replace this stub with the real HTTP call."
    )


# ---------------------------------------------------------------------------
# Object Detection — YOLOv12-S endpoint
# ---------------------------------------------------------------------------


@tool
async def object_detection(image_base64: str = "") -> str:
    """Detect and localise objects in an image using the YOLOv12-S model.

    Args:
        image_base64: Base64-encoded image data (JPEG/PNG).

    Returns:
        A JSON-like string listing detected objects with bounding boxes and
        confidence scores from the YOLOv12-S endpoint.
    """
    logger.info("object_detection called | image_provided=%s", bool(image_base64))
    # TODO: Replace with actual HTTP call to YOLOv12-S inference endpoint.
    # Example:
    #   async with httpx.AsyncClient() as client:
    #       resp = await client.post(
    #           YOLOV12_URL,
    #           json={"image": image_base64},
    #           timeout=60,
    #       )
    #       return resp.json()["detections"]
    return (
        "[STUB — YOLOv12-S] Object detection results: "
        "This tool sends the image to the YOLOv12-S endpoint at YOLOV12_ENDPOINT_URL "
        "and returns detected objects with bounding boxes and confidence scores. "
        "Replace this stub with the real HTTP call."
    )


# ---------------------------------------------------------------------------
# Tool registry (used by graph.py)
# ---------------------------------------------------------------------------

TOOLS = [medical_qa, legal_qa, vision_llm, object_detection]
TOOL_MAP: dict[str, object] = {t.name: t for t in TOOLS}
