"""Tool definitions for the multi-modal AI agent.

Each tool is decorated with ``@tool`` from ``langchain_core.tools``.

Live endpoints
--------------
- medical_qa    : MedGemma vLLM server   — http://<host>:8000/v1/chat/completions
- vision_llm    : Qwen3-VL-2B vLLM server — http://<host>:8001/v1/chat/completions

Stubs (endpoint not yet available)
-----------------------------------
- legal_qa        : Pinecone RAG + Qwen via AWS Bedrock
- object_detection: YOLOv12-S inference server

Endpoint base URLs are read from environment variables so the same image
works locally and in Docker / EC2 without code changes.
"""

import logging
import os

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint configuration (override via env vars; defaults match the live IPs)
# ---------------------------------------------------------------------------

_MEDGEMMA_URL: str = os.getenv(
    "MEDGEMMA_ENDPOINT_URL", "http://137.74.88.197:8000"
).rstrip("/") + "/v1/chat/completions"

_QWEN_VL_URL: str = os.getenv(
    "QWEN_VL_ENDPOINT_URL", "http://137.74.88.197:8001"
).rstrip("/") + "/v1/chat/completions"

_YOLOV12_URL: str = os.getenv(
    "YOLOV12_ENDPOINT_URL", "http://localhost:9003"
).rstrip("/") + "/v1/detect"

_HTTP_TIMEOUT: int = int(os.getenv("TOOL_HTTP_TIMEOUT", "60"))


# ---------------------------------------------------------------------------
# Medical QA — MedGemma endpoint
# ---------------------------------------------------------------------------


@tool
async def medical_qa(query: str) -> str:
    """Answer medical and health-related questions using the MedGemma model.

    Calls the MedGemma vLLM-hosted OpenAI-compatible endpoint at
    ``MEDGEMMA_ENDPOINT_URL`` (default: ``http://137.74.88.197:8000``).

    Args:
        query: The medical question or clinical text to analyse.

    Returns:
        A medically-informed response string from the MedGemma endpoint.

    Raises:
        httpx.HTTPStatusError: On 4xx / 5xx responses from the model server.
        httpx.TimeoutException: When the model server does not respond in time.
    """
    logger.info("medical_qa called | url=%s query[:120]=%r", _MEDGEMMA_URL, query[:120])
    payload = {
        "model": "google/medgemma-1.5-4b-it",
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 512,
        "temperature": 0.2,
    }
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(_MEDGEMMA_URL, json=payload)
            resp.raise_for_status()
        data = resp.json()
        answer: str = data["choices"][0]["message"]["content"]
        logger.info("medical_qa | tokens_used=%s", data.get("usage"))
        return answer
    except httpx.HTTPStatusError as exc:
        logger.error(
            "medical_qa | HTTP %s from %s: %s",
            exc.response.status_code,
            _MEDGEMMA_URL,
            exc.response.text[:200],
        )
        raise
    except httpx.TimeoutException:
        logger.error("medical_qa | request timed out after %ds", _HTTP_TIMEOUT)
        raise


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
    # TODO: Wire up Pinecone vector search + AWS Bedrock Qwen invocation.
    # Steps:
    #   1. Embed the query: vector = embed_fn(query)
    #   2. Query Pinecone: docs = index.query(vector=vector, top_k=5)
    #   3. Build context: context = "\n".join([d.metadata["text"] for d in docs.matches])
    #   4. Call Bedrock Qwen with context + query
    return (
        f"[STUB — Legal RAG + Qwen/Bedrock] Legal answer for: '{query}'. "
        "Endpoint not yet configured. Set PINECONE_API_KEY, PINECONE_INDEX, "
        "and AWS credentials, then replace this stub with the real calls."
    )


# ---------------------------------------------------------------------------
# Vision LLM — Qwen3-VL-2B endpoint
# ---------------------------------------------------------------------------


@tool
async def vision_llm(query: str, image_base64: str = "") -> str:
    """Analyse an image and answer a natural-language question about it using
    the Qwen3-VL-2B vision-language model.

    Calls the Qwen3-VL vLLM-hosted OpenAI-compatible endpoint at
    ``QWEN_VL_ENDPOINT_URL`` (default: ``http://137.74.88.197:8001``).
    When ``image_base64`` is provided the message content is sent as a
    multi-modal list (text + image_url); otherwise plain text is used.

    Args:
        query: The question or instruction about the image.
        image_base64: Base64-encoded image data (JPEG/PNG/WebP).

    Returns:
        A vision-language response string from the Qwen3-VL-2B endpoint.

    Raises:
        httpx.HTTPStatusError: On 4xx / 5xx responses from the model server.
        httpx.TimeoutException: When the model server does not respond in time.
    """
    logger.info(
        "vision_llm called | url=%s query[:120]=%r image_provided=%s",
        _QWEN_VL_URL,
        query[:120],
        bool(image_base64),
    )

    # Build message content — multi-modal when an image is present.
    if image_base64:
        content: list = [
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
    else:
        content = query  # type: ignore[assignment]

    payload = {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
    }
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(_QWEN_VL_URL, json=payload)
            resp.raise_for_status()
        data = resp.json()
        answer: str = data["choices"][0]["message"]["content"]
        logger.info("vision_llm | tokens_used=%s", data.get("usage"))
        return answer
    except httpx.HTTPStatusError as exc:
        logger.error(
            "vision_llm | HTTP %s from %s: %s",
            exc.response.status_code,
            _QWEN_VL_URL,
            exc.response.text[:200],
        )
        raise
    except httpx.TimeoutException:
        logger.error("vision_llm | request timed out after %ds", _HTTP_TIMEOUT)
        raise


# ---------------------------------------------------------------------------
# Object Detection — YOLOv12-S endpoint
# ---------------------------------------------------------------------------


@tool
async def object_detection(image_base64: str = "") -> str:
    """Detect and localise objects in an image using the YOLOv12-S model.

    Args:
        image_base64: Base64-encoded image data (JPEG/PNG).

    Returns:
        A string listing detected objects with bounding boxes and confidence
        scores from the YOLOv12-S endpoint.
    """
    logger.info("object_detection called | image_provided=%s", bool(image_base64))
    # TODO: Replace with real HTTP call once YOLOv12-S endpoint is deployed.
    # The endpoint is expected to accept {"image": "<base64>"} and return
    # {"detections": [{"label": ..., "confidence": ..., "bbox": [...]}]}
    # Example:
    #   async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
    #       resp = await client.post(_YOLOV12_URL, json={"image": image_base64})
    #       resp.raise_for_status()
    #   return str(resp.json()["detections"])
    return (
        "[STUB — YOLOv12-S] Object detection endpoint not yet deployed. "
        f"Set YOLOV12_ENDPOINT_URL (currently: {_YOLOV12_URL}) and replace this stub."
    )


# ---------------------------------------------------------------------------
# Tool registry (used by graph.py)
# ---------------------------------------------------------------------------

TOOLS = [medical_qa, legal_qa, vision_llm, object_detection]
TOOL_MAP: dict[str, object] = {t.name: t for t in TOOLS}
