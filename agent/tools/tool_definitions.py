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

import base64 as _base64
import io
import json
import logging
import os

import httpx
from langchain_core.tools import tool
from PIL import Image

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
    "YOLOV12_ENDPOINT_URL", "http://137.74.88.197:8002"
).rstrip("/") + "/predict"

_HTTP_TIMEOUT: int = int(os.getenv("TOOL_HTTP_TIMEOUT", "60"))
# Max pixel dimension sent to the YOLO server — keeps memory usage predictable.
_YOLO_MAX_DIM: int = int(os.getenv("YOLO_MAX_DIM", "640"))


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
    """Detect and localise objects in an image using the YOLOv8 model.

    Calls the YOLOv8 inference endpoint at ``YOLOV12_ENDPOINT_URL``
    (default: ``http://137.74.88.197:8002``) via a multipart file upload.
    Returns a JSON string containing the detection list and a human-readable
    summary so the synthesiser can describe results and the API layer can
    draw bounding boxes onto the original image.

    Args:
        image_base64: Base64-encoded image data (JPEG/PNG/WebP).

    Returns:
        JSON string with keys ``summary`` (str), ``detections`` (list),
        ``count`` (int), ``image_width`` (int), ``image_height`` (int).

    Raises:
        httpx.HTTPStatusError: On 4xx / 5xx responses from the model server.
        httpx.TimeoutException: When the model server does not respond in time.
    """
    logger.info("object_detection called | url=%s image_provided=%s", _YOLOV12_URL, bool(image_base64))

    if not image_base64:
        return json.dumps({
            "summary": "No image was provided. Please attach an image to run object detection.",
            "detections": [],
            "count": 0,
            "image_width": 0,
            "image_height": 0,
        })

    try:
        # Remove data-URI prefix if present (e.g. "data:image/jpeg;base64,...")
        raw = image_base64.split(",", 1)[-1]
        # Remove ALL whitespace — b64decode silently ignores spaces/newlines
        # and produces garbage bytes rather than raising an error.
        raw = "".join(raw.split())
        # Normalise URL-safe base64 chars (- → +, _ → /) to standard alphabet
        raw = raw.replace("-", "+").replace("_", "/")
        # Restore any stripped padding
        raw += "=" * (-len(raw) % 4)
        logger.info(
            "object_detection | base64 len=%d first30=%r", len(raw), raw[:30]
        )
        img_bytes = _base64.b64decode(raw, validate=True)
    except Exception:
        logger.exception("object_detection | failed to decode base64 image")
        return json.dumps({"summary": "Invalid image data.", "detections": [], "count": 0,
                           "image_width": 0, "image_height": 0})

    # Resize to _YOLO_MAX_DIM to avoid OOM on the inference server
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = pil_img.size
        if max(orig_w, orig_h) > _YOLO_MAX_DIM:
            scale = _YOLO_MAX_DIM / max(orig_w, orig_h)
            new_size = (int(orig_w * scale), int(orig_h * scale))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            logger.info(
                "object_detection | resized %dx%d -> %dx%d (max_dim=%d)",
                orig_w, orig_h, new_size[0], new_size[1], _YOLO_MAX_DIM,
            )
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()
    except Exception:
        logger.exception("object_detection | resize failed — sending original bytes")

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(
                _YOLOV12_URL,
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
            )
            resp.raise_for_status()
        data = resp.json()

        detections: list[dict] = data.get("detections", [])
        count: int = data.get("count", len(detections))

        if not detections:
            summary = "No objects were detected in the image."
        else:
            parts = [f"{d['class_name']} ({d['confidence']:.0%})" for d in detections]
            summary = f"Detected {count} object(s): {', '.join(parts)}."

        logger.info("object_detection | count=%d detections=%s", count,
                    [d['class_name'] for d in detections])
        return json.dumps({
            "summary": summary,
            "detections": detections,
            "count": count,
            "image_width": data.get("image_width", 0),
            "image_height": data.get("image_height", 0),
        })

    except httpx.HTTPStatusError as exc:
        logger.error(
            "object_detection | HTTP %s from %s: %s",
            exc.response.status_code,
            _YOLOV12_URL,
            exc.response.text[:400],
        )
        return json.dumps({
            "summary": (
                f"Object detection failed (server returned HTTP {exc.response.status_code}). "
                "The inference server encountered an error processing the image."
            ),
            "detections": [],
            "count": 0,
            "image_width": 0,
            "image_height": 0,
        })
    except httpx.TimeoutException:
        logger.error("object_detection | request timed out after %ds", _HTTP_TIMEOUT)
        return json.dumps({
            "summary": f"Object detection timed out after {_HTTP_TIMEOUT}s. Try again.",
            "detections": [],
            "count": 0,
            "image_width": 0,
            "image_height": 0,
        })


# ---------------------------------------------------------------------------
# Tool registry (used by graph.py)
# ---------------------------------------------------------------------------

TOOLS = [medical_qa, legal_qa, vision_llm, object_detection]
TOOL_MAP: dict[str, object] = {t.name: t for t in TOOLS}
