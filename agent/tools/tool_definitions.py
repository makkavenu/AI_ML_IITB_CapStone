"""Tool definitions for the multi-modal AI agent.

Each tool is decorated with ``@tool`` from ``langchain_core.tools``.

Live endpoints
--------------
- medical_qa     : MedGemma vLLM server   — http://<host>:8000/v1/chat/completions
- vision_llm     : Qwen3-VL-235B via AWS Bedrock Converse API
- object_detection: YOLOv8 inference server — http://<host>:8002/predict

Stubs (endpoint not yet available)
-----------------------------------
- legal_qa : Pinecone RAG + Qwen via AWS Bedrock

Endpoint base URLs are read from environment variables so the same image
works locally and in Docker / EC2 without code changes.
"""

import base64 as _base64
import io
import json
import logging
import os
import threading

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

_YOLOV12_URL: str = os.getenv(
    "YOLOV12_ENDPOINT_URL", "http://137.74.88.197:8002"
).rstrip("/") + "/predict"

_HTTP_TIMEOUT: int = int(os.getenv("TOOL_HTTP_TIMEOUT", "60"))
# Max pixel dimension sent to the YOLO server — keeps memory usage predictable.
_YOLO_MAX_DIM: int = int(os.getenv("YOLO_MAX_DIM", "640"))

# Bedrock vision model — Qwen3-VL-235B served by AWS Bedrock Converse API.
# The exact model id is shown in the Bedrock playground "View API request"
# panel. As of May 2026 it is ``qwen.qwen3-vl-235b-a22b`` (no version suffix,
# no geo prefix) served from ``us-west-2``.
_QWEN_VL_BEDROCK_MODEL_ID: str = os.getenv(
    "QWEN_VL_BEDROCK_MODEL_ID", "qwen.qwen3-vl-235b-a22b"
)
# Region for the vision model — defaults to us-west-2 where Qwen3-VL is
# currently available. Falls back to AWS_DEFAULT_REGION if not set.
_QWEN_VL_BEDROCK_REGION: str = os.getenv(
    "QWEN_VL_BEDROCK_REGION",
    os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
)
# Max pixel dimension sent to Bedrock — keeps payload size predictable and
# well under the per-image limit for the Converse API.
_VISION_MAX_DIM: int = int(os.getenv("VISION_MAX_DIM", "1280"))

# Lazy-built boto3 Bedrock client (reused across calls in the process).
_BEDROCK_CLIENT = None
_BEDROCK_CLIENT_LOCK = threading.Lock()


def _get_bedrock_client():
    """Build (once) and return the boto3 ``bedrock-runtime`` client for the
    vision model.

    Uses standard AWS credentials from the environment
    (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``) but pins the region to
    ``QWEN_VL_BEDROCK_REGION`` (default: ``us-west-2``) since Qwen3-VL is
    only available in specific regions.

    Returns:
        A configured ``bedrock-runtime`` boto3 client.
    """
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is not None:
        return _BEDROCK_CLIENT
    with _BEDROCK_CLIENT_LOCK:
        if _BEDROCK_CLIENT is not None:
            return _BEDROCK_CLIENT
        import boto3  # local import keeps tool module light if unused

        _BEDROCK_CLIENT = boto3.client(
            "bedrock-runtime", region_name=_QWEN_VL_BEDROCK_REGION
        )
        logger.info(
            "_get_bedrock_client | initialised in region=%s for model=%s",
            _QWEN_VL_BEDROCK_REGION, _QWEN_VL_BEDROCK_MODEL_ID,
        )
        return _BEDROCK_CLIENT


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

    The blocking RAG pipeline (OpenAI embed → Pinecone query → Bedrock
    invoke_model) is delegated to a worker thread via ``asyncio.to_thread``
    so the FastAPI event loop remains responsive.

    Args:
        query: The legal question or case description to research.

    Returns:
        A RAG-augmented legal response string sourced from Indian law
        (IPC / CrPC / Constitution).
    """
    import asyncio
    from agent.tools.query_legal_rag import run_legal_rag

    logger.info("legal_qa called | query[:120]=%r", query[:120])
    try:
        answer: str = await asyncio.to_thread(run_legal_rag, query)
        logger.info("legal_qa | response_chars=%d", len(answer or ""))
        return answer or "The legal RAG pipeline returned an empty response."
    except ValueError as exc:
        # Missing env vars — surface clearly without leaking internals.
        logger.error("legal_qa | configuration error: %s", exc)
        return f"Legal RAG is not configured: {exc}"
    except Exception:
        logger.exception("legal_qa | RAG pipeline failed")
        return (
            "The legal RAG pipeline encountered an error while processing "
            "this question. Please try again or rephrase the query."
        )


# ---------------------------------------------------------------------------
# Vision LLM — Qwen3-VL-2B endpoint
# ---------------------------------------------------------------------------


@tool
async def vision_llm(query: str, image_base64: str = "") -> str:
    """Analyse an image and answer a natural-language question about it using
    the Qwen3-VL-235B vision-language model served by AWS Bedrock.

    Calls the AWS Bedrock ``Converse`` API with the model id
    ``QWEN_VL_BEDROCK_MODEL_ID`` (default: ``qwen.qwen3-vl-235b-a22b-v1:0``).
    The blocking boto3 call is delegated to a worker thread via
    ``asyncio.to_thread`` so the FastAPI event loop stays responsive.

    Args:
        query: The question or instruction about the image.
        image_base64: Base64-encoded image data (JPEG/PNG/WebP/GIF).

    Returns:
        A vision-language response string from Bedrock.
    """
    import asyncio

    logger.info(
        "vision_llm called | model=%s query[:120]=%r image_provided=%s",
        _QWEN_VL_BEDROCK_MODEL_ID,
        query[:120],
        bool(image_base64),
    )

    if not image_base64:
        # No image — fall back to a text-only Converse call so the tool still
        # behaves reasonably if invoked without an attachment.
        try:
            return await asyncio.to_thread(_bedrock_vision_text_only, query)
        except Exception:
            logger.exception("vision_llm | text-only Bedrock call failed")
            return (
                "No image was attached and the vision model could not be "
                "reached. Please attach an image and try again."
            )

    # Decode + normalise the image bytes (Bedrock Converse expects raw bytes
    # in the content block, plus an explicit format tag).
    try:
        raw = image_base64.split(",", 1)[-1]
        raw = "".join(raw.split())  # strip whitespace
        raw = raw.replace("-", "+").replace("_", "/")  # URL-safe → standard
        raw += "=" * (-len(raw) % 4)
        img_bytes = _base64.b64decode(raw, validate=True)
    except Exception:
        logger.exception("vision_llm | failed to decode base64 image")
        return "The attached image could not be decoded."

    # Detect format via Pillow; re-encode to JPEG if anything else (and resize
    # large images so we stay well under the Converse per-image size limit).
    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_format = (pil_img.format or "").upper()
        if pil_format == "JPEG":
            fmt = "jpeg"
        elif pil_format == "PNG":
            fmt = "png"
        elif pil_format == "GIF":
            fmt = "gif"
        elif pil_format == "WEBP":
            fmt = "webp"
        else:
            # Unknown format — normalise to JPEG.
            pil_img = pil_img.convert("RGB")
            fmt = "jpeg"

        w, h = pil_img.size
        if max(w, h) > _VISION_MAX_DIM:
            scale = _VISION_MAX_DIM / max(w, h)
            pil_img = pil_img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )
            buf = io.BytesIO()
            save_kwargs = {"quality": 90} if fmt == "jpeg" else {}
            pil_img.convert("RGB" if fmt == "jpeg" else pil_img.mode).save(
                buf, format=fmt.upper(), **save_kwargs
            )
            img_bytes = buf.getvalue()
            logger.info(
                "vision_llm | resized image to max_dim=%d (final_bytes=%d)",
                _VISION_MAX_DIM, len(img_bytes),
            )
    except Exception:
        logger.exception(
            "vision_llm | image normalisation failed — sending original bytes as JPEG"
        )
        fmt = "jpeg"

    def _invoke_bedrock_vision() -> str:
        """Blocking Bedrock Converse call (run in a worker thread)."""
        client = _get_bedrock_client()
        response = client.converse(
            modelId=_QWEN_VL_BEDROCK_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": query or "Describe this image."},
                        {
                            "image": {
                                "format": fmt,
                                "source": {"bytes": img_bytes},
                            }
                        },
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 512, "temperature": 0.2},
        )
        # Converse response shape:
        # {"output": {"message": {"role": "assistant",
        #             "content": [{"text": "..."}]}}, "usage": {...}, ...}
        content_blocks = (
            response.get("output", {}).get("message", {}).get("content", []) or []
        )
        parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        answer = "".join(parts).strip()
        logger.info(
            "vision_llm | usage=%s response_chars=%d",
            response.get("usage"),
            len(answer),
        )
        return answer or "The vision model returned an empty response."

    try:
        return await asyncio.to_thread(_invoke_bedrock_vision)
    except Exception as exc:
        logger.exception("vision_llm | Bedrock Converse call failed")
        return (
            "The vision model on AWS Bedrock encountered an error processing "
            f"the image (model={_QWEN_VL_BEDROCK_MODEL_ID}). "
            f"Details: {type(exc).__name__}: {exc}"
        )


def _bedrock_vision_text_only(query: str) -> str:
    """Text-only Converse call (used when no image is attached)."""
    client = _get_bedrock_client()
    response = client.converse(
        modelId=_QWEN_VL_BEDROCK_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": query}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.2},
    )
    blocks = response.get("output", {}).get("message", {}).get("content", []) or []
    parts = [b["text"] for b in blocks if isinstance(b, dict) and isinstance(b.get("text"), str)]
    return "".join(parts).strip() or "The vision model returned an empty response."


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
