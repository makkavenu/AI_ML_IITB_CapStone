import os
import io
from typing import Optional
from urllib.parse import unquote

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, ImageOps

router = APIRouter()

RETFOUND_ENDPOINT_URL = os.getenv(
    "RETFOUND_ENDPOINT_URL",
    "http://137.74.88.197:8005"
).rstrip("/")

HTTP_TIMEOUT = int(os.getenv("TOOL_HTTP_TIMEOUT", "120"))


class RETFoundRequest(BaseModel):
    request_id: str
    model: str = "retfound-cfp"
    dataset: str
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    task: str
    classes: list[str] = Field(default_factory=list)
    prompt_for_explainer: Optional[str] = None
    return_fields: list[str] = Field(default_factory=list, alias="return")


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _image_id_from_url(url: str) -> str:
    return unquote(url.rstrip("/").split("/")[-1])


def _ground_truth_from_dataset(dataset: str, image_source: str) -> Optional[str]:
    decoded = unquote(image_source)

    filename = decoded.rstrip("/").split("/")[-1]

    if dataset == "HRF":
        lower = filename.lower()
        if lower.endswith("_h.jpg") or lower.endswith("_h.jpeg") or lower.endswith("_h.png"):
            return "healthy"
        if "_dr" in lower:
            return "diabetic_retinopathy"
        if lower.endswith("_g.jpg") or lower.endswith("_g.jpeg") or lower.endswith("_g.png"):
            return "glaucoma"

    if dataset == "Paraguay_DR":
        folders = [
            "No DR signs",
            "Mild (or early) NPDR",
            "Mild NPDR",
            "Moderate NPDR",
            "Severe NPDR",
            "Very Severe NPDR",
            "PDR",
            "Advanced PDR",
        ]
        for folder in folders:
            if f"/{folder}/" in decoded:
                return folder.replace("Mild (or early) NPDR", "Mild NPDR")

    return None


async def _download_image_as_jpeg(url: str) -> tuple[bytes, str, str]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")

    # If S3 returns XML/HTML AccessDenied or wrong file, fail clearly.
    if "xml" in content_type or "html" in content_type:
        raise HTTPException(
            status_code=400,
            detail=f"S3 URL did not return an image. Content-Type={content_type}. Check public access and URL."
        )

    try:
        image = Image.open(io.BytesIO(response.content))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Downloaded file is not a valid image: {exc}"
        )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()

    original_filename = _image_id_from_url(url)
    safe_filename = original_filename.rsplit(".", 1)[0] + ".jpg"

    return image_bytes, safe_filename, "image/jpeg"


@router.get("/health")
async def health() -> dict:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.get(f"{RETFOUND_ENDPOINT_URL}/health")
        response.raise_for_status()
    return response.json()


@router.post("/infer")
async def infer(request: RETFoundRequest) -> dict:
    image_source = request.image_url or request.image_path

    if not image_source:
        raise HTTPException(status_code=400, detail="image_url is required")

    if not _is_url(image_source):
        raise HTTPException(
            status_code=400,
            detail="This wrapper expects a public image_url, not a local path."
        )

    try:
        image_bytes, filename, mime_type = await _download_image_as_jpeg(image_source)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not download image_url: {exc}")

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            raw_response = await client.post(
                f"{RETFOUND_ENDPOINT_URL}/infer",
                params={"output": "embedding"},
                files={"file": (filename, image_bytes, mime_type)},
            )
            raw_response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Raw RETFound endpoint failed",
                "status_code": exc.response.status_code,
                "response": exc.response.text[:500],
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not call RETFound: {exc}")

    raw = raw_response.json()
    embedding = raw.get("embedding", [])
    embedding_dim = raw.get("embedding_dim", len(embedding))

    ground_truth = _ground_truth_from_dataset(request.dataset, image_source)

    demo_label = ground_truth

    return {
        "request_id": request.request_id,
        "model_used": "RETFound",
        "dataset": request.dataset,
        "image_id": filename,
        "input_type": "retinal_fundus_image",
        "task": request.task,

        "retfound_output": {
            "feature_type": "retinal_foundation_embedding",
            "embedding_dim": embedding_dim,
            "embedding_preview": embedding[:12],
            "embedding": embedding if "embedding" in request.return_fields else None
        },

        "demo_dataset_label": {
            "label": demo_label,
            "source": "dataset_filename_or_folder_label"
        },

        "structured_result": {
            "retinal_image_type": "color_fundus",
            "model_family": "RETFound",
            "demo_task": request.task,
            "classes_requested": request.classes
        },

        "explanation": (
            f"RETFound processed this retinal fundus image and extracted a "
            f"{embedding_dim}-dimensional retinal feature embedding. "
            f"For this research demo, the dataset label associated with this image is "
            f"'{demo_label}'. These features can be used for downstream retinal analysis, "
            f"similarity search, retrieval, or classifier-based workflows."
        ),

        "research_warning": [
            "Research demo only",
            "Not a clinical diagnosis",
            "Dataset labels are used only for academic demonstration"
        ]
    }
