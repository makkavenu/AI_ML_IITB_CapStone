import base64
import io
import json
import os
from typing import Optional

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

router = APIRouter()

RAW_SAM_MED2D_URL = os.getenv(
    "SAM_MED2D_ENDPOINT_URL",
    "http://137.74.88.197:8004"
).rstrip("/") + "/predict"

HTTP_TIMEOUT = int(os.getenv("TOOL_HTTP_TIMEOUT", "120"))


class SamMed2DRequest(BaseModel):
    request_id: str
    model: str = "sam-med2d"

    # Use image_url for S3/public URLs.
    image_url: Optional[str] = None

    # Keep this only for compatibility if GPT sends image_path as URL.
    image_path: Optional[str] = None

    task: str = "medical_2d_segmentation"
    target_label: str
    prompt_for_demo_ui: Optional[str] = None
    prompt_type: str = "bbox"
    bbox: list[int]

    reference_mask_url: Optional[str] = None
    reference_mask_path_for_evaluation: Optional[str] = None

    return_fields: list[str] = Field(default_factory=list, alias="return")


class SamMed2DResponse(BaseModel):
    request_id: str
    model_used: str
    image_id: str
    target_label: str
    input_type: str
    output_type: str
    sam_prompt_used: dict
    prediction: dict
    evaluation: dict
    explanation_for_professor: str


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


async def _download_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.get(url)
        response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def _normalize_rgb_uint8(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    arr = np.array(image)

    if arr.dtype != np.uint8:
        arr = arr.astype("float32")
        mn, mx = float(arr.min()), float(arr.max())
        arr = ((arr - mn) / (mx - mn + 1e-6) * 255).astype("uint8")
        image = Image.fromarray(arr)

    return image.convert("RGB")


def _validate_bbox(box: list[int], width: int, height: int) -> list[int]:
    if len(box) != 4:
        raise HTTPException(status_code=400, detail="bbox must be [x1, y1, x2, y2]")

    x1, y1, x2, y2 = map(int, box)

    if x1 < 0 or y1 < 0:
        raise HTTPException(status_code=400, detail="bbox x1/y1 cannot be negative")

    if x2 >= width or y2 >= height:
        raise HTTPException(
            status_code=400,
            detail=f"bbox outside image. Image size={width}x{height}, bbox={box}",
        )

    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=400, detail=f"Invalid bbox: {box}")

    return [x1, y1, x2, y2]


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_to_base64_png(image: Image.Image) -> str:
    return base64.b64encode(_image_to_png_bytes(image)).decode("utf-8")


def _create_overlay(base_image: Image.Image, mask_image: Image.Image) -> Image.Image:
    base = base_image.convert("RGBA")
    mask = mask_image.convert("L").resize(base.size, Image.Resampling.NEAREST)

    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha = mask.point(lambda p: 110 if p > 0 else 0)
    overlay.putalpha(alpha)

    return Image.alpha_composite(base, overlay)


def _compute_metrics(pred_mask: Image.Image, ref_mask: Optional[Image.Image]) -> dict:
    if ref_mask is None:
        return {"dice_score": None, "iou_score": None}

    pred = np.array(pred_mask.convert("L")) > 0
    ref = np.array(
        ref_mask.convert("L").resize(pred_mask.size, Image.Resampling.NEAREST)
    ) > 0

    intersection = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    pred_sum = int(pred.sum())
    ref_sum = int(ref.sum())

    dice = round((2 * intersection) / (pred_sum + ref_sum), 4) if pred_sum + ref_sum else None
    iou = round(intersection / union, 4) if union else None

    return {"dice_score": dice, "iou_score": iou}


@router.post("/predict", response_model=SamMed2DResponse)
async def predict(request: SamMed2DRequest) -> SamMed2DResponse:
    image_source = request.image_url or request.image_path

    if not image_source:
        raise HTTPException(status_code=400, detail="image_url is required")

    if not _is_url(image_source):
        raise HTTPException(
            status_code=400,
            detail="For this wrapper, use a public image_url. Local server paths are not enabled.",
        )

    try:
        image = await _download_image(image_source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not download image: {exc}")

    image = _normalize_rgb_uint8(image)

    width, height = image.size
    bbox = _validate_bbox(request.bbox, width, height)

    image_bytes = _image_to_png_bytes(image)

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            raw_response = await client.post(
                RAW_SAM_MED2D_URL,
                files={"file": ("image.png", image_bytes, "image/png")},
                data={"prompt": json.dumps({"box": bbox})},
            )
            raw_response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Raw SAM-Med2D endpoint failed",
                "status_code": exc.response.status_code,
                "response": exc.response.text[:500],
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not call SAM-Med2D: {exc}")

    try:
        pred_mask = Image.open(io.BytesIO(raw_response.content)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"SAM-Med2D response was not an image: {exc}")

    if pred_mask.size != image.size:
        pred_mask = pred_mask.resize(image.size, Image.Resampling.NEAREST)

    area_pixels = int((np.array(pred_mask) > 0).sum())

    reference_source = request.reference_mask_url or request.reference_mask_path_for_evaluation
    ref_mask = None

    if reference_source:
        if _is_url(reference_source):
            try:
                ref_mask = await _download_image(reference_source)
            except Exception:
                ref_mask = None

    metrics = _compute_metrics(pred_mask, ref_mask)
    overlay = _create_overlay(image, pred_mask)

    mask_base64 = _image_to_base64_png(pred_mask)
    overlay_base64 = _image_to_base64_png(overlay)

    return SamMed2DResponse(
        request_id=request.request_id,
        model_used="SAM-Med2D",
        image_id=image_source.split("/")[-1],
        target_label=request.target_label,
        input_type="2D medical image",
        output_type="binary_segmentation_mask",
        sam_prompt_used={
            "prompt_type": request.prompt_type,
            "box": bbox,
        },
        prediction={
            "mask_base64": mask_base64,
            "mask_data_url": f"data:image/png;base64,{mask_base64}",
            "overlay_base64": overlay_base64,
            "overlay_data_url": f"data:image/png;base64,{overlay_base64}",
            "area_pixels": area_pixels,
        },
        evaluation={
            "reference_mask_used": reference_source,
            "dice_score": metrics["dice_score"],
            "iou_score": metrics["iou_score"],
        },
        explanation_for_professor=(
            f"SAM-Med2D received a 2D medical image and a bounding-box prompt for "
            f"{request.target_label}. It returned a pixel-level binary segmentation mask. "
            f"This is a research segmentation demo, not a clinical diagnosis."
        ),
    )
