"""Helpers for uploaded S3 files.

The chat router accepts arbitrary file formats. For routing accuracy we extract
lightweight context from file types that are safe and cheap to inspect:
images are converted to base64 for vision models, text-like files are decoded,
and PDFs are text-extracted with pypdf. Unknown formats are still represented by
metadata so the router can make a conservative decision.
"""

import base64
import io
import logging
import mimetypes
from dataclasses import dataclass
from typing import Any

from botocore.exceptions import ClientError
from PIL import Image, ImageOps
from pypdf import PdfReader

from app.services.aws_clients import S3_BUCKET_NAME, s3_client

logger = logging.getLogger(__name__)

MAX_FILE_BYTES_TO_DOWNLOAD: int = 20 * 1024 * 1024
MAX_TEXT_CHARS_PER_FILE: int = 6000
MAX_IMAGES_FOR_ROUTER: int = 10

TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".log",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rs",
    ".sql", ".yaml", ".yml",
}


@dataclass
class ProcessedFileContext:
    """Processed context for one uploaded file."""

    filename: str
    s3_key: str
    s3_uri: str
    content_type: str
    size_bytes: int | None
    category: str
    extracted_text: str = ""
    image_base64: str = ""
    error: str = ""


def guess_category(filename: str, content_type: str) -> str:
    """Classify a file into a coarse routing category."""
    lower = filename.lower()
    guessed = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    if guessed.startswith("image/"):
        return "image"
    if guessed.startswith("video/"):
        return "video"
    if guessed.startswith("text/") or any(lower.endswith(ext) for ext in TEXT_EXTENSIONS):
        return "text"
    if guessed == "application/pdf" or lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith((".nii", ".nii.gz", ".dcm")):
        return "medical_volume"
    return "unknown"


def download_s3_file(s3_key: str, max_bytes: int = MAX_FILE_BYTES_TO_DOWNLOAD) -> bytes:
    """Download an S3 object with a size cap."""
    head = s3_client().head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
    size = int(head.get("ContentLength", 0))
    if size > max_bytes:
        raise ValueError(f"File is too large to inspect synchronously: {size} bytes")
    obj = s3_client().get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
    return obj["Body"].read()


def image_bytes_to_base64_jpeg(raw: bytes) -> str:
    """Normalise an image to JPEG base64 for vision APIs."""
    image = Image.open(io.BytesIO(raw))
    image = ImageOps.exif_transpose(image).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_pdf_text(raw: bytes) -> str:
    """Extract limited text from a PDF byte string."""
    reader = PdfReader(io.BytesIO(raw))
    parts: list[str] = []
    for page in reader.pages[:10]:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())
        if sum(len(p) for p in parts) >= MAX_TEXT_CHARS_PER_FILE:
            break
    return "\n\n".join(parts)[:MAX_TEXT_CHARS_PER_FILE]


def extract_text_like(raw: bytes) -> str:
    """Decode text-like bytes with a fallback error strategy."""
    return raw.decode("utf-8", errors="replace")[:MAX_TEXT_CHARS_PER_FILE]


def build_file_context(uploaded_files: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], str]:
    """Inspect uploaded files and return metadata, image b64 list, and text context."""
    processed: list[ProcessedFileContext] = []
    image_base64_list: list[str] = []

    for f in uploaded_files[:MAX_IMAGES_FOR_ROUTER]:
        filename = f.get("filename") or f.get("name") or "uploaded-file"
        s3_key = f.get("s3_key") or ""
        s3_uri = f.get("s3_uri") or f"s3://{S3_BUCKET_NAME}/{s3_key}"
        content_type = f.get("content_type") or "application/octet-stream"
        size_bytes = f.get("size_bytes")
        category = guess_category(filename, content_type)
        ctx = ProcessedFileContext(
            filename=filename,
            s3_key=s3_key,
            s3_uri=s3_uri,
            content_type=content_type,
            size_bytes=size_bytes,
            category=category,
        )

        try:
            raw = download_s3_file(s3_key)
            if category == "image" and len(image_base64_list) < MAX_IMAGES_FOR_ROUTER:
                ctx.image_base64 = image_bytes_to_base64_jpeg(raw)
                image_base64_list.append(ctx.image_base64)
            elif category == "pdf":
                ctx.extracted_text = extract_pdf_text(raw)
            elif category == "text":
                ctx.extracted_text = extract_text_like(raw)
        except Exception as exc:
            # Keep metadata even when extraction fails; the router can still see file name/type.
            logger.warning("File context extraction failed | key=%s error=%s", s3_key, exc)
            ctx.error = str(exc)

        processed.append(ctx)

    # Do not persist base64 image payloads in DynamoDB. DynamoDB has a hard
    # 400 KB item limit; image_base64 is returned separately in image_base64_list
    # for model calls and should not be copied into processed_files metadata.
    processed_dicts = []
    for c in processed:
        item = c.__dict__.copy()
        image_b64 = item.pop("image_base64", "") or ""
        item["image_loaded_for_model"] = bool(image_b64)
        item["image_base64_chars_omitted"] = len(image_b64)
        processed_dicts.append(item)

    context_lines: list[str] = []
    for idx, c in enumerate(processed, start=1):
        context_lines.append(
            f"File {idx}: filename={c.filename!r}, type={c.content_type!r}, "
            f"category={c.category!r}, size_bytes={c.size_bytes}, s3_uri={c.s3_uri!r}"
        )
        if c.extracted_text:
            context_lines.append(f"Extracted text preview for file {idx}:\n{c.extracted_text}")
        if c.image_base64:
            context_lines.append(f"Image file {idx} was loaded for vision analysis.")
        if c.error:
            context_lines.append(f"File {idx} extraction warning: {c.error}")

    return processed_dicts, image_base64_list, "\n\n".join(context_lines)


def verify_uploaded_file_references(uploaded_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verify S3 references before accepting a chat request.

    Checks that every referenced object exists in the configured bucket, lives
    under the ``inputs/`` prefix, and matches the client-provided size when the
    frontend supplied one. This prevents a user from submitting arbitrary S3
    paths or stale/unuploaded keys to the chat worker.
    """
    verified: list[dict[str, Any]] = []
    if len(uploaded_files) > MAX_IMAGES_FOR_ROUTER:
        raise ValueError(f"Maximum {MAX_IMAGES_FOR_ROUTER} files are allowed per chat request")

    for idx, raw_ref in enumerate(uploaded_files, start=1):
        ref = dict(raw_ref)
        bucket = ref.get("s3_bucket") or S3_BUCKET_NAME
        key = ref.get("s3_key") or ""
        if bucket != S3_BUCKET_NAME:
            raise ValueError(f"File {idx}: unexpected S3 bucket {bucket!r}")
        if not key or not key.startswith("inputs/") or ".." in key:
            raise ValueError(f"File {idx}: unsafe or missing S3 key {key!r}")

        try:
            head = s3_client().head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "Unknown")
            raise ValueError(f"File {idx}: S3 object does not exist or is not accessible ({code})") from exc

        actual_size = int(head.get("ContentLength", 0))
        if actual_size <= 0:
            raise ValueError(f"File {idx}: uploaded S3 object is empty")
        expected_size = ref.get("size_bytes")
        if expected_size is not None and int(expected_size) != actual_size:
            raise ValueError(
                f"File {idx}: size mismatch; frontend={expected_size} bytes, S3={actual_size} bytes"
            )

        actual_content_type = head.get("ContentType") or ref.get("content_type") or "application/octet-stream"
        expected_content_type = ref.get("content_type") or "application/octet-stream"
        if (
            expected_content_type != "application/octet-stream"
            and actual_content_type != "binary/octet-stream"
            and actual_content_type.split(";", 1)[0].lower() != expected_content_type.split(";", 1)[0].lower()
        ):
            raise ValueError(
                f"File {idx}: content-type mismatch; frontend={expected_content_type!r}, S3={actual_content_type!r}"
            )

        ref.update({
            "s3_bucket": bucket,
            "s3_key": key,
            "s3_uri": ref.get("s3_uri") or f"s3://{bucket}/{key}",
            "size_bytes": actual_size,
            "content_type": actual_content_type,
            "verified": True,
            "verification_method": "s3_head_object",
        })
        verified.append(ref)

    return verified
