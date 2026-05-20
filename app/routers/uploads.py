"""Upload orchestration endpoints.

Frontend flow:
1. POST /api/uploads/presign with file metadata.
2. Upload each file directly to S3 using the returned presigned PUT URL.
3. POST /api/chat/messages with the returned S3 file references.
"""

import logging
import mimetypes
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.services.aws_clients import AWS_INDIA_REGION, S3_BUCKET_NAME, s3_client
from app.services.metrics import FILE_PRESIGN_FILES_TOTAL, FILE_PRESIGN_REQUESTS_TOTAL

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILES_PER_REQUEST: int = 10
DEFAULT_PRESIGN_EXPIRES_SECONDS: int = 900
MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024  # accepted upload cap per file


class UploadFileDescriptor(BaseModel):
    """Metadata for a file the frontend wants to upload."""

    filename: str
    content_type: Optional[str] = None
    size_bytes: Optional[int] = Field(default=None, ge=0, le=MAX_FILE_SIZE_BYTES)


class PresignUploadRequest(BaseModel):
    """Request body for POST /api/uploads/presign."""

    session_id: Optional[str] = None
    files: list[UploadFileDescriptor]

    @field_validator("files")
    @classmethod
    def validate_files(cls, files: list[UploadFileDescriptor]) -> list[UploadFileDescriptor]:
        if not files:
            raise ValueError("At least one file is required")
        if len(files) > MAX_FILES_PER_REQUEST:
            raise ValueError(f"Maximum {MAX_FILES_PER_REQUEST} files are allowed")
        return files


class PresignedFile(BaseModel):
    """One presigned S3 PUT target."""

    upload_id: str
    filename: str
    content_type: str
    size_bytes: Optional[int]
    s3_bucket: str
    s3_key: str
    s3_uri: str
    presigned_put_url: str
    upload_headers: dict[str, str]
    public_url: str
    expires_in_seconds: int


class PresignUploadResponse(BaseModel):
    """Response body for POST /api/uploads/presign."""

    upload_batch_id: str
    files: list[PresignedFile]


def _safe_filename(filename: str) -> str:
    """Return a path-safe filename while preserving the extension."""
    cleaned = filename.replace("\\", "_").replace("/", "_").strip() or "file"
    return cleaned[:180]


@router.post("/uploads/presign", response_model=PresignUploadResponse)
async def presign_uploads(request: PresignUploadRequest) -> PresignUploadResponse:
    """Create presigned PUT URLs for direct-to-S3 frontend uploads."""
    upload_batch_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    date_prefix = now.strftime("%Y/%m/%d")
    response_files: list[PresignedFile] = []

    for file_desc in request.files:
        upload_id = str(uuid.uuid4())
        filename = _safe_filename(file_desc.filename)
        content_type = (
            file_desc.content_type
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )
        
        s3_key = f"inputs/{date_prefix}/{upload_batch_id}/{upload_id}-{filename}"
        metadata = {
            "original-filename": quote(filename),
            "upload-batch-id": upload_batch_id,
        }
        upload_headers = {
            "Content-Type": content_type,
            "x-amz-meta-original-filename": metadata["original-filename"],
            "x-amz-meta-upload-batch-id": metadata["upload-batch-id"],
        }

        try:
            presigned_put_url = s3_client().generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": S3_BUCKET_NAME,
                    "Key": s3_key,
                    "ContentType": content_type,
                    "Metadata": metadata,
                },
                ExpiresIn=DEFAULT_PRESIGN_EXPIRES_SECONDS,
            )
        except Exception as exc:
            logger.exception("Could not generate S3 presigned URL")
            FILE_PRESIGN_REQUESTS_TOTAL.labels(status="failed").inc()
            raise HTTPException(
                status_code=500,
                detail=f"Could not generate S3 presigned URL: {exc}",
            )

        response_files.append(
            PresignedFile(
                upload_id=upload_id,
                filename=filename,
                content_type=content_type,
                size_bytes=file_desc.size_bytes,
                s3_bucket=S3_BUCKET_NAME,
                s3_key=s3_key,
                s3_uri=f"s3://{S3_BUCKET_NAME}/{s3_key}",
                presigned_put_url=presigned_put_url,
                upload_headers=upload_headers,
                public_url=f"https://{S3_BUCKET_NAME}.s3.{AWS_INDIA_REGION}.amazonaws.com/{s3_key}",
                expires_in_seconds=DEFAULT_PRESIGN_EXPIRES_SECONDS,
            )
        )

    FILE_PRESIGN_REQUESTS_TOTAL.labels(status="success").inc()
    FILE_PRESIGN_FILES_TOTAL.inc(len(response_files))
    return PresignUploadResponse(upload_batch_id=upload_batch_id, files=response_files)
