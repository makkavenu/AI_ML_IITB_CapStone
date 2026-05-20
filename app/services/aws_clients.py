"""Shared AWS client factory helpers.

The project stores request metadata in DynamoDB and uploads user files to S3.
AWS region is fixed to ap-south-1 by default for this research demo, while
allowing an environment override for future deployments.
"""

import os
from functools import lru_cache
from typing import Any

import boto3

AWS_INDIA_REGION: str = os.getenv("AWS_INDIA_REGION", "ap-south-1")
S3_BUCKET_NAME: str = os.getenv(
    "S3_BUCKET_NAME", "ai-agent-project-requests-uploads"
)
DDB_TABLE_NAME: str = os.getenv("DDB_TABLE_NAME", "ai-agent-requests")


@lru_cache(maxsize=1)
def s3_client() -> Any:
    """Return a cached boto3 S3 client."""
    return boto3.client("s3", region_name=AWS_INDIA_REGION)


@lru_cache(maxsize=1)
def dynamodb_resource() -> Any:
    """Return a cached boto3 DynamoDB resource."""
    return boto3.resource("dynamodb", region_name=AWS_INDIA_REGION)


@lru_cache(maxsize=1)
def request_table() -> Any:
    """Return the configured DynamoDB table resource."""
    return dynamodb_resource().Table(DDB_TABLE_NAME)
