import asyncio
import os

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_REDIS_INTEGRATION_TESTS") != "1",
    reason="Set RUN_REDIS_INTEGRATION_TESTS=1 to run against a real Redis service.",
)
def test_real_redis_response_cache_round_trip():
    redis_asyncio = pytest.importorskip("redis.asyncio")
    from app.services.cache import build_response_cache_key, get_cached_response, set_cached_response

    async def scenario():
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis_asyncio.from_url(redis_url, encoding="utf-8", decode_responses=True)
        key = build_response_cache_key(message="redis integration", orchestrator_model="qwen3-32b")
        payload = {"response": "ok", "answer_model": "Qwen3-32B (AWS Bedrock)", "tools_chain": []}
        try:
            await set_cached_response(key, payload, ttl_seconds=30, redis_client=client)
            assert await get_cached_response(key, redis_client=client) == payload
        finally:
            await client.delete(key)
            await client.aclose()

    asyncio.run(scenario())
