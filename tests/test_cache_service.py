import asyncio
import json

from app.services.cache import build_response_cache_key, get_cached_response, set_cached_response


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttl = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.ttl[key] = ttl
        self.store[key] = value
        return True


def test_response_cache_key_is_stable_for_equivalent_payloads():
    files_a = [{"filename": "scan.png", "s3_key": "inputs/a/scan.png", "content_type": "image/png", "size_bytes": 123}]
    files_b = [{"size_bytes": 123, "content_type": "image/png", "s3_key": "inputs/a/scan.png", "filename": "scan.png"}]

    key_a = build_response_cache_key(
        message="  Analyse this image  ",
        orchestrator_model="qwen3-32b",
        history=[{"role": "user", "content": "hello"}],
        files=files_a,
    )
    key_b = build_response_cache_key(
        message="Analyse this image",
        orchestrator_model="qwen3-32b",
        history=[{"content": "hello", "role": "user"}],
        files=files_b,
    )

    assert key_a == key_b
    assert key_a.startswith("ai_agent:response:")


def test_response_cache_key_changes_when_model_changes():
    key_qwen = build_response_cache_key(message="hello", orchestrator_model="qwen3-32b")
    key_gpt = build_response_cache_key(message="hello", orchestrator_model="gpt-4o")

    assert key_qwen != key_gpt


def test_get_set_cached_response_round_trip_with_fake_redis():
    async def scenario():
        fake = FakeRedis()
        key = build_response_cache_key(message="What is this?", orchestrator_model="qwen3-32b")
        payload = {
            "response": "demo answer",
            "tool_used": "medical_qa",
            "tools_chain": ["medical_qa"],
            "answer_model": "MedGemma 1.5 4B",
            "answer_model_chain": ["MedGemma 1.5 4B"],
            "guardrail_flagged": False,
        }

        assert await get_cached_response(key, redis_client=fake) is None
        await set_cached_response(key, payload, ttl_seconds=60, redis_client=fake)
        cached = await get_cached_response(key, redis_client=fake)

        assert cached == payload
        assert fake.ttl[key] == 60
        assert json.loads(fake.store[key]) == payload

    asyncio.run(scenario())
