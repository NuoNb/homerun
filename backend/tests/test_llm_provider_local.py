from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.ai import llm_provider
from services.ai.llm_provider import (
    DeepSeekProvider,
    LLMManager,
    LLMMessage,
    LLMProvider,
    LMStudioProvider,
    OpenAIProvider,
    OllamaProvider,
    ToolDefinition,
    ZhipuProvider,
    _ensure_openai_compatible_base_url,
    _normalize_model_name_for_provider,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._payload


# Dummy value for provider constructors in tests — never a real credential.
FAKE_PROVIDER_KEY = "test-key"


def test_detect_provider_for_explicit_local_prefixes():
    manager = LLMManager()

    assert manager.detect_provider("ollama/llama3.2:latest") == LLMProvider.OLLAMA
    assert manager.detect_provider("lmstudio/qwen2.5-7b-instruct") == LLMProvider.LMSTUDIO
    assert manager.detect_provider("glm-4.6") == LLMProvider.ZHIPU
    assert manager.detect_provider("zhipu/glm-5.3") == LLMProvider.ZHIPU


def test_detect_provider_prefers_selected_local_provider_for_generic_model_names():
    manager = LLMManager()
    manager._providers = {LLMProvider.OLLAMA: object()}  # type: ignore[assignment]
    manager._preferred_provider = LLMProvider.OLLAMA

    assert manager.detect_provider("llama3.2:latest") == LLMProvider.OLLAMA


def test_model_normalization_strips_local_provider_prefixes():
    assert _normalize_model_name_for_provider("ollama/llama3.2:latest", LLMProvider.OLLAMA) == "llama3.2:latest"
    assert (
        _normalize_model_name_for_provider("lmstudio/qwen2.5-7b-instruct", LLMProvider.LMSTUDIO)
        == "qwen2.5-7b-instruct"
    )
    assert _normalize_model_name_for_provider("gpt-4o-mini", LLMProvider.OPENAI) == "gpt-4o-mini"
    assert _normalize_model_name_for_provider("zhipu/glm-5.3", LLMProvider.ZHIPU) == "glm-5.3"
    assert _normalize_model_name_for_provider("glm-5.3", LLMProvider.ZHIPU) == "glm-5.3"


def test_openai_compatible_base_url_normalization():
    assert (
        _ensure_openai_compatible_base_url("http://localhost:11434", "http://localhost:11434/v1")
        == "http://localhost:11434/v1"
    )
    assert (
        _ensure_openai_compatible_base_url("http://localhost:1234/v1", "http://localhost:1234/v1")
        == "http://localhost:1234/v1"
    )


def test_openai_format_messages_normalizes_internal_tool_calls():
    provider = OpenAIProvider(api_key=None, base_url="http://localhost:1234/v1", model_prefixes=None)
    messages = [
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "abc123",
                    "name": "get_market_details",
                    "arguments": {"market_id": "PM_2024_Fed_Rate_Cut_April"},
                }
            ],
        )
    ]

    formatted = provider._format_messages(messages)

    assert len(formatted) == 1
    tc = formatted[0]["tool_calls"][0]
    assert tc["id"] == "abc123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_market_details"
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"market_id": "PM_2024_Fed_Rate_Cut_April"}


def test_openai_parse_tool_calls_accepts_dict_arguments():
    provider = OpenAIProvider(api_key=None, base_url="http://localhost:1234/v1", model_prefixes=None)
    raw_tool_calls = [
        {
            "id": "abc123",
            "type": "function",
            "function": {
                "name": "get_market_details",
                "arguments": {"market_id": "PM_2024_Fed_Rate_Cut_April"},
            },
        }
    ]

    parsed = provider._parse_tool_calls(raw_tool_calls)

    assert len(parsed) == 1
    assert parsed[0].id == "abc123"
    assert parsed[0].name == "get_market_details"
    assert parsed[0].arguments == {"market_id": "PM_2024_Fed_Rate_Cut_April"}


@pytest.mark.asyncio
async def test_openai_structured_output_handles_string_error_payload(monkeypatch):
    provider = OpenAIProvider(api_key=None, base_url="http://localhost:1234/v1", model_prefixes=None)
    responses = [_FakeResponse(400, {"error": "Model not loaded"})]

    async def fake_retry(coro_factory, max_retries=3, base_delay=1.0):
        return responses.pop(0)

    monkeypatch.setattr(llm_provider, "_retry_with_backoff", fake_retry)

    with pytest.raises(RuntimeError) as exc_info:
        await provider.structured_output(
            messages=[LLMMessage(role="user", content="Return JSON only.")],
            schema={"type": "object"},
            model="mistralai/ministral-3-3b",
        )

    msg = str(exc_info.value)
    assert "Model not loaded" in msg
    assert "object has no attribute 'get'" not in msg


@pytest.mark.asyncio
async def test_openai_structured_output_uses_json_schema_response_format(monkeypatch):
    provider = OpenAIProvider(api_key=None, base_url="http://localhost:1234/v1", model_prefixes=None)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    responses = [_FakeResponse(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})]
    requests = []

    async def fake_retry(coro_factory, max_retries=3, base_delay=1.0):
        return await coro_factory()

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            requests.append({"url": url, "headers": headers, "json": json})
            return responses.pop(0)

    monkeypatch.setattr(llm_provider, "_retry_with_backoff", fake_retry)
    monkeypatch.setattr(llm_provider.httpx, "AsyncClient", FakeAsyncClient)

    result = await provider.structured_output(
        messages=[LLMMessage(role="user", content="Return JSON only.")],
        schema=schema,
        model="mistralai/ministral-3-3b",
    )

    assert len(requests) == 1
    assert requests[0]["json"]["response_format"]["type"] == "json_schema"
    assert requests[0]["json"]["response_format"]["json_schema"]["schema"] == schema
    assert requests[0]["json"]["response_format"]["json_schema"]["strict"] is True
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_lmstudio_structured_output_uses_json_schema_response_format(monkeypatch):
    provider = LMStudioProvider(base_url="http://localhost:1234/v1", api_key=None)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    responses = [_FakeResponse(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})]
    requests = []

    async def fake_retry(coro_factory, max_retries=3, base_delay=1.0):
        return await coro_factory()

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            requests.append({"url": url, "headers": headers, "json": json})
            return responses.pop(0)

    monkeypatch.setattr(llm_provider, "_retry_with_backoff", fake_retry)
    monkeypatch.setattr(llm_provider.httpx, "AsyncClient", FakeAsyncClient)

    result = await provider.structured_output(
        messages=[LLMMessage(role="user", content="Return JSON only.")],
        schema=schema,
        model="mistralai/ministral-3-3b",
    )

    assert len(requests) == 1
    assert requests[0]["url"].endswith("/v1/chat/completions")
    assert requests[0]["json"]["response_format"]["type"] == "json_schema"
    assert requests[0]["json"]["response_format"]["json_schema"]["schema"] == schema
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_ollama_structured_output_uses_native_format_schema(monkeypatch):
    provider = OllamaProvider(base_url="http://localhost:11434", api_key=None)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    responses = [_FakeResponse(200, {"message": {"content": '{"ok": true}'}})]
    requests = []

    async def fake_retry(coro_factory, max_retries=3, base_delay=1.0):
        return await coro_factory()

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            requests.append({"url": url, "headers": headers, "json": json})
            return responses.pop(0)

    monkeypatch.setattr(llm_provider, "_retry_with_backoff", fake_retry)
    monkeypatch.setattr(llm_provider.httpx, "AsyncClient", FakeAsyncClient)

    result = await provider.structured_output(
        messages=[LLMMessage(role="user", content="Return JSON only.")],
        schema=schema,
        model="llama3.2:latest",
    )

    assert len(requests) == 1
    assert requests[0]["url"].endswith("/api/chat")
    assert requests[0]["json"]["format"] == schema
    assert "response_format" not in requests[0]["json"]
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_deepseek_structured_output_uses_json_object_response_format(monkeypatch):
    provider = DeepSeekProvider(api_key="test-key")
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    responses = [_FakeResponse(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})]
    requests = []

    async def fake_retry(coro_factory, max_retries=3, base_delay=1.0):
        return await coro_factory()

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            requests.append({"url": url, "headers": headers, "json": json})
            return responses.pop(0)

    monkeypatch.setattr(llm_provider, "_retry_with_backoff", fake_retry)
    monkeypatch.setattr(llm_provider.httpx, "AsyncClient", FakeAsyncClient)

    result = await provider.structured_output(
        messages=[LLMMessage(role="user", content="Return JSON only.")],
        schema=schema,
        model="deepseek-chat",
    )

    assert len(requests) == 1
    assert requests[0]["json"]["response_format"]["type"] == "json_object"
    assert "json_schema" not in requests[0]["json"]["response_format"]
    assert result == {"ok": True}


def test_zhipu_provider_preserves_v4_base_url():
    provider = ZhipuProvider(api_key=FAKE_PROVIDER_KEY)

    assert provider.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert provider._delegate.base_url == "https://open.bigmodel.cn/api/paas/v4"

    custom = ZhipuProvider(
        api_key=FAKE_PROVIDER_KEY,
        base_url="https://api.z.ai/api/paas/v4/",
    )
    assert custom._delegate.base_url == "https://api.z.ai/api/paas/v4"


@pytest.mark.asyncio
async def test_zhipu_structured_output_uses_json_object_response_format(monkeypatch):
    provider = ZhipuProvider(api_key=FAKE_PROVIDER_KEY)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    responses = [_FakeResponse(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})]
    requests = []

    async def fake_retry(coro_factory, max_retries=3, base_delay=1.0):
        return await coro_factory()

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            requests.append({"url": url, "headers": headers, "json": json})
            return responses.pop(0)

    monkeypatch.setattr(llm_provider, "_retry_with_backoff", fake_retry)
    monkeypatch.setattr(llm_provider.httpx, "AsyncClient", FakeAsyncClient)

    result = await provider.structured_output(
        messages=[LLMMessage(role="user", content="Return JSON only.")],
        schema=schema,
        model="glm-5.3",
    )

    assert len(requests) == 1
    assert requests[0]["url"] == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert requests[0]["json"]["model"] == "glm-5.3"
    assert requests[0]["json"]["response_format"]["type"] == "json_object"
    assert "json_schema" not in requests[0]["json"]["response_format"]
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_zhipu_chat_dials_down_thinking_without_tools(monkeypatch):
    provider = ZhipuProvider(api_key=FAKE_PROVIDER_KEY)
    responses = [
        _FakeResponse(200, {"choices": [{"message": {"content": '{"ok": true}'}}]}),
        _FakeResponse(200, {"choices": [{"message": {"content": '{"ok": true}'}}]}),
        _FakeResponse(200, {"choices": [{"message": {"content": '{"ok": true}'}}]}),
        _FakeResponse(200, {"choices": [{"message": {"content": '{"ok": true}'}}]}),
    ]
    requests = []

    async def fake_retry(coro_factory, max_retries=3, base_delay=1.0):
        return await coro_factory()

    class FakeAsyncClient:
        is_closed = False

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            requests.append({"url": url, "headers": headers, "json": json})
            return responses.pop(0)

    monkeypatch.setattr(llm_provider, "_retry_with_backoff", fake_retry)
    monkeypatch.setattr(llm_provider.httpx, "AsyncClient", FakeAsyncClient)

    await provider.chat(
        messages=[LLMMessage(role="user", content="draft")],
        model="glm-5.3",
    )
    await provider.chat(
        messages=[LLMMessage(role="user", content="draft")],
        model="glm-4.6",
    )
    await provider.chat(
        messages=[LLMMessage(role="user", content="draft")],
        model="glm-5.3",
        tools=[ToolDefinition(name="lookup", description="look up", parameters={"type": "object", "properties": {}})],
    )
    await provider.chat(
        messages=[LLMMessage(role="user", content="draft")],
        model="glm-4.6",
        tools=[ToolDefinition(name="lookup", description="look up", parameters={"type": "object", "properties": {}})],
    )

    # GLM-5 models are forced-thinking; effort low instead of a switch —
    # for tool-free AND tool-calling requests alike.
    assert requests[0]["json"]["reasoning_effort"] == "low"
    assert "thinking" not in requests[0]["json"]
    assert requests[2]["json"]["reasoning_effort"] == "low"
    assert "thinking" not in requests[2]["json"]
    # GLM-4.5/4.6 accept the classic switch when tool-free; tool calling
    # there requires thinking enabled, so nothing is injected.
    assert requests[1]["json"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in requests[1]["json"]
    assert "thinking" not in requests[3]["json"]
    assert "reasoning_effort" not in requests[3]["json"]
