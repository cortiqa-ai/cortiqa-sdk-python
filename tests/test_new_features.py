"""Unit tests for new SDK features:
1. Default model & client-level default
2. Reasoning fields and thought property
3. Tool calling payload normalization and ChatMessage serialization
4. Granular error handling (param, BadRequestError, UnprocessableEntityError)
5. One-liner client.prompt(...) shortcut
"""

import pytest
import httpx
from cortiqa import (
    Cortiqa,
    AsyncCortiqa,
    ChatMessage,
    Tool,
    FunctionDefinition,
    APIError,
    BadRequestError,
    UnprocessableEntityError,
)
from cortiqa.types import ChatCompletion


def test_default_model_and_client_override():
    # 1. Default should be openai/gpt-oss-120b
    client = Cortiqa(api_key="sk-test")
    assert client.default_model == "openai/gpt-oss-120b"

    # 2. Client-level default override
    client_custom = Cortiqa(api_key="sk-test", default_model="custom/model-v1")
    assert client_custom.default_model == "custom/model-v1"


def test_reasoning_fields():
    # ChatMessage with reasoning
    msg = ChatMessage(role="assistant", content="The answer is 42.", reasoning="First consider the question...")
    assert msg.reasoning == "First consider the question..."
    assert msg.thought == "First consider the question..."

    # ChatMessage with reasoning_content alias
    msg2 = ChatMessage(role="assistant", content="Hello", reasoning_content="Thinking process...")
    assert msg2.reasoning_content == "Thinking process..."
    assert msg2.thought == "Thinking process..."

    # ChatCompletion.reasoning shortcut
    completion = ChatCompletion.model_validate({
        "id": "cmpl-reasoning-1",
        "object": "chat.completion",
        "created": 123456,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Calculated value: 100",
                    "reasoning": "Step 1: 50 + 50 = 100",
                },
                "finish_reason": "stop",
            }
        ],
    })
    assert completion.content == "Calculated value: 100"
    assert completion.reasoning == "Step 1: 50 + 50 = 100"


def test_prompt_helper_and_model_fallback():
    captured_payload = {}

    def custom_transport(request: httpx.Request) -> httpx.Response:
        import json
        nonlocal captured_payload
        captured_payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "cmpl-prompt-1",
                "object": "chat.completion",
                "created": 123456,
                "model": captured_payload.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Fast answer!"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(custom_transport))
    client = Cortiqa(api_key="sk-test", http_client=mock_client)

    # Calling prompt() without model should use default "openai/gpt-oss-120b"
    res = client.prompt("What is 2+2?")
    assert res == "Fast answer!"
    assert captured_payload["model"] == "openai/gpt-oss-120b"
    assert captured_payload["messages"] == [{"role": "user", "content": "What is 2+2?"}]

    # Calling prompt() with system prompt
    res2 = client.prompt("Explain quantum", system="Be concise.")
    assert res2 == "Fast answer!"
    assert len(captured_payload["messages"]) == 2
    assert captured_payload["messages"][0] == {"role": "system", "content": "Be concise."}
    assert captured_payload["messages"][1] == {"role": "user", "content": "Explain quantum"}


def test_tool_and_message_normalization():
    captured_payload = {}

    def custom_transport(request: httpx.Request) -> httpx.Response:
        import json
        nonlocal captured_payload
        captured_payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "cmpl-tool-1",
                "object": "chat.completion",
                "created": 123456,
                "model": captured_payload.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Checking weather..."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(custom_transport))
    client = Cortiqa(api_key="sk-test", http_client=mock_client)

    # 1. Pass ChatMessage objects in messages (tests Pydantic serialization)
    # 2. Pass Tool object and shorthand tool dict
    tool_pydantic = Tool(
        type="function",
        function=FunctionDefinition(
            name="get_current_weather",
            description="Get current temperature",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
        ),
    )
    raw_tool = {
        "name": "calculate_tax",
        "description": "Calculate tax rate",
        "parameters": {"type": "object", "properties": {"amount": {"type": "number"}}},
    }

    client.chat.completions.create(
        messages=[
            ChatMessage(role="user", content="What is the weather in Tokyo?"),
        ],
        tools=[tool_pydantic, raw_tool],
    )

    # Messages should be serializable dicts
    assert isinstance(captured_payload["messages"][0], dict)
    assert captured_payload["messages"][0]["role"] == "user"
    assert captured_payload["messages"][0]["content"] == "What is Tokyo weather?" or "Tokyo" in captured_payload["messages"][0]["content"]

    # Tools should all be normalized to {"type": "function", "function": {...}}
    tools = captured_payload["tools"]
    assert len(tools) == 2
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_current_weather"
    assert tools[1]["type"] == "function"
    assert tools[1]["function"]["name"] == "calculate_tax"


def test_error_details_parsing():
    # 400 Bad Request with param
    def bad_request_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Invalid temperature value, must be between 0.0 and 2.0",
                    "param": "temperature",
                    "code": "invalid_parameter",
                    "type": "invalid_request_error",
                }
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(bad_request_transport))
    client = Cortiqa(api_key="sk-test", http_client=mock_client)

    with pytest.raises(BadRequestError) as exc_info:
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            temperature=5.0,
        )

    err = exc_info.value
    assert err.status_code == 400
    assert err.param == "temperature"
    assert err.code == "invalid_parameter"
    assert err.error_type == "invalid_request_error"
    assert "[temperature]" in str(err)

    # 422 Unprocessable Entity with FastAPI detail array
    def unprocessable_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "messages"],
                        "msg": "field required",
                        "type": "value_error.missing",
                    }
                ]
            },
        )

    mock_client2 = httpx.Client(transport=httpx.MockTransport(unprocessable_transport))
    client2 = Cortiqa(api_key="sk-test", http_client=mock_client2)

    with pytest.raises(UnprocessableEntityError) as exc_info2:
        client2.chat.completions.create(messages=[])

    err2 = exc_info2.value
    assert err2.status_code == 422
    assert err2.param == "messages"
    assert "messages" in str(err2)
