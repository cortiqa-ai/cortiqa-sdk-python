"""Unit tests for Chat Completions parsing and validation."""

import httpx
from cortiqa import Cortiqa
from cortiqa.types import ChatCompletion


def test_chat_completion_model_validate():
    raw_response = {
        "id": "chatcmpl-test1234",
        "object": "chat.completion",
        "created": 1726900000,
        "model": "falin-01",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I am Falin, an AI by Cortiqa.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 10,
            "total_tokens": 22,
        },
    }

    completion = ChatCompletion.model_validate(raw_response)
    assert completion.id == "chatcmpl-test1234"
    assert completion.model == "falin-01"
    assert len(completion.choices) == 1
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content == "Hello! I am Falin, an AI by Cortiqa."
    assert completion.usage.total_tokens == 22


def test_chat_completion_mock_http():
    def custom_transport(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer sk-cortiqa-dummy"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 1726900000,
                "model": "falin-01",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Mocked response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(custom_transport))
    client = Cortiqa(api_key="sk-cortiqa-dummy", http_client=mock_client)

    response = client.chat.completions.create(
        model="falin-01",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert response.id == "chatcmpl-mock"
    assert response.choices[0].message.content == "Mocked response"
    client.close()
