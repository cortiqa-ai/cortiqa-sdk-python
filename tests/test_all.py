"""Comprehensive unittest suite for Cortiqa SDK."""

import sys
import unittest
import httpx

sys.path.insert(0, ".")
from cortiqa import Cortiqa, AsyncCortiqa
from cortiqa.exceptions import AuthenticationError
from cortiqa.types import ChatCompletion, ToolCall, FunctionCall


class TestCortiqaClient(unittest.TestCase):
    def test_missing_api_key(self):
        with self.assertRaises(AuthenticationError):
            Cortiqa(api_key="")

    def test_client_custom_options(self):
        client = Cortiqa(
            api_key="sk-cortiqa-test-12345",
            base_url="https://custom.api.cortiqa.co",
            timeout=30.0,
            max_retries=3,
        )
        self.assertEqual(client.api_key, "sk-cortiqa-test-12345")
        self.assertEqual(client.base_url, "https://custom.api.cortiqa.co")
        self.assertEqual(client.timeout, 30.0)
        self.assertEqual(client.max_retries, 3)
        self.assertEqual(client._headers["Authorization"], "Bearer sk-cortiqa-test-12345")
        self.assertIn("cortiqa-python", client._headers["User-Agent"])
        client.close()

    def test_chat_completion_model_validate(self):
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
        self.assertEqual(completion.id, "chatcmpl-test1234")
        self.assertEqual(completion.model, "falin-01")
        self.assertEqual(len(completion.choices), 1)
        self.assertEqual(completion.choices[0].message.role, "assistant")
        self.assertEqual(completion.choices[0].message.content, "Hello! I am Falin, an AI by Cortiqa.")
        # Content property test (Anthropic style)
        self.assertEqual(completion.content, "Hello! I am Falin, an AI by Cortiqa.")
        self.assertEqual(completion.usage.total_tokens, 22)

    def test_tool_calls_parsing(self):
        raw_response = {
            "id": "chatcmpl-tools",
            "object": "chat.completion",
            "created": 1726900000,
            "model": "falin-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "lookup_stock_price",
                                    "arguments": '{"ticker": "TATAMOTORS"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        completion = ChatCompletion.model_validate(raw_response)
        self.assertEqual(completion.choices[0].finish_reason, "tool_calls")
        self.assertIsNotNone(completion.choices[0].message.tool_calls)
        tool_call = completion.choices[0].message.tool_calls[0]
        self.assertEqual(tool_call.id, "call_123")
        self.assertEqual(tool_call.function.name, "lookup_stock_price")

    def test_messages_resource_alias(self):
        """Test client.messages.create alias works exactly like chat.completions.create."""
        def custom_transport(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer sk-cortiqa-dummy")
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-messages",
                    "object": "chat.completion",
                    "created": 1726900000,
                    "model": "falin-01",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Messages resource output"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        mock_client = httpx.Client(transport=httpx.MockTransport(custom_transport))
        client = Cortiqa(api_key="sk-cortiqa-dummy", http_client=mock_client)

        response = client.messages.create(
            model="falin-01",
            messages=[{"role": "user", "content": "Hi"}],
        )

        self.assertEqual(response.id, "chatcmpl-messages")
        self.assertEqual(response.content, "Messages resource output")
        client.close()


if __name__ == "__main__":
    unittest.main()
