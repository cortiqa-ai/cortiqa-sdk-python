# Cortiqa Python SDK API Reference

This document provides a comprehensive reference for all client classes, resources, methods, types, and exceptions in the Cortiqa Python SDK.

---

## Table of Contents

- [Client Initialization](#client-initialization)
  - [`Cortiqa`](#cortiqa)
  - [`AsyncCortiqa`](#asynccortiqa)
- [Resources & Endpoints](#resources--endpoints)
  - [`client.messages.create()`](#clientmessagescreate)
  - [`client.messages.stream()`](#clientmessagesstream)
  - [`client.chat.completions.create()`](#clientchatcompletionscreate)
  - [`client.models.list()`](#clientmodelslist)
- [Data Types & Models](#data-types--models)
  - [`ChatCompletion`](#chatcompletion)
  - [`ChatMessage`](#chatmessage)
  - [`ChatCompletionChunk`](#chatcompletionchunk)
  - [`Tool` & `ToolCall`](#tool--toolcall)
  - [`ModelInfo`](#modelinfo)
- [Exceptions & Error Handling](#exceptions--error-handling)

---

## Client Initialization

### `Cortiqa`

The synchronous client for calling Cortiqa AI models.

```python
from cortiqa import Cortiqa

client = Cortiqa(
    api_key="sk-cortiqa-...",            # Optional if CORTIQA_API_KEY environment variable is set
    base_url="https://api.cortiqa.co",   # Optional custom endpoint (default: https://api.cortiqa.co)
    timeout=60.0,                        # Request timeout in seconds (default: 60.0)
    max_retries=2,                       # Number of retries on 429/5xx (default: 2)
    http_client=None,                    # Optional custom httpx.Client instance
)
```

### `AsyncCortiqa`

The asynchronous client for high-throughput and non-blocking asyncio applications.

```python
from cortiqa import AsyncCortiqa

client = AsyncCortiqa(
    api_key="sk-cortiqa-...",
    base_url="https://api.cortiqa.co",
    timeout=60.0,
    max_retries=2,
    http_client=None,                    # Optional custom httpx.AsyncClient instance
)
```

---

## Resources & Endpoints

### `client.messages.create()`

Creates a text or tool completion using an Anthropic-compatible parameter interface.

#### Parameters

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `model` | `str` | Yes | `"falin-01"` | Target model ID (`falin-01`, `falin-pro`, `falin-vision`, `falin-ultra`) |
| `messages` | `List[Dict[str, Any]]` | Yes | - | List of message dicts with `role` and `content` |
| `max_tokens` | `int` | No | `None` | The maximum number of tokens to generate |
| `temperature` | `float` | No | `0.7` | Sampling temperature (between 0.0 and 2.0) |
| `top_p` | `float` | No | `None` | Nucleus sampling probability threshold |
| `stream` | `bool` | No | `False` | When `True`, returns an iterator yielding `ChatCompletionChunk` |
| `tools` | `List[Dict[str, Any]]` | No | `None` | Tools / functions available for model execution |
| `tool_choice` | `str` or `dict` | No | `None` | Control tool selection (`'auto'`, `'required'`, or specific tool) |

#### Return Value
* When `stream=False`: Returns a [`ChatCompletion`](#chatcompletion) object.
* When `stream=True`: Returns an `Iterator[ChatCompletionChunk]`.

#### Example

```python
message = client.messages.create(
    model="falin-01",
    max_tokens=500,
    messages=[{"role": "user", "content": "Explain machine learning in 10 words."}],
)
print(message.content)
```

---

### `client.messages.stream()`

A context manager that streamlines token streaming and message accumulation.

```python
with client.messages.stream(
    model="falin-01",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a guide on Docker."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

    final_message = stream.get_final_message()
    print(f"\nTotal tokens: {final_message.usage.total_tokens}")
```

---

### `client.chat.completions.create()`

OpenAI-compatible completion endpoint matching standard `client.chat.completions.create`.

```python
completion = client.chat.completions.create(
    model="falin-pro",
    messages=[
        {"role": "system", "content": "You are an expert software engineer."},
        {"role": "user", "content": "Refactor this SQL query for performance."}
    ],
    temperature=0.3,
)
print(completion.choices[0].message.content)
```

---

### `client.models.list()`

List all foundation models available in your account.

#### Returns
`List[ModelInfo]`

```python
models = client.models.list()
for model in models:
    print(f"{model.id} - {model.description} (Free tier: {model.free})")
```

---

## Data Types & Models

### `ChatCompletion`

| Attribute | Type | Description |
|---|---|---|
| `id` | `str` | Unique completion ID (e.g. `chatcmpl-a1b2c3d4`) |
| `object` | `str` | Always `"chat.completion"` |
| `created` | `int` | Unix timestamp of creation |
| `model` | `str` | Model identifier used |
| `choices` | `List[ChatCompletionChoice]` | List of completion choices |
| `usage` | `Optional[Usage]` | Token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`) |
| `content` | `str` | Convenience property returning `choices[0].message.content` |

### `ChatMessage`

| Attribute | Type | Description |
|---|---|---|
| `role` | `str` | Message role: `"system"`, `"user"`, `"assistant"`, or `"tool"` |
| `content` | `Optional[str]` | The text content of the message |
| `tool_calls` | `Optional[List[ToolCall]]` | List of tool calls initiated by the assistant |
| `tool_call_id`| `Optional[str]` | The tool call ID when sending role `"tool"` |

---

## Exceptions & Error Handling

All SDK exceptions inherit from `CortiqaError`.

```
CortiqaError
 ├── APIConnectionError
 │    └── APITimeoutError
 └── APIError
      ├── AuthenticationError    (HTTP 401)
      ├── PermissionDeniedError  (HTTP 403)
      ├── NotFoundError          (HTTP 404)
      ├── RateLimitError         (HTTP 429)
      └── InternalServerError    (HTTP 500+)
```

### Catching Specific Exceptions

```python
from cortiqa import Cortiqa
from cortiqa.exceptions import AuthenticationError, RateLimitError, APIError

client = Cortiqa()

try:
    response = client.messages.create(
        model="falin-01",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except AuthenticationError as e:
    print(f"Auth error: {e.message}")
except RateLimitError as e:
    print(f"Rate limited: {e.message}")
except APIError as e:
    print(f"API failed with code {e.status_code}: {e.message}")
```
