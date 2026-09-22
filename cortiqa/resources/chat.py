"""Chat and Messages resource supporting OpenAI & Anthropic style APIs."""

import json
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator, TYPE_CHECKING
from cortiqa.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatCompletionChoice,
    Usage,
)

if TYPE_CHECKING:
    from cortiqa.client import Cortiqa, AsyncCortiqa


class StreamManager:
    """Convenience context manager for streaming completions (Anthropic-style)."""

    def __init__(self, iterator: Iterator[ChatCompletionChunk]) -> None:
        self._iterator = iterator
        self._collected_tokens: List[str] = []
        self._final_completion: Optional[ChatCompletion] = None
        self._model: str = ""
        self._id: str = ""

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self

    def __next__(self) -> ChatCompletionChunk:
        chunk = next(self._iterator)
        if chunk.id:
            self._id = chunk.id
        if chunk.model:
            self._model = chunk.model
        if chunk.choices and chunk.choices[0].delta.content:
            self._collected_tokens.append(chunk.choices[0].delta.content)
        return chunk

    @property
    def text_stream(self) -> Iterator[str]:
        """Yield text tokens directly without chunk envelope."""
        for chunk in self:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_final_message(self) -> ChatCompletion:
        """Return assembled full completion after streaming finishes."""
        full_text = "".join(self._collected_tokens)
        return ChatCompletion(
            id=self._id or "chatcmpl-streamed",
            object="chat.completion",
            created=0,
            model=self._model or "openai/gpt-oss-120b",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=len(self._collected_tokens), total_tokens=len(self._collected_tokens)),
        )

    def __enter__(self) -> "StreamManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class AsyncStreamManager:
    """Async convenience context manager for streaming completions."""

    def __init__(self, iterator: AsyncIterator[ChatCompletionChunk]) -> None:
        self._iterator = iterator
        self._collected_tokens: List[str] = []
        self._model: str = ""
        self._id: str = ""

    def __aiter__(self) -> "AsyncStreamManager":
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        chunk = await self._iterator.__anext__()
        if chunk.id:
            self._id = chunk.id
        if chunk.model:
            self._model = chunk.model
        if chunk.choices and chunk.choices[0].delta.content:
            self._collected_tokens.append(chunk.choices[0].delta.content)
        return chunk

    async def text_stream(self) -> AsyncIterator[str]:
        """Async yield text tokens directly."""
        async for chunk in self:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_final_message(self) -> ChatCompletion:
        """Return assembled full completion after streaming finishes."""
        full_text = "".join(self._collected_tokens)
        return ChatCompletion(
            id=self._id or "chatcmpl-streamed",
            object="chat.completion",
            created=0,
            model=self._model or "openai/gpt-oss-120b",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=len(self._collected_tokens), total_tokens=len(self._collected_tokens)),
        )

    async def __aenter__(self) -> "AsyncStreamManager":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


def _normalize_messages(messages: List[Union[Dict[str, Any], ChatMessage]]) -> List[Dict[str, Any]]:
    """Ensure all messages are serializable dicts (handling Pydantic ChatMessage objects)."""
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            dumped = msg.model_dump(exclude_none=True)
            normalized.append(dumped)
        elif isinstance(msg, dict):
            normalized.append(msg)
        else:
            normalized.append(msg)
    return normalized


def _normalize_tools(tools: Optional[List[Union[Dict[str, Any], Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Ensure tools are in the expected OpenAI schema: {"type": "function", "function": {...}}."""
    if tools is None:
        return None
    normalized: List[Dict[str, Any]] = []
    for tool in tools:
        if hasattr(tool, "model_dump"):
            t_dict = tool.model_dump(exclude_none=True)
        elif isinstance(tool, dict):
            t_dict = dict(tool)
        else:
            t_dict = tool

        if isinstance(t_dict, dict):
            if t_dict.get("type") == "function" and "function" in t_dict:
                normalized.append(t_dict)
            elif "function" in t_dict and "type" not in t_dict:
                normalized.append({"type": "function", "function": t_dict["function"]})
            elif "name" in t_dict and "function" not in t_dict:
                desc = t_dict.get("description")
                params = t_dict.get("parameters", {})
                func_dict: Dict[str, Any] = {"name": t_dict["name"]}
                if desc is not None:
                    func_dict["description"] = desc
                if params is not None:
                    func_dict["parameters"] = params
                normalized.append({
                    "type": "function",
                    "function": func_dict,
                })
            else:
                normalized.append(t_dict)
        else:
            normalized.append(t_dict)
    return normalized


class CompletionsResource:
    """Synchronous chat completions manager."""

    def __init__(self, client: "Cortiqa") -> None:
        self._client = client

    def create(
        self,
        *,
        model: Optional[str] = None,
        messages: List[Union[Dict[str, Any], ChatMessage]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        tools: Optional[List[Union[Dict[str, Any], Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **extra_params: Any,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Create a chat or message completion.

        Args:
            model: Model identifier (defaults to client default or 'openai/gpt-oss-120b')
            messages: List of message dictionaries or ChatMessage objects
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream back partial tokens
            tools: Optional tool/function specifications
            tool_choice: Control tool usage ('auto', 'required', or specific tool)
            **extra_params: Additional vendor parameters

        Returns:
            ChatCompletion object if stream=False, or an iterator of ChatCompletionChunk if stream=True.
        """
        effective_model = model or getattr(self._client, "default_model", "openai/gpt-oss-120b")
        norm_messages = _normalize_messages(messages)
        norm_tools = _normalize_tools(tools)

        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": norm_messages,
            **extra_params,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if norm_tools is not None:
            payload["tools"] = norm_tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if stream:
            payload["stream"] = True

        if stream:
            return self._stream_completions(payload)

        response = self._client._post("/v1/chat/completions", json_data=payload)
        return ChatCompletion.model_validate(response)

    def stream(
        self,
        *,
        model: Optional[str] = None,
        messages: List[Union[Dict[str, Any], ChatMessage]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Union[Dict[str, Any], Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **extra_params: Any,
    ) -> StreamManager:
        """Stream chat completions using a high-level context manager (Anthropic style)."""
        it = self.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
            **extra_params,
        )
        return StreamManager(it)  # type: ignore

    def _stream_completions(self, payload: Dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        """Stream chat completions chunk by chunk."""
        with self._client._http_client.stream(
            "POST",
            f"{self._client.base_url}/v1/chat/completions",
            json=payload,
            headers=self._client._headers,
            timeout=self._client.timeout,
        ) as response:
            self._client._handle_response_status(response)
            for line in response.iter_lines():
                if not line or not line.strip():
                    continue
                line_str = line.strip()
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        yield ChatCompletionChunk.model_validate(chunk_data)
                    except json.JSONDecodeError:
                        continue


class ChatResource:
    """Top-level chat resource providing completions."""

    def __init__(self, client: "Cortiqa") -> None:
        self.completions = CompletionsResource(client)


class MessagesResource:
    """Anthropic-style messages resource alias."""

    def __init__(self, client: "Cortiqa") -> None:
        self._completions = CompletionsResource(client)

    def create(self, **kwargs: Any) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Anthropic-style message creation."""
        return self._completions.create(**kwargs)

    def stream(self, **kwargs: Any) -> StreamManager:
        """Anthropic-style message streaming context manager."""
        return self._completions.stream(**kwargs)


class AsyncCompletionsResource:
    """Asynchronous chat completions manager."""

    def __init__(self, client: "AsyncCortiqa") -> None:
        self._client = client

    async def create(
        self,
        *,
        model: Optional[str] = None,
        messages: List[Union[Dict[str, Any], ChatMessage]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        tools: Optional[List[Union[Dict[str, Any], Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **extra_params: Any,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """Create an asynchronous chat completion."""
        effective_model = model or getattr(self._client, "default_model", "openai/gpt-oss-120b")
        norm_messages = _normalize_messages(messages)
        norm_tools = _normalize_tools(tools)

        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": norm_messages,
            **extra_params,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if norm_tools is not None:
            payload["tools"] = norm_tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if stream:
            payload["stream"] = True

        if stream:
            return self._stream_completions(payload)

        response = await self._client._post("/v1/chat/completions", json_data=payload)
        return ChatCompletion.model_validate(response)

    def stream(
        self,
        *,
        model: Optional[str] = None,
        messages: List[Union[Dict[str, Any], ChatMessage]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Union[Dict[str, Any], Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **extra_params: Any,
    ) -> AsyncStreamManager:
        """Stream chat completions using an async context manager."""
        async def gen() -> AsyncIterator[ChatCompletionChunk]:
            it = await self.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True,
                tools=tools,
                tool_choice=tool_choice,
                **extra_params,
            )
            async for chunk in it:  # type: ignore
                yield chunk

        return AsyncStreamManager(gen())

    async def _stream_completions(self, payload: Dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        """Stream chat completions chunk by chunk asynchronously."""
        async with self._client._http_client.stream(
            "POST",
            f"{self._client.base_url}/v1/chat/completions",
            json=payload,
            headers=self._client._headers,
            timeout=self._client.timeout,
        ) as response:
            self._client._handle_response_status(response)
            async for line in response.aiter_lines():
                if not line or not line.strip():
                    continue
                line_str = line.strip()
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        yield ChatCompletionChunk.model_validate(chunk_data)
                    except json.JSONDecodeError:
                        continue


class AsyncChatResource:
    """Top-level async chat resource providing completions."""

    def __init__(self, client: "AsyncCortiqa") -> None:
        self.completions = AsyncCompletionsResource(client)


class AsyncMessagesResource:
    """Anthropic-style async messages resource alias."""

    def __init__(self, client: "AsyncCortiqa") -> None:
        self._completions = AsyncCompletionsResource(client)

    async def create(self, **kwargs: Any) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """Anthropic-style async message creation."""
        return await self._completions.create(**kwargs)

    def stream(self, **kwargs: Any) -> AsyncStreamManager:
        """Anthropic-style async message streaming context manager."""
        return self._completions.stream(**kwargs)
