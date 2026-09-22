"""Core client implementations for Cortiqa AI SDK."""

import os
import time
import asyncio
from typing import Optional, Dict, Any, Union
import httpx

from cortiqa.exceptions import (
    CortiqaError,
    APIError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    APIConnectionError,
    APITimeoutError,
)
from cortiqa.resources.chat import (
    ChatResource,
    AsyncChatResource,
    MessagesResource,
    AsyncMessagesResource,
)
from cortiqa.resources.models import ModelsResource, AsyncModelsResource
from cortiqa.version import __version__

DEFAULT_BASE_URL = "https://api.cortiqa.co"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MODEL = "openai/gpt-oss-120b"


def _extract_error_details(response: httpx.Response) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[Any]]:
    """Extract (message, param, code, error_type, body) from error response."""
    status_code = response.status_code
    param: Optional[str] = None
    code: Optional[str] = None
    error_type: Optional[str] = None
    body: Optional[Any] = None

    try:
        body = response.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                message = err.get("message")
                param = err.get("param")
                code = err.get("code")
                error_type = err.get("type")
            elif isinstance(err, str):
                message = err
            else:
                message = None

            # Handle FastAPI/Pydantic validation error lists
            if not message and isinstance(body.get("detail"), list) and body["detail"]:
                first_err = body["detail"][0]
                if isinstance(first_err, dict):
                    loc = first_err.get("loc", [])
                    if loc:
                        param = str(loc[-1])
                    msg = first_err.get("msg")
                    error_type = first_err.get("type")
                    if param and msg:
                        message = f"Parameter '{param}': {msg}"
                    else:
                        message = msg

            if not message:
                message = body.get("message") or response.text
    except Exception:
        body = None
        message = response.text or f"HTTP {status_code} Error"

    if not message:
        message = f"HTTP {status_code} Error"

    if param and f"'{param}'" not in message and f"[{param}]" not in message:
        message = f"[{param}] {message}"

    return message, param, code, error_type, body


class Cortiqa:
    """Synchronous client for Cortiqa AI APIs.

    Supports both OpenAI style (`client.chat.completions.create(...)`)
    and Anthropic style (`client.messages.create(...)`), plus one-liner `client.prompt(...)`.

    Example:
        ```python
        from cortiqa import Cortiqa

        client = Cortiqa(api_key="sk-cortiqa-...")

        # Option A: One-liner prompt
        answer = client.prompt("Explain quantum computing in 1 sentence.")
        print(answer)

        # Option B: OpenAI style
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello Cortiqa!"}]
        )
        print(completion.choices[0].message.content)
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_model: str = DEFAULT_MODEL,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("CORTIQA_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "No API key provided. Pass `api_key` or set the `CORTIQA_API_KEY` environment variable."
            )

        raw_base_url = base_url or os.environ.get("CORTIQA_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = raw_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_model = default_model

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"cortiqa-python/{__version__}",
        }

        self._http_client = http_client or httpx.Client(
            timeout=self.timeout,
            headers=self._headers,
        )

        # Resources
        self.chat = ChatResource(self)
        self.messages = MessagesResource(self)
        self.models = ModelsResource(self)

    def prompt(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **extra_params: Any,
    ) -> str:
        """Convenience shortcut to send a prompt and get the text response directly.

        Example:
            answer = client.prompt("Explain quantum computing in 1 sentence.")
            print(answer)
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        completion = self.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_params,
        )
        return completion.content

    def _handle_response_status(self, response: httpx.Response) -> None:
        """Inspect HTTP response and raise appropriate typed exception on error."""
        if response.is_success:
            return

        status_code = response.status_code
        message, param, code, error_type, body = _extract_error_details(response)

        kwargs = {
            "status_code": status_code,
            "body": body,
            "param": param,
            "code": code,
            "error_type": error_type,
        }

        if status_code == 400:
            raise BadRequestError(message, **kwargs)
        if status_code == 401:
            raise AuthenticationError(message, **kwargs)
        if status_code == 403:
            raise PermissionDeniedError(message, **kwargs)
        if status_code == 404:
            raise NotFoundError(message, **kwargs)
        if status_code == 422:
            raise UnprocessableEntityError(message, **kwargs)
        if status_code == 429:
            raise RateLimitError(message, **kwargs)
        if status_code >= 500:
            raise InternalServerError(message, **kwargs)

        raise APIError(message, **kwargs)

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        retries = 0
        while True:
            try:
                response = self._http_client.get(url, params=params, headers=self._headers)
                self._handle_response_status(response)
                return response.json()
            except (RateLimitError, InternalServerError, httpx.RequestError) as exc:
                if retries < self.max_retries:
                    retries += 1
                    time.sleep(0.5 * (2 ** (retries - 1)))
                    continue
                if isinstance(exc, (RateLimitError, InternalServerError)):
                    raise
                if isinstance(exc, httpx.TimeoutException):
                    raise APITimeoutError(f"Request to {url} timed out.") from exc
                raise APIConnectionError(f"Failed to connect to Cortiqa API: {exc}") from exc

    def _post(self, path: str, json_data: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        retries = 0
        while True:
            try:
                response = self._http_client.post(url, json=json_data, headers=self._headers)
                self._handle_response_status(response)
                return response.json()
            except (RateLimitError, InternalServerError, httpx.RequestError) as exc:
                if retries < self.max_retries:
                    retries += 1
                    time.sleep(0.5 * (2 ** (retries - 1)))
                    continue
                if isinstance(exc, (RateLimitError, InternalServerError)):
                    raise
                if isinstance(exc, httpx.TimeoutException):
                    raise APITimeoutError(f"Request to {url} timed out.") from exc
                raise APIConnectionError(f"Failed to connect to Cortiqa API: {exc}") from exc

    def close(self) -> None:
        """Close internal HTTP client connections."""
        self._http_client.close()

    def __enter__(self) -> "Cortiqa":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class AsyncCortiqa:
    """Asynchronous client for Cortiqa AI APIs.

    Example:
        ```python
        import asyncio
        from cortiqa import AsyncCortiqa

        async def main():
            async with AsyncCortiqa(api_key="sk-cortiqa-...") as client:
                message = await client.messages.create(
                    model="falin-01",
                    max_tokens=256,
                    messages=[{"role": "user", "content": "Explain quantum computing in 1 line."}]
                )
                print(message.content)

        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_model: str = DEFAULT_MODEL,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("CORTIQA_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "No API key provided. Pass `api_key` or set the `CORTIQA_API_KEY` environment variable."
            )

        raw_base_url = base_url or os.environ.get("CORTIQA_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = raw_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_model = default_model

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"cortiqa-python/{__version__}",
        }

        self._http_client = http_client or httpx.AsyncClient(
            timeout=self.timeout,
            headers=self._headers,
        )

        # Resources
        self.chat = AsyncChatResource(self)
        self.messages = AsyncMessagesResource(self)
        self.models = AsyncModelsResource(self)

    async def prompt(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **extra_params: Any,
    ) -> str:
        """Asynchronously send a prompt and return the text response directly.

        Example:
            answer = await client.prompt("Explain quantum computing in 1 sentence.")
            print(answer)
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        completion = await self.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_params,
        )
        return completion.content

    def _handle_response_status(self, response: httpx.Response) -> None:
        """Inspect HTTP response and raise appropriate typed exception on error."""
        if response.is_success:
            return

        status_code = response.status_code
        message, param, code, error_type, body = _extract_error_details(response)

        kwargs = {
            "status_code": status_code,
            "body": body,
            "param": param,
            "code": code,
            "error_type": error_type,
        }

        if status_code == 400:
            raise BadRequestError(message, **kwargs)
        if status_code == 401:
            raise AuthenticationError(message, **kwargs)
        if status_code == 403:
            raise PermissionDeniedError(message, **kwargs)
        if status_code == 404:
            raise NotFoundError(message, **kwargs)
        if status_code == 422:
            raise UnprocessableEntityError(message, **kwargs)
        if status_code == 429:
            raise RateLimitError(message, **kwargs)
        if status_code >= 500:
            raise InternalServerError(message, **kwargs)

        raise APIError(message, **kwargs)

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        retries = 0
        while True:
            try:
                response = await self._http_client.get(url, params=params, headers=self._headers)
                self._handle_response_status(response)
                return response.json()
            except (RateLimitError, InternalServerError, httpx.RequestError) as exc:
                if retries < self.max_retries:
                    retries += 1
                    await asyncio.sleep(0.5 * (2 ** (retries - 1)))
                    continue
                if isinstance(exc, (RateLimitError, InternalServerError)):
                    raise
                if isinstance(exc, httpx.TimeoutException):
                    raise APITimeoutError(f"Request to {url} timed out.") from exc
                raise APIConnectionError(f"Failed to connect to Cortiqa API: {exc}") from exc

    async def _post(self, path: str, json_data: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        retries = 0
        while True:
            try:
                response = await self._http_client.post(url, json=json_data, headers=self._headers)
                self._handle_response_status(response)
                return response.json()
            except (RateLimitError, InternalServerError, httpx.RequestError) as exc:
                if retries < self.max_retries:
                    retries += 1
                    await asyncio.sleep(0.5 * (2 ** (retries - 1)))
                    continue
                if isinstance(exc, (RateLimitError, InternalServerError)):
                    raise
                if isinstance(exc, httpx.TimeoutException):
                    raise APITimeoutError(f"Request to {url} timed out.") from exc
                raise APIConnectionError(f"Failed to connect to Cortiqa API: {exc}") from exc

    async def aclose(self) -> None:
        """Close internal async HTTP client connections."""
        await self._http_client.aclose()

    async def __aenter__(self) -> "AsyncCortiqa":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()
