"""Core client implementations for Cortiqa AI SDK."""

import os
import time
import asyncio
from typing import Optional, Dict, Any, Union
import httpx

from cortiqa.exceptions import (
    CortiqaError,
    APIError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
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


class Cortiqa:
    """Synchronous client for Cortiqa AI APIs.

    Supports both OpenAI style (`client.chat.completions.create(...)`)
    and Anthropic style (`client.messages.create(...)`).

    Example:
        ```python
        from cortiqa import Cortiqa

        client = Cortiqa(api_key="sk-cortiqa-...")

        # Option A: Anthropic style
        message = client.messages.create(
            model="falin-01",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello Cortiqa!"}]
        )
        print(message.content)

        # Option B: OpenAI style
        completion = client.chat.completions.create(
            model="falin-01",
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

    def _handle_response_status(self, response: httpx.Response) -> None:
        """Inspect HTTP response and raise appropriate typed exception on error."""
        if response.is_success:
            return

        status_code = response.status_code
        try:
            body = response.json()
            message = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("error")
            if not message:
                message = body.get("message") or response.text
        except Exception:
            body = None
            message = response.text or f"HTTP {status_code} Error"

        if status_code == 401:
            raise AuthenticationError(message, status_code=status_code, body=body)
        if status_code == 403:
            raise PermissionDeniedError(message, status_code=status_code, body=body)
        if status_code == 404:
            raise NotFoundError(message, status_code=status_code, body=body)
        if status_code == 429:
            raise RateLimitError(message, status_code=status_code, body=body)
        if status_code >= 500:
            raise InternalServerError(message, status_code=status_code, body=body)

        raise APIError(message, status_code=status_code, body=body)

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

    def _handle_response_status(self, response: httpx.Response) -> None:
        """Inspect HTTP response and raise appropriate typed exception on error."""
        if response.is_success:
            return

        status_code = response.status_code
        try:
            body = response.json()
            message = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("error")
            if not message:
                message = body.get("message") or response.text
        except Exception:
            body = None
            message = response.text or f"HTTP {status_code} Error"

        if status_code == 401:
            raise AuthenticationError(message, status_code=status_code, body=body)
        if status_code == 403:
            raise PermissionDeniedError(message, status_code=status_code, body=body)
        if status_code == 404:
            raise NotFoundError(message, status_code=status_code, body=body)
        if status_code == 429:
            raise RateLimitError(message, status_code=status_code, body=body)
        if status_code >= 500:
            raise InternalServerError(message, status_code=status_code, body=body)

        raise APIError(message, status_code=status_code, body=body)

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
