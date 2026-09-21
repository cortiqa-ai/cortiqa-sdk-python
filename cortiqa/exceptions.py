"""Custom exceptions for Cortiqa Python SDK."""

from typing import Optional, Any


class CortiqaError(Exception):
    """Base exception class for all Cortiqa errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class APIError(CortiqaError):
    """Exception raised when an API request fails with an error response."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        body: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(status_code={self.status_code}, message={self.message!r})"


class AuthenticationError(APIError):
    """Exception raised for 401 Unauthorized errors (invalid or missing API key)."""


class PermissionDeniedError(APIError):
    """Exception raised for 403 Forbidden errors."""


class NotFoundError(APIError):
    """Exception raised for 404 Not Found errors."""


class RateLimitError(APIError):
    """Exception raised for 429 Too Many Requests errors."""


class InternalServerError(APIError):
    """Exception raised for 500+ Internal Server errors."""


class APIConnectionError(CortiqaError):
    """Exception raised when failing to connect to the Cortiqa API."""


class APITimeoutError(APIConnectionError):
    """Exception raised when a request to Cortiqa API times out."""
