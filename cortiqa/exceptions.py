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
        *,
        param: Optional[str] = None,
        code: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.param = param
        self.code = code
        self.error_type = error_type

    def __repr__(self) -> str:
        param_str = f", param={self.param!r}" if self.param else ""
        return f"{self.__class__.__name__}(status_code={self.status_code}, message={self.message!r}{param_str})"


class BadRequestError(APIError):
    """Exception raised for 400 Bad Request errors (e.g. invalid parameter)."""


class AuthenticationError(APIError):
    """Exception raised for 401 Unauthorized errors (invalid or missing API key)."""


class PermissionDeniedError(APIError):
    """Exception raised for 403 Forbidden errors."""


class NotFoundError(APIError):
    """Exception raised for 404 Not Found errors."""


class UnprocessableEntityError(APIError):
    """Exception raised for 422 Unprocessable Entity errors (e.g. validation failure)."""


class RateLimitError(APIError):
    """Exception raised for 429 Too Many Requests errors."""


class InternalServerError(APIError):
    """Exception raised for 500+ Internal Server errors."""


class APIConnectionError(CortiqaError):
    """Exception raised when failing to connect to the Cortiqa API."""


class APITimeoutError(APIConnectionError):
    """Exception raised when a request to Cortiqa API times out."""
