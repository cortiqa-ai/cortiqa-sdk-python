"""Cortiqa AI Python SDK.

Official client library for Cortiqa AI platform and Falin Foundation Models.
"""

from cortiqa.version import __version__
from cortiqa.client import Cortiqa, AsyncCortiqa
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
from cortiqa.types.chat import (
    ChatMessage,
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    Usage,
    Tool,
    FunctionDefinition,
    ToolCall,
    FunctionCall,
)
from cortiqa.types.models import ModelInfo, ModelListResponse

__all__ = [
    "__version__",
    "Cortiqa",
    "AsyncCortiqa",
    "CortiqaError",
    "APIError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "ChatMessage",
    "ChatCompletion",
    "ChatCompletionChoice",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkDelta",
    "Usage",
    "Tool",
    "FunctionDefinition",
    "ToolCall",
    "FunctionCall",
    "ModelInfo",
    "ModelListResponse",
]
