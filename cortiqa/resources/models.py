"""Models resource for listing and inspecting Cortiqa models."""

from typing import List, TYPE_CHECKING
from cortiqa.types.models import ModelInfo

if TYPE_CHECKING:
    from cortiqa.client import Cortiqa, AsyncCortiqa


class ModelsResource:
    """Synchronous models manager."""

    def __init__(self, client: "Cortiqa") -> None:
        self._client = client

    def list(self) -> List[ModelInfo]:
        """List all available Cortiqa AI models.

        Returns:
            List of ModelInfo objects describing model id, name, description, and tier.
        """
        response = self._client._get("/api/v1/ai/models")
        data = response.get("data", []) if isinstance(response, dict) else response
        return [ModelInfo.model_validate(m) for m in data]


class AsyncModelsResource:
    """Asynchronous models manager."""

    def __init__(self, client: "AsyncCortiqa") -> None:
        self._client = client

    async def list(self) -> List[ModelInfo]:
        """List all available Cortiqa AI models asynchronously.

        Returns:
            List of ModelInfo objects describing model id, name, description, and tier.
        """
        response = await self._client._get("/api/v1/ai/models")
        data = response.get("data", []) if isinstance(response, dict) else response
        return [ModelInfo.model_validate(m) for m in data]
