"""Type definitions for AI Models."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about a Cortiqa AI Model."""

    id: str = Field(..., description="Unique model identifier (e.g. falin-01, falin-pro)")
    name: Optional[str] = Field(None, description="Display name of the model")
    description: Optional[str] = Field(None, description="Capabilities and summary")
    provider: Optional[str] = Field("Cortiqa", description="Provider organization")
    free: Optional[bool] = Field(None, description="Whether the model is available on free tier")


class ModelListResponse(BaseModel):
    """Response containing list of available models."""

    success: bool = True
    data: List[ModelInfo] = Field(default_factory=list)
