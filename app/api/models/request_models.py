from pydantic import BaseModel, Field


class TextDetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text content to analyze")
