from typing import List, Optional, Any, Dict
from pydantic import BaseModel, ConfigDict

class PerModelResult(BaseModel):
    success: bool
    detector: Optional[str] = None
    model_used: Optional[str] = None
    ai_probability: Optional[float] = None
    real_probability: Optional[float] = None
    confidence: Optional[float] = None
    inference_time_ms: Optional[float] = None
    error: Optional[str] = None
    raw_results: Optional[Any] = None

    model_config = ConfigDict(extra='allow')

class DetectionData(BaseModel):
    success: bool
    ai_probability: Optional[float] = None
    real_probability: Optional[float] = None
    weighted_ai_probability: Optional[float] = None
    average_ai_probability: Optional[float] = None
    average_real_probability: Optional[float] = None
    is_ai_generated: Optional[bool] = None
    confidence: Optional[float] = None
    models_used: Optional[List[str]] = None
    filename: str
    error: Optional[str] = None
    per_model: Optional[List[PerModelResult]] = None

class DetectionResponse(BaseModel):
    success: bool
    data: Optional[DetectionData] = None
    error: Optional[str] = None

class BatchDetectionResponse(BaseModel):
    success: bool
    data: List[DetectionData]
    total: int
    error: Optional[str] = None

class EnsembleInfo(BaseModel):
    combiner: str
    weights: Dict[str, float]
    return_per_model: bool

class HealthResponse(BaseModel):
    status: str
    service: str
    models: Optional[List[str]] = None
    ensemble: Optional[EnsembleInfo] = None
    error: Optional[str] = None


class TextPerModelResult(BaseModel):
    success: bool
    detector: Optional[str] = None
    model_used: Optional[str] = None
    ai_probability: Optional[float] = None
    human_probability: Optional[float] = None
    confidence: Optional[float] = None
    inference_time_ms: Optional[float] = None
    error: Optional[str] = None
    raw_results: Optional[Any] = None

    model_config = ConfigDict(extra='allow')


class TextDetectionData(BaseModel):
    success: bool
    ai_probability: Optional[float] = None
    human_probability: Optional[float] = None
    weighted_ai_probability: Optional[float] = None
    average_ai_probability: Optional[float] = None
    average_human_probability: Optional[float] = None
    is_ai_generated: Optional[bool] = None
    confidence: Optional[float] = None
    models_used: Optional[List[str]] = None
    text_length: int
    error: Optional[str] = None
    per_model: Optional[List[TextPerModelResult]] = None


class TextDetectionResponse(BaseModel):
    success: bool
    data: Optional[TextDetectionData] = None
    error: Optional[str] = None


class VideoFrameResult(BaseModel):
    frame_index: int
    probability: float


class VideoDetectionResponse(BaseModel):
    ai_probability: float
    frame_results: List[VideoFrameResult]
    aggregation: str
    num_frames_used: int
    processing_time_ms: Optional[float] = None
