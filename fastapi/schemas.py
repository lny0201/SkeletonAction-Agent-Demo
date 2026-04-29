from typing import List, Optional

from pydantic import BaseModel, Field


class JsonSample(BaseModel):
    keypoint: List = Field(..., description="Raw NTU keypoint with semantic shape (M,T,V,C)")
    total_frames: Optional[int] = Field(default=None)
    sample_name: str = Field(default="")


class PredictJsonRequest(BaseModel):
    samples: List[JsonSample]
    topk: int = Field(default=5, ge=1, le=120)


class TopKItem(BaseModel):
    label: int
    name: str
    score: float


class PredictItem(BaseModel):
    sample_name: str
    pred_label: int
    pred_name: str
    score: float
    topk: List[TopKItem]
    input_shape: List[int]
    model_input_shape: List[int]


class PredictResponse(BaseModel):
    num_samples: int
    predictions: List[PredictItem]


class VideoDemoMapping(BaseModel):
    video_filename: str
    map_key: str
    sample_name: str
    input_type: str
    input_path: str


class PredictVideoDemoResponse(BaseModel):
    mapping: VideoDemoMapping
    num_samples: int
    predictions: List[PredictItem]
    metrics: Optional[dict] = None


class ExplanationResult(BaseModel):
    mode: str
    text: str
    error: str = ""
    prompt_preview: str = ""


class ExplainRequest(BaseModel):
    prediction: PredictItem
    language: str = Field(default="zh")
    include_prompt_preview: bool = Field(default=False)


class QARequest(BaseModel):
    prediction: PredictItem
    question: str = Field(..., min_length=1)
    language: str = Field(default="zh")
    include_prompt_preview: bool = Field(default=False)


class ExplainResponse(BaseModel):
    explanation: ExplanationResult


class QAResponse(BaseModel):
    answer: ExplanationResult


class AnalyzeVideoDemoResponse(BaseModel):
    mapping: VideoDemoMapping
    num_samples: int
    predictions: List[PredictItem]
    explanation: ExplanationResult
    metrics: Optional[dict] = None
