from pydantic import BaseModel


class VideoResponse(BaseModel):
    id: str
    original_name: str
    file_size: int
    duration_sec: float | None
    status: str
    error_message: str | None
    has_annotated_video: bool
    created_at: str
    completed_at: str | None

    model_config = {"from_attributes": True}


class VideoListResponse(BaseModel):
    videos: list[VideoResponse]
    total: int


class FootTrafficPoint(BaseModel):
    timestamp_sec: float
    count: int
    male: int
    female: int
    unknown: int


class AgeGroupDetail(BaseModel):
    male: int
    female: int
    total: int


class PersonResult(BaseModel):
    track_id: int
    age: int | None
    age_group: str | None
    gender: str
    confidence: float


class PipelineConfig(BaseModel):
    device: str = ""
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    yolo_model: str = ""
    demographics_model: str = ""
    yolo_batch_size: int = 0
    mivolo_batch_size: int | None = None
    frame_skip: int | None = None


class AnalyticsResponse(BaseModel):
    video_id: str
    total_unique: int
    total_analyzed: int
    foot_traffic: list[FootTrafficPoint]
    age_distribution: dict[str, AgeGroupDetail]
    gender_distribution: dict[str, int]
    persons: list[PersonResult]
    pipeline_config: PipelineConfig
    processing_time_sec: float | None


class UploadResponse(BaseModel):
    id: str
    status: str
    message: str
