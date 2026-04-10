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


class AnalyticsResponse(BaseModel):
    video_id: str
    total_unique: int
    total_analyzed: int
    foot_traffic: list[FootTrafficPoint]
    age_distribution: dict[str, AgeGroupDetail]
    gender_distribution: dict[str, int]
    processing_time_sec: float | None


class UploadResponse(BaseModel):
    id: str
    status: str
    message: str
