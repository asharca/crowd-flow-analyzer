import json

import redis
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Video
from app.schemas import AgeGroupDetail, AnalyticsResponse, FootTrafficPoint, PersonResult, PipelineConfig

router = APIRouter()

_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _get_video_with_analysis(video_id: str, db: Session) -> Video:
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status == "processing" or video.status == "queued":
        raise HTTPException(status_code=202, detail="Video is still being processed")
    if video.status == "failed":
        raise HTTPException(status_code=422, detail=f"Processing failed: {video.error_message}")
    if not video.analysis:
        raise HTTPException(status_code=404, detail="Analysis results not found")
    return video


@router.get("/{video_id}/analytics", response_model=AnalyticsResponse)
def get_analytics(video_id: str, db: Session = Depends(get_db)):
    video = _get_video_with_analysis(video_id, db)
    result = video.analysis
    age_raw = json.loads(result.age_distribution)
    age_dist = {k: AgeGroupDetail(**v) if isinstance(v, dict) else AgeGroupDetail(male=0, female=0, total=v) for k, v in age_raw.items()}

    persons_raw = json.loads(result.persons) if result.persons else []
    config_raw = json.loads(result.pipeline_config) if result.pipeline_config else {}

    return AnalyticsResponse(
        video_id=video_id,
        total_unique=result.total_unique,
        total_analyzed=result.total_analyzed,
        foot_traffic=[FootTrafficPoint(**p) for p in json.loads(result.foot_traffic)],
        age_distribution=age_dist,
        gender_distribution=json.loads(result.gender_distribution),
        persons=[PersonResult(**p) for p in persons_raw],
        pipeline_config=PipelineConfig(**config_raw),
        processing_time_sec=result.processing_time_sec,
    )


@router.get("/{video_id}/progress")
def get_progress(video_id: str, db: Session = Depends(get_db)):
    """Get real-time processing progress from Redis."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if video.status == "completed":
        return {
            "stage": "completed",
            "stage_percent": 100,
            "overall_percent": 100,
            "detail": "Processing complete",
            "system_info": {},
        }
    if video.status == "failed":
        return {
            "stage": "failed",
            "stage_percent": 0,
            "overall_percent": 0,
            "detail": video.error_message or "Unknown error",
            "system_info": {},
        }

    raw = _get_redis().get(f"cfa:progress:{video_id}")
    if raw:
        return json.loads(raw)

    return {
        "stage": "queued",
        "stage_percent": 0,
        "overall_percent": 0,
        "detail": "Waiting in queue...",
        "system_info": {},
    }


@router.get("/{video_id}/analytics/foot-traffic")
def get_foot_traffic(video_id: str, db: Session = Depends(get_db)):
    video = _get_video_with_analysis(video_id, db)
    return {
        "video_id": video_id,
        "foot_traffic": json.loads(video.analysis.foot_traffic),
    }


@router.get("/{video_id}/analytics/demographics")
def get_demographics(video_id: str, db: Session = Depends(get_db)):
    video = _get_video_with_analysis(video_id, db)
    return {
        "video_id": video_id,
        "age_distribution": json.loads(video.analysis.age_distribution),
        "gender_distribution": json.loads(video.analysis.gender_distribution),
    }
