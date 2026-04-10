import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Video
from app.schemas import AgeGroupDetail, AnalyticsResponse, FootTrafficPoint

router = APIRouter()


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

    return AnalyticsResponse(
        video_id=video_id,
        total_unique=result.total_unique,
        total_analyzed=result.total_analyzed,
        foot_traffic=[FootTrafficPoint(**p) for p in json.loads(result.foot_traffic)],
        age_distribution=age_dist,
        gender_distribution=json.loads(result.gender_distribution),
        processing_time_sec=result.processing_time_sec,
    )


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
