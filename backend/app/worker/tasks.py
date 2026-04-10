import json
import logging
import time
from datetime import datetime, timezone

import cv2

from app.config import settings
from app.database import SessionLocal
from app.ml.pipeline import run_pipeline
from app.models import AnalysisResult, Video
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


def _probe_video(video_path: str) -> float | None:
    """Extract video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if frame_count > 0 and fps > 0:
        return frame_count / fps
    return None


@celery_app.task(bind=True, max_retries=2)
def process_video(self, video_id: str):
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            return {"error": "Video not found"}

        video.status = "processing"
        db.commit()

        video_path = str(settings.upload_dir / video.filename)

        # Probe video metadata
        duration = _probe_video(video_path)
        if duration is not None:
            video.duration_sec = round(duration, 2)
            db.commit()

        start = time.time()
        logger.info("Processing video %s: %s", video_id, video.original_name)

        # Annotated video output path
        annotated_filename = f"{video_id}_annotated.mp4"
        annotated_path = str(settings.upload_dir / annotated_filename)

        result = run_pipeline(video_path, annotated_output_path=annotated_path)
        elapsed = time.time() - start

        analysis = AnalysisResult(
            video_id=video_id,
            total_unique=result["total_unique"],
            total_analyzed=result["total_analyzed"],
            foot_traffic=json.dumps(result["foot_traffic"]),
            age_distribution=json.dumps(result["age_distribution"]),
            gender_distribution=json.dumps(result["gender_distribution"]),
            processing_time_sec=round(elapsed, 2),
        )
        db.add(analysis)

        video.annotated_filename = annotated_filename
        video.status = "completed"
        video.completed_at = datetime.now(timezone.utc).isoformat()
        db.commit()

        return {"status": "completed", "video_id": video_id}

    except Exception as exc:
        db.rollback()
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = "failed"
            video.error_message = str(exc)[:500]
            db.commit()
        raise self.retry(exc=exc, countdown=30)
    finally:
        db.close()
