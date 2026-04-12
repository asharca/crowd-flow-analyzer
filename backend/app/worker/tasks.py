import json
import logging
import time
from datetime import datetime, timezone

import cv2
import redis

from app.config import settings
from app.database import SessionLocal
from app.ml.pipeline import get_system_info, run_pipeline
from app.models import AnalysisResult, Video
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _progress_key(video_id: str) -> str:
    return f"cfa:progress:{video_id}"


def _make_progress_callback(video_id: str, yolo_model: str | None = None):
    """Return a callback that pushes progress updates to Redis."""
    r = _get_redis()
    key = _progress_key(video_id)
    sys_info = get_system_info(yolo_model)

    def on_progress(stage: str, percent: int, detail: str) -> None:
        stages = ["detection", "tracking", "demographics", "annotation", "aggregation"]
        stage_idx = stages.index(stage) if stage in stages else 0
        # Overall progress: weight each stage
        weights = [30, 5, 40, 20, 5]  # detection heaviest + demographics
        completed_weight = sum(weights[:stage_idx])
        current_weight = weights[stage_idx] if stage_idx < len(weights) else 0
        overall = completed_weight + int(current_weight * percent / 100)

        data = {
            "stage": stage,
            "stage_percent": percent,
            "overall_percent": min(overall, 100),
            "detail": detail,
            "system_info": sys_info,
            "updated_at": time.time(),
        }
        r.set(key, json.dumps(data), ex=3600)

    return on_progress


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

        # Resolve YOLO model from user selection
        yolo_model_file = None
        if video.yolo_model:
            from app.ml.models import get_model
            yolo_model_file = get_model(video.yolo_model).filename

        # Parse user pipeline params (0 values = use server default)
        pp = json.loads(video.pipeline_params) if video.pipeline_params else {}

        # Update progress callback with correct model info
        on_progress = _make_progress_callback(video_id, yolo_model_file)

        result = run_pipeline(
            video_path,
            annotated_output_path=annotated_path,
            on_progress=on_progress,
            yolo_model=yolo_model_file,
            frame_skip=pp.get("frame_skip"),
            yolo_batch_size=pp.get("yolo_batch_size"),
            mivolo_batch_size=pp.get("mivolo_batch_size"),
            max_crops=pp.get("max_crops"),
        )
        elapsed = time.time() - start

        analysis = AnalysisResult(
            video_id=video_id,
            total_unique=result["total_unique"],
            total_analyzed=result["total_analyzed"],
            foot_traffic=json.dumps(result["foot_traffic"]),
            age_distribution=json.dumps(result["age_distribution"]),
            gender_distribution=json.dumps(result["gender_distribution"]),
            persons=json.dumps(result["persons"]),
            pipeline_config=json.dumps(result["pipeline_config"]),
            processing_time_sec=round(elapsed, 2),
        )
        db.add(analysis)

        video.annotated_filename = annotated_filename
        video.status = "completed"
        video.completed_at = datetime.now(timezone.utc).isoformat()
        db.commit()

        # Clean up progress key
        try:
            _get_redis().delete(_progress_key(video_id))
        except Exception:
            pass

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
