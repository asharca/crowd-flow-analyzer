import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.ml.models import get_default_model_id, get_model, list_models
from app.models import Video
from app.schemas import UploadResponse, VideoListResponse, VideoResponse


def _video_to_response(video: Video) -> VideoResponse:
    return VideoResponse(
        id=video.id,
        original_name=video.original_name,
        file_size=video.file_size,
        duration_sec=video.duration_sec,
        status=video.status,
        error_message=video.error_message,
        has_annotated_video=video.annotated_filename is not None,
        created_at=video.created_at,
        completed_at=video.completed_at,
    )

router = APIRouter()


@router.get("/models")
def get_available_models():
    """List all available detection models and server defaults."""
    from app.ml.device import DEVICE, FRAME_SKIP, MIVOLO_BATCH_SIZE, YOLO_BATCH_SIZE
    return {
        "models": list_models(),
        "default": get_default_model_id(DEVICE),
        "defaults": {
            "frame_skip": FRAME_SKIP,
            "yolo_batch_size": YOLO_BATCH_SIZE,
            "mivolo_batch_size": MIVOLO_BATCH_SIZE,
            "max_crops": 5,
            "device": DEVICE,
        },
    }


@router.post("/upload", response_model=UploadResponse)
def upload_video(
    file: UploadFile,
    model: str = Form(default=""),
    frame_skip: int = Form(default=0),
    yolo_batch_size: int = Form(default=0),
    mivolo_batch_size: int = Form(default=0),
    max_crops: int = Form(default=0),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {settings.allowed_extensions}",
        )

    # Save file
    video_id = str(uuid.uuid4())
    filename = f"{video_id}.{ext}"
    file_path = settings.upload_dir / filename

    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    file_size = file_path.stat().st_size
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_bytes:
        file_path.unlink()
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    # Resolve model selection
    if not model:
        from app.ml.device import DEVICE
        model = get_default_model_id(DEVICE)
    model_info = get_model(model)

    # Build pipeline params (0 = auto/server default)
    import json
    params: dict = {}
    if frame_skip > 0:
        params["frame_skip"] = frame_skip
    if yolo_batch_size > 0:
        params["yolo_batch_size"] = yolo_batch_size
    if mivolo_batch_size > 0:
        params["mivolo_batch_size"] = mivolo_batch_size
    if max_crops > 0:
        params["max_crops"] = max_crops

    # Create DB record
    video = Video(
        id=video_id,
        filename=filename,
        original_name=file.filename,
        file_size=file_size,
        yolo_model=model_info.id,
        pipeline_params=json.dumps(params),
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    # Dispatch Celery task
    from app.worker.tasks import process_video

    task = process_video.delay(video_id)

    video.celery_task_id = task.id
    db.commit()

    return UploadResponse(id=video_id, status="queued", message="Video uploaded, processing started")


@router.get("", response_model=VideoListResponse)
def list_videos(status: str | None = None, db: Session = Depends(get_db)):
    query = db.query(Video).order_by(Video.created_at.desc())
    if status:
        query = query.filter(Video.status == status)
    videos = query.all()
    return VideoListResponse(
        videos=[_video_to_response(v) for v in videos],
        total=len(videos),
    )


@router.get("/{video_id}", response_model=VideoResponse)
def get_video(video_id: str, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return _video_to_response(video)


@router.get("/{video_id}/stream")
def stream_video(video_id: str, db: Session = Depends(get_db)):
    """Stream the original uploaded video."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    file_path = settings.upload_dir / video.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=video.original_name,
    )


@router.get("/{video_id}/stream/annotated")
def stream_annotated_video(video_id: str, db: Session = Depends(get_db)):
    """Stream the annotated video with bounding boxes and labels."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.annotated_filename:
        raise HTTPException(status_code=404, detail="Annotated video not available")

    file_path = settings.upload_dir / video.annotated_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video file not found")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=f"annotated_{video.original_name}",
    )


@router.delete("/{video_id}")
def delete_video(video_id: str, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete files
    file_path = settings.upload_dir / video.filename
    if file_path.exists():
        file_path.unlink()
    if video.annotated_filename:
        annotated_path = settings.upload_dir / video.annotated_filename
        if annotated_path.exists():
            annotated_path.unlink()

    db.delete(video)
    db.commit()
    return {"message": "Video deleted"}
