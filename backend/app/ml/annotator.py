import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.config import settings
from app.ml.demographics import DemographicResult

logger = logging.getLogger(__name__)

# Color palette: male=blue, female=pink, unknown=gray
GENDER_COLORS = {
    "male": sv.Color(66, 133, 244),
    "female": sv.Color(233, 80, 142),
}
DEFAULT_COLOR = sv.Color(158, 158, 158)

AGE_GROUP_LABELS = {
    "0-18": "Youth",
    "19-30": "Young",
    "31-45": "Adult",
    "46-60": "Middle",
    "60+": "Senior",
}


def _build_label(track_id: int, demo: DemographicResult | None) -> str:
    if demo is None:
        return f"#{track_id}"
    gender_icon = "M" if demo.gender == "male" else "F"
    return f"#{track_id} {gender_icon} {demo.age_group}"


def _get_color(demo: DemographicResult | None) -> sv.Color:
    if demo is None:
        return DEFAULT_COLOR
    return GENDER_COLORS.get(demo.gender, DEFAULT_COLOR)


def generate_annotated_video(
    video_path: str,
    output_path: str,
    demographics: dict[int, DemographicResult],
    frame_skip: int | None = None,
) -> None:
    """Re-run detection + tracking on the video and render annotated frames to output."""
    if frame_skip is None:
        frame_skip = settings.frame_skip

    model = YOLO(settings.yolo_model)
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write to a temp file first (mp4v), then re-encode to H.264 for browser playback
    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    # Annotators - use TRACK color lookup since Detections have no class_id
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color_lookup=sv.ColorLookup.TRACK,
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=5,
        color_lookup=sv.ColorLookup.TRACK,
    )

    frame_idx = 0
    last_tracked: sv.Detections | None = None
    last_labels: list[str] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model(frame, classes=[0], verbose=False)
            result = results[0]

            if len(result.boxes) > 0:
                bboxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
            else:
                bboxes = np.empty((0, 4))
                confs = np.empty((0,))

            detections = sv.Detections(xyxy=bboxes, confidence=confs)
            tracked = tracker.update_with_detections(detections)

            if len(tracked) > 0 and tracked.tracker_id is not None:
                labels = [
                    _build_label(int(tid), demographics.get(int(tid)))
                    for tid in tracked.tracker_id
                ]
                # Assign per-box colors based on gender
                last_tracked = tracked
                last_labels = labels
            else:
                last_tracked = None
                last_labels = []

        # Draw annotations (use last detection for non-sampled frames)
        if last_tracked is not None and len(last_tracked) > 0:
            annotated = box_annotator.annotate(frame.copy(), last_tracked)
            annotated = label_annotator.annotate(annotated, last_tracked, last_labels)
        else:
            annotated = frame

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    # Re-encode to H.264 for browser compatibility
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_path,
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        Path(tmp_path).unlink(missing_ok=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg not available - fall back to mp4v (may not play in some browsers)
        logger.warning("ffmpeg not available, using mp4v codec (limited browser support)")
        Path(tmp_path).rename(output_path)

    logger.info("Annotated video saved to %s (%d frames)", output_path, frame_idx)
