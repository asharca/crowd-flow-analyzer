from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import settings


@dataclass
class FrameDetection:
    frame_index: int
    timestamp_sec: float
    bboxes: np.ndarray  # shape (N, 4) in xyxy format
    confidences: np.ndarray  # shape (N,)


def detect_persons(video_path: str, frame_skip: int | None = None) -> list[FrameDetection]:
    """Run YOLOv8 person detection on video frames.

    Returns a list of FrameDetection, one per sampled frame.
    """
    if frame_skip is None:
        frame_skip = settings.frame_skip

    model = YOLO(settings.yolo_model)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detections: list[FrameDetection] = []
    frame_idx = 0

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

            detections.append(
                FrameDetection(
                    frame_index=frame_idx,
                    timestamp_sec=frame_idx / fps,
                    bboxes=bboxes,
                    confidences=confs,
                )
            )

        frame_idx += 1

    cap.release()
    return detections
