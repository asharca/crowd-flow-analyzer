from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import settings
from app.ml.device import DEVICE, YOLO_BATCH_SIZE


@dataclass
class FrameDetection:
    frame_index: int
    timestamp_sec: float
    bboxes: np.ndarray  # shape (N, 4) in xyxy format
    confidences: np.ndarray  # shape (N,)


def detect_persons(video_path: str, frame_skip: int | None = None) -> list[FrameDetection]:
    """Run YOLOv8 person detection on video frames using batch inference.

    Uses GPU (CUDA/MPS) when available, with FP16 on CUDA for ~2x throughput.
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

    # Accumulate sampled frames then run batch inference
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    frame_idx = 0

    use_half = DEVICE == "cuda"

    def _flush_batch() -> None:
        if not batch_frames:
            return
        batch_results = model(
            batch_frames,
            classes=[0],
            verbose=False,
            device=DEVICE,
            half=use_half,
        )
        for fi, result in zip(batch_indices, batch_results):
            if len(result.boxes) > 0:
                bboxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
            else:
                bboxes = np.empty((0, 4))
                confs = np.empty((0,))
            detections.append(
                FrameDetection(
                    frame_index=fi,
                    timestamp_sec=fi / fps,
                    bboxes=bboxes,
                    confidences=confs,
                )
            )
        batch_frames.clear()
        batch_indices.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            if len(batch_frames) >= YOLO_BATCH_SIZE:
                _flush_batch()

        frame_idx += 1

    _flush_batch()
    cap.release()
    return detections
