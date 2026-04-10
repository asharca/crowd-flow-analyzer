from dataclasses import dataclass

import numpy as np
import supervision as sv

from app.ml.detector import FrameDetection


@dataclass
class TrackedFrame:
    frame_index: int
    timestamp_sec: float
    bboxes: np.ndarray  # shape (N, 4)
    tracker_ids: np.ndarray  # shape (N,) int


def track_persons(frame_detections: list[FrameDetection]) -> list[TrackedFrame]:
    """Assign persistent track IDs across frames using ByteTrack."""
    tracker = sv.ByteTrack()
    tracked_frames: list[TrackedFrame] = []

    for fd in frame_detections:
        detections = sv.Detections(
            xyxy=fd.bboxes,
            confidence=fd.confidences,
        )
        tracked = tracker.update_with_detections(detections)

        if len(tracked) > 0 and tracked.tracker_id is not None:
            bboxes = tracked.xyxy
            ids = tracked.tracker_id
        else:
            bboxes = np.empty((0, 4))
            ids = np.empty((0,), dtype=int)

        tracked_frames.append(
            TrackedFrame(
                frame_index=fd.frame_index,
                timestamp_sec=fd.timestamp_sec,
                bboxes=bboxes,
                tracker_ids=ids,
            )
        )

    return tracked_frames
