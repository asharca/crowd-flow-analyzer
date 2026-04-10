import logging
from dataclasses import dataclass

import cv2
import numpy as np
from deepface import DeepFace

from app.ml.tracker import TrackedFrame

logger = logging.getLogger(__name__)

# Number of top crops to try per track (fallback if face detection fails on best crop)
_MAX_CROPS_PER_TRACK = 3


@dataclass
class DemographicResult:
    age: int
    age_group: str
    gender: str
    confidence: float


def _classify_age_group(age: int) -> str:
    if age <= 18:
        return "0-18"
    if age <= 30:
        return "19-30"
    if age <= 45:
        return "31-45"
    if age <= 60:
        return "46-60"
    return "60+"


def _find_top_crops(
    video_path: str, tracked_frames: list[TrackedFrame], max_per_track: int = _MAX_CROPS_PER_TRACK
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """For each unique track ID, find the top N frames (by bbox area)
    and return cropped regions. Multiple crops give us fallback options
    if face detection fails on one crop.

    Returns dict[track_id -> [(frame_index, crop_array), ...]].
    """
    # Collect all (frame_idx, bbox, area) per track
    track_candidates: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for tf in tracked_frames:
        for i, tid in enumerate(tf.tracker_ids):
            tid = int(tid)
            bbox = tf.bboxes[i]
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            track_candidates.setdefault(tid, []).append((tf.frame_index, bbox, area))

    # Keep top N by area for each track
    top_per_track: dict[int, list[tuple[int, np.ndarray]]] = {}
    for tid, candidates in track_candidates.items():
        candidates.sort(key=lambda x: x[2], reverse=True)
        top_per_track[tid] = [(fi, bb) for fi, bb, _ in candidates[:max_per_track]]

    # Group by frame to minimize seeking
    frame_to_tracks: dict[int, list[tuple[int, np.ndarray]]] = {}
    for tid, entries in top_per_track.items():
        for frame_idx, bbox in entries:
            frame_to_tracks.setdefault(frame_idx, []).append((tid, bbox))

    # Read frames and crop
    cap = cv2.VideoCapture(video_path)
    crops: dict[int, list[tuple[int, np.ndarray]]] = {}

    for target_frame in sorted(frame_to_tracks.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        for tid, bbox in frame_to_tracks[target_frame]:
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.setdefault(tid, []).append((target_frame, crop))

    cap.release()
    return crops


def _analyze_single_crop(crop: np.ndarray) -> dict | None:
    """Try to analyze a single person crop with proper face detection."""
    # First try with opencv face detector (most reliable for this use case)
    try:
        analyses = DeepFace.analyze(
            img_path=crop,
            actions=["age", "gender"],
            enforce_detection=True,
            detector_backend="opencv",
            silent=True,
        )
        result = analyses[0] if isinstance(analyses, list) else analyses
        face_conf = result.get("face_confidence", 0)
        if face_conf and face_conf > 0.5:
            return result
    except (ValueError, Exception):
        pass

    # Fallback: take upper 40% of body crop as rough head region, analyze without detection
    h = crop.shape[0]
    head_crop = crop[0 : int(h * 0.4), :]
    if head_crop.size == 0:
        return None

    try:
        analyses = DeepFace.analyze(
            img_path=head_crop,
            actions=["age", "gender"],
            enforce_detection=False,
            detector_backend="skip",
            silent=True,
        )
        return analyses[0] if isinstance(analyses, list) else analyses
    except Exception:
        return None


def analyze_demographics(
    video_path: str, tracked_frames: list[TrackedFrame]
) -> dict[int, DemographicResult]:
    """Run DeepFace age/gender analysis on the best crops for each tracked person."""
    all_crops = _find_top_crops(video_path, tracked_frames)
    results: dict[int, DemographicResult] = {}

    for tid, crop_list in all_crops.items():
        analysis = None
        for frame_idx, crop in crop_list:
            analysis = _analyze_single_crop(crop)
            if analysis is not None:
                break

        if analysis is None:
            logger.warning("Demographics failed for track %d (tried %d crops)", tid, len(crop_list))
            continue

        age = int(analysis["age"])
        dominant_gender = analysis.get("dominant_gender", "Man")
        gender_scores = analysis.get("gender", {})
        confidence = gender_scores.get(dominant_gender, 50.0) / 100.0
        gender = "male" if dominant_gender == "Man" else "female"

        results[tid] = DemographicResult(
            age=age,
            age_group=_classify_age_group(age),
            gender=gender,
            confidence=confidence,
        )

    logger.info("Demographics: %d/%d tracks analyzed", len(results), len(all_crops))
    return results
