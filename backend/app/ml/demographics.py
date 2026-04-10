import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np
from deepface import DeepFace

from app.ml.device import DEMOGRAPHICS_WORKERS
from app.ml.tracker import TrackedFrame

logger = logging.getLogger(__name__)

# Number of top crops to try per track (more crops = better voting accuracy)
_MAX_CROPS_PER_TRACK = 3
# Minimum gender confidence to accept a prediction (0-1)
_MIN_GENDER_CONFIDENCE = 0.65


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
    """Try to analyze a single person crop with proper face detection.

    Uses ssd for speed, then falls back to opencv.
    Does NOT use detector_backend='skip' to avoid low-quality predictions.
    """
    backends = ["ssd", "opencv"]
    for backend in backends:
        try:
            analyses = DeepFace.analyze(
                img_path=crop,
                actions=["age", "gender"],
                enforce_detection=True,
                detector_backend=backend,
                silent=True,
            )
            result = analyses[0] if isinstance(analyses, list) else analyses
            face_conf = result.get("face_confidence", 0)
            if face_conf and face_conf > 0.6:
                return result
        except (ValueError, Exception):
            continue

    # Fallback: take upper 40% of body crop as rough head region
    h = crop.shape[0]
    head_crop = crop[0 : int(h * 0.4), :]
    if head_crop.size == 0:
        return None

    # Try head crop with ssd (still enforce detection to avoid garbage predictions)
    try:
        analyses = DeepFace.analyze(
            img_path=head_crop,
            actions=["age", "gender"],
            enforce_detection=True,
            detector_backend="ssd",
            silent=True,
        )
        result = analyses[0] if isinstance(analyses, list) else analyses
        face_conf = result.get("face_confidence", 0)
        if face_conf and face_conf > 0.5:
            return result
    except (ValueError, Exception):
        pass

    return None


def _analyze_track(tid: int, crop_list: list[tuple[int, np.ndarray]]) -> tuple[int, DemographicResult | None]:
    """Analyze all crops for a single track and return (tid, result)."""
    successful_analyses: list[dict] = []
    for _frame_idx, crop in crop_list:
        analysis = _analyze_single_crop(crop)
        if analysis is not None:
            successful_analyses.append(analysis)

    if not successful_analyses:
        logger.warning("Demographics failed for track %d (tried %d crops)", tid, len(crop_list))
        return tid, None

    gender_votes: list[tuple[str, float]] = []
    ages: list[int] = []
    for analysis in successful_analyses:
        dominant_gender = analysis.get("dominant_gender", "Man")
        gender_scores = analysis.get("gender", {})
        conf = gender_scores.get(dominant_gender, 50.0) / 100.0
        gender_votes.append(("male" if dominant_gender == "Man" else "female", conf))
        ages.append(int(analysis["age"]))

    vote_counts = Counter(g for g, _ in gender_votes)
    winner_gender = vote_counts.most_common(1)[0][0]
    winner_confidences = [c for g, c in gender_votes if g == winner_gender]
    avg_confidence = sum(winner_confidences) / len(winner_confidences)

    if avg_confidence < _MIN_GENDER_CONFIDENCE:
        logger.info(
            "Track %d: gender confidence %.2f below threshold %.2f, skipping",
            tid, avg_confidence, _MIN_GENDER_CONFIDENCE,
        )
        return tid, None

    avg_age = round(sum(ages) / len(ages))
    return tid, DemographicResult(
        age=avg_age,
        age_group=_classify_age_group(avg_age),
        gender=winner_gender,
        confidence=avg_confidence,
    )


def analyze_demographics(
    video_path: str, tracked_frames: list[TrackedFrame]
) -> dict[int, DemographicResult]:
    """Run DeepFace age/gender analysis on the best crops for each tracked person.

    Tracks are analyzed in parallel using a thread pool.
    """
    all_crops = _find_top_crops(video_path, tracked_frames)
    results: dict[int, DemographicResult] = {}

    with ThreadPoolExecutor(max_workers=DEMOGRAPHICS_WORKERS) as executor:
        futures = {
            executor.submit(_analyze_track, tid, crop_list): tid
            for tid, crop_list in all_crops.items()
        }
        for future in as_completed(futures):
            tid, result = future.result()
            if result is not None:
                results[tid] = result

    logger.info("Demographics: %d/%d tracks analyzed", len(results), len(all_crops))
    return results
