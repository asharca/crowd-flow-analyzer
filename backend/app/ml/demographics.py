import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np

from app.ml.device import DEMOGRAPHICS_WORKERS, DEVICE, _cuda_device_index
from app.ml.tracker import TrackedFrame

logger = logging.getLogger(__name__)

# Number of top crops to try per track (by bbox area — largest = closest to camera)
_MAX_CROPS_PER_TRACK = 3
# Minimum face confidence to accept InsightFace prediction
_MIN_FACE_CONFIDENCE = 0.5
# Minimum gender vote ratio to accept result (e.g. 2/3 crops agree)
_MIN_VOTE_RATIO = 0.5


@dataclass
class DemographicResult:
    age: int
    age_group: str
    gender: str
    confidence: float


def _get_face_analyzer():
    """Return a thread-local InsightFace FaceAnalysis instance.

    InsightFace models are not thread-safe to share, so each worker gets its own.
    Uses CUDA when available, falls back to CPU.
    """
    import threading

    import onnxruntime as rt

    local = threading.local()
    if not hasattr(local, "app"):
        from insightface.app import FaceAnalysis

        available = rt.get_available_providers()
        if "CUDAExecutionProvider" in available and DEVICE.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = _cuda_device_index(DEVICE)
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        local.app = app
    return local.app


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
    and return cropped regions.

    Returns dict[track_id -> [(frame_index, crop_array), ...]].
    """
    track_candidates: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for tf in tracked_frames:
        for i, tid in enumerate(tf.tracker_ids):
            tid = int(tid)
            bbox = tf.bboxes[i]
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            track_candidates.setdefault(tid, []).append((tf.frame_index, bbox, area))

    top_per_track: dict[int, list[tuple[int, np.ndarray]]] = {}
    for tid, candidates in track_candidates.items():
        candidates.sort(key=lambda x: x[2], reverse=True)
        top_per_track[tid] = [(fi, bb) for fi, bb, _ in candidates[:max_per_track]]

    # Group by frame to minimize video seeking
    frame_to_tracks: dict[int, list[tuple[int, np.ndarray]]] = {}
    for tid, entries in top_per_track.items():
        for frame_idx, bbox in entries:
            frame_to_tracks.setdefault(frame_idx, []).append((tid, bbox))

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


def _analyze_crop(crop: np.ndarray) -> dict | None:
    """Run InsightFace on a single person crop.

    InsightFace's SCRFD detector handles small faces (down to ~16px) much better
    than SSD/RetinaFace. Returns gender ('M'/'F') and age, or None if no face found.
    """
    app = _get_face_analyzer()
    # Resize small crops to give the detector a better chance
    h, w = crop.shape[:2]
    if h < 128 or w < 64:
        scale = max(128 / h, 64 / w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    faces = app.get(crop)
    if not faces:
        return None

    # Pick the face with highest detection score
    face = max(faces, key=lambda f: f.det_score)
    if face.det_score < _MIN_FACE_CONFIDENCE:
        return None

    return {"gender": face.sex, "age": int(face.age), "confidence": float(face.det_score)}


def _analyze_track(tid: int, crop_list: list[tuple[int, np.ndarray]]) -> tuple[int, DemographicResult | None]:
    """Analyze all crops for a single track via multi-crop voting."""
    gender_votes: list[tuple[str, float]] = []
    ages: list[int] = []

    for _frame_idx, crop in crop_list:
        result = _analyze_crop(crop)
        if result is None:
            continue
        gender = "male" if result["gender"] == "M" else "female"
        gender_votes.append((gender, result["confidence"]))
        ages.append(result["age"])

    if not gender_votes:
        logger.warning("Demographics failed for track %d (tried %d crops)", tid, len(crop_list))
        return tid, None

    vote_counts = Counter(g for g, _ in gender_votes)
    winner_gender = vote_counts.most_common(1)[0][0]
    total_votes = len(gender_votes)
    winner_votes = vote_counts[winner_gender]

    if winner_votes / total_votes < _MIN_VOTE_RATIO:
        logger.info("Track %d: ambiguous gender votes %s, skipping", tid, dict(vote_counts))
        return tid, None

    winner_confidences = [c for g, c in gender_votes if g == winner_gender]
    avg_confidence = sum(winner_confidences) / len(winner_confidences)
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
    """Run InsightFace age/gender analysis on the best crops for each tracked person.

    Uses SCRFD face detector (designed for small/crowd faces) instead of SSD/RetinaFace.
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
