"""Demographics analysis using MiVOLO v2 (body-based age/gender estimation).

MiVOLO estimates age and gender from full-body crops without requiring
face detection, making it robust for crowd scenes where faces are often
not visible (back-facing, masked, distant).

Pipeline: reuse YOLO person bboxes -> crop body regions -> batch MiVOLO inference.
"""

import logging
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from app.ml.device import DEVICE, MIVOLO_BATCH_SIZE
from app.ml.tracker import TrackedFrame

logger = logging.getLogger(__name__)

# Number of top crops to try per track (by bbox area — largest = closest to camera)
_MAX_CROPS_PER_TRACK = 5


@dataclass
class DemographicResult:
    age: int
    age_group: str
    gender: str
    confidence: float


# ── Lazy-loaded MiVOLO singleton ────────────────────────────────────────────

_model = None
_processor = None
_config = None
_face_zero: torch.Tensor | None = None  # pre-computed zero tensor for body-only mode


def _load_model() -> None:
    """Load MiVOLO v2 from HuggingFace Hub (cached after first call)."""
    global _model, _processor, _config, _face_zero
    if _model is not None:
        return

    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForImageClassification,
    )

    model_id = "iitolstykh/mivolo_v2"
    use_fp16 = DEVICE.startswith("cuda")
    dtype = torch.float16 if use_fp16 else torch.float32

    logger.info("Loading MiVOLO v2 (device=%s, dtype=%s) ...", DEVICE, dtype)

    _config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    _processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    _model = (
        AutoModelForImageClassification.from_pretrained(
            model_id, trust_remote_code=True, dtype=dtype,
        )
        .to(DEVICE)
        .eval()
    )

    # Pre-compute a single "no face" zero tensor — reused for every body-only call
    face_null = _processor(images=[None])["pixel_values"]
    if isinstance(face_null, list):
        face_null = torch.stack(face_null)
    _face_zero = face_null.to(device=DEVICE, dtype=dtype)

    logger.info("MiVOLO v2 ready")


# ── Helpers ─────────────────────────────────────────────────────────────────


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
    video_path: str,
    tracked_frames: list[TrackedFrame],
    max_per_track: int = _MAX_CROPS_PER_TRACK,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """For each track, find the top N frames (by bbox area) and return body crops.

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

    # Group by frame index to minimize video seeking
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


# ── Batch inference ─────────────────────────────────────────────────────────


@torch.no_grad()
def _infer_batch(crops: list[np.ndarray]) -> list[dict]:
    """Run MiVOLO on a batch of body crops (no face detection needed).

    Returns list of {"age": float, "gender": str, "confidence": float}.
    """
    body_processed = _processor(images=crops)["pixel_values"]
    if isinstance(body_processed, list):
        body_processed = torch.stack(body_processed)
    body_batch = body_processed.to(device=DEVICE, dtype=_model.dtype)

    # Expand the pre-computed face-zero tensor to match batch size
    face_batch = _face_zero.expand(len(crops), -1, -1, -1)

    output = _model(faces_input=face_batch, body_input=body_batch)

    results = []
    for i in range(len(crops)):
        age = output.age_output[i].item()
        gender_idx = output.gender_class_idx[i].item()
        gender = _config.gender_id2label[gender_idx]
        confidence = output.gender_probs[i].item()
        results.append({"age": age, "gender": gender, "confidence": confidence})

    return results


# ── Public API ──────────────────────────────────────────────────────────────


def analyze_demographics(
    video_path: str,
    tracked_frames: list[TrackedFrame],
    batch_size: int | None = None,
    max_crops_per_track: int | None = None,
) -> dict[int, DemographicResult]:
    """Run MiVOLO body-based age/gender on each tracked person.

    Uses batch GPU inference on body crops from existing YOLO detections.
    No face detection required — works even when people face away from camera.
    """
    _load_model()
    effective_max_crops = max_crops_per_track or _MAX_CROPS_PER_TRACK
    all_crops = _find_top_crops(video_path, tracked_frames, max_per_track=effective_max_crops)

    # Flatten all crops for batch processing: [(tid, crop), ...]
    batch_items: list[tuple[int, np.ndarray]] = []
    for tid, crop_list in all_crops.items():
        for _, crop in crop_list:
            batch_items.append((tid, crop))

    if not batch_items:
        logger.info("Demographics: no crops to analyze")
        return {}

    # Run batch inference
    raw_results: dict[int, list[dict]] = {}
    effective_batch = batch_size or MIVOLO_BATCH_SIZE
    for start in range(0, len(batch_items), effective_batch):
        batch = batch_items[start : start + effective_batch]
        crops = [item[1] for item in batch]
        predictions = _infer_batch(crops)

        for (tid, _), pred in zip(batch, predictions):
            raw_results.setdefault(tid, []).append(pred)

    # Aggregate per track via multi-crop voting
    results: dict[int, DemographicResult] = {}
    for tid, predictions in raw_results.items():
        genders = [p["gender"] for p in predictions]
        winner_gender = Counter(genders).most_common(1)[0][0]

        avg_age = round(sum(p["age"] for p in predictions) / len(predictions))
        winner_confs = [p["confidence"] for p in predictions if p["gender"] == winner_gender]
        avg_conf = sum(winner_confs) / len(winner_confs)

        results[tid] = DemographicResult(
            age=avg_age,
            age_group=_classify_age_group(avg_age),
            gender=winner_gender,
            confidence=avg_conf,
        )

    logger.info("Demographics: %d/%d tracks analyzed", len(results), len(all_crops))
    return results
