import json
import logging
import math
from typing import Callable

from app.ml.demographics import DemographicResult, analyze_demographics
from app.ml.detector import detect_persons
from app.ml.tracker import TrackedFrame, track_persons

logger = logging.getLogger(__name__)

# Progress callback type: (stage, percent, detail)
ProgressCallback = Callable[[str, int, str], None]

def _noop_progress(stage: str, percent: int, detail: str) -> None:
    pass


def _aggregate_foot_traffic(
    tracked_frames: list[TrackedFrame],
    demo_results: dict[int, DemographicResult],
) -> list[dict]:
    """Compute per-second people count with gender breakdown."""
    if not tracked_frames:
        return []

    max_time = max(tf.timestamp_sec for tf in tracked_frames)
    total_seconds = int(math.ceil(max_time)) + 1

    second_ids: list[set[int]] = [set() for _ in range(total_seconds)]
    for tf in tracked_frames:
        sec = int(tf.timestamp_sec)
        if sec < total_seconds:
            for tid in tf.tracker_ids:
                second_ids[sec].add(int(tid))

    result = []
    for s, ids in enumerate(second_ids):
        male_count = 0
        female_count = 0
        for tid in ids:
            demo = demo_results.get(tid)
            if demo and demo.gender == "male":
                male_count += 1
            elif demo and demo.gender == "female":
                female_count += 1
        result.append({
            "timestamp_sec": float(s),
            "count": len(ids),
            "male": male_count,
            "female": female_count,
            "unknown": len(ids) - male_count - female_count,
        })
    return result


def _aggregate_demographics(
    demo_results: dict[int, DemographicResult],
) -> tuple[dict[str, dict], dict[str, int]]:
    """Aggregate age distribution by gender and overall gender distribution."""
    age_groups = ["0-18", "19-30", "31-45", "46-60", "60+"]

    # Age distribution with gender breakdown per group
    age_by_gender: dict[str, dict[str, int]] = {
        group: {"male": 0, "female": 0, "total": 0} for group in age_groups
    }
    gender_dist = {"male": 0, "female": 0}

    for result in demo_results.values():
        group = result.age_group
        if group not in age_by_gender:
            group = "60+"  # fallback
        age_by_gender[group][result.gender] += 1
        age_by_gender[group]["total"] += 1
        if result.gender in gender_dist:
            gender_dist[result.gender] += 1

    return age_by_gender, gender_dist


def get_system_info(yolo_model_override: str | None = None) -> dict:
    """Return device and model info for the current environment."""
    from app.ml.device import DEVICE, FRAME_SKIP, MIVOLO_BATCH_SIZE, YOLO_BATCH_SIZE, YOLO_MODEL

    effective_model = yolo_model_override or YOLO_MODEL
    info: dict = {
        "device": DEVICE,
        "yolo_model": effective_model.replace(".pt", ""),
        "demographics_model": "MiVOLO v2",
        "yolo_batch_size": YOLO_BATCH_SIZE,
        "mivolo_batch_size": MIVOLO_BATCH_SIZE,
        "frame_skip": FRAME_SKIP,
    }

    if DEVICE.startswith("cuda"):
        import torch
        from app.ml.device import _cuda_device_index
        idx = _cuda_device_index(DEVICE)
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        info["gpu_vram_gb"] = round(vram, 1)
    return info


def run_pipeline(
    video_path: str,
    annotated_output_path: str | None = None,
    on_progress: ProgressCallback = _noop_progress,
    yolo_model: str | None = None,
    frame_skip: int | None = None,
    yolo_batch_size: int | None = None,
    mivolo_batch_size: int | None = None,
    max_crops: int | None = None,
) -> dict:
    """Full ML pipeline: detect -> track -> demographics -> annotate -> aggregate."""
    on_progress("detection", 0, "Starting person detection (YOLO)")

    logger.info("Starting detection on %s", video_path)
    frame_detections = detect_persons(
        video_path,
        yolo_model=yolo_model,
        frame_skip=frame_skip,
        yolo_batch_size=yolo_batch_size,
    )
    logger.info("Detection complete: %d frames processed", len(frame_detections))
    on_progress("detection", 100, f"{len(frame_detections)} frames processed")

    on_progress("tracking", 0, "Assigning track IDs (ByteTrack)")
    logger.info("Starting tracking")
    tracked_frames = track_persons(frame_detections)
    all_ids: set[int] = set()
    for tf in tracked_frames:
        for tid in tf.tracker_ids:
            all_ids.add(int(tid))
    logger.info("Tracking complete: %d unique people", len(all_ids))
    on_progress("tracking", 100, f"{len(all_ids)} unique people found")

    on_progress("demographics", 0, "Analyzing age & gender (MiVOLO v2)")
    logger.info("Starting demographics analysis")
    demo_results = analyze_demographics(
        video_path, tracked_frames,
        batch_size=mivolo_batch_size,
        max_crops_per_track=max_crops,
    )
    logger.info("Demographics complete: %d/%d analyzed", len(demo_results), len(all_ids))
    on_progress("demographics", 100, f"{len(demo_results)}/{len(all_ids)} analyzed")

    if annotated_output_path:
        on_progress("annotation", 0, "Generating annotated video")
        from app.ml.annotator import generate_annotated_video
        logger.info("Generating annotated video")
        generate_annotated_video(
            video_path, annotated_output_path, demo_results,
            frame_skip=frame_skip, yolo_model=yolo_model,
        )
        logger.info("Annotated video saved")
        on_progress("annotation", 100, "Annotated video ready")

    on_progress("aggregation", 0, "Computing statistics")
    foot_traffic = _aggregate_foot_traffic(tracked_frames, demo_results)
    age_by_gender, gender_distribution = _aggregate_demographics(demo_results)

    # Per-person list for frontend detail view
    persons = []
    for tid in sorted(all_ids):
        demo = demo_results.get(tid)
        entry: dict = {"track_id": tid}
        if demo:
            entry.update({
                "age": demo.age,
                "age_group": demo.age_group,
                "gender": demo.gender,
                "confidence": round(demo.confidence, 3),
            })
        else:
            entry.update({"age": None, "age_group": None, "gender": "unknown", "confidence": 0})
        persons.append(entry)

    on_progress("aggregation", 100, "Done")

    return {
        "total_unique": len(all_ids),
        "total_analyzed": len(demo_results),
        "foot_traffic": foot_traffic,
        "age_distribution": age_by_gender,
        "gender_distribution": gender_distribution,
        "persons": persons,
        "pipeline_config": get_system_info(yolo_model),
    }
