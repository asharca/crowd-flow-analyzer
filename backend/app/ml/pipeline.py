import logging
import math

from app.ml.demographics import DemographicResult, analyze_demographics
from app.ml.detector import detect_persons
from app.ml.tracker import TrackedFrame, track_persons

logger = logging.getLogger(__name__)


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


def run_pipeline(video_path: str, annotated_output_path: str | None = None) -> dict:
    """Full ML pipeline: detect -> track -> demographics -> annotate -> aggregate."""
    logger.info("Starting detection on %s", video_path)
    frame_detections = detect_persons(video_path)
    logger.info("Detection complete: %d frames processed", len(frame_detections))

    logger.info("Starting tracking")
    tracked_frames = track_persons(frame_detections)
    all_ids: set[int] = set()
    for tf in tracked_frames:
        for tid in tf.tracker_ids:
            all_ids.add(int(tid))
    logger.info("Tracking complete: %d unique people", len(all_ids))

    logger.info("Starting demographics analysis")
    demo_results = analyze_demographics(video_path, tracked_frames)
    logger.info("Demographics complete: %d/%d analyzed", len(demo_results), len(all_ids))

    # Generate annotated video
    if annotated_output_path:
        from app.ml.annotator import generate_annotated_video

        logger.info("Generating annotated video")
        generate_annotated_video(video_path, annotated_output_path, demo_results)
        logger.info("Annotated video saved")

    foot_traffic = _aggregate_foot_traffic(tracked_frames, demo_results)
    age_by_gender, gender_distribution = _aggregate_demographics(demo_results)

    return {
        "total_unique": len(all_ids),
        "total_analyzed": len(demo_results),
        "foot_traffic": foot_traffic,
        "age_distribution": age_by_gender,
        "gender_distribution": gender_distribution,
    }
