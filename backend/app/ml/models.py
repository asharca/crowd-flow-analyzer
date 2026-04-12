"""Available detection models and their metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str          # e.g. "yolo11m"
    filename: str    # e.g. "yolo11m.pt"
    name: str        # display name
    family: str      # YOLO11, YOLOv8, etc.
    size: str        # nano, small, medium, large, xlarge
    params_m: float  # millions of parameters
    map50_95: float  # COCO val mAP
    recommended: bool = False


# Ordered by family then size. Only include practical choices.
MODEL_REGISTRY: list[ModelInfo] = [
    # YOLO11 — latest stable, recommended
    ModelInfo("yolo11n", "yolo11n.pt", "YOLO11 Nano",   "YOLO11", "nano",   2.6, 39.5),
    ModelInfo("yolo11s", "yolo11s.pt", "YOLO11 Small",  "YOLO11", "small",  9.4, 47.0),
    ModelInfo("yolo11m", "yolo11m.pt", "YOLO11 Medium", "YOLO11", "medium", 20.1, 51.5, recommended=True),
    ModelInfo("yolo11l", "yolo11l.pt", "YOLO11 Large",  "YOLO11", "large",  25.3, 53.4),
    ModelInfo("yolo11x", "yolo11x.pt", "YOLO11 XLarge", "YOLO11", "xlarge", 56.9, 54.7),
    # YOLOv10 — NMS-free, efficient
    ModelInfo("yolov10n", "yolov10n.pt", "YOLOv10 Nano",   "YOLOv10", "nano",   2.3, 38.5),
    ModelInfo("yolov10s", "yolov10s.pt", "YOLOv10 Small",  "YOLOv10", "small",  7.2, 46.3),
    ModelInfo("yolov10m", "yolov10m.pt", "YOLOv10 Medium", "YOLOv10", "medium", 15.4, 51.1),
    ModelInfo("yolov10l", "yolov10l.pt", "YOLOv10 Large",  "YOLOv10", "large",  24.4, 53.2),
    ModelInfo("yolov10x", "yolov10x.pt", "YOLOv10 XLarge", "YOLOv10", "xlarge", 29.5, 54.4),
    # YOLOv9
    ModelInfo("yolov9t", "yolov9t.pt", "YOLOv9 Tiny",    "YOLOv9", "nano",   2.0, 38.3),
    ModelInfo("yolov9s", "yolov9s.pt", "YOLOv9 Small",   "YOLOv9", "small",  7.2, 46.8),
    ModelInfo("yolov9m", "yolov9m.pt", "YOLOv9 Medium",  "YOLOv9", "medium", 20.1, 51.4),
    ModelInfo("yolov9e", "yolov9e.pt", "YOLOv9 Extended", "YOLOv9", "xlarge", 58.1, 55.6),
    # YOLOv8 — proven stable
    ModelInfo("yolov8n", "yolov8n.pt", "YOLOv8 Nano",   "YOLOv8", "nano",   3.2, 37.3),
    ModelInfo("yolov8s", "yolov8s.pt", "YOLOv8 Small",  "YOLOv8", "small",  11.2, 44.9),
    ModelInfo("yolov8m", "yolov8m.pt", "YOLOv8 Medium", "YOLOv8", "medium", 25.9, 50.2),
    ModelInfo("yolov8l", "yolov8l.pt", "YOLOv8 Large",  "YOLOv8", "large",  43.7, 52.9),
    ModelInfo("yolov8x", "yolov8x.pt", "YOLOv8 XLarge", "YOLOv8", "xlarge", 68.2, 53.9),
]

_BY_ID: dict[str, ModelInfo] = {m.id: m for m in MODEL_REGISTRY}


def get_model(model_id: str) -> ModelInfo:
    """Look up a model by ID. Falls back to yolo11m if not found."""
    return _BY_ID.get(model_id, _BY_ID["yolo11m"])


def get_default_model_id(device: str) -> str:
    """Return the default model ID based on device capability."""
    if device.startswith("cuda"):
        return "yolo11m"
    return "yolo11n"


def list_models() -> list[dict]:
    """Return all models as dicts for API response."""
    return [
        {
            "id": m.id,
            "name": m.name,
            "family": m.family,
            "size": m.size,
            "params_m": m.params_m,
            "map50_95": m.map50_95,
            "recommended": m.recommended,
        }
        for m in MODEL_REGISTRY
    ]
