from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Crowd Flow Analyzer"
    debug: bool = False

    # Storage
    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: Path = base_dir / "storage" / "uploads"
    db_dir: Path = base_dir / "storage" / "db"

    # Database
    database_url: str = ""

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = ""
    celery_result_backend: str = ""

    # Upload limits
    max_upload_size_mb: int = 500
    allowed_extensions: set[str] = {"mp4", "avi", "mov", "mkv"}

    # ML pipeline
    frame_skip: int = 3
    yolo_model: str = "yolov8n.pt"

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = {"env_prefix": "CFA_", "env_file": ".env"}

    def model_post_init(self, __context: object) -> None:
        if not self.database_url:
            self.database_url = f"sqlite:///{self.db_dir / 'crowd_flow.db'}"
        if not self.celery_broker_url:
            self.celery_broker_url = self.redis_url
        if not self.celery_result_backend:
            self.celery_result_backend = self.redis_url
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
