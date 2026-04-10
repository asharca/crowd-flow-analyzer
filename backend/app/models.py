import uuid
from datetime import datetime, timezone

from sqlalchemy import Float, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    original_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="queued")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    annotated_filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, default=_utcnow)
    completed_at: Mapped[str | None] = mapped_column(Text, nullable=True)

    analysis: Mapped["AnalysisResult | None"] = relationship(
        back_populates="video", cascade="all, delete-orphan", uselist=False
    )


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=_uuid)
    video_id: Mapped[str] = mapped_column(
        Text, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    total_unique: Mapped[int] = mapped_column(Integer, nullable=False)
    total_analyzed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    foot_traffic: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    age_distribution: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    gender_distribution: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    processing_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, default=_utcnow)

    video: Mapped["Video"] = relationship(back_populates="analysis")
