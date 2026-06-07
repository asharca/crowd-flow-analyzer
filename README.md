# Crowd Flow Analyzer

Upload a video, get back crowd analytics: unique person count, foot traffic over time, and age/gender demographics — powered by YOLO11 detection and MiVOLO v2 estimation.

## Features

- **Video upload & processing** — drag-drop upload, async processing via Celery
- **Real-time progress** — live updates streamed from Redis while the pipeline runs
- **Person tracking** — multi-object tracking across frames, deduplicates head counts
- **Demographics** — per-person age and gender estimation (MiVOLO v2)
- **Annotated video** — watch the original or the bbox-annotated output
- **Analytics dashboard** — foot traffic chart, age/gender distribution, per-person table
- **GPU acceleration** — auto-selects CUDA / MPS / CPU; tunable batch sizes

## Architecture

```
frontend (React + Vite)
    │  REST + polling
    ▼
backend (FastAPI)  ──── SQLite (videos + results)
    │                   /app/storage/uploads
    │ Celery task
    ▼
worker (Celery)  ──── Redis (task queue + progress)
    │
    ▼
ML pipeline
  ├── YOLO11       — person detection
  ├── Tracker      — multi-object tracking (Supervision)
  ├── MiVOLO v2    — age / gender estimation
  └── Annotator    — renders bboxes onto output video
```

## Quick Start (Docker)

**Prerequisites:** Docker with Compose, NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
git clone https://github.com/asharca/crowd-flow-analyzer.git
cd crowd-flow-analyzer
docker compose up
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

> **First run**: the backend downloads YOLO11 (~44 MB) and MiVOLO v2 (~3 GB) model weights into a persistent volume. Subsequent starts skip this step.

## Development Setup

**Backend**

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

**Worker** (separate terminal)

```bash
cd backend
uv run celery -A app.worker.celery_app worker --loglevel=info --pool=solo
```

**Frontend** (separate terminal)

```bash
cd frontend
bun install
bun run dev
```

Requires a running Redis instance (`docker run -p 6379:6379 redis:7-alpine`).

## Configuration

All settings are environment variables with sensible defaults.

| Variable | Default | Description |
|---|---|---|
| `CFA_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CFA_CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins (JSON array) |
| `CFA_DATABASE_URL` | SQLite in `storage/db/` | SQLAlchemy database URL |
| `CFA_ML_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `CFA_MAX_UPLOAD_SIZE_MB` | `500` | Max video file size |
| `CFA_FRAME_SKIP` | `0` (auto) | Process every Nth frame; 0 = 1 on GPU, 3 on CPU |
| `CFA_YOLO_BATCH_SIZE` | `0` (auto) | YOLO inference batch size |
| `CFA_MIVOLO_BATCH_SIZE` | `0` (auto) | MiVOLO inference batch size |
| `CFA_DEMOGRAPHICS_WORKERS` | `0` (auto) | CPU threads for demographics |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU index for the worker container |

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/videos/upload` | Upload video (multipart/form-data) |
| `GET` | `/api/videos` | List videos |
| `GET` | `/api/videos/{id}` | Video metadata |
| `DELETE` | `/api/videos/{id}/delete` | Delete video and results |
| `GET` | `/api/videos/{id}/stream` | Stream original video |
| `GET` | `/api/videos/{id}/stream/annotated` | Stream annotated video |
| `GET` | `/api/videos/{id}/progress` | Processing progress (Redis) |
| `GET` | `/api/videos/{id}/analytics` | Full analytics results |
| `GET` | `/api/videos/{id}/analytics/foot-traffic` | Foot traffic time series |
| `GET` | `/api/videos/{id}/analytics/demographics` | Age / gender breakdown |
| `GET` | `/api/videos/models` | Available detection models |

## Scaling

To process multiple videos in parallel, scale the worker and switch to PostgreSQL:

```bash
# Scale workers
docker compose up --scale worker=3

# Use PostgreSQL instead of SQLite
CFA_DATABASE_URL=postgresql://user:pass@host:5432/cfa docker compose up
```

The worker uses `--pool=solo` (one task at a time per process) to keep SQLite safe. With PostgreSQL, you can switch to `--pool=prefork`.
