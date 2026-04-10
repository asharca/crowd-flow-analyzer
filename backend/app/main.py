from contextlib import asynccontextmanager

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run migrations on startup
    alembic_cfg = Config(str(settings.base_dir / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(settings.base_dir / "alembic"))
    command.upgrade(alembic_cfg, "head")
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api import analytics, videos  # noqa: E402

app.include_router(videos.router, prefix="/api/videos", tags=["videos"])
app.include_router(analytics.router, prefix="/api/videos", tags=["analytics"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
