from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from .config import get_settings
from .database import SessionLocal, init_db
from .models import Project
from .project_types import ProjectStatus
from .routers import auth, projects, users


async def recover_stale_embedding_jobs() -> None:
    """Make embedding jobs retryable when the process that owned them restarted."""
    async with SessionLocal() as db:
        result = await db.execute(select(Project).where(
            Project.status == ProjectStatus.EMBEDDING,
        ).with_for_update())
        stale_projects = list(result.scalars().all())
        for project in stale_projects:
            project.status = ProjectStatus.PROCESSED
            project.status_message = "Previous embedding job stopped before completion. Rebuild the AI index to retry."
        if stale_projects:
            await db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await recover_stale_embedding_jobs()
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=True, allow_methods=["GET", "POST", "PATCH", "OPTIONS"], allow_headers=["Authorization", "Content-Type"])
app.include_router(auth.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(projects.router, prefix="/api")


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}
