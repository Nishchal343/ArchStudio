import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, Response, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import SessionLocal, get_db
from ..dependencies import get_current_user
from ..models import Project, ProjectAnalysis, ProjectArtifact, ProjectSourceType, ProjectStatus
from ..config import get_settings
from ..schemas import (AnalysisResult, AnalysisStartResponse, AnalysisStatusResponse, GitHubProjectRequest,
                       ProcessingStartResponse, ProjectCreateResponse, RetrievalCallbackRequest,
                       RetrievalStartResponse, SearchRequest, SearchResponse, SearchResult,
                       ArtifactStartResponse, ArtifactStatusResponse, ArtifactUpdateRequest, ChatRequest, ChatResponse,
                       EditableDocument, DiagramPayload)
from ..project_types import AnalysisStatus, ArtifactKind, ArtifactStatus
from ..services.github_service import GitHubNotFoundError, GitHubPrivateError, GitHubValidationError, validate_public_repository
from ..services.n8n_service import N8nNotConfiguredError, N8nResponseError, N8nTimeoutError, N8nUnavailableError, RetrievalRequest, trigger_github_retrieval
from ..services.n8n_auth import valid_webhook_secret
from ..services.code_processing import ProcessingFatalError, process_project_files
from ..services.project_service import get_owned_project
from ..services.storage_service import StorageValidationError, extract_zip_archive, project_directory, remove_project_directory, save_upload_to_temp, validate_zip_archive
from ..services.rag.ingestion import EmbeddingIngestionError, ingest_project
from ..services.rag.retrieval import search_project
from ..services.embeddings.base import EmbeddingProviderError
from ..services.analysis.service import AnalysisServiceError, analyze_project
from ..services.llm.base import LLMProviderError
from ..services.artifacts.reports import generate_report
from ..services.artifacts.diagrams import architecture_diagram, database_diagram, enrich_diagram_explanation, render_diagram
from ..services.artifacts.export import export_document
from ..services.chat.service import answer_project_question
from ..services.literature_review import generate_literature_review

router = APIRouter(prefix="/projects", tags=["projects"])
logger = logging.getLogger(__name__)
EMBEDDING_ESTIMATE_SECONDS_PER_CHUNK = 6.5
STALE_EMBEDDING_MINUTES = 20


@router.post("/{project_id}/literature-review")
async def literature_review(project_id: UUID, urls: str = Form("[]"), files: list[UploadFile] = File(default=[]),
                            current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    project = await get_owned_project(db, project_id, current_user["id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.status != ProjectStatus.EMBEDDED:
        raise HTTPException(status_code=409, detail="Project must be embedded before literature comparison.")
    try:
        parsed_urls = json.loads(urls)
        if not isinstance(parsed_urls, list):
            raise ValueError("URLs must be a JSON array.")
        return await generate_literature_review(db, project_id, [str(url) for url in parsed_urls if str(url).strip()], files)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Literature review failed")
        raise HTTPException(status_code=422, detail=str(error)) from error


def initial_embedding_status(chunks_path: Path) -> str:
    try:
        total_chunks = sum(1 for line in chunks_path.open("r", encoding="utf-8") if line.strip())
    except OSError:
        total_chunks = 0
    if not total_chunks:
        return "Embedding generation started. Estimating time remaining…"
    seconds = max(60, round(total_chunks * EMBEDDING_ESTIMATE_SECONDS_PER_CHUNK))
    minutes = max(1, round(seconds / 60))
    return f"Embedding generation started. Your project should be ready in about {minutes} minutes."


def project_name(value: str | None, filename: str | None = None) -> str:
    if value:
        cleaned = " ".join(value.split())
        if cleaned:
            return cleaned
    safe_filename = (filename or "").replace("\\", "/")
    stem = Path(safe_filename).stem.strip()
    return stem or "Untitled project"


async def recover_stale_embedding_project(project: Project, db: AsyncSession) -> bool:
    if project.status != ProjectStatus.EMBEDDING or not project.updated_at:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_EMBEDDING_MINUTES)
    updated_at = project.updated_at
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if updated_at >= cutoff:
        return False
    project.status = ProjectStatus.PROCESSED
    project.status_message = "Previous embedding job stopped before completion. Rebuild the AI index to retry."
    await db.commit()
    logger.warning("Recovered stale embedding project_id=%s", project.id)
    return True


def upload_error(error: StorageValidationError) -> HTTPException:
    if "exceeds the maximum allowed size" in str(error):
        return HTTPException(status_code=413, detail=str(error))
    return HTTPException(status_code=400, detail=str(error))


@router.post("/upload", response_model=ProjectCreateResponse, status_code=201, summary="Create a project from a ZIP archive")
async def upload_project(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    temporary_archive = None
    project: Project | None = None
    try:
        temporary_archive, _ = await save_upload_to_temp(file)
        validate_zip_archive(temporary_archive)
        project = Project(
            id=uuid.uuid4(),
            user_id=current_user["id"],
            name=project_name(name, file.filename),
            source_type=ProjectSourceType.ZIP,
            original_filename=Path((file.filename or "upload.zip").replace("\\", "/")).name,
            status=ProjectStatus.VALIDATING,
        )
        project.storage_path = str(project_directory(str(project.id)))
        db.add(project)
        await db.commit()
        await db.refresh(project)
        extract_zip_archive(temporary_archive, project_directory(str(project.id)))
        project.status = ProjectStatus.READY
        await db.commit()
        await db.refresh(project)
        return project
    except StorageValidationError as error:
        if project:
            await db.rollback()
            await db.execute(delete(Project).where(Project.id == project.id))
            await db.commit()
            remove_project_directory(str(project.id))
        raise upload_error(error)
    except Exception:
        if project:
            await db.rollback()
            project.status = ProjectStatus.FAILED
            db.add(project)
            await db.commit()
        raise HTTPException(status_code=500, detail="Project upload could not be completed")
    finally:
        if temporary_archive:
            temporary_archive.unlink(missing_ok=True)


@router.post("/github", response_model=ProjectCreateResponse, status_code=201, summary="Create a project from a public GitHub repository")
async def create_github_project(
    payload: GitHubProjectRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        normalized_url, repository_name = await validate_public_repository(payload.url)
    except GitHubNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except GitHubPrivateError as error:
        raise HTTPException(status_code=403, detail=str(error))
    except GitHubValidationError as error:
        message = str(error)
        code = 422 if message.startswith("Invalid GitHub repository URL") else 400
        raise HTTPException(status_code=code, detail=message)

    project = Project(
        id=uuid.uuid4(),
        user_id=current_user["id"],
        name=project_name(payload.name, repository_name),
        source_type=ProjectSourceType.GITHUB,
        source_url=normalized_url,
        status=ProjectStatus.CREATED,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


def retrieval_failure(project: Project, db: AsyncSession, message: str) -> None:
    project.status = ProjectStatus.FAILED
    project.status_message = message


async def process_project_in_background(project_id: str) -> None:
    started = asyncio.get_running_loop().time()
    async with SessionLocal() as db:
        try:
            manifest = await asyncio.to_thread(process_project_files, project_id)
            result = await db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
            project = result.scalar_one_or_none()
            if not project:
                return
            project.status = ProjectStatus.PROCESSED
            project.status_message = "Codebase processing completed."
            await db.commit()
            logger.info(
                "Codebase processing completed project_id=%s processed_files=%s skipped_files=%s total_chunks=%s duration_seconds=%.3f",
                project_id, manifest["processed_files"], manifest["skipped_files"], manifest["total_chunks"],
                asyncio.get_running_loop().time() - started,
            )
        except ProcessingFatalError:
            result = await db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
            project = result.scalar_one_or_none()
            if project:
                project.status = ProjectStatus.FAILED
                project.status_message = "Codebase processing failed."
                await db.commit()
            logger.exception("Codebase processing failed project_id=%s", project_id)
        except Exception:
            await db.rollback()
            result = await db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
            project = result.scalar_one_or_none()
            if project:
                project.status = ProjectStatus.FAILED
                project.status_message = "Codebase processing failed."
                await db.commit()
            logger.exception("Unexpected codebase processing failure project_id=%s", project_id)


async def embed_project_in_background(project_id: str) -> None:
    started = asyncio.get_running_loop().time()
    try:
        async def report_progress(stats) -> None:
            completed = stats.embedded_chunks + stats.skipped_chunks
            if not stats.total_chunks or not completed:
                return
            elapsed = asyncio.get_running_loop().time() - started
            remaining = max(0, stats.total_chunks - completed)
            estimate_seconds = int(elapsed * remaining / completed)
            if estimate_seconds >= 60:
                estimate = f"about {max(1, round(estimate_seconds / 60))} min remaining"
            else:
                estimate = f"about {max(1, estimate_seconds)} sec remaining"
            async with SessionLocal() as progress_db:
                result = await progress_db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
                progress_project = result.scalar_one_or_none()
                if progress_project and progress_project.status == ProjectStatus.EMBEDDING:
                    progress_project.status_message = f"Embedding {completed}/{stats.total_chunks} chunks; {estimate}."
                    await progress_db.commit()

        # The worker owns this session. It never inherits the request-scoped
        # session, and the context closes it even when embedding or PostgreSQL fails.
        async with SessionLocal() as db:
            stats = await ingest_project(db, uuid.UUID(project_id), progress=report_progress)
        async with SessionLocal() as db:
            result = await db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
            project = result.scalar_one_or_none()
            if project:
                project.status = ProjectStatus.EMBEDDED
                project.status_message = "Embedding index completed."
                await db.commit()
        logger.info("Embedding job completed project_id=%s embedded_chunks=%s extraction_seconds=%.3f chunking_seconds=%.3f embedding_seconds=%.3f pgvector_seconds=%.3f duration_seconds=%.3f batch_size=%s concurrency=%s throughput_chunks_per_second=%.3f",
                    project_id, stats.embedded_chunks, stats.extraction_seconds, stats.chunking_seconds,
                    stats.embedding_seconds, stats.pgvector_seconds, asyncio.get_running_loop().time() - started,
                    stats.batch_size, stats.concurrency, stats.throughput_chunks_per_second)
    except EmbeddingIngestionError as error:
        try:
            async with SessionLocal() as db:
                result = await db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
                project = result.scalar_one_or_none()
                if project:
                    project.status = ProjectStatus.FAILED
                    if isinstance(error.__cause__, EmbeddingProviderError):
                        project.status_message = "Local embedding service is unavailable."
                    else:
                        project.status_message = "Embedding index failed. Retry after reviewing processed output."
                    await db.commit()
        except Exception:
            logger.exception("Could not persist embedding failure status project_id=%s", project_id)
        logger.exception("Embedding job failed project_id=%s", project_id)
    except Exception:
        try:
            async with SessionLocal() as db:
                result = await db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
                project = result.scalar_one_or_none()
                if project:
                    project.status = ProjectStatus.FAILED
                    project.status_message = "Embedding index failed."
                    await db.commit()
        except Exception:
            logger.exception("Could not persist unexpected embedding failure status project_id=%s", project_id)
        logger.exception("Unexpected embedding failure project_id=%s", project_id)


async def analyze_project_in_background(analysis_id: str) -> None:
    async with SessionLocal() as db:
        try:
            result = await db.execute(select(ProjectAnalysis).where(ProjectAnalysis.id == uuid.UUID(analysis_id)))
            analysis = result.scalar_one_or_none()
            if not analysis:
                return
            analysis.status = AnalysisStatus.ANALYZING
            await db.commit()
            result = await db.execute(select(ProjectAnalysis).where(ProjectAnalysis.id == uuid.UUID(analysis_id)))
            analysis = result.scalar_one_or_none()
            if not analysis:
                return
            generated = await analyze_project(db, analysis.project_id)
            analysis.result = generated.model_dump(mode="json")
            analysis.status = AnalysisStatus.COMPLETED
            analysis.error_message = None
            await db.commit()
            logger.info("Project analysis completed project_id=%s analysis_id=%s", analysis.project_id, analysis_id)
        except (AnalysisServiceError, LLMProviderError) as error:
            await db.rollback()
            result = await db.execute(select(ProjectAnalysis).where(ProjectAnalysis.id == uuid.UUID(analysis_id)))
            analysis = result.scalar_one_or_none()
            if analysis:
                analysis.status = AnalysisStatus.FAILED
                if isinstance(error.__cause__, LLMProviderError) or isinstance(error, LLMProviderError):
                    analysis.error_message = "Local LLM service is unavailable."
                else:
                    analysis.error_message = "Project analysis failed."
                await db.commit()
            logger.exception("Project analysis failed analysis_id=%s", analysis_id)
        except Exception:
            await db.rollback()
            result = await db.execute(select(ProjectAnalysis).where(ProjectAnalysis.id == uuid.UUID(analysis_id)))
            analysis = result.scalar_one_or_none()
            if analysis:
                analysis.status = AnalysisStatus.FAILED
                analysis.error_message = "Project analysis failed."
                await db.commit()
            logger.exception("Unexpected project analysis failure analysis_id=%s", analysis_id)


@router.post("/{project_id}/embed", response_model=ProcessingStartResponse, status_code=202, summary="Build the project's vector index")
async def embed_project(project_id: UUID, background_tasks: BackgroundTasks,
                        current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).where(Project.id == project_id, Project.user_id == current_user["id"]).with_for_update())
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    chunks_path = project_directory(str(project.id)) / "processed" / "chunks" / "chunks.jsonl"
    if not chunks_path.is_file():
        raise HTTPException(status_code=422, detail="Project has no processed Step 4 output.")
    if project.status == ProjectStatus.EMBEDDING:
        raise HTTPException(status_code=409, detail="Project embedding is already in progress.")
    if project.status not in {ProjectStatus.PROCESSED, ProjectStatus.EMBEDDED, ProjectStatus.FAILED}:
        raise HTTPException(status_code=409, detail="Project must be processed before embedding.")
    project.status = ProjectStatus.EMBEDDING
    project.status_message = initial_embedding_status(chunks_path)
    await db.commit()
    background_tasks.add_task(embed_project_in_background, str(project.id))
    logger.info("Embedding job started project_id=%s model=%s device=%s", project.id,
                get_settings().embedding_model, get_settings().embedding_device)
    return ProcessingStartResponse(project_id=project.id, status=ProjectStatus.EMBEDDING, message="Embedding generation started.")


@router.post("/{project_id}/search", response_model=SearchResponse, summary="Search the project's indexed code")
async def search_project_endpoint(project_id: UUID, payload: SearchRequest,
                                  current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    project = await get_owned_project(db, project_id, current_user["id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.status != ProjectStatus.EMBEDDED:
        raise HTTPException(status_code=409, detail="Project is not indexed for search.")
    settings = get_settings()
    if payload.top_k > settings.rag_max_top_k:
        raise HTTPException(status_code=422, detail="top_k exceeds the configured maximum.")
    if len(payload.query) > settings.rag_max_query_length:
        raise HTTPException(status_code=422, detail="Query is too long.")
    try:
        results = await search_project(db, project_id, payload.query, payload.top_k, payload.min_similarity)
    except (EmbeddingProviderError, EmbeddingIngestionError) as error:
        raise HTTPException(status_code=502, detail=str(error))
    return SearchResponse(project_id=project_id, query=payload.query,
                          results=[SearchResult(**result) for result in results])


@router.post("/{project_id}/analyze", response_model=AnalysisStartResponse, status_code=202,
             summary="Queue grounded local AI project analysis")
async def analyze_project_endpoint(project_id: UUID, background_tasks: BackgroundTasks,
                                   current_user: dict = Depends(get_current_user),
                                   db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).where(
        Project.id == project_id, Project.user_id == current_user["id"]
    ).with_for_update())
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.status != ProjectStatus.EMBEDDED:
        raise HTTPException(status_code=409, detail="Project must be embedded before analysis.")
    active = await db.execute(select(ProjectAnalysis).where(
        ProjectAnalysis.project_id == project_id,
        ProjectAnalysis.status.in_([AnalysisStatus.QUEUED, AnalysisStatus.ANALYZING]),
    ))
    if active.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Project analysis is already in progress.")
    analysis = ProjectAnalysis(project_id=project_id, status=AnalysisStatus.QUEUED)
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)
    background_tasks.add_task(analyze_project_in_background, str(analysis.id))
    logger.info("Project analysis started project_id=%s analysis_id=%s model=%s",
                project_id, analysis.id, get_settings().groq_model)
    return AnalysisStartResponse(project_id=project_id, analysis_id=analysis.id,
                                 status=AnalysisStatus.QUEUED.value,
                                 message="Project analysis started.")


@router.get("/{project_id}/analysis/{analysis_id}", response_model=AnalysisStatusResponse,
            summary="Get project analysis status or result")
async def get_analysis(project_id: UUID, analysis_id: UUID,
                       current_user: dict = Depends(get_current_user),
                       db: AsyncSession = Depends(get_db)):
    project = await get_owned_project(db, project_id, current_user["id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    result = await db.execute(select(ProjectAnalysis).where(
        ProjectAnalysis.id == analysis_id, ProjectAnalysis.project_id == project_id
    ))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    parsed = AnalysisResult.model_validate(analysis.result) if analysis.result else None
    return AnalysisStatusResponse(project_id=project_id, analysis_id=analysis_id,
                                  status=analysis.status.value, error=analysis.error_message,
                                  analysis=parsed)


async def generate_artifact_in_background(artifact_id: str) -> None:
    async with SessionLocal() as db:
        try:
            result = await db.execute(select(ProjectArtifact).where(ProjectArtifact.id == uuid.UUID(artifact_id)))
            artifact = result.scalar_one_or_none()
            if not artifact:
                return
            artifact.status = ArtifactStatus.GENERATING
            await db.commit()
            started = asyncio.get_running_loop().time()
            logger.info("Artifact generation started artifact_id=%s kind=%s project_id=%s",
                        artifact.id, artifact.kind.value, artifact.project_id)
            if artifact.kind in {ArtifactKind.REPORT, ArtifactKind.IEEE_REPORT, ArtifactKind.RESEARCH_PAPER}:
                payload = (await generate_report(db, artifact.project_id, artifact.kind.value)).model_dump(mode="json")
            elif artifact.kind == ArtifactKind.DATABASE_DIAGRAM:
                diagram = await enrich_diagram_explanation(database_diagram(artifact.project_id))
                diagram.mermaid = f"%% ARCHSTUDIO_EXPLANATION {diagram.explanation.replace(chr(10), ' ')}\n{diagram.mermaid}"
                payload = diagram.model_dump(mode="json")
            else:
                diagram = await architecture_diagram(db, artifact.project_id)
                diagram.mermaid = f"%% ARCHSTUDIO_EXPLANATION {diagram.explanation.replace(chr(10), ' ')}\n{diagram.mermaid}"
                payload = diagram.model_dump(mode="json")
            artifact.payload = payload
            artifact.status = ArtifactStatus.COMPLETED
            artifact.error_message = None
            await db.commit()
            logger.info("Artifact generation completed artifact_id=%s kind=%s duration_seconds=%.3f",
                        artifact.id, artifact.kind.value,
                        asyncio.get_running_loop().time() - started)
        except Exception as error:
            await db.rollback()
            result = await db.execute(select(ProjectArtifact).where(ProjectArtifact.id == uuid.UUID(artifact_id)))
            artifact = result.scalar_one_or_none()
            if artifact:
                artifact.status = ArtifactStatus.FAILED
                artifact.error_message = artifact_generation_error_message(error)
                await db.commit()
            logger.exception("Artifact generation failed artifact_id=%s", artifact_id)


def artifact_generation_error_message(error: Exception) -> str:
    """Expose actionable provider failures without leaking internal details."""
    cause = error
    while cause:
        if isinstance(cause, LLMProviderError):
            return str(cause)
        if isinstance(cause, EmbeddingProviderError):
            return str(cause)
        cause = cause.__cause__
    if isinstance(error, AnalysisServiceError):
        return "Grounded project analysis could not be generated."
    return "Artifact generation failed. Please try again."


async def queue_artifact(project_id: UUID, kind: ArtifactKind, background_tasks: BackgroundTasks,
                         current_user: dict, db: AsyncSession) -> ArtifactStartResponse:
    result = await db.execute(select(Project).where(Project.id == project_id, Project.user_id == current_user["id"]).with_for_update())
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.status != ProjectStatus.EMBEDDED:
        raise HTTPException(status_code=409, detail="Project must be embedded before generation.")
    active = await db.execute(select(ProjectArtifact).where(
        ProjectArtifact.project_id == project_id, ProjectArtifact.kind == kind,
        ProjectArtifact.status.in_([ArtifactStatus.QUEUED, ArtifactStatus.GENERATING])))
    if active.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="This artifact is already being generated.")
    artifact = ProjectArtifact(project_id=project_id, kind=kind, status=ArtifactStatus.QUEUED)
    db.add(artifact)
    await db.commit()
    await db.refresh(artifact)
    background_tasks.add_task(generate_artifact_in_background, str(artifact.id))
    return ArtifactStartResponse(project_id=project_id, artifact_id=artifact.id, kind=kind,
                                 status=ArtifactStatus.QUEUED.value, message="Artifact generation started.")


@router.post("/{project_id}/reports", response_model=ArtifactStartResponse, status_code=202)
async def create_report(project_id: UUID, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    return await queue_artifact(project_id, ArtifactKind.REPORT, background_tasks, current_user, db)


@router.post("/{project_id}/ieee-report", response_model=ArtifactStartResponse, status_code=202)
async def create_ieee_report(project_id: UUID, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    return await queue_artifact(project_id, ArtifactKind.IEEE_REPORT, background_tasks, current_user, db)


@router.post("/{project_id}/research-paper", response_model=ArtifactStartResponse, status_code=202)
async def create_research_paper(project_id: UUID, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    return await queue_artifact(project_id, ArtifactKind.RESEARCH_PAPER, background_tasks, current_user, db)


@router.post("/{project_id}/database-diagram", response_model=ArtifactStartResponse, status_code=202)
async def create_database_diagram(project_id: UUID, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    return await queue_artifact(project_id, ArtifactKind.DATABASE_DIAGRAM, background_tasks, current_user, db)


@router.post("/{project_id}/architecture-diagram", response_model=ArtifactStartResponse, status_code=202)
async def create_architecture_diagram(project_id: UUID, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    return await queue_artifact(project_id, ArtifactKind.ARCHITECTURE_DIAGRAM, background_tasks, current_user, db)


@router.get("/{project_id}/artifacts/{artifact_id}", response_model=ArtifactStatusResponse)
async def get_artifact(project_id: UUID, artifact_id: UUID, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    project = await get_owned_project(db, project_id, current_user["id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    result = await db.execute(select(ProjectArtifact).where(ProjectArtifact.id == artifact_id, ProjectArtifact.project_id == project_id))
    artifact = result.scalar_one_or_none()
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    document = EditableDocument.model_validate(artifact.payload) if artifact.payload and artifact.kind in {ArtifactKind.REPORT, ArtifactKind.IEEE_REPORT, ArtifactKind.RESEARCH_PAPER} else None
    diagram = DiagramPayload.model_validate(artifact.payload) if artifact.payload and artifact.kind in {ArtifactKind.DATABASE_DIAGRAM, ArtifactKind.ARCHITECTURE_DIAGRAM} else None
    return ArtifactStatusResponse(project_id=project_id, artifact_id=artifact.id, kind=artifact.kind,
                                  status=artifact.status.value, error=artifact.error_message, document=document, diagram=diagram)


@router.patch("/{project_id}/artifacts/{artifact_id}", response_model=ArtifactStatusResponse)
async def update_artifact(project_id: UUID, artifact_id: UUID, payload: ArtifactUpdateRequest,
                          current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    project = await get_owned_project(db, project_id, current_user["id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    result = await db.execute(select(ProjectArtifact).where(ProjectArtifact.id == artifact_id, ProjectArtifact.project_id == project_id).with_for_update())
    artifact = result.scalar_one_or_none()
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.status != ArtifactStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Only completed artifacts can be edited.")
    if artifact.kind in {ArtifactKind.REPORT, ArtifactKind.IEEE_REPORT, ArtifactKind.RESEARCH_PAPER}:
        if not payload.document or payload.document.document_type != artifact.kind:
            raise HTTPException(status_code=422, detail="A matching editable document is required.")
        artifact.payload = payload.document.model_dump(mode="json")
    else:
        if not payload.diagram or payload.diagram.diagram_type != artifact.kind:
            raise HTTPException(status_code=422, detail="A matching diagram is required.")
        artifact.payload = payload.diagram.model_dump(mode="json")
    await db.commit()
    return await get_artifact(project_id, artifact_id, current_user, db)


@router.get("/{project_id}/artifacts/{artifact_id}/download")
async def download_artifact(project_id: UUID, artifact_id: UUID, format: str,
                            current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    status = await get_artifact(project_id, artifact_id, current_user, db)
    if status.status != ArtifactStatus.COMPLETED.value:
        raise HTTPException(status_code=409, detail="Artifact is not ready for download.")
    if status.document:
        if format not in {"pdf", "docx"}:
            raise HTTPException(status_code=422, detail="Documents support pdf or docx downloads.")
        content, media_type, extension = export_document(status.document, format)
    elif status.diagram:
        if format not in {"svg", "png", "pdf"}:
            raise HTTPException(status_code=422, detail="Diagrams support svg, png, or pdf downloads.")
        content, media_type, extension = render_diagram(status.diagram, format)
    else:
        raise HTTPException(status_code=422, detail="Artifact has no content.")
    return Response(content=content, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{status.kind.value}.{extension}"'})


@router.post("/{project_id}/chat", response_model=ChatResponse)
async def project_chat(project_id: UUID, payload: ChatRequest, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    project = await get_owned_project(db, project_id, current_user["id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.status != ProjectStatus.EMBEDDED:
        raise HTTPException(status_code=409, detail="Project must be embedded before chat.")
    try:
        return await answer_project_question(db, project_id, payload.query, payload.top_k)
    except (EmbeddingProviderError, LLMProviderError) as error:
        raise HTTPException(status_code=502, detail=str(error))


@router.post("/{project_id}/retrieve", response_model=RetrievalStartResponse, summary="Queue GitHub repository retrieval in n8n")
async def retrieve_github_project(project_id: UUID, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    project = await get_owned_project(db, project_id, current_user["id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.source_type != ProjectSourceType.GITHUB:
        raise HTTPException(status_code=422, detail="Only GitHub projects can start repository retrieval.")
    if not project.source_url:
        raise HTTPException(status_code=422, detail="This GitHub project does not have a valid source URL.")
    if project.status in {ProjectStatus.QUEUED, ProjectStatus.PROCESSING}:
        raise HTTPException(status_code=409, detail="Repository retrieval is already in progress.")

    project.status = ProjectStatus.QUEUED
    project.status_message = "Repository retrieval queued."
    project.storage_path = project.storage_path or str(project_directory(str(project.id)))
    await db.commit()

    request = RetrievalRequest(
        project_id=str(project.id),
        user_id=str(project.user_id),
        source_type=project.source_type.value,
        source_url=project.source_url,
    )
    try:
        await trigger_github_retrieval(request)
    except N8nTimeoutError:
        retrieval_failure(project, db, "Repository retrieval service timed out.")
        await db.commit()
        raise HTTPException(status_code=504, detail="Repository retrieval service timed out. Please try again later.")
    except (N8nNotConfiguredError, N8nResponseError, N8nUnavailableError):
        retrieval_failure(project, db, "Repository retrieval service is unavailable.")
        await db.commit()
        raise HTTPException(status_code=502, detail="Repository retrieval service is unavailable. Please try again later.")

    return RetrievalStartResponse(project_id=project.id, status=project.status, message="Repository retrieval started.")


@router.post("/{project_id}/process", response_model=ProcessingStartResponse, summary="Start safe codebase processing")
async def process_project(
    project_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Project)
        .where(Project.id == project_id, Project.user_id == current_user["id"])
        .with_for_update()
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.status == ProjectStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="Project processing is already in progress.")
    if project.status == ProjectStatus.PROCESSED:
        raise HTTPException(status_code=409, detail="Project has already been processed.")
    if project.status in {ProjectStatus.QUEUED, ProjectStatus.VALIDATING, ProjectStatus.CREATED}:
        raise HTTPException(status_code=409, detail="Project source is not ready for processing.")
    source_directory = project_directory(str(project.id))
    if not source_directory.is_dir():
        project.status = ProjectStatus.FAILED
        project.status_message = "Project source directory is missing."
        await db.commit()
        raise HTTPException(status_code=422, detail="Project source is not available for processing.")

    project.status = ProjectStatus.PROCESSING
    project.status_message = "Codebase processing started."
    await db.commit()
    background_tasks.add_task(process_project_in_background, str(project.id))
    logger.info("Codebase processing started project_id=%s", project.id)
    return ProcessingStartResponse(project_id=project.id, status=ProjectStatus.PROCESSING, message="Codebase processing started.")


def verify_callback_secret(value: str | None) -> None:
    configured = get_settings().n8n_webhook_secret
    if not valid_webhook_secret(value, configured):
        raise HTTPException(status_code=401, detail="Invalid workflow callback authentication")


@router.post("/retrieval/callback", status_code=204, summary="Receive an authenticated n8n retrieval status callback")
async def retrieval_callback(
    payload: RetrievalCallbackRequest,
    callback_secret: str | None = Header(default=None, alias="X-ArchStudio-Webhook-Secret"),
    db: AsyncSession = Depends(get_db),
):
    verify_callback_secret(callback_secret)
    if payload.source_type != ProjectSourceType.GITHUB:
        raise HTTPException(status_code=422, detail="Invalid retrieval source type.")
    if payload.status not in {ProjectStatus.PROCESSING, ProjectStatus.READY, ProjectStatus.FAILED}:
        raise HTTPException(status_code=422, detail="Invalid retrieval callback status.")

    result = await db.execute(
        select(Project).where(
            Project.id == payload.project_id,
            Project.user_id == payload.user_id,
            Project.source_type == ProjectSourceType.GITHUB,
        )
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.status = payload.status
    project.status_message = {
        ProjectStatus.PROCESSING: "Repository retrieval is processing.",
        ProjectStatus.READY: "Repository retrieval completed.",
        ProjectStatus.FAILED: "Repository retrieval failed.",
    }[payload.status]
    await db.commit()


@router.get("", response_model=list[ProjectCreateResponse], summary="List the current user's projects")
async def list_projects(current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).where(Project.user_id == current_user["id"]).order_by(Project.created_at.desc()))
    return list(result.scalars().all())


@router.get("/{project_id}", response_model=ProjectCreateResponse, summary="Get one of the current user's projects")
async def get_project(project_id: UUID, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    project = await get_owned_project(db, project_id, current_user["id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    await recover_stale_embedding_project(project, db)
    return project


@router.get("/{project_id}/artifacts", response_model=list[ArtifactStatusResponse])
async def list_artifacts(project_id: UUID, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    project = await get_owned_project(db, project_id, current_user["id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    result = await db.execute(select(ProjectArtifact).where(ProjectArtifact.project_id == project_id).order_by(ProjectArtifact.updated_at.desc()))
    return [await get_artifact(project_id, artifact.id, current_user, db) for artifact in result.scalars().all()]


@router.delete("/{project_id}", status_code=204, summary="Delete one of the current user's projects")
async def delete_project(project_id: UUID, current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    project = await get_owned_project(db, project_id, current_user["id"])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    await db.delete(project)
    await db.commit()
    remove_project_directory(str(project_id))
