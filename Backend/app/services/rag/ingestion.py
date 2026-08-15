import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ...config import get_settings
from ...services.embeddings.base import EmbeddingProvider
from ...services.embeddings.service import get_embedding_provider
from ...services.storage_service import project_directory
if TYPE_CHECKING:
    from ..vector_store.pgvector import PgVectorStore

logger = logging.getLogger(__name__)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class EmbeddingIngestionError(RuntimeError):
    pass


@dataclass
class IngestionStats:
    total_chunks: int = 0
    embedded_chunks: int = 0
    skipped_chunks: int = 0
    failed_chunks: int = 0
    api_batches: int = 0
    duration_seconds: float = 0.0


def normalize_chunk(raw: dict, project_id: str) -> dict:
    if not isinstance(raw, dict):
        raise EmbeddingIngestionError("Malformed chunk record")
    if raw.get("project_id") and str(raw["project_id"]) != project_id:
        raise EmbeddingIngestionError("Chunk project does not match ingestion project")
    file_path = raw.get("file_path")
    index = raw.get("chunk_index")
    content = raw.get("content")
    content_hash = raw.get("content_hash")
    if not isinstance(file_path, str) or not file_path or Path(file_path).is_absolute() or ".." in Path(file_path).parts:
        raise EmbeddingIngestionError("Chunk has an invalid file path")
    if not isinstance(index, int) or index < 0:
        raise EmbeddingIngestionError("Chunk is missing a valid chunk index")
    if not isinstance(content, str):
        raise EmbeddingIngestionError("Chunk is missing content")
    if not isinstance(content_hash, str) or not HASH_RE.fullmatch(content_hash):
        raise EmbeddingIngestionError("Chunk has an invalid content hash")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != content_hash:
        raise EmbeddingIngestionError("Chunk content hash is invalid")
    chunk_id = raw.get("chunk_id") or f"{file_path}:{index}"
    if not isinstance(chunk_id, str) or not chunk_id or len(chunk_id) > 512:
        raise EmbeddingIngestionError("Chunk is missing a valid chunk ID")
    return {"chunk_id": chunk_id, "content_hash": content_hash, "content": content,
            "file_path": file_path, "language": raw.get("language"),
            "start_line": raw.get("start_line"), "end_line": raw.get("end_line")}


async def ingest_project(db: AsyncSession, project_id: UUID, provider: EmbeddingProvider | None = None,
                         store: "PgVectorStore | None" = None) -> IngestionStats:
    settings = get_settings()
    chunks_path = project_directory(str(project_id)) / "processed" / "chunks" / "chunks.jsonl"
    if not chunks_path.is_file():
        raise EmbeddingIngestionError("Processed chunks are not available")
    provider = provider or get_embedding_provider()
    if store is None:
        from ..vector_store.pgvector import PgVectorStore
        store = PgVectorStore()
    stats = IngestionStats()
    started = time.monotonic()
    seen_ids: set[str] = set()
    batch: list[dict] = []
    try:
        with chunks_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    chunk = normalize_chunk(raw, str(project_id))
                except (json.JSONDecodeError, EmbeddingIngestionError) as error:
                    stats.failed_chunks += 1
                    raise EmbeddingIngestionError(f"Invalid chunk record at line {line_number}") from error
                if chunk["chunk_id"] in seen_ids:
                    raise EmbeddingIngestionError("Duplicate chunk ID in processed output")
                seen_ids.add(chunk["chunk_id"])
                stats.total_chunks += 1
                batch.append(chunk)
                if len(batch) >= settings.ollama_embedding_batch_size:
                    await _process_batch(db, project_id, batch, provider, store, stats)
                    batch = []
        if batch:
            await _process_batch(db, project_id, batch, provider, store, stats)
        await store.delete_missing(db, project_id, seen_ids)
    except EmbeddingIngestionError:
        stats.duration_seconds = time.monotonic() - started
        raise
    except Exception as error:
        stats.duration_seconds = time.monotonic() - started
        raise EmbeddingIngestionError("Embedding ingestion failed") from error
    stats.duration_seconds = time.monotonic() - started
    logger.info("Embedding ingestion completed project_id=%s total_chunks=%s embedded_chunks=%s skipped_chunks=%s failed_chunks=%s api_batches=%s duration_seconds=%.3f model=%s",
                project_id, stats.total_chunks, stats.embedded_chunks, stats.skipped_chunks,
                stats.failed_chunks, stats.api_batches, stats.duration_seconds, settings.ollama_embedding_model)
    return stats


async def _process_batch(db, project_id, batch, provider, store, stats):
    settings = get_settings()
    existing = await store.existing(db, project_id, [item["chunk_id"] for item in batch])
    changed = [item for item in batch if not _is_reusable(existing.get(item["chunk_id"]), item, settings)]
    stats.skipped_chunks += len(batch) - len(changed)
    if not changed:
        return
    vectors = await provider.embed_documents([item["content"] for item in changed])
    if len(vectors) != len(changed) or any(len(vector) != settings.embedding_dimension for vector in vectors):
        raise EmbeddingIngestionError("Embedding provider returned an invalid batch")
    await store.upsert(db, project_id, [
        {key: item[key] for key in ("chunk_id", "file_path", "language", "start_line", "end_line", "content_hash")}
        | {"embedding": vector, "embedding_model": settings.ollama_embedding_model,
           "embedding_dimension": settings.embedding_dimension}
        for item, vector in zip(changed, vectors)
    ])
    stats.embedded_chunks += len(changed)
    stats.api_batches += 1


def _is_reusable(existing, item, settings) -> bool:
    return bool(existing and existing.content_hash == item["content_hash"] and
                existing.embedding_model == settings.ollama_embedding_model and
                existing.embedding_dimension == settings.embedding_dimension)
