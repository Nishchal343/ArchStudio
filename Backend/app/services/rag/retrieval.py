import json
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ...services.embeddings.base import EmbeddingProvider
from ...services.embeddings.service import get_embedding_provider
from ...services.storage_service import project_directory
if TYPE_CHECKING:
    from ..vector_store.pgvector import PgVectorStore


async def search_project(db: AsyncSession, project_id: UUID, query: str, top_k: int, min_similarity: float,
                         provider: EmbeddingProvider | None = None, store: "PgVectorStore | None" = None) -> list[dict]:
    provider = provider or get_embedding_provider()
    if store is None:
        from ..vector_store.pgvector import PgVectorStore
        store = PgVectorStore()
    vector = await provider.embed_query(query)
    embeddings = await store.search(db, project_id, vector, top_k, min_similarity)
    by_id = {item.chunk_id: item for item in embeddings}
    chunks_path = project_directory(str(project_id)) / "processed" / "chunks" / "chunks.jsonl"
    if not chunks_path.is_file():
        return []
    content_by_id: dict[str, str] = {}
    with chunks_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = chunk.get("chunk_id") or f"{chunk.get('file_path')}:{chunk.get('chunk_index')}"
            if chunk_id in by_id:
                content_by_id[chunk_id] = chunk.get("content", "")
    return [{"chunk_id": item.chunk_id, "file_path": item.file_path, "language": item.language,
             "start_line": item.start_line, "end_line": item.end_line,
             "similarity": getattr(item, "search_similarity", 0.0),
             "content": content_by_id[item.chunk_id]}
            for item in embeddings if item.chunk_id in content_by_id]
