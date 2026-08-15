import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from ...config import get_settings
from ...models import CodeEmbedding


@dataclass(frozen=True)
class StoredEmbedding:
    chunk_id: str
    content_hash: str
    file_path: str
    language: str | None
    start_line: int | None
    end_line: int | None


class PgVectorStore:
    """Repository boundary for pgvector; APIs do not depend on SQL details."""

    async def existing(self, db: AsyncSession, project_id: uuid.UUID, chunk_ids: list[str]) -> dict[str, CodeEmbedding]:
        if not chunk_ids:
            return {}
        result = await db.execute(select(CodeEmbedding).where(CodeEmbedding.project_id == project_id,
                                                              CodeEmbedding.chunk_id.in_(chunk_ids)))
        return {row.chunk_id: row for row in result.scalars().all()}

    async def upsert(self, db: AsyncSession, project_id: uuid.UUID, records: list[dict]) -> None:
        if not records:
            return
        values = [{**record, "id": uuid.uuid4(), "project_id": project_id} for record in records]
        statement = insert(CodeEmbedding).values(values)
        excluded = statement.excluded
        statement = statement.on_conflict_do_update(
            index_elements=[CodeEmbedding.project_id, CodeEmbedding.chunk_id],
            set_={"file_path": excluded.file_path, "language": excluded.language,
                  "start_line": excluded.start_line, "end_line": excluded.end_line,
                  "content_hash": excluded.content_hash, "embedding": excluded.embedding,
                  "embedding_model": excluded.embedding_model, "embedding_dimension": excluded.embedding_dimension,
                  "updated_at": datetime.utcnow()},
        )
        await db.execute(statement)
        await db.commit()

    async def delete_missing(self, db: AsyncSession, project_id: uuid.UUID, chunk_ids: set[str]) -> int:
        query = delete(CodeEmbedding).where(CodeEmbedding.project_id == project_id)
        if chunk_ids:
            query = query.where(CodeEmbedding.chunk_id.not_in(chunk_ids))
        result = await db.execute(query)
        await db.commit()
        return result.rowcount or 0

    async def search(self, db: AsyncSession, project_id: uuid.UUID, vector: list[float], top_k: int,
                     min_similarity: float) -> list[CodeEmbedding]:
        distance = CodeEmbedding.embedding.cosine_distance(vector)
        similarity = (1 - distance).label("similarity")
        query: Select = (select(CodeEmbedding, similarity)
                         .where(CodeEmbedding.project_id == project_id, similarity >= min_similarity)
                         .order_by(distance).limit(top_k))
        result = await db.execute(query)
        rows = []
        for embedding, score in result.all():
            embedding.search_similarity = float(score)  # type: ignore[attr-defined]
            rows.append(embedding)
        return rows
