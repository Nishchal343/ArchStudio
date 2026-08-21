import hashlib
import asyncio
import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import SearchRequest
from app.services.embeddings.base import EmbeddingProviderError
from app.services.embeddings.sentence_transformers import SentenceTransformerEmbeddingProvider
from app.services.rag.ingestion import EmbeddingIngestionError, ingest_project
from app.services.rag.retrieval import search_project
from app.services.storage_service import project_directory


class FakeModel:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [[0.1] * self.dimension for _ in texts]


def run(coro):
    return asyncio.run(coro)


def test_sentence_transformer_provider_initialization_and_document_embedding():
    model = FakeModel()
    provider = SentenceTransformerEmbeddingProvider(model=model, dimension=3, max_retries=0)
    run(provider.embed_documents(["a", "b"]))
    run(provider.embed_query("find auth"))
    assert model.calls[0][0] == ["search_document: a", "search_document: b"]
    assert model.calls[1][0] == ["search_query: find auth"]
    assert model.calls[0][1]["normalize_embeddings"] is True


def test_provider_rejects_dimension_mismatch():
    provider = SentenceTransformerEmbeddingProvider(model=FakeModel(dimension=2), dimension=3, max_retries=0)
    with pytest.raises(EmbeddingProviderError):
        run(provider.embed_documents(["a"]))


class FakeStore:
    def __init__(self, records=None):
        self.records = records or {}
        self.upserted = []
        self.deleted = None

    async def existing(self, db, project_id, chunk_ids):
        return {key: self.records[key] for key in chunk_ids if key in self.records}

    async def upsert(self, db, project_id, records):
        self.upserted.extend(records)
        for record in records:
            self.records[record["chunk_id"]] = SimpleNamespace(**record)

    async def delete_missing(self, db, project_id, chunk_ids):
        self.deleted = set(self.records) - chunk_ids
        for key in self.deleted:
            del self.records[key]
        return len(self.deleted)

    async def search(self, db, project_id, vector, top_k, min_similarity):
        return []


def write_chunks(tmp_path, monkeypatch, project_id, contents):
    monkeypatch.setattr(get_settings(), "project_storage_root", str(tmp_path / "projects"))
    root = project_directory(str(project_id)) / "processed" / "chunks"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "chunks.jsonl").open("w", encoding="utf-8") as output:
        for index, content in enumerate(contents):
            output.write(json.dumps({"project_id": str(project_id), "file_path": "auth.py",
                                     "language": "python", "chunk_index": index,
                                     "start_line": index + 1, "end_line": index + 1,
                                     "content": content,
                                     "content_hash": hashlib.sha256(content.encode()).hexdigest()}) + "\n")


class FakeProvider:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.calls = []

    async def embed_documents(self, texts):
        self.calls.append(texts)
        return [[float(len(text))] * self.dimension for text in texts]

    async def embed_query(self, text):
        return [1.0] * self.dimension


def test_ingestion_batches_and_is_idempotent(tmp_path, monkeypatch):
    project_id = UUID("11111111-1111-1111-1111-111111111111")
    write_chunks(tmp_path, monkeypatch, project_id, ["one", "two", "three"])
    monkeypatch.setattr(get_settings(), "embedding_batch_size", 2)
    monkeypatch.setattr(get_settings(), "embedding_dimension", 3)
    provider = FakeProvider()
    store = FakeStore()
    first = run(ingest_project(None, project_id, provider, store))
    second = run(ingest_project(None, project_id, provider, store))
    assert first.embedded_chunks == 3 and first.api_batches == 2
    assert second.embedded_chunks == 0 and second.skipped_chunks == 3
    assert len(provider.calls) == 2


def test_changed_and_deleted_chunks_are_incremental(tmp_path, monkeypatch):
    project_id = UUID("22222222-2222-2222-2222-222222222222")
    write_chunks(tmp_path, monkeypatch, project_id, ["one", "two"])
    monkeypatch.setattr(get_settings(), "embedding_dimension", 3)
    provider, store = FakeProvider(), FakeStore()
    run(ingest_project(None, project_id, provider, store))
    write_chunks(tmp_path, monkeypatch, project_id, ["changed"])
    stats = run(ingest_project(None, project_id, provider, store))
    assert stats.embedded_chunks == 1
    assert store.deleted == {"auth.py:1"}
    assert set(store.records) == {"auth.py:0"}


def test_missing_and_malformed_processed_output_fail(tmp_path, monkeypatch):
    project_id = UUID("33333333-3333-3333-3333-333333333333")
    monkeypatch.setattr(get_settings(), "project_storage_root", str(tmp_path / "projects"))
    with pytest.raises(EmbeddingIngestionError):
        run(ingest_project(None, project_id, FakeProvider(), FakeStore()))
    write_chunks(tmp_path, monkeypatch, project_id, [])
    path = project_directory(str(project_id)) / "processed" / "chunks" / "chunks.jsonl"
    path.write_text('{"file_path":"x.py"}\n', encoding="utf-8")
    with pytest.raises(EmbeddingIngestionError):
        run(ingest_project(None, project_id, FakeProvider(), FakeStore()))


def test_search_request_validation():
    assert SearchRequest(query="  auth  ").query == "auth"
    with pytest.raises(ValidationError):
        SearchRequest(query="x", top_k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="x", min_similarity=1.1)


def test_search_loads_only_project_scoped_ranked_chunks(tmp_path, monkeypatch):
    project_id = UUID("44444444-4444-4444-4444-444444444444")
    write_chunks(tmp_path, monkeypatch, project_id, ["authentication implementation"])
    item = SimpleNamespace(chunk_id="auth.py:0", file_path="auth.py", language="python",
                           start_line=1, end_line=1, search_similarity=0.87)

    class SearchStore(FakeStore):
        async def search(self, db, requested_project, vector, top_k, min_similarity):
            assert requested_project == project_id
            assert top_k == 1 and min_similarity == 0.5
            return [item]

    results = run(search_project(None, project_id, "where auth", 1, 0.5, FakeProvider(), SearchStore()))
    assert results[0]["content"] == "authentication implementation"
    assert results[0]["similarity"] == 0.87
