from sqlalchemy import select
from sqlalchemy.dialects.postgresql import dialect, insert

from app.models import CodeEmbedding


def test_embedding_column_uses_pgvector_and_binds_python_list():
    column_type = CodeEmbedding.__table__.c.embedding.type
    assert column_type.__class__.__name__ == "VECTOR"
    assert column_type.dim == 768

    vector = [0.1] * 768
    processor = column_type.bind_processor(dialect())
    bound_value = processor(vector)
    assert isinstance(bound_value, str)
    assert bound_value.startswith("[") and bound_value.endswith("]")


def test_upsert_and_cosine_query_retain_vector_bind_type():
    vector = [0.1] * 768
    statement = insert(CodeEmbedding).values(
        project_id="00000000-0000-0000-0000-000000000001",
        chunk_id="file.py:0",
        file_path="file.py",
        content_hash="a" * 64,
        embedding=vector,
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_dimension=768,
    )
    compiled_insert = statement.compile(dialect=dialect())
    assert compiled_insert.binds["embedding"].type.dim == 768

    compiled_search = select(CodeEmbedding).where(
        CodeEmbedding.embedding.cosine_distance(vector) < 0.5
    ).compile(dialect=dialect())
    assert "<=>" in str(compiled_search)
    assert compiled_search.binds["embedding_1"].type.dim == 768
