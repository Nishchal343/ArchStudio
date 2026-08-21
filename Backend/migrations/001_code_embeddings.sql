-- Step 5: reversible pgvector index for Sentence Transformers nomic-embed-text-v1.5 (768 dimensions).
-- Run with a database role allowed to manage extensions.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS code_embeddings (
    id uuid PRIMARY KEY,
    project_id uuid NOT NULL,
    chunk_id varchar(512) NOT NULL,
    file_path varchar(2048) NOT NULL,
    language varchar(80),
    start_line integer,
    end_line integer,
    content_hash varchar(64) NOT NULL,
    embedding vector(768) NOT NULL,
    embedding_model varchar(200) NOT NULL,
    embedding_dimension integer NOT NULL DEFAULT 768,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_code_embeddings_project_chunk UNIQUE (project_id, chunk_id),
    CONSTRAINT uq_code_embeddings_project_chunk_hash UNIQUE (project_id, chunk_id, content_hash)
);

CREATE INDEX IF NOT EXISTS ix_code_embeddings_project_id ON code_embeddings (project_id);
CREATE INDEX IF NOT EXISTS ix_code_embeddings_embedding_hnsw
    ON code_embeddings USING hnsw (embedding vector_cosine_ops);

-- Rollback (only for an explicit migration rollback):
-- DROP INDEX IF EXISTS ix_code_embeddings_embedding_hnsw;
-- DROP INDEX IF EXISTS ix_code_embeddings_project_id;
-- DROP TABLE IF EXISTS code_embeddings;
-- DROP EXTENSION IF EXISTS vector;
