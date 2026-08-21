# ArchStudio Step 1 — SaaS authentication

## Setup

1. Create a Python 3.11+ virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and fill in the Supabase values below.
3. Run `uvicorn app.main:app --reload` from `Backend`.

The API is at `http://localhost:8000`; OpenAPI is at `/docs`.

## API

- `POST /api/auth/register` — creates a Supabase user and returns `confirmation_required=true` when email confirmation is pending.
- `POST /api/auth/login` — email/password login.
- `POST /api/auth/forgot-password` — sends recovery email.
- `POST /api/auth/update-password` — updates password during a Supabase recovery session.
- `GET /api/auth/oauth/{google|facebook|apple|x}` — returns the provider redirect URL.
- `GET /api/me` — validates a Supabase bearer token.
- `GET /api/me/profile` — creates/returns the PostgreSQL application profile.

## Step 2 project ingestion API

- `POST /api/projects/upload` — authenticated multipart ZIP upload. The archive is validated, safely extracted into local controlled storage, and marked `ready`.
- `POST /api/projects/github` — authenticated public GitHub repository validation. No repository clone or download occurs in Step 2.
- `GET /api/projects` — lists only the authenticated user's projects.
- `GET /api/projects/{project_id}` — retrieves only an owned project.
- `DELETE /api/projects/{project_id}` — deletes an owned project and its local storage directory.

## Step 4 codebase processing API

- `POST /api/projects/{project_id}/process` — authenticated, ownership-scoped processing start. It returns `processing` immediately and runs bounded discovery/chunking in a background task.

Processing moves a ready project through `processing` to `processed`, or to `failed` with a safe status message. Duplicate processing returns `409`. The processor excludes generated/dependency directories, environment files, binaries, archives, databases, certificates, and hidden files; redacts common secrets; detects source language, writes SHA-256 metadata, and stores `processed/manifest.json` plus `processed/chunks/chunks.jsonl`. It never executes repository code or invokes package managers.

## Step 5 embedding and RAG retrieval

Step 4 is deterministic code discovery, redaction, and chunking. Step 5 is embedding, vector indexing, and semantic retrieval. Step 6 uses focused retrieved context for local LLM analysis and structured report data.

Run `migrations/001_code_embeddings.sql` once against PostgreSQL/Supabase before deploying. It enables `vector`, creates only the application-owned `code_embeddings` table, and adds a project index plus HNSW cosine index. The database role must be permitted to create the extension; if Supabase has not enabled it for the project, enable the **vector** extension in Database → Extensions first, then run the migration. Do not run the rollback statements during normal setup.

Step 5 uses the in-process Sentence Transformers model `nomic-ai/nomic-embed-text-v1.5`. It produces 768-dimensional vectors, so document and query embeddings remain compatible with the existing pgvector schema. Configure `EMBEDDING_MODEL`, `EMBEDDING_DEVICE=auto|cpu|cuda`, `EMBEDDING_BATCH_SIZE`, and `EMBEDDING_CONCURRENCY` in `.env.example`. The model is loaded once and reused; batching, bounded concurrency, content-hash reuse, progress updates, and timing metrics are retained. No embedding HTTP service is required, and keys, tokens, embeddings, and source contents are never logged.

On first use, Sentence Transformers downloads the configured model into the container's model cache. `EMBEDDING_DEVICE=auto` selects CUDA when PyTorch reports a usable GPU and otherwise falls back to CPU. If the model cannot be loaded, the embedding job fails with a safe status message and can be retried.

The existing migration remains `vector(768)`, which is the dimension for `nomic-ai/nomic-embed-text-v1.5`. Stored rows are reused only when their content hash, model name, and dimension match; rows from another provider are regenerated on the next embed run. No vector dimension migration is required.

- `POST /api/projects/{project_id}/embed` — authenticated, ownership-scoped background indexing; returns `202`, rejects duplicate active jobs with `409`, and transitions `processed` → `embedding` → `embedded` (or `failed`).
- `POST /api/projects/{project_id}/search` — authenticated, ownership-scoped cosine search with bounded `top_k`, threshold filtering, metadata, and chunk content loaded from Step 4 JSONL storage.
- `POST /api/projects/{project_id}/analyze` — authenticated, ownership-scoped background Step 6 analysis using focused Step 5 retrieval and local `qwen2.5-coder:7b`; returns an analysis job ID.
- `GET /api/projects/{project_id}/analysis/{analysis_id}` — authenticated analysis job status/result endpoint.

Step 6 never sends the entire repository to the LLM. It performs focused retrieval for architecture/structure, technology/dependencies, quality/maintainability, security, and performance, selects a small diverse evidence set (one top chunk per task), caps the evidence context to a CPU/model-friendly default of 5 chunks and 12,000 characters, and asks the configured Groq API for validated JSON. Analysis uses its own configurable `ANALYSIS_MIN_SIMILARITY` default of `0.0` so focused tasks still receive their nearest evidence when the repository's similarity scores are modest; interactive `/search` thresholds remain unchanged. The prompt requires file/line evidence and explicitly forbids invented repository facts. Configure `GROQ_API_KEY`, `GROQ_MODEL`, and `GROQ_TIMEOUT` for local analysis.

### Step 6 user-facing generation

After a project is embedded, the following authenticated, ownership-scoped endpoints queue background artifacts: `POST /api/projects/{project_id}/reports`, `/ieee-report`, `/research-paper`, `/database-diagram`, and `/architecture-diagram`. Poll `GET /api/projects/{project_id}/artifacts/{artifact_id}` for structured editable sections or Mermaid payloads. Save edits with `PATCH` on the same resource and download the latest content with `GET .../download?format=pdf|docx|svg|png`. Rendering is local; no Mermaid cloud service is used. Database diagrams only include schema declarations found in processed repository files and report “No database schema detected” when none are present. Reports explicitly mark unavailable evidence and do not fabricate research or references.

`POST /api/projects/{project_id}/chat` performs synchronous, project-scoped Step 5 retrieval followed by grounded local Qwen generation. The response contains the answer plus file/line evidence. It is intentionally not a general LLM endpoint and does not read another project's vectors.

Migration `002_project_analyses.sql` also creates `project_artifacts`. Apply it after `001_code_embeddings.sql` (or run the existing `init_db` against a development database). PostgreSQL must have the `vector` extension enabled; managed Supabase deployments may require the one-time SQL `CREATE EXTENSION IF NOT EXISTS vector;` by a database owner. No Supabase-managed authentication tables are changed.

Step 4 remains deterministic code processing/chunking. Step 5 remains embedding, vector indexing, and semantic retrieval. Step 6 adds grounded analysis, editable reports/diagrams, exports, and project chat. LLM answer generation beyond these project features is not implemented.

Ingestion streams JSONL and upserts only changed `(project_id, chunk_id)` records. An unchanged content hash/model/dimension makes no provider call; changed chunks are replaced and disappeared chunks are deleted. Provider calls are batched and retried with bounded exponential backoff. Tests should run from `Backend` with `pytest`; all embedding calls in tests must be mocked.

ZIP uploads are limited by `MAX_UPLOAD_SIZE_MB` and `MAX_EXTRACTED_SIZE_MB`, reject malformed archives, reject traversal and symbolic-link entries, and are never executed. Project files are stored under `PROJECT_STORAGE_ROOT`, which is ignored by Git. GitHub validation calls only the fixed `api.github.com/repos/{owner}/{repo}` endpoint and accepts public repositories without cloning them; retrieval/orchestration is deferred to Step 3.

## Step 3 n8n integration

- `POST /api/projects/{project_id}/retrieve` — queues owned GitHub projects for asynchronous n8n retrieval.
- `POST /api/projects/retrieval/callback` — accepts authenticated n8n status callbacks.

See [docs/n8n-github-retrieval.md](docs/n8n-github-retrieval.md) for the workflow contract and manual setup. n8n is not configured or claimed to be live by this codebase.

## Environment and external configuration

Create a Supabase project at https://supabase.com/dashboard. From the project dashboard, copy the Project URL into `SUPABASE_URL` and the public anon key into `SUPABASE_ANON_KEY`. Both are used by the backend; the anon key is safe to use in a browser, but this backend does not expose it in responses.

From the Supabase dashboard's **Connect** dialog, copy the PostgreSQL connection string. Use the pooled connection for application traffic when appropriate, and convert its scheme for SQLAlchemy/asyncpg by changing `postgresql://` to `postgresql+asyncpg://`. Put the resulting full string, including the database password, in `DATABASE_URL`. This value is server-only and must never be committed.

The backend does not currently require `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_JWT_SECRET`. It validates bearer tokens with Supabase Auth's `get_user` endpoint using the anon-key client, so neither key belongs in `.env` at this stage.

In Supabase **Authentication** settings, enable email/password sign-up, email confirmation, and password recovery. Leave the standard **Confirm signup** email template unchanged. Supabase generates and sends the confirmation link, and Supabase marks the user verified after the link is clicked.

Custom SMTP and custom email templates are not required for this flow. Configure SMTP only if you later choose a production email provider. Set the Supabase Site URL and redirect URL to `http://localhost:5173/auth/callback` for the Vite frontend. The frontend callback should handle both successful and expired/invalid confirmation links and provide a clear return-to-login action.

For Google, Facebook, Apple, and X, create provider applications in the respective developer consoles, copy their client IDs/secrets into Supabase **Authentication → Providers**, and register the Supabase callback URL shown there with each provider. These OAuth secrets are server-side Supabase configuration and are not application `.env` values.

Set `ALLOWED_ORIGINS=http://localhost:5173`. No credentials are committed.

Supabase owns password storage, confirmation-link email verification, recovery sessions, and provider OAuth. PostgreSQL stores only a profile keyed by the Supabase user UUID. For production, replace `create_all` with Alembic migrations.
