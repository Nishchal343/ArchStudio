# ArchStudio

A project-intelligence workspace that turns software repositories into searchable, explainable, and AI-assisted engineering workspaces.

## Project Overview

ArchStudio imports a ZIP archive or public GitHub repository, processes its source code, creates a semantic index, and generates project-grounded answers, analysis, diagrams, reports, and literature comparisons.

## Live Demo

Coming soon.

## Features

- ZIP upload and public GitHub integration
- Safe code processing, secret redaction, language detection, and chunking
- Architecture, dependency, quality, security, and performance analysis
- Project-grounded search and chat with file/line evidence
- Editable reports and architecture/database diagrams
- Literature Review from multiple URLs, PDFs, or mixed sources
- PDF, DOCX, SVG, and PNG export support where applicable
- Email/password, Google OAuth, and GitHub OAuth authentication

## Architecture

~~~mermaid
flowchart LR
    U[User] --> F[React + Nginx]
    F --> A[FastAPI]
    A --> S[Supabase Auth]
    A --> P[(Supabase PostgreSQL)]
    A --> C[Code processing]
    C --> E[Sentence Transformers]
    E --> V[(pgvector)]
    A --> R[Retrieval]
    V --> R
    R --> G[Groq LLM]
    G --> A
~~~

## Request/Data Flow

Requests are authenticated by Supabase, scoped to the selected project, matched against indexed source chunks, and sent to Groq with focused repository evidence. Results return to the React interface with supporting file and line references where available.

## Tech Stack

| Area | Technology |
|---|---|
| Frontend | React, React Router, Vite, Mermaid |
| Backend | Python, FastAPI, Uvicorn |
| Database | Supabase PostgreSQL, SQLAlchemy, asyncpg |
| Retrieval | pgvector cosine search |
| LLM | Groq, default qwen/qwen3.6-27b |
| Embeddings | Sentence Transformers, nomic-ai/nomic-embed-text-v1.5 |
| Infrastructure | Docker Compose, Nginx, n8n integration |

## Authentication

Supabase Auth provides email/password, Google OAuth, GitHub OAuth, email confirmation, and password recovery.

Callback paths:

- OAuth: /auth/callback
- Email confirmation: /auth/confirm
- Password reset: /auth/reset-password

OAuth is implemented but is not described as production-ready. Production requires configuring deployed redirect URLs with Supabase, Google, and GitHub.

## Codebase Workflow

1. Import a ZIP archive or validate a public GitHub URL.
2. Safely extract and process supported files.
3. Redact common secrets and create line-aware JSONL chunks.
4. Generate 768-dimensional local embeddings.
5. Store vectors and metadata in PostgreSQL pgvector.
6. Retrieve relevant code for search, chat, analysis, reports, diagrams, and Literature Review.

Repository code is not executed and package managers are not run during processing.

## Literature Review

The workflow accepts multiple public URLs, PDFs, or mixed sources. Paper text is extracted, kept source-specific, and compared with evidence retrieved from the selected project's indexed codebase.

It generates Literature Review, Research Gap & Motivation, Comparison with Existing System, and per-paper methodology, architecture, algorithms, technologies, results, limitations, and future work. Generated sections can be copied. URLs and results use browser session storage; uploaded PDFs must be attached again after refresh.

## AI / Embeddings / Retrieval

Sentence Transformers runs in the backend and produces 768-dimensional embeddings. pgvector performs project-scoped cosine retrieval. Groq generates grounded analysis, chat, reports, diagrams, and Literature Review content.

The current Docker setup uses CPU embeddings. Free cloud GPU deployment with automatic scale-to-zero is not implemented and remains future work.

## Getting Started

### Prerequisites

- Docker Desktop with Docker Compose
- Supabase PostgreSQL and Auth
- Groq API key
- Optional public GitHub repository and n8n setup

### Configure and run

~~~powershell
Copy-Item Backend/.env.example Backend/.env
docker compose up --build
~~~

Set database, Supabase, redirect, Groq, and processing values in Backend/.env. Do not commit secrets.

Services:

- Frontend: http://localhost:8080
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs
- n8n: http://localhost:5678

Stop with:

~~~powershell
docker compose down
~~~

## Environment Variables

Use Backend/.env.example as the source of truth. Main groups cover database, Supabase, redirect URLs, Groq, embeddings, processing limits, n8n, retrieval, and analysis. Never place API keys, passwords, OAuth secrets, or webhook secrets in this README.

## Docker / Deployment

| Service | Port | Purpose |
|---|---:|---|
| frontend | 8080 | React build served by Nginx |
| backend | 8000 | FastAPI API and AI pipeline |
| n8n | 5678 | Optional GitHub retrieval orchestration |
| Supabase PostgreSQL | External | Application data and pgvector |

The repository provides a local Docker Compose setup. Production still requires durable storage, production URLs, OAuth configuration, database verification, monitoring, and final end-to-end testing. No GPU embedding service is currently deployed.

## Project Structure

~~~text
Backend/             FastAPI app, services, migrations, and tests
Frontend/            React application and Nginx build
docker-compose.yml   Local service orchestration
Backend/.env.example Backend configuration template
storage/             Project processing storage
~~~

## Testing

~~~powershell
cd Backend
pytest
~~~

Backend tests cover authentication, validation, project services, code processing, n8n, embeddings, vector search, analysis, and artifacts. No separate frontend test command is defined.

## Project Status

Core functionality is implemented and locally Dockerized. Production deployment, OAuth provider configuration/testing, GPU embedding infrastructure, and final production verification are pending.

## Future Improvements

- Production deployment and monitoring
- Production Google/GitHub OAuth configuration
- Durable project and model storage
- Optional GPU embeddings with scale-to-zero
- Broader frontend and end-to-end test coverage
- Better extraction support for access-controlled publisher pages

## License

License not yet specified.

