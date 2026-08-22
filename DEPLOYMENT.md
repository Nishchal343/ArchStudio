# AWS EC2 Ubuntu deployment

This uses the existing React/Nginx, FastAPI, n8n, Supabase PostgreSQL/pgvector, and CPU embedding architecture.

## Requirements

- Ubuntu 22.04 or 24.04 LTS
- At least 2 vCPUs and 8 GB RAM; 4 vCPUs and 16 GB is recommended for larger repositories
- 30 GB or more EBS storage, plus room for uploaded repositories and model cache
- Security group: TCP 22 from your IP, TCP 80 and 443 from the internet
- Do not expose ports 8000, 5678, or database ports publicly

The deployment is CPU-only; no NVIDIA runtime or GPU is required.

## Install Docker

    sudo apt-get update
    sudo apt-get install -y ca-certificates curl git
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER"
    newgrp docker
    docker --version
    docker compose version

## Clone and configure

    git clone <YOUR_REPOSITORY_URL> ArchStudioAi
    cd ArchStudioAi
    cp Backend/.env.example Backend/.env
    nano Backend/.env

Set `ENVIRONMENT=production`, the Supabase `DATABASE_URL`, `SUPABASE_URL`, and `SUPABASE_ANON_KEY`, the public `FRONTEND_URL`, `ALLOWED_ORIGINS`, all three redirect URLs, `GROQ_API_KEY`, `N8N_ENCRYPTION_KEY`, and `N8N_WEBHOOK_SECRET`. Set `N8N_GITHUB_WORKFLOW_WEBHOOK` if GitHub retrieval is used. Use `openssl rand -hex 32` for random secrets. Never commit `Backend/.env`.

Configure the same redirect URLs in Supabase Auth and Google/GitHub OAuth applications. For ZIP-only use, n8n retrieval variables may remain empty.

## Build, start, and stop

    docker compose --env-file Backend/.env -f docker-compose.prod.yml build
    docker compose --env-file Backend/.env -f docker-compose.prod.yml up -d
    docker compose --env-file Backend/.env -f docker-compose.prod.yml down

The frontend is available at `http://<EC2_PUBLIC_IP>/` or the configured domain.

## Database initialization

The current backend has no versioned Alembic migrations. On startup it initializes SQLAlchemy metadata. Run the same initialization explicitly:

    docker compose --env-file Backend/.env -f docker-compose.prod.yml exec backend python -c "import asyncio; from app.database import init_db; asyncio.run(init_db())"

This does not reset or delete data. The Supabase role must be able to create the `vector` extension and application tables.

## Status, logs, and checks

    docker compose --env-file Backend/.env -f docker-compose.prod.yml ps
    docker compose --env-file Backend/.env -f docker-compose.prod.yml logs --tail=200 backend
    docker compose --env-file Backend/.env -f docker-compose.prod.yml logs --tail=200 frontend
    docker compose --env-file Backend/.env -f docker-compose.prod.yml logs --tail=200 n8n
    curl -I http://127.0.0.1/
    curl http://127.0.0.1/health

The browser frontend uses same-origin `/api` requests; it must not use `localhost:8000` in production.

## Updates and persistence

    git pull --ff-only
    docker compose --env-file Backend/.env -f docker-compose.prod.yml build
    docker compose --env-file Backend/.env -f docker-compose.prod.yml up -d
    docker image prune -f

Named volumes preserve `project_storage`, `embedding_model_cache`, and `n8n_data`. Do not run `docker compose down -v` unless you intend to delete them.

## HTTPS and troubleshooting

Use an AWS load balancer or host reverse proxy to terminate TLS and forward to the frontend on port 80. Then change `FRONTEND_URL`, `ALLOWED_ORIGINS`, and Supabase/OAuth redirects to `https://<domain>`. Keep n8n internal unless public webhooks are required; if exposed later, protect it and set `N8N_PUBLIC_URL`.

- Backend unhealthy: inspect logs and verify `DATABASE_URL`, Supabase reachability, and pgvector permissions.
- API calls fail: inspect Nginx/backend logs; browser requests should target `/api`.
- GitHub retrieval stays queued: verify the n8n workflow, webhook path, callback `http://backend:8000/api/projects/retrieval/callback`, and secret.
- Embeddings are slow: CPU is intentional; increase EC2 CPU/RAM or tune existing batch/concurrency values.
- Files disappear: verify `project_storage` and avoid `down -v`.
