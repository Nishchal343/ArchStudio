# Step 3: n8n GitHub retrieval workflow

This repository contains the FastAPI integration contract only. No live n8n instance, webhook, credential, or workflow activation is included.

## Install n8n

Install n8n using the official [n8n installation documentation](https://docs.n8n.io/hosting/), using either a self-hosted Docker/npm deployment or n8n Cloud. For local development, the n8n editor must be reachable by the FastAPI process, commonly at `http://localhost:5678`. Protect the n8n editor with its normal user management and do not expose an unauthenticated production webhook; n8n also provides a security audit for detecting unprotected webhooks and risky filesystem nodes.

## Environment

Set these values in the backend's local `.env` file; never commit them:

```env
N8N_BASE_URL=http://localhost:5678
N8N_GITHUB_WORKFLOW_WEBHOOK=webhook/github-retrieval
N8N_WEBHOOK_SECRET=<generate-a-long-random-secret>
N8N_CALLBACK_URL=http://localhost:8000/api/projects/retrieval/callback
N8N_REQUEST_TIMEOUT_SECONDS=30
```

`N8N_GITHUB_WORKFLOW_WEBHOOK` may also be a complete HTTPS webhook URL. The shared secret is sent by FastAPI in the `X-ArchStudio-Webhook-Secret` header and must be configured as an n8n webhook-header authentication check. The same secret authenticates the n8n callback to FastAPI. Do not put it in a frontend or Git repository.

## Workflow contract

Create and activate one n8n Webhook workflow:

1. Webhook trigger, accepting `POST` JSON.
2. Validate that `project_id`, `user_id`, `source_type`, and `source_url` are present, that `source_type` is `github`, and that the URL is the normalized HTTPS GitHub URL received from FastAPI. Reject all other hosts, schemes, paths, localhost, private-network destinations, and arbitrary URLs.
3. Retrieve the public repository using an official GitHub/n8n integration or an HTTP Request node against the GitHub API. Do not configure a GitHub token for public repositories.
4. Store the retrieved source under the shared controlled location `storage/projects/<project_id>/` (or the equivalent mounted storage path). The n8n process must have access only to the intended project storage root. Do not write outside it, execute repository files, install dependencies, run package managers, or run shell/project startup commands.
5. On success, call `N8N_CALLBACK_URL` with the JSON below and the same shared-secret header. On failure, call it with `status=failed` and no internal stack trace, token, credential, or filesystem path.

FastAPI sends exactly:

```json
{
  "project_id": "<UUID>",
  "user_id": "<Supabase UUID>",
  "source_type": "github",
  "source_url": "https://github.com/owner/repository"
}
```

The callback body is:

```json
{
  "project_id": "<UUID>",
  "user_id": "<Supabase UUID>",
  "source_type": "github",
  "status": "ready",
  "error": null
}
```

Allowed callback statuses are `processing`, `ready`, and `failed`. FastAPI stores only safe fixed status messages and never stores the callback's raw error text.

## FastAPI endpoints

`POST /api/projects/{project_id}/retrieve` requires the existing Supabase bearer token. It verifies ownership and GitHub source type, rejects duplicate `queued`/`processing` jobs with `409`, changes the project to `queued`, and returns immediately.

`POST /api/projects/retrieval/callback` is not a public unauthenticated status endpoint. It requires the shared `X-ArchStudio-Webhook-Secret` header, verifies the project and user UUID, and updates only that GitHub project.

The ZIP path is intentionally excluded from this workflow because Step 2 already stores ZIP source locally.
