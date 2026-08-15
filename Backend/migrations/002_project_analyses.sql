-- Step 6: persisted local Qwen analysis jobs/results.
CREATE TABLE IF NOT EXISTS project_analyses (
    id uuid PRIMARY KEY,
    project_id uuid NOT NULL,
    status varchar(32) NOT NULL,
    result json,
    error_message varchar(500),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_project_analyses_project_id ON project_analyses (project_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_project_analysis_active
    ON project_analyses (project_id)
    WHERE status IN ('queued', 'analyzing');

CREATE TABLE IF NOT EXISTS project_artifacts (
    id uuid PRIMARY KEY,
    project_id uuid NOT NULL,
    kind varchar(40) NOT NULL,
    status varchar(32) NOT NULL,
    payload json,
    error_message varchar(500),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_project_artifacts_project_id ON project_artifacts (project_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_project_artifact_active
    ON project_artifacts (project_id, kind)
    WHERE status IN ('queued', 'generating');
