from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ArchStudio API"
    environment: str = "development"
    database_url: str
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_pool_timeout: float = Field(default=30.0, gt=0, le=300)
    db_pool_recycle: int = Field(default=1800, ge=60, le=86400)
    db_connect_timeout: float = Field(default=30.0, gt=0, le=300)
    supabase_url: str
    supabase_anon_key: str
    frontend_url: str = "http://localhost:5173"
    allowed_origins: Annotated[list[str], NoDecode] = []
    auth_redirect_url: str = "http://localhost:5173/auth/callback"
    email_confirmation_redirect_url: str = "http://localhost:5173/auth/confirm"
    password_reset_redirect_url: str = "http://localhost:5173/auth/reset-password"
    project_storage_root: str = "storage/projects"
    max_upload_size_mb: int = 50
    max_extracted_size_mb: int = 200
    n8n_base_url: str | None = None
    n8n_github_workflow_webhook: str | None = None
    n8n_webhook_secret: str | None = None
    n8n_callback_url: str | None = None
    n8n_request_timeout_seconds: float = 30.0
    processing_excluded_directories: Annotated[list[str], NoDecode] = [
        ".git", ".github", "node_modules", "venv", ".venv", "env", "__pycache__",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", "target",
        "coverage", ".next", ".nuxt", ".cache", "vendor", "processed", "generated",
        "generated-artifacts", "artifacts", "out", "release", "__macosx",
    ]
    processing_excluded_file_patterns: Annotated[list[str], NoDecode] = [
        ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.exe", "*.dll",
        "*.so", "*.dylib", "*.bin", "*.zip", "*.tar", "*.gz", "*.7z", "*.rar",
        "*.db", "*.sqlite", "*.sqlite3", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp",
        "*.ico", "*.bmp", "*.tiff", "*.mp3", "*.mp4", "*.mov", "*.avi", "*.woff", "*.woff2",
        "*.ttf", "*.otf", "*.map", "*.min.js", "*.min.css",
    ]
    max_processing_file_size_mb: int = 5
    max_project_files: int = 10000
    max_total_source_size_mb: int = 500
    max_chunk_size: int = 4000
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_device: str = "auto"
    embedding_batch_size: int = Field(default=32, gt=0, le=128)
    embedding_concurrency: int = Field(default=2, ge=1, le=4)
    embedding_max_retries: int = Field(default=2, ge=0, le=5)
    groq_api_key: str | None = None
    groq_model: str = "qwen/qwen3.6-27b"
    groq_timeout: float = Field(default=120.0, gt=0, le=900)
    groq_max_retries: int = Field(default=2, ge=0, le=5)
    embedding_dimension: int = Field(default=768, gt=0)
    rag_default_top_k: int = Field(default=8, gt=0)
    rag_max_top_k: int = Field(default=20, gt=0, le=100)
    rag_min_similarity: float = Field(default=0.5, ge=0, le=1)
    rag_max_query_length: int = Field(default=2000, gt=0, le=10000)
    analysis_max_context_chunks: int = Field(default=5, gt=0, le=100)
    analysis_max_context_chars: int = Field(default=12000, gt=1000, le=200000)
    analysis_min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("processing_excluded_directories", "processing_excluded_file_patterns", mode="before")
    @classmethod
    def split_processing_lists(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

@lru_cache
def get_settings() -> Settings:
    return Settings()
