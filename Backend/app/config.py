from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ArchStudio API"
    environment: str = "development"
    database_url: str
    supabase_url: str
    supabase_anon_key: str
    frontend_url: str = "http://localhost:5173"
    allowed_origins: Annotated[list[str], NoDecode] = []
    auth_redirect_url: str = "http://localhost:5173/auth/callback"
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
        "coverage", ".next", ".nuxt", ".cache", "vendor", "processed",
    ]
    processing_excluded_file_patterns: Annotated[list[str], NoDecode] = [
        ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.exe", "*.dll",
        "*.so", "*.dylib", "*.bin", "*.zip", "*.tar", "*.gz", "*.7z", "*.rar",
        "*.db", "*.sqlite", "*.sqlite3",
    ]
    max_processing_file_size_mb: int = 5
    max_project_files: int = 10000
    max_total_source_size_mb: int = 500
    max_chunk_size: int = 4000
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_llm_model: str = "qwen2.5-coder:7b"
    ollama_llm_timeout: float = Field(default=300.0, gt=0, le=900)
    ollama_llm_max_retries: int = Field(default=2, ge=0, le=5)
    ollama_embedding_timeout: float = Field(default=120.0, gt=0, le=600)
    ollama_embedding_batch_size: int = Field(default=4, gt=0, le=64)
    embedding_dimension: int = Field(default=768, gt=0)
    embedding_max_retries: int = Field(default=3, ge=0, le=10)
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
