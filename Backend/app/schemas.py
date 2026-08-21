from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from .project_types import ArtifactKind, ProjectSourceType, ProjectStatus


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Name is required")
        return value


class EmailRequest(BaseModel):
    email: EmailStr


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UpdatePasswordRequest(BaseModel):
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class ProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    supabase_user_id: UUID
    display_name: str | None
    avatar_url: str | None
    created_at: datetime


class AuthResponse(BaseModel):
    session: dict | None = None
    user: dict | None = None
    confirmation_required: bool = False


class ProjectCreateResponse(BaseModel):
    id: UUID
    name: str
    source_type: ProjectSourceType
    source_url: str | None = None
    original_filename: str | None = None
    status: ProjectStatus
    status_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class GitHubProjectRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    name: str | None = Field(default=None, min_length=1, max_length=200)


class RetrievalStartResponse(BaseModel):
    project_id: UUID
    status: ProjectStatus
    message: str


class ProcessingStartResponse(BaseModel):
    project_id: UUID
    status: ProjectStatus
    message: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query is required")
        return value


class SearchResult(BaseModel):
    chunk_id: str
    file_path: str
    language: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    similarity: float
    content: str


class SearchResponse(BaseModel):
    project_id: UUID
    query: str
    results: list[SearchResult]


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    project_id: UUID
    overview: str = ""
    project_structure: str = ""
    architecture: str = ""
    technology_stack: list[str] = Field(default_factory=list)
    components: list[dict] = Field(default_factory=list)
    component_relationships: list[dict] = Field(default_factory=list)
    dependencies: list[dict] = Field(default_factory=list)
    code_quality: list[dict] = Field(default_factory=list)
    security_findings: list[dict] = Field(default_factory=list)
    performance_findings: list[dict] = Field(default_factory=list)
    maintainability_findings: list[dict] = Field(default_factory=list)
    code_smells: list[dict] = Field(default_factory=list)
    recommendations: list[dict] = Field(default_factory=list)
    summary: str = ""


class AnalysisStartResponse(BaseModel):
    project_id: UUID
    analysis_id: UUID
    status: str
    message: str


class AnalysisStatusResponse(BaseModel):
    project_id: UUID
    analysis_id: UUID
    status: str
    error: str | None = None
    analysis: AnalysisResult | None = None


class ReportSection(BaseModel):
    heading: str
    content: str


class EditableDocument(BaseModel):
    title: str
    document_type: ArtifactKind
    authors: list[dict[str, str]] = Field(default_factory=lambda: [
        {"name": "[AUTHOR 1]", "institution": "[INSTITUTION / AFFILIATION]"},
        {"name": "[AUTHOR 2]", "institution": "[INSTITUTION / AFFILIATION]"},
        {"name": "[AUTHOR 3]", "institution": "[INSTITUTION / AFFILIATION]"},
        {"name": "[AUTHOR 4]", "institution": "[INSTITUTION / AFFILIATION]"},
        {"name": "[AUTHOR 5]", "institution": "[INSTITUTION / AFFILIATION]"},
    ])
    date: str = "[DATE]"
    sections: list[ReportSection] = Field(default_factory=list)
    layout: dict = Field(default_factory=dict)


class DiagramPayload(BaseModel):
    title: str
    diagram_type: ArtifactKind
    mermaid: str
    explanation: str = ""
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ArtifactStartResponse(BaseModel):
    project_id: UUID
    artifact_id: UUID
    kind: ArtifactKind
    status: str
    message: str


class ArtifactStatusResponse(BaseModel):
    project_id: UUID
    artifact_id: UUID
    kind: ArtifactKind
    status: str
    error: str | None = None
    document: EditableDocument | None = None
    diagram: DiagramPayload | None = None


class ArtifactUpdateRequest(BaseModel):
    document: EditableDocument | None = None
    diagram: DiagramPayload | None = None


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def clean_chat_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question is required")
        return value


class ChatEvidence(BaseModel):
    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    similarity: float


class ChatResponse(BaseModel):
    project_id: UUID
    query: str
    answer: str
    evidence: list[ChatEvidence] = Field(default_factory=list)


class RetrievalCallbackRequest(BaseModel):
    project_id: UUID
    user_id: UUID
    source_type: ProjectSourceType
    status: ProjectStatus
    error: str | None = Field(default=None, max_length=500)
