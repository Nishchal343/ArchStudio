import enum


class ProjectSourceType(str, enum.Enum):
    ZIP = "zip"
    GITHUB = "github"


class ProjectStatus(str, enum.Enum):
    CREATED = "created"
    VALIDATING = "validating"
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    PROCESSED = "processed"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"


class AnalysisStatus(str, enum.Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactKind(str, enum.Enum):
    REPORT = "report"
    IEEE_REPORT = "ieee_report"
    RESEARCH_PAPER = "research_paper"
    DATABASE_DIAGRAM = "database_diagram"
    ARCHITECTURE_DIAGRAM = "architecture_diagram"


class ArtifactStatus(str, enum.Enum):
    QUEUED = "queued"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
