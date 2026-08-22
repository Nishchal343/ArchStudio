import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..config import get_settings
from ..services.storage_service import project_directory

logger = logging.getLogger(__name__)


LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".c": "c", ".h": "c", ".cc": "c++", ".cpp": "c++", ".cxx": "c++", ".hpp": "c++",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".php": "php", ".rb": "ruby", ".kt": "kotlin",
    ".kts": "kotlin", ".swift": "swift", ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
    ".sql": "sql", ".sh": "shell", ".bash": "shell", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".xml": "xml", ".md": "markdown", ".markdown": "markdown",
}

DEFAULT_EXCLUDED_DIRECTORIES = {
    ".git", ".github", "node_modules", "venv", ".venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", "target", "coverage",
    ".next", ".nuxt", ".cache", "vendor", "processed", "generated", "generated-artifacts",
    "artifacts", "out", "release", "__macosx",
}
DEFAULT_EXCLUDED_FILE_PATTERNS = {
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.ico", "*.bmp", "*.tiff", "*.mp3",
    "*.mp4", "*.mov", "*.avi", "*.woff", "*.woff2", "*.ttf", "*.otf", "*.map", "*.min.js", "*.min.css",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "composer.lock", "Gemfile.lock",
}

_PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----", re.IGNORECASE)
_JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_TOKEN_PATTERN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_\-]{20,}|github_pat_[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16})\b")
_DATABASE_URL_PATTERN = re.compile(r"(\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^:/\s]+:)[^@\s]+(@)", re.IGNORECASE)
_ASSIGNMENT_PATTERN = re.compile(
    r"(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|bearer|client[_-]?secret|database[_-]?password|db[_-]?password|password|passwd|secret|secret[_-]?key)\b\s*[:=]\s*[\"']?)([^\s\"'`,;]+)",
    re.IGNORECASE,
)


class ProcessingFatalError(RuntimeError):
    pass


def detect_language(path: Path) -> str:
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "unknown")


def redact_secrets(content: str) -> tuple[str, bool]:
    redacted = content
    redacted, private_changed = _PRIVATE_KEY_PATTERN.subn("[REDACTED_SECRET]", redacted)
    redacted, jwt_changed = _JWT_PATTERN.subn("[REDACTED_SECRET]", redacted)
    redacted, token_changed = _TOKEN_PATTERN.subn("[REDACTED_SECRET]", redacted)
    redacted, database_changed = _DATABASE_URL_PATTERN.subn(r"\1[REDACTED_SECRET]\2", redacted)
    redacted, assignment_changed = _ASSIGNMENT_PATTERN.subn(r"\1[REDACTED_SECRET]", redacted)
    return redacted, any((private_changed, jwt_changed, token_changed, database_changed, assignment_changed))


def _chunk_lines(content: str, max_chunk_size: int) -> list[tuple[int, int, str]]:
    if max_chunk_size < 1:
        raise ValueError("MAX_CHUNK_SIZE must be positive")
    lines = content.splitlines(keepends=True)
    if not lines:
        return []
    chunks: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_length = 0
    start_line = 1
    for line_number, line in enumerate(lines, start=1):
        if current and current_length + len(line) > max_chunk_size:
            chunks.append((start_line, line_number - 1, "".join(current)))
            current = []
            current_length = 0
            start_line = line_number
        if len(line) > max_chunk_size:
            if current:
                chunks.append((start_line, line_number - 1, "".join(current)))
                current = []
                current_length = 0
            for offset in range(0, len(line), max_chunk_size):
                piece = line[offset:offset + max_chunk_size]
                chunks.append((line_number, line_number, piece))
            start_line = line_number + 1
            continue
        current.append(line)
        current_length += len(line)
    if current:
        chunks.append((start_line, len(lines), "".join(current)))
    return chunks


def _is_excluded(path: Path, excluded_patterns: list[str]) -> bool:
    name = path.name.lower()
    if name.startswith("."):
        return True
    return any(fnmatch.fnmatch(name, pattern.lower()) for pattern in excluded_patterns)


def _safe_relative_file(path: Path, root: Path) -> str | None:
    if path.is_symlink():
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return None


def _read_text(path: Path, maximum: int) -> str:
    data = bytearray()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            data.extend(chunk)
            if len(data) > maximum:
                raise ValueError("file exceeds configured size limit")
    if b"\x00" in data:
        raise UnicodeError("binary content")
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return bytes(data).decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError("unsupported text encoding")


def _warning(warnings: list[str], message: str) -> None:
    if len(warnings) < 1000:
        warnings.append(message)


def process_project_files(project_id: str) -> dict:
    settings = get_settings()
    root = project_directory(project_id).resolve()
    if not root.exists() or not root.is_dir():
        raise ProcessingFatalError("Project source directory is missing")
    processed_directory = (root / "processed").resolve()
    try:
        processed_directory.relative_to(root)
    except ValueError as error:
        raise ProcessingFatalError("Invalid project processing directory") from error

    max_file_size = settings.max_processing_file_size_mb * 1024 * 1024
    max_total_size = settings.max_total_source_size_mb * 1024 * 1024
    excluded_directories = {item.lower() for item in settings.processing_excluded_directories} | DEFAULT_EXCLUDED_DIRECTORIES | {"processed"}
    excluded_file_patterns = list(settings.processing_excluded_file_patterns) + list(DEFAULT_EXCLUDED_FILE_PATTERNS)
    temp_directory = Path(tempfile.mkdtemp(prefix=".processed-", dir=root))
    excluded_directories.add(temp_directory.name.lower())
    chunks_directory = temp_directory / "chunks"
    chunks_directory.mkdir()
    warnings: list[str] = []
    files_metadata: list[dict] = []
    language_counts: Counter[str] = Counter()
    total_files = processed_files = skipped_files = total_chunks = 0
    total_source_size = 0
    started = time.monotonic()

    try:
        chunk_path = chunks_directory / "chunks.jsonl"
        with chunk_path.open("w", encoding="utf-8", newline="\n") as chunk_output:
            for current_root, directory_names, filenames in os.walk(root, topdown=True, followlinks=False):
                current_path = Path(current_root)
                directory_names[:] = [
                    name for name in directory_names
                    if name.lower() not in excluded_directories and not (current_path / name).is_symlink()
                ]
                if current_path == processed_directory or processed_directory in current_path.parents:
                    directory_names[:] = []
                    continue
                for filename in filenames:
                    candidate = current_path / filename
                    relative_path = _safe_relative_file(candidate, root)
                    if not relative_path:
                        skipped_files += 1
                        _warning(warnings, "Skipped unsafe or symbolic-link path")
                        continue
                    total_files += 1
                    if total_files > settings.max_project_files:
                        skipped_files += 1
                        _warning(warnings, "Project file count limit exceeded")
                        continue
                    if _is_excluded(candidate, excluded_file_patterns):
                        skipped_files += 1
                        continue
                    try:
                        file_size = candidate.stat().st_size
                    except OSError:
                        skipped_files += 1
                        _warning(warnings, "Skipped file whose metadata could not be read")
                        continue
                    if file_size > max_file_size:
                        skipped_files += 1
                        _warning(warnings, f"Skipped oversized file: {relative_path}")
                        continue
                    if total_source_size + file_size > max_total_size:
                        skipped_files += 1
                        _warning(warnings, "Total source size limit exceeded")
                        continue
                    try:
                        original_content = _read_text(candidate, max_file_size)
                        content, secret_found = redact_secrets(original_content)
                        if secret_found:
                            _warning(warnings, f"Redacted potential secret in: {relative_path}")
                        chunks = _chunk_lines(content, settings.max_chunk_size)
                        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                        language = detect_language(candidate)
                        for index, (start_line, end_line, chunk_content) in enumerate(chunks):
                            chunk_output.write(json.dumps({
                                "project_id": project_id,
                                "file_path": relative_path,
                                "language": language,
                                "chunk_index": index,
                                "start_line": start_line,
                                "end_line": end_line,
                                "content": chunk_content,
                                "content_hash": hashlib.sha256(chunk_content.encode("utf-8")).hexdigest(),
                            }, ensure_ascii=False) + "\n")
                        files_metadata.append({
                            "project_id": project_id,
                            "file_path": relative_path,
                            "language": language,
                            "file_size": file_size,
                            "line_count": len(content.splitlines()),
                            "content_hash": content_hash,
                            "chunk_count": len(chunks),
                        })
                        processed_files += 1
                        total_chunks += len(chunks)
                        total_source_size += file_size
                        language_counts[language] += 1
                    except (UnicodeError, OSError, ValueError) as error:
                        skipped_files += 1
                        _warning(warnings, f"Skipped unreadable file: {relative_path}")
                        logger.debug("File processing warning for project_id=%s type=%s", project_id, type(error).__name__)

        manifest = {
            "project_id": project_id,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "total_files": total_files,
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "total_chunks": total_chunks,
            "languages": dict(sorted(language_counts.items())),
            "warnings": warnings,
            "files": files_metadata,
        }
        with (temp_directory / "manifest.json").open("w", encoding="utf-8") as manifest_output:
            json.dump(manifest, manifest_output, indent=2, ensure_ascii=False)
        if processed_directory.exists():
            shutil.rmtree(processed_directory)
        temp_directory.rename(processed_directory)
        logger.info(
            "Codebase processing completed project_id=%s processed_files=%s skipped_files=%s total_chunks=%s duration_seconds=%.3f",
            project_id, processed_files, skipped_files, total_chunks, time.monotonic() - started,
        )
        return manifest
    except ProcessingFatalError:
        shutil.rmtree(temp_directory, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(temp_directory, ignore_errors=True)
        raise ProcessingFatalError("Project processing failed") from error
