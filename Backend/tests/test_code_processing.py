import json
from pathlib import Path

import pytest

from app.services.code_processing import ProcessingFatalError, process_project_files, redact_secrets
from app.services.storage_service import project_directory


def setup_project(tmp_path, monkeypatch, project_id="project-1") -> Path:
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "project_storage_root", str(tmp_path / "projects"))
    monkeypatch.setattr(settings, "max_chunk_size", 40)
    root = project_directory(project_id)
    root.mkdir(parents=True)
    return root


def test_processing_discovers_filters_redacts_and_chunks(tmp_path, monkeypatch):
    root = setup_project(tmp_path, monkeypatch)
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text('password = "do-not-persist"\n' + "print('hello')\n" * 8, encoding="utf-8")
    (root / "src" / ".env").write_text("SECRET=ignored", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "ignored.js").write_text("ignored", encoding="utf-8")
    (root / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")

    manifest = process_project_files("project-1")
    assert manifest["processed_files"] == 1
    assert manifest["skipped_files"] >= 1
    assert manifest["languages"] == {"python": 1}
    assert manifest["files"][0]["chunk_count"] > 1

    chunks = (root / "processed" / "chunks" / "chunks.jsonl").read_text(encoding="utf-8")
    assert "do-not-persist" not in chunks
    assert "[REDACTED_SECRET]" in chunks
    assert "project-1" in chunks
    assert "package-lock.json" not in chunks


def test_processing_is_deterministic_for_metadata_and_chunks(tmp_path, monkeypatch):
    root = setup_project(tmp_path, monkeypatch)
    (root / "example.ts").write_text("const value = 1;\n", encoding="utf-8")
    first = process_project_files("project-1")
    first_chunks = (root / "processed" / "chunks" / "chunks.jsonl").read_text(encoding="utf-8")
    second = process_project_files("project-1")
    second_chunks = (root / "processed" / "chunks" / "chunks.jsonl").read_text(encoding="utf-8")
    assert first["files"] == second["files"]
    assert first["languages"] == second["languages"]
    assert first_chunks == second_chunks


def test_processing_skips_binary_and_symlink_when_supported(tmp_path, monkeypatch):
    root = setup_project(tmp_path, monkeypatch)
    (root / "image.bin").write_bytes(b"\x00\x01\x02")
    try:
        (root / "link.py").symlink_to(root / "missing.py")
    except (OSError, NotImplementedError):
        pass
    manifest = process_project_files("project-1")
    assert manifest["processed_files"] == 0
    assert manifest["skipped_files"] >= 1


def test_processing_missing_project_directory_is_fatal(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "project_storage_root", str(tmp_path / "projects"))
    with pytest.raises(ProcessingFatalError):
        process_project_files("missing")


def test_processing_file_size_limit_is_enforced(tmp_path, monkeypatch):
    root = setup_project(tmp_path, monkeypatch)
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_processing_file_size_mb", 0)
    (root / "large.py").write_text("content", encoding="utf-8")
    manifest = process_project_files("project-1")
    assert manifest["processed_files"] == 0
    assert manifest["skipped_files"] == 1


def test_secret_patterns_are_redacted_without_returning_values():
    source = """api_key = \"key-value\"\njwt = eyJabcdefghijk.lmnopqrstuv.wxyz123456\nurl = postgres://admin:database-password@db.example.test/app\n-----BEGIN PRIVATE KEY-----\nprivate-value\n-----END PRIVATE KEY-----"""
    redacted, changed = redact_secrets(source)
    assert changed is True
    assert "key-value" not in redacted
    assert "database-password" not in redacted
    assert "private-value" not in redacted
    assert redacted.count("[REDACTED_SECRET]") >= 4
