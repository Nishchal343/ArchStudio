import io
import asyncio
import zipfile

import httpx
import pytest
from starlette.datastructures import UploadFile

from app.services.github_service import GitHubNotFoundError, GitHubPrivateError, GitHubValidationError, normalize_repository_url, validate_public_repository
from app.services.storage_service import StorageValidationError, extract_zip_archive, save_upload_to_temp, validate_zip_archive


def make_zip(*entries: tuple[str, bytes]) -> io.BytesIO:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for filename, content in entries:
            archive.writestr(filename, content)
    stream.seek(0)
    return stream


def test_valid_zip_is_validated_and_extracted(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "project_storage_root", str(tmp_path / "projects"))
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(make_zip(("src/main.py", b"print('safe')")).getvalue())
    destination = tmp_path / "project"

    validate_zip_archive(archive_path)
    extract_zip_archive(archive_path, destination)
    assert (destination / "src" / "main.py").read_bytes() == b"print('safe')"


def test_non_zip_and_malformed_zip_are_rejected(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "project_storage_root", str(tmp_path / "projects"))
    text_file = tmp_path / "not-a-zip.zip"
    text_file.write_text("not a zip")
    with pytest.raises(StorageValidationError):
        validate_zip_archive(text_file)

    malformed = tmp_path / "malformed.zip"
    malformed.write_bytes(b"PK\x03\x04broken")
    with pytest.raises(StorageValidationError):
        validate_zip_archive(malformed)


def test_zip_slip_entry_is_rejected(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "project_storage_root", str(tmp_path / "projects"))
    archive_path = tmp_path / "unsafe.zip"
    archive_path.write_bytes(make_zip(("../../outside.txt", b"unsafe")).getvalue())
    with pytest.raises(StorageValidationError):
        validate_zip_archive(archive_path)


def test_upload_size_limit_is_enforced(tmp_path, monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "project_storage_root", str(tmp_path / "projects"))
    monkeypatch.setattr(settings, "max_upload_size_mb", 0)
    upload = UploadFile(file=io.BytesIO(b"data"), filename="source.zip")
    with pytest.raises(StorageValidationError):
        asyncio.run(save_upload_to_temp(upload))


def test_github_url_normalization():
    assert normalize_repository_url("https://github.com/owner/repo/") == ("https://github.com/owner/repo", "owner", "repo")
    assert normalize_repository_url("https://www.github.com/owner/repo.git") == ("https://github.com/owner/repo", "owner", "repo")
    with pytest.raises(GitHubValidationError):
        normalize_repository_url("https://example.com/owner/repo")


class FakeGitHubClient:
    def __init__(self, response):
        self.response = response

    async def get(self, *_args, **_kwargs):
        return self.response


def test_github_not_found_and_private_errors_are_clean():
    not_found = httpx.Response(404, request=httpx.Request("GET", "https://api.github.com/repos/a/b"))
    with pytest.raises(GitHubNotFoundError):
        asyncio.run(validate_public_repository("https://github.com/a/b", FakeGitHubClient(not_found)))

    private = httpx.Response(200, json={"private": True}, request=httpx.Request("GET", "https://api.github.com/repos/a/b"))
    with pytest.raises(GitHubPrivateError):
        asyncio.run(validate_public_repository("https://github.com/a/b", FakeGitHubClient(private)))
