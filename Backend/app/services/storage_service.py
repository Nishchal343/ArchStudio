import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from fastapi import UploadFile

from ..config import get_settings


class StorageValidationError(ValueError):
    pass


def storage_root() -> Path:
    root = Path(get_settings().project_storage_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


async def save_upload_to_temp(upload: UploadFile) -> tuple[Path, int]:
    maximum = get_settings().max_upload_size_mb * 1024 * 1024
    root = storage_root()
    fd, raw_path = tempfile.mkstemp(prefix="upload-", suffix=".tmp", dir=root)
    path = Path(raw_path)
    size = 0
    try:
        with os.fdopen(fd, "wb") as output:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > maximum:
                    raise StorageValidationError("ZIP upload exceeds the maximum allowed size.")
                output.write(chunk)
        if size == 0:
            raise StorageValidationError("Uploaded ZIP file is empty.")
        return path, size
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def _safe_member_path(member_name: str, destination: Path) -> Path:
    normalized = member_name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if not normalized or relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise StorageValidationError("ZIP archive contains an unsafe path.")
    if any(":" in part for part in relative.parts):
        raise StorageValidationError("ZIP archive contains an unsafe path.")
    target = (destination / Path(*relative.parts)).resolve()
    try:
        target.relative_to(destination.resolve())
    except ValueError as error:
        raise StorageValidationError("ZIP archive contains an unsafe path.") from error
    return target


def validate_zip_archive(archive_path: Path) -> None:
    if not zipfile.is_zipfile(archive_path):
        raise StorageValidationError("Uploaded file is not a valid ZIP archive.")

    maximum_extracted = get_settings().max_extracted_size_mb * 1024 * 1024
    regular_file_count = 0
    extracted_size = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                _safe_member_path(member.filename, storage_root())
                mode = (member.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise StorageValidationError("ZIP archive contains an unsafe symbolic link.")
                if not member.is_dir():
                    regular_file_count += 1
                    extracted_size += member.file_size
                    if extracted_size > maximum_extracted:
                        raise StorageValidationError("ZIP archive expands beyond the maximum allowed size.")
    except (zipfile.BadZipFile, OSError, RuntimeError) as error:
        raise StorageValidationError("Uploaded ZIP archive is malformed or corrupted.") from error

    if regular_file_count == 0:
        raise StorageValidationError("ZIP archive does not contain any files.")


def extract_zip_archive(archive_path: Path, project_directory: Path) -> None:
    project_directory.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = _safe_member_path(member.filename, project_directory)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except Exception:
        shutil.rmtree(project_directory, ignore_errors=True)
        raise


def project_directory(project_id: str) -> Path:
    root = storage_root()
    directory = (root / project_id).resolve()
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise StorageValidationError("Invalid project storage path.") from error
    return directory


def remove_project_directory(project_id: str) -> None:
    directory = project_directory(project_id)
    if directory.exists():
        shutil.rmtree(directory)
