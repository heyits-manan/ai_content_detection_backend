from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import aiofiles
from fastapi import UploadFile


@dataclass(frozen=True)
class SavedUpload:
    original_filename: str
    content_type: str
    suffix: str
    temp_path: str
    size_bytes: int


def normalize_content_type(content_type: str | None) -> str:
    return (content_type or "").strip().lower()


def get_file_suffix(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def resolve_upload_suffix(
    *,
    filename: str | None,
    content_type: str | None,
    allowed_extensions: Iterable[str],
    allowed_content_types: Iterable[str],
    content_type_suffix_map: dict[str, str] | None = None,
    generic_prefix: str | None = None,
    default_suffix: str | None = None,
) -> str:
    allowed_extensions_set = {ext.lower() for ext in allowed_extensions}
    allowed_content_types_set = {item.lower() for item in allowed_content_types}
    suffix = get_file_suffix(filename)
    normalized_content_type = normalize_content_type(content_type)
    mapped_suffixes = {k.lower(): v for k, v in (content_type_suffix_map or {}).items()}

    if suffix in allowed_extensions_set:
        return suffix

    mapped_suffix = mapped_suffixes.get(normalized_content_type)
    if mapped_suffix:
        return mapped_suffix

    if generic_prefix and normalized_content_type.startswith(generic_prefix):
        return suffix or (default_suffix or "")

    if not suffix and normalized_content_type in allowed_content_types_set:
        return default_suffix or ""

    return ""


async def save_upload_to_temp(
    *,
    file: UploadFile,
    upload_dir: str,
    prefix: str,
    suffix: str,
    chunk_size: int,
    max_size_bytes: int,
) -> SavedUpload:
    os.makedirs(upload_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix=prefix,
        dir=upload_dir,
    ) as temp_file:
        temp_path = temp_file.name

    total_bytes = 0
    try:
        await file.seek(0)
        async with aiofiles.open(temp_path, "wb") as output_file:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > max_size_bytes:
                    raise ValueError(f"File too large. Max size: {max_size_bytes / (1024 * 1024)}MB")
                await output_file.write(chunk)
    except Exception:
        remove_file_if_exists(temp_path)
        raise

    if total_bytes == 0:
        remove_file_if_exists(temp_path)
        raise ValueError("Uploaded file is empty")

    return SavedUpload(
        original_filename=file.filename or "",
        content_type=normalize_content_type(file.content_type),
        suffix=suffix,
        temp_path=temp_path,
        size_bytes=total_bytes,
    )


def remove_file_if_exists(path: str | None) -> None:
    if path and os.path.exists(path):
        os.remove(path)
