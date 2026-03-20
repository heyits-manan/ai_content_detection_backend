import asyncio
from io import BytesIO

import pytest
from starlette.datastructures import Headers, UploadFile

from app.core.file_handler import resolve_upload_suffix, save_upload_to_temp


def test_resolve_upload_suffix_accepts_extension():
    suffix = resolve_upload_suffix(
        filename="sample.png",
        content_type="application/octet-stream",
        allowed_extensions=[".png"],
        allowed_content_types=["application/octet-stream"],
    )

    assert suffix == ".png"


def test_resolve_upload_suffix_accepts_content_type_mapping_without_extension():
    suffix = resolve_upload_suffix(
        filename="blob",
        content_type="video/mp4",
        allowed_extensions=[".mp4"],
        allowed_content_types=["video/mp4"],
        content_type_suffix_map={"video/mp4": ".mp4"},
        generic_prefix="video/",
        default_suffix=".mp4",
    )

    assert suffix == ".mp4"


def test_save_upload_to_temp_rejects_empty_upload(tmp_path):
    upload = UploadFile(
        file=BytesIO(b""),
        filename="empty.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )

    async def run_test():
        with pytest.raises(ValueError, match="Uploaded file is empty"):
            await save_upload_to_temp(
                file=upload,
                upload_dir=str(tmp_path),
                prefix="audio_",
                suffix=".wav",
                chunk_size=1024,
                max_size_bytes=2048,
            )

    asyncio.run(run_test())


def test_save_upload_to_temp_writes_file_and_tracks_size(tmp_path):
    upload = UploadFile(
        file=BytesIO(b"123456"),
        filename="sample.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )

    async def run_test():
        return await save_upload_to_temp(
            file=upload,
            upload_dir=str(tmp_path),
            prefix="audio_",
            suffix=".wav",
            chunk_size=2,
            max_size_bytes=1024,
        )

    saved = asyncio.run(run_test())
    assert saved.size_bytes == 6
    assert saved.suffix == ".wav"
