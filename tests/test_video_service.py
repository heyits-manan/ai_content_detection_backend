import math

from io import BytesIO

import cv2
import numpy as np
import pytest
from starlette.datastructures import Headers, UploadFile

from app.core.file_handler import resolve_upload_suffix
from app.services.video_service import VideoDetectionService, aggregate_scores, extract_frames, process_frames_parallel


def test_aggregate_scores_uses_max_mean_hybrid():
    frame_results = [
        {"frame_index": 0, "probability": 0.2},
        {"frame_index": 1, "probability": 0.5},
        {"frame_index": 2, "probability": 0.9},
    ]

    expected = (0.7 * 0.9) + (0.3 * ((0.2 + 0.5 + 0.9) / 3))
    assert math.isclose(aggregate_scores(frame_results), expected)


def test_process_frames_parallel_sorts_results_by_frame_index():
    frames = [
        (5, np.zeros((8, 8, 3), dtype=np.uint8)),
        (1, np.full((8, 8, 3), 255, dtype=np.uint8)),
    ]

    def fake_inference(frame: np.ndarray) -> float:
        return float(frame.mean() / 255.0)

    results = process_frames_parallel(
        frames=frames,
        max_workers=2,
        inference_timeout_seconds=5.0,
        inference_fn=fake_inference,
    )

    assert results == [
        {"frame_index": 1, "probability": 1.0},
        {"frame_index": 5, "probability": 0.0},
    ]


def test_extract_frames_uses_all_available_frames_for_short_videos(tmp_path):
    video_path = tmp_path / "short.avi"
    frame_size = (16, 16)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        frame_size,
    )

    if not writer.isOpened():
        pytest.skip("OpenCV video writer codec is unavailable in this environment")

    for value in range(4):
        frame = np.full((frame_size[1], frame_size[0], 3), value * 50, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames = extract_frames(str(video_path), num_frames=16)

    assert [frame_index for frame_index, _ in frames] == [0, 1, 2, 3]
    assert all(frame.shape == (16, 16, 3) for _, frame in frames)


def test_validate_upload_accepts_mp4_content_type_without_extension():
    upload = UploadFile(
        file=BytesIO(b"fake video bytes"),
        filename="blob",
        headers=Headers({"content-type": "video/mp4"}),
    )

    assert resolve_upload_suffix(
        filename=upload.filename,
        content_type=upload.content_type,
        allowed_extensions=[".mp4"],
        allowed_content_types=["video/mp4"],
        content_type_suffix_map={"video/mp4": ".mp4"},
        generic_prefix="video/",
        default_suffix=".mp4",
    ) == ".mp4"
