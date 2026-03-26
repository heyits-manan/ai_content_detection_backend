from __future__ import annotations

from pathlib import Path

from app.workers import tasks


def test_process_video_job_returns_mock_result_and_removes_temp_file(monkeypatch, tmp_path):
    video_path = tmp_path / "queued-video.mp4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(tasks.settings, "ENABLE_REAL_INFERENCE", False)
    monkeypatch.setattr(tasks.settings, "MOCK_VIDEO_PROCESSING_SECONDS", 0.0)

    result = tasks.process_video_job(str(video_path), 4)

    assert result["aggregation"] == "mock_mean"
    assert result["num_frames_used"] == 4
    assert len(result["frame_results"]) == 4
    assert not video_path.exists()
