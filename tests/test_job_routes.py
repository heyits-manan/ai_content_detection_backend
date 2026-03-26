from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import job_routes
from app.core.rate_limit import limiter


class FakeVideoService:
    async def save_validated_upload(self, file):
        return "/tmp/video_job_test.mp4"


class FakeQueueJob:
    def __init__(self, job_id: str):
        self.id = job_id


class FakeQueue:
    def enqueue(self, func, temp_path, num_frames, **kwargs):
        assert temp_path == "/tmp/video_job_test.mp4"
        assert num_frames == 8
        assert kwargs["meta"] == {"status": "queued"}
        return FakeQueueJob("job-123")


class FakeFetchedJob:
    def __init__(self, status: str, result=None, exc_info: str | None = None, meta=None):
        self._status = status
        self.result = result
        self.exc_info = exc_info
        self.meta = meta or {}

    def get_status(self, refresh: bool = False) -> str:
        return self._status


def create_test_client() -> TestClient:
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(job_routes.router, prefix="/api/v1")
    app.dependency_overrides[job_routes.get_video_service] = lambda: FakeVideoService()
    return TestClient(app)


def test_create_video_job_returns_enqueued_job_id(monkeypatch):
    client = create_test_client()
    monkeypatch.setattr(job_routes, "get_video_queue", lambda: FakeQueue())

    response = client.post(
        "/api/v1/video/jobs",
        files={"file": ("clip.mp4", b"video-bytes", "video/mp4")},
        data={"num_frames": "8"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "job_id": "job-123",
        "status": "queued",
    }


def test_get_job_status_returns_completed_result(monkeypatch):
    client = create_test_client()
    monkeypatch.setattr(
        job_routes,
        "fetch_job_by_id",
        lambda job_id: FakeFetchedJob(
            "finished",
            result={
                "ai_probability": 0.81,
                "frame_results": [{"frame_index": 0, "probability": 0.81}],
                "aggregation": "max_mean_hybrid",
                "num_frames_used": 1,
                "processing_time_ms": 123.45,
            },
        ),
    )

    response = client.get("/api/v1/jobs/job-123")

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "job_id": "job-123",
        "status": "completed",
        "result": {
            "ai_probability": 0.81,
            "frame_results": [{"frame_index": 0, "probability": 0.81}],
            "aggregation": "max_mean_hybrid",
            "num_frames_used": 1,
            "processing_time_ms": 123.45,
        },
        "error": None,
    }


def test_get_job_status_returns_failed_error(monkeypatch):
    client = create_test_client()
    monkeypatch.setattr(
        job_routes,
        "fetch_job_by_id",
        lambda job_id: FakeFetchedJob(
            "failed",
            exc_info="Traceback\nRuntimeError: model crashed",
        ),
    )

    response = client.get("/api/v1/jobs/job-123")

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "job_id": "job-123",
        "status": "failed",
        "result": None,
        "error": "RuntimeError: model crashed",
    }
