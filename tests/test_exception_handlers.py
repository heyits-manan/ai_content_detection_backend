from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded

from app.core.exceptions import (
    AppError,
    BadRequestError,
    ErrorCode,
    InferenceFailedError,
    REQUEST_ID_HEADER,
    UnprocessableEntityError,
    add_request_id_middleware,
    app_error_handler,
    http_exception_handler,
    rate_limit_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)


def create_test_app() -> TestClient:
    app = FastAPI()
    app.middleware("http")(add_request_id_middleware)
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.get("/app-error")
    async def app_error():
        raise BadRequestError("Invalid media payload", details={"field": "file"})

    @app.get("/inference-error")
    async def inference_error():
        raise InferenceFailedError("Inference timed out", details={"kind": "timeout"})

    @app.get("/unprocessable")
    async def unprocessable():
        raise UnprocessableEntityError("Media could not be decoded")

    @app.get("/http-error")
    async def http_error():
        raise HTTPException(status_code=404, detail="Not found")

    @app.get("/boom")
    async def boom():
        raise RuntimeError("unexpected")

    return TestClient(app, raise_server_exceptions=False)


def test_app_error_uses_stable_error_schema():
    client = create_test_app()

    response = client.get("/app-error", headers={REQUEST_ID_HEADER: "req-123"})

    assert response.status_code == 400
    assert response.headers[REQUEST_ID_HEADER] == "req-123"
    assert response.json() == {
        "success": False,
        "error": {
            "code": "bad_request",
            "message": "Invalid media payload",
            "request_id": "req-123",
            "details": {"field": "file"},
        },
    }


def test_http_exception_uses_stable_error_schema():
    client = create_test_app()

    response = client.get("/http-error")
    body = response.json()

    assert response.status_code == 404
    assert body["success"] is False
    assert body["error"]["code"] == "bad_request"
    assert body["error"]["message"] == "Not found"
    assert body["error"]["request_id"]


def test_inference_error_uses_typed_error_code():
    client = create_test_app()

    response = client.get("/inference-error")
    body = response.json()

    assert response.status_code == 500
    assert body["error"]["code"] == "inference_failed"
    assert body["error"]["message"] == "Inference timed out"
    assert body["error"]["details"] == {"kind": "timeout"}


def test_unprocessable_error_uses_422_status():
    client = create_test_app()

    response = client.get("/unprocessable")
    body = response.json()

    assert response.status_code == 422
    assert body["error"]["code"] == "unprocessable_entity"
    assert body["error"]["message"] == "Media could not be decoded"


def test_unhandled_exception_uses_internal_server_error_shape():
    client = create_test_app()

    response = client.get("/boom")
    body = response.json()

    assert response.status_code == 500
    assert body["success"] is False
    assert body["error"]["code"] == "internal_server_error"
    assert body["error"]["message"] == "Internal server error"
    assert body["error"]["request_id"]
