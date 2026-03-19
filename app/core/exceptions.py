from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette import status

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_STATE_KEY = "request_id"


class ErrorCode(StrEnum):
    BAD_REQUEST = "bad_request"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNPROCESSABLE_ENTITY = "unprocessable_entity"
    INFERENCE_FAILED = "inference_failed"
    INTERNAL_SERVER_ERROR = "internal_server_error"


class AppError(Exception):
    def __init__(
        self,
        *,
        code: ErrorCode,
        message: str,
        status_code: int,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


class BadRequestError(AppError):
    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(
            code=ErrorCode.BAD_REQUEST,
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class ValidationFailedError(AppError):
    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class UnprocessableEntityError(AppError):
    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(
            code=ErrorCode.UNPROCESSABLE_ENTITY,
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class InferenceFailedError(AppError):
    def __init__(
        self,
        message: str,
        details: Any | None = None,
        *,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ) -> None:
        super().__init__(
            code=ErrorCode.INFERENCE_FAILED,
            message=message,
            status_code=status_code,
            details=details,
        )


def get_request_id(request: Request) -> str:
    existing = getattr(request.state, REQUEST_ID_STATE_KEY, None)
    if existing:
        return existing

    header_value = request.headers.get(REQUEST_ID_HEADER)
    request_id = header_value.strip() if header_value else uuid4().hex
    setattr(request.state, REQUEST_ID_STATE_KEY, request_id)
    return request_id


def build_error_response(
    *,
    request: Request,
    status_code: int,
    code: ErrorCode,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
    request_id = get_request_id(request)
    payload = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": details,
        },
    }
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers={REQUEST_ID_HEADER: request_id},
    )


async def add_request_id_middleware(request: Request, call_next):
    request_id = get_request_id(request)
    response = await call_next(request)
    response.headers.setdefault(REQUEST_ID_HEADER, request_id)
    return response


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning(
        "Application error: code=%s status_code=%s path=%s request_id=%s details=%s",
        exc.code,
        exc.status_code,
        request.url.path,
        get_request_id(request),
        exc.details,
    )
    return build_error_response(
        request=request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    details = exc.detail if isinstance(exc.detail, dict | list) else None
    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return build_error_response(
        request=request,
        status_code=exc.status_code,
        code=ErrorCode.BAD_REQUEST if exc.status_code < 500 else ErrorCode.INTERNAL_SERVER_ERROR,
        message=message,
        details=details,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return build_error_response(
        request=request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        details=exc.errors(),
    )


async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    detail = getattr(exc, "detail", None)
    message = detail if isinstance(detail, str) else "Rate limit exceeded"
    return build_error_response(
        request=request,
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        code=ErrorCode.RATE_LIMIT_EXCEEDED,
        message=message,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = get_request_id(request)
    logger.exception(
        "Unhandled exception: path=%s request_id=%s",
        request.url.path,
        request_id,
        exc_info=exc,
    )
    return build_error_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code=ErrorCode.INTERNAL_SERVER_ERROR,
        message="Internal server error",
    )
