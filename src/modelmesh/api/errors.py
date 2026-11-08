"""Renders every failure as the contract's error envelope."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from modelmesh.core.context import RequestContext, get_request_context
from modelmesh.core.errors import ApiError

logger = logging.getLogger("modelmesh.api")


def error_body(error: ApiError, ctx: RequestContext) -> dict[str, Any]:
    return {
        "error": {
            "type": error.error_type,
            "code": error.code,
            "message": error.message,
            "param": error.param,
            "request_id": ctx.request_id,
            "client_request_id": ctx.client_request_id,
            "retryable": error.retryable,
            "details": error.details,
        }
    }


def _context(request: Request) -> RequestContext:
    ctx = getattr(request.state, "request_context", None) or get_request_context()
    return ctx if ctx is not None else RequestContext()


def render_error(request: Request, error: ApiError) -> JSONResponse:
    headers = dict(error.headers)
    if error.retry_after_s is not None:
        headers["Retry-After"] = str(error.retry_after_s)
    return JSONResponse(error_body(error, _context(request)), status_code=error.status_code, headers=headers)


def invalid_query(exc: ValidationError | RequestValidationError) -> ApiError:
    first = exc.errors()[0]
    location = [str(part) for part in first.get("loc", ()) if part not in ("query", "path", "body", "header")]
    param = ".".join(location) or None
    return ApiError("invalid_request", f"Invalid value for '{param}'.", code="invalid_value", param=param)


def install_error_handlers(app: FastAPI) -> None:
    async def on_api_error(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, ApiError)
        return render_error(request, exc)

    async def on_validation(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, RequestValidationError)
        return render_error(request, invalid_query(exc))

    async def on_http(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, StarletteHTTPException)
        if exc.status_code == 404:
            error = ApiError("not_found", "No such endpoint.", code="endpoint_not_found")
        elif exc.status_code == 405:
            error = ApiError("invalid_request", "Method not allowed for this endpoint.", code="invalid_value")
        else:
            error = ApiError("internal_error", "Unexpected error.", retryable=True)
        return render_error(request, error)

    async def on_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return render_error(request, ApiError("internal_error", "An unexpected error occurred.", retryable=True))

    app.add_exception_handler(ApiError, on_api_error)
    app.add_exception_handler(RequestValidationError, on_validation)
    app.add_exception_handler(StarletteHTTPException, on_http)
    app.add_exception_handler(Exception, on_unexpected)
