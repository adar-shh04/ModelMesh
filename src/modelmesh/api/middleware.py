"""Pure ASGI middleware: request identity, contract headers and the body-size limit.

Written at the ASGI level because BaseHTTPMiddleware interferes with streaming and client-disconnect handling.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from modelmesh.api.errors import error_body
from modelmesh.core.context import RequestContext, set_request_context
from modelmesh.core.errors import ApiError
from modelmesh.core.settings import CONTRACT_VERSION

_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class RequestIdentityMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope["headers"]}
        supplied = headers.get("x-request-id")
        client_request_id = supplied if supplied and _VALID_REQUEST_ID.match(supplied) else None
        ctx = RequestContext(client_request_id=client_request_id, generated_client_request_id=str(uuid.uuid4()))
        set_request_context(ctx)
        scope.setdefault("state", {})["request_context"] = ctx

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                extra = [
                    (b"x-modelmesh-request-id", ctx.request_id.encode()),
                    (b"x-request-id", ctx.effective_client_request_id.encode()),
                    (b"x-modelmesh-contract", CONTRACT_VERSION.encode()),
                ]
                message["headers"] = [*message.get("headers", []), *extra]
            await send(message)

        declared = headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            await self._reject(ctx, send_with_headers)
            return

        received = 0
        too_large = False

        async def limited_receive() -> Message:
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    too_large = True
                    message = {"type": "http.request", "body": b"", "more_body": False}
            return message

        scope["state"]["body_limit_exceeded"] = lambda: too_large
        await self.app(scope, limited_receive, send_with_headers)

    @staticmethod
    async def _reject(ctx: RequestContext, send: Any) -> None:
        error = ApiError("request_too_large", "The request body is too large.", code="body_too_large")
        payload = json.dumps(error_body(error, ctx)).encode()
        await send(
            {
                "type": "http.response.start",
                "status": error.status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})
