"""Per-request context, available to every layer through a context variable."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

from modelmesh.core.ids import new_request_id


@dataclass
class RequestContext:
    request_id: str = field(default_factory=new_request_id)
    client_request_id: str | None = None
    generated_client_request_id: str | None = None

    @property
    def effective_client_request_id(self) -> str:
        return self.client_request_id or self.generated_client_request_id or self.request_id


_current: ContextVar[RequestContext | None] = ContextVar("modelmesh_request_context", default=None)


def set_request_context(ctx: RequestContext) -> None:
    _current.set(ctx)


def get_request_context() -> RequestContext | None:
    return _current.get()
