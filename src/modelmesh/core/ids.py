"""Identifier generation."""

from __future__ import annotations

import secrets

from ulid import ULID


def new_request_id() -> str:
    return f"mmr_{ULID()}"


def new_feedback_id() -> str:
    return f"fb_{ULID()}"


def new_tenant_id() -> str:
    return f"ten_{secrets.token_hex(12)}"


def new_api_key_id() -> str:
    return f"key_{secrets.token_hex(12)}"


def new_completion_id(request_id: str) -> str:
    return f"chatcmpl-{request_id}"
