"""A minimal Server-Sent Events parser for upstream OpenAI-compatible streams."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator


async def iter_sse_data(lines: AsyncIterable[str]) -> AsyncIterator[str]:
    """Yield each event's data payload. Comment lines such as keep-alives are skipped."""
    buffer: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if buffer:
                yield "\n".join(buffer)
                buffer = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            buffer.append(value[1:] if value.startswith(" ") else value)
    if buffer:
        yield "\n".join(buffer)
