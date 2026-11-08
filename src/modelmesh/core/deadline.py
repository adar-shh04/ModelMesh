"""A wall-clock budget shared across all attempts of a request."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Deadline:
    expires_at: float

    @classmethod
    def in_ms(cls, ms: int) -> Deadline:
        return cls(time.monotonic() + ms / 1000)

    def remaining_s(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    @property
    def expired(self) -> bool:
        return self.remaining_s() <= 0
