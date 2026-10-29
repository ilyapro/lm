"""Timestamp parsing shared by memory storage and maintenance."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def parse_timestamp(value: Any) -> datetime | None:
    """Parse common ISO-like timestamps used by memory nodes."""

    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
