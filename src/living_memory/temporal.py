"""Temporal pattern helpers for memory consolidation."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any, Iterable

WEEKDAY_HINTS: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


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


def detect_weekly_hint(
    timestamps: Iterable[Any],
    *,
    min_observations: int = 3,
    dominance: float = 0.75,
) -> str | None:
    """Return a weekly hint when one weekday dominates the observations."""

    weekdays: list[int] = []
    for timestamp in timestamps:
        parsed = parse_timestamp(timestamp)
        if parsed is not None:
            weekdays.append(parsed.weekday())

    if len(weekdays) < min_observations:
        return None

    weekday, count = Counter(weekdays).most_common(1)[0]
    if count / len(weekdays) < dominance:
        return None
    return f"weekly:{WEEKDAY_HINTS[weekday]}"


def detect_temporal_hint(observations: Iterable[Any]) -> str | None:
    """Detect temporal hints from nodes or raw timestamp values."""

    timestamps: list[Any] = []
    for observation in observations:
        timestamps.append(getattr(observation, "timestamp", observation))
    return detect_weekly_hint(timestamps)
