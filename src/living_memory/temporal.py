"""Temporal pattern helpers for memory consolidation."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any, Iterable, Sequence

WEEKDAY_HINTS: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

ERA_MIN_GAP_SECONDS = 7 * 24 * 3600
ERA_GAP_DOMINANCE = 4.0


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


def split_time_regimes(
    moments: Sequence[datetime | None],
    *,
    min_gap_seconds: float = ERA_MIN_GAP_SECONDS,
    dominance: float = ERA_GAP_DOMINANCE,
) -> list[list[int]]:
    """Partition datetimes into disjoint activity regimes, oldest regime first.

    Returns groups of input indices; ``None`` moments are omitted. A regime
    boundary is a silence of at least ``min_gap_seconds`` that also dwarfs the
    cluster's own rhythm (``dominance`` times the median of the other gaps):
    a steady cadence — however slow — never splits, while bursts of activity
    separated by weeks of silence do.
    """

    dated = sorted(
        (moment, index) for index, moment in enumerate(moments) if moment is not None
    )
    if not dated:
        return []

    gaps = [
        (dated[position + 1][0] - dated[position][0]).total_seconds()
        for position in range(len(dated) - 1)
    ]
    boundaries: set[int] = set()
    for position, gap in enumerate(gaps):
        if gap < min_gap_seconds:
            continue
        others = gaps[:position] + gaps[position + 1 :]
        if not others or gap >= dominance * _median(others):
            boundaries.add(position)

    groups: list[list[int]] = [[]]
    for position, (_moment, index) in enumerate(dated):
        groups[-1].append(index)
        if position in boundaries:
            groups.append([])
    return groups


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
