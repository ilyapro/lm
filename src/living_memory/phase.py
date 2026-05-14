"""Automatic trace-count phase detection."""

from __future__ import annotations

from collections.abc import Mapping

from living_memory.config import DEFAULT_PHASE_THRESHOLDS
from living_memory.models import PhaseInfo


PHASE_NAMES: dict[int, str] = {
    0: "passthrough",
    1: "full-text",
    2: "semantic-ready",
    3: "graph-activation",
    4: "hierarchical-routing",
    5: "federation-ready",
}

PHASE_FEATURES: dict[int, tuple[str, ...]] = {
    0: ("append-only trace logging", "linear inspection"),
    1: ("fts5 bm25 recall", "scope filtering"),
    2: ("lazy local embeddings", "pattern detection"),
    3: ("concept graph", "activation traversal"),
    4: ("hierarchical sharding hooks", "routed retrieval"),
    5: ("federation hooks", "learned retrieval policy"),
}


class PhaseManager:
    """Maps trace counts to additive memory capability phases."""

    def __init__(self, thresholds: Mapping[int, int] | None = None) -> None:
        configured = dict(thresholds or DEFAULT_PHASE_THRESHOLDS)
        if 0 not in configured or configured[0] != 0:
            raise ValueError("phase 0 threshold must be 0")
        self._thresholds = dict(sorted((int(k), int(v)) for k, v in configured.items()))

    @property
    def thresholds(self) -> dict[int, int]:
        return dict(self._thresholds)

    def detect(self, trace_count: int) -> PhaseInfo:
        if trace_count < 0:
            raise ValueError("trace_count must be non-negative")

        phase_number = 0
        min_traces = 0
        for candidate, threshold in self._thresholds.items():
            if trace_count >= threshold and candidate >= phase_number:
                phase_number = candidate
                min_traces = threshold

        features: list[str] = []
        for phase in range(phase_number + 1):
            features.extend(PHASE_FEATURES.get(phase, ()))

        return PhaseInfo(
            number=phase_number,
            name=PHASE_NAMES.get(phase_number, f"custom-{phase_number}"),
            trace_count=trace_count,
            min_traces=min_traces,
            features=tuple(features),
        )
