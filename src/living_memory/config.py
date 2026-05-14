"""Configuration loading for the Living Memory storage foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

from living_memory.embeddings import DEFAULT_EMBEDDING_MODEL


@dataclass(frozen=True, slots=True)
class RetrievalWeightConfig:
    bm25: float
    vector: float
    graph: float
    learning_rate: float = 0.05


DEFAULT_RETRIEVAL_WEIGHTS: dict[str, RetrievalWeightConfig] = {
    "default": RetrievalWeightConfig(bm25=1.0, vector=0.0, graph=0.0),
    "project": RetrievalWeightConfig(bm25=0.7, vector=0.3, graph=0.0),
    "global": RetrievalWeightConfig(bm25=0.4, vector=0.4, graph=0.2),
    "session": RetrievalWeightConfig(bm25=0.8, vector=0.2, graph=0.0),
}

DEFAULT_PHASE_THRESHOLDS: dict[int, int] = {
    0: 0,
    1: 1,
    2: 100,
    3: 10_000,
    4: 100_000,
    5: 1_000_000,
}


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    db_path: Path = Path("living_memory.sqlite3")
    default_scope: str = "global"
    trace_ttl_days: int = 180
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    phase_thresholds: dict[int, int] = field(default_factory=lambda: dict(DEFAULT_PHASE_THRESHOLDS))
    retrieval_weights: dict[str, RetrievalWeightConfig] = field(
        default_factory=lambda: dict(DEFAULT_RETRIEVAL_WEIGHTS)
    )

    @classmethod
    def from_toml(cls, path: str | Path) -> "MemoryConfig":
        return load_config(path)


def load_config(path: str | Path | None = None) -> MemoryConfig:
    """Load TOML configuration, returning defaults when no path is provided."""

    if path is None:
        return MemoryConfig()

    config_path = Path(path)
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    defaults = MemoryConfig()
    storage = _table(raw, "storage")
    embeddings = _table(raw, "embeddings")
    phases = _table(raw, "phases")
    retrieval = _table(raw, "retrieval_weights")

    phase_thresholds = dict(DEFAULT_PHASE_THRESHOLDS)
    for key, value in phases.items():
        phase_thresholds[int(key)] = int(value)

    retrieval_weights = dict(DEFAULT_RETRIEVAL_WEIGHTS)
    for scope, values in retrieval.items():
        if not isinstance(values, dict):
            raise ValueError(f"retrieval_weights.{scope} must be a TOML table")
        retrieval_weights[scope] = RetrievalWeightConfig(
            bm25=float(values.get("bm25", DEFAULT_RETRIEVAL_WEIGHTS["default"].bm25)),
            vector=float(values.get("vector", DEFAULT_RETRIEVAL_WEIGHTS["default"].vector)),
            graph=float(values.get("graph", DEFAULT_RETRIEVAL_WEIGHTS["default"].graph)),
            learning_rate=float(
                values.get("learning_rate", DEFAULT_RETRIEVAL_WEIGHTS["default"].learning_rate)
            ),
        )

    return MemoryConfig(
        db_path=Path(storage.get("db_path", defaults.db_path)),
        default_scope=str(storage.get("default_scope", defaults.default_scope)),
        trace_ttl_days=int(storage.get("trace_ttl_days", defaults.trace_ttl_days)),
        embedding_model=str(
            embeddings.get(
                "model",
                embeddings.get(
                    "embedding_model",
                    storage.get(
                        "embedding_model",
                        raw.get("embedding_model", defaults.embedding_model),
                    ),
                ),
            )
        ),
        phase_thresholds=phase_thresholds,
        retrieval_weights=retrieval_weights,
    )


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a TOML table")
    return value
