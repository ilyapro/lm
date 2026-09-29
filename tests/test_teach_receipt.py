"""``memory_teach`` answers with a receipt, not an echo of what it stored.

The caller already holds both texts: the original came from recall, the
correction it wrote itself. The response carries only what the caller lacks —
ids, the supersedes edge, feedback counters — so its size does not grow with
the correction. The stored node, edge and corrections are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from living_memory.server import create_mcp_server


class FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        return lambda func: func

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        return (lambda inner: inner) if func is None else func


ORIGINAL = "harbor gauge uses chart datum " + "o" * 600
CORRECTION = "harbor gauge uses the local tide datum " + "c" * 1961


def _teach(tmp_path: Path) -> tuple[Any, dict[str, Any], str]:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    original_id = mcp.tools["memory_remember"](ORIGINAL, {"scope": "project:tide"})["node"]["id"]
    mcp.tools["memory_recall"]("harbor gauge datum", scope="project:tide")
    taught = mcp.tools["memory_teach"](original_id, CORRECTION, context={"agent": "a"})
    return mcp.memory_store, taught, original_id


def test_teach_response_is_a_small_receipt_without_either_text(tmp_path: Path) -> None:
    assert len(CORRECTION) == 2000
    _store, taught, original_id = _teach(tmp_path)
    body = json.dumps(taught, ensure_ascii=False)

    assert len(body.encode()) < 1024, body
    assert CORRECTION[:60] not in body
    assert ORIGINAL[:30] not in body
    # The ids ae readers take from the response keep their paths.
    assert taught["corrective_trace"]["id"]
    assert taught["original"]["id"] == original_id
    assert taught["supersedes"]["type"] == "supersedes"


def test_teach_still_stores_node_edge_and_corrections(tmp_path: Path) -> None:
    store, taught, original_id = _teach(tmp_path)
    corrective_id = taught["corrective_trace"]["id"]

    corrective = store.get_node(corrective_id)
    assert corrective.content == CORRECTION
    assert corrective.scope == "project:tide"
    assert corrective.provenance.get("prior_recalls")

    edges = [
        edge
        for edge in store.list_connections(source_id=corrective_id)
        if edge.type == "supersedes" and edge.target_id == original_id
    ]
    assert len(edges) == 1 and edges[0].id == taught["supersedes"]["id"]
    assert edges[0].source_id == corrective_id

    original = store.get_node(original_id)
    assert original.content == ORIGINAL
    assert original.corrections[-1]["old"] == ORIGINAL
    assert original.corrections[-1]["new"] == CORRECTION
