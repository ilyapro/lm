"""Background runner for the consolidation pass a write makes due.

The write path only schedules; one worker thread per server runs the passes.
Pending scopes are saved before their requests become runnable. A later
committed write retries failed admission or startup; server start resumes
durable pending work.

Guarantees:

- One pass per scope at a time. A scope made due while queued joins that pass;
  while running it is coalesced into a single follow-up pass.
- A failed pass loses no writes (the traces were committed before the pass was
  scheduled) and does not leave the scope without a next pass: it is retried
  after a back-off, and the pending mark is persisted, so a pass that never
  finished — failed or killed with the process — is resumed by the next server
  start.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "PENDING_KV_KEY",
    "AutoConsolidationScheduler",
]

PENDING_KV_KEY = "auto_consolidation_pending"
DEFAULT_RETRY_DELAYS: tuple[float, ...] = (30.0, 300.0, 1800.0)

_LOG = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class _QueuedPass:
    # An accepted pass can retain its backoff while a new request awaits save.
    ready_at: float | None = None
    requested_at: float | None = None


class AutoConsolidationScheduler:
    """Queue of due scopes drained by one daemon worker thread.

    ``run_pass(scope)`` performs the pass and returns its summary dict; it owns
    its own locking. ``load_pending`` / ``save_pending`` persist the set of
    scopes whose pass has not completed yet (a JSON object in the store's kv,
    through the server's lock); either may be ``None`` for an in-memory queue.
    ``persist_guard`` is held around every save, snapshot included, so saves
    from the write path and from the worker cannot land out of order; pass
    the lock ``save_pending`` itself needs, so it is always taken first.
    """

    def __init__(
        self,
        run_pass: Callable[[str], dict[str, Any]],
        *,
        load_pending: Callable[[], str | None] | None = None,
        save_pending: Callable[[str], None] | None = None,
        persist_guard: Any | None = None,
        retry_delays: Iterable[float] = DEFAULT_RETRY_DELAYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._run_pass = run_pass
        self._load_pending = load_pending
        self._save_pending = save_pending
        self._persist_guard = persist_guard if persist_guard is not None else threading.RLock()
        self._retry_delays = tuple(float(delay) for delay in retry_delays)
        self._clock = clock
        self._cond = threading.Condition()
        # Accepted readiness and unsaved request time; insertion order is FIFO.
        self._queue: dict[str, _QueuedPass] = {}
        # scope -> first request time (ISO) of the not-yet-completed pass.
        self._persisted: dict[str, str] = {}
        self._attempts: dict[str, int] = {}
        self._running: str | None = None
        self._last: dict[str, dict[str, Any]] = {}
        self._passes_completed = 0
        self._thread: threading.Thread | None = None
        self._closed = False

    # -- write-path side -------------------------------------------------

    def request(self, scope: str, *, trace_count: int | None = None) -> dict[str, Any]:
        """Schedule a pass for ``scope`` and return at once.

        The returned dict is what the write reports: the pass is scheduled,
        not run. ``coalesced`` says the request merged into a pass that was
        already queued or is running (a running pass is followed by exactly
        one more).
        """

        # Requests survive save failure, but only saved requests become ready.
        # The store's guard always precedes the condition.
        with self._persist_guard, self._cond:
            if self._closed:
                raise RuntimeError("auto-consolidation scheduler is closed")
            already_queued = scope in self._queue
            running = self._running == scope
            queued = self._queue.get(scope, _QueuedPass())
            self._queue[scope] = _QueuedPass(
                ready_at=queued.ready_at,
                requested_at=(
                    queued.requested_at if queued.requested_at is not None else self._clock()
                ),
            )
            self._persisted.setdefault(scope, _utc_now())
            # Save coalesced requests too: an earlier save may have failed.
            # Let failure reach the nonfatal post-commit boundary; no durable
            # scheduling confirmation can be made in that case.
            self._save_and_admit()
            worker_started = self._ensure_worker()
            self._cond.notify_all()
        report: dict[str, Any] = {
            "status": "scheduled",
            "scope": scope,
            "coalesced": already_queued or running,
        }
        if trace_count is not None:
            report["trace_count"] = trace_count
        if not worker_started:
            report["worker_start_pending"] = True
        return report

    def retry_pending(self) -> None:
        """Retry unsaved requests or restart a missing worker.

        Ordinary writes may call this even when they make no new pass due.
        Only an unsaved due request can override an accepted retry's backoff.
        """

        with self._persist_guard, self._cond:
            if self._closed or not self._queue:
                return
            if (
                self._thread is not None and self._thread.is_alive()
                and all(queued.requested_at is None for queued in self._queue.values())
            ):
                return
            self._save_and_admit()
            self._ensure_worker()
            self._cond.notify_all()

    def resume(self) -> list[str]:
        """Queue every scope a previous process left pending. Returns them."""

        if self._load_pending is None:
            return []
        try:
            raw = self._load_pending()
            stored = json.loads(raw) if raw else {}
        except Exception:
            _LOG.warning("could not read pending auto-consolidation scopes", exc_info=True)
            return []
        if not isinstance(stored, dict):
            return []
        scopes = [str(scope) for scope in stored]
        with self._cond:
            if self._closed:
                return []
            for scope in scopes:
                self._persisted.setdefault(scope, str(stored[scope]))
                self._queue.setdefault(scope, _QueuedPass(ready_at=self._clock()))
            if scopes:
                self._ensure_worker()
                self._cond.notify_all()
        return scopes

    # -- observation -----------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._cond:
            return {
                "running": self._running,
                "queued": list(self._queue),
                "passes_completed": self._passes_completed,
                "last": {scope: dict(entry) for scope, entry in self._last.items()},
            }

    def last_result(self, scope: str) -> dict[str, Any] | None:
        with self._cond:
            entry = self._last.get(scope)
            return dict(entry) if entry is not None else None

    def wait_idle(self, timeout: float | None = None, *, include_retries: bool = False) -> bool:
        """Block until nothing runs and nothing is ready to run.

        Scopes waiting out a retry back-off count as idle unless
        ``include_retries``. Returns False on timeout.
        """

        deadline = None if timeout is None else self._clock() + timeout
        with self._cond:
            while True:
                if self._running is None and not self._ready_or_waiting(include_retries):
                    return True
                remaining = None if deadline is None else deadline - self._clock()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining if remaining is not None else 1.0)

    def close(self, timeout: float | None = 5.0) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    # -- worker ----------------------------------------------------------

    def _ready_or_waiting(self, include_retries: bool) -> bool:
        if include_retries:
            return bool(self._queue)
        now = self._clock()
        return any(
            queued.requested_at is not None
            or (queued.ready_at is not None and queued.ready_at <= now)
            or scope not in self._attempts
            for scope, queued in self._queue.items()
        )

    def _ensure_worker(self) -> bool:
        if self._closed:
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        try:
            thread = threading.Thread(
                target=self._work,
                name="living-memory-auto-consolidation",
                daemon=True,
            )
            thread.start()
        except Exception:
            _LOG.warning("could not start auto-consolidation worker; scopes remain pending", exc_info=True)
            return False
        # Only started threads may be joined by close(). A failed start leaves
        # the queue intact for a later write or resume, without a retry loop.
        self._thread = thread
        return True

    def _next_scope(self) -> str | None:
        """Pop the first ready scope, waiting for one; None once closed."""

        with self._cond:
            while not self._closed:
                now = self._clock()
                for scope, queued in self._queue.items():
                    if queued.ready_at is not None and queued.ready_at <= now:
                        if queued.requested_at is None:
                            del self._queue[scope]
                        else:
                            self._queue[scope] = _QueuedPass(requested_at=queued.requested_at)
                        self._running = scope
                        return scope
                wake = min(
                    (queued.ready_at for queued in self._queue.values() if queued.ready_at is not None),
                    default=None,
                )
                self._cond.wait(timeout=None if wake is None else max(0.0, wake - now))
            return None

    def _work(self) -> None:
        while True:
            scope = self._next_scope()
            if scope is None:
                return
            started = self._clock()
            started_at = _utc_now()
            try:
                summary = self._run_pass(scope)
            except Exception as exc:  # the worker must outlive any one pass
                _LOG.warning("auto-consolidation of %s failed", scope, exc_info=True)
                self._finish_failed(scope, exc, started, started_at)
            else:
                self._finish_ok(scope, summary, started, started_at)

    def _finish_ok(
        self, scope: str, summary: dict[str, Any], started: float, started_at: str
    ) -> None:
        with self._cond:
            self._attempts.pop(scope, None)
            self._passes_completed += 1
            self._last[scope] = {
                "status": "completed",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_seconds": round(self._clock() - started, 3),
                "summary": summary,
            }
            persist = False
            if scope not in self._queue:
                # Only a pass that started after the last request settles it.
                persist = self._persisted.pop(scope, None) is not None
        if persist:
            self._persist()
        # Idle only once the settled mark is written: a waiter that sees the
        # worker idle must also see the pending set it left behind.
        with self._cond:
            self._running = None
            self._cond.notify_all()

    def _finish_failed(
        self, scope: str, exc: BaseException, started: float, started_at: str
    ) -> None:
        with self._cond:
            self._running = None
            attempt = self._attempts.get(scope, 0) + 1
            self._attempts[scope] = attempt
            retry_in: float | None = None
            queued = self._queue.get(scope, _QueuedPass())
            if queued.ready_at is None and attempt <= len(self._retry_delays):
                retry_in = self._retry_delays[attempt - 1]
                self._queue[scope] = _QueuedPass(
                    ready_at=self._clock() + retry_in,
                    requested_at=queued.requested_at,
                )
            self._last[scope] = {
                "status": "failed",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_seconds": round(self._clock() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "attempt": attempt,
                "retry_in_seconds": retry_in,
            }
            # The persisted mark stays: a restart resumes the scope even when
            # the in-process retries are exhausted.
            self._cond.notify_all()

    def _save_and_admit(self) -> None:
        """Save and admit requests with both the guard and condition held."""

        if self._save_pending is not None:
            self._save_pending(json.dumps(self._persisted, sort_keys=True))
        for scope, queued in self._queue.items():
            if queued.requested_at is not None:
                # A saved due request overrides backoff and starts a fresh pass.
                ready_at = queued.requested_at
                if queued.ready_at is not None:
                    ready_at = min(ready_at, queued.ready_at)
                self._queue[scope] = _QueuedPass(ready_at=ready_at)
                self._attempts.pop(scope, None)

    def _persist(self) -> None:
        """Save the current pending set and admit any requests it includes.

        Lock order is guard first, then ``_cond``:
        the write path calls in holding the guard already, and the worker
        never holds ``_cond`` while taking the guard.
        """

        with self._persist_guard, self._cond:
            try:
                self._save_and_admit()
            except Exception:
                _LOG.warning(
                    "could not persist pending auto-consolidation scopes", exc_info=True
                )
