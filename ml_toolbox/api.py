# -*- coding: utf-8 -*-
"""Stable, headless facade for ML Toolbox.

This module is intentionally independent from Qt and from any AI/agent layer.  It
wraps the existing Dataset/Pipeline/registry/runner contracts so external Python
callers have one small, versioned entry point while the legacy modules remain
available for plugin authors.

The facade is deliberately conservative: it does not pickle estimators, change
persistence formats, or promise exact optimization resume.  It provides a clear
request/record/status boundary and leaves those capabilities for later schema
versions.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from .core import registry, runner
from .core.contracts import DataSpec, MLMethod, RunConfig
from .core.runner import RunRecord
from .core.dataset import Dataset
from .core.pipeline import Pipeline


API_VERSION = "1.0"


class RunState(str, Enum):
    """Observable states for one facade run request."""

    DRAFT = "draft"
    VALIDATING = "validating"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class TerminalReason(str, Enum):
    """Why a request reached a terminal state."""

    COMPLETED = "completed"
    METHOD_ERROR = "method_error"
    RESOURCE_LIMIT = "resource_limit"
    CONFIG_ERROR = "config_error"
    USER_CANCELLED = "user_cancelled"
    INTERRUPTED = "interrupted"


class SessionError(RuntimeError):
    """Raised when a request cannot be dispatched or configured."""


@dataclass(frozen=True)
class RunRequest:
    """Immutable-by-convention input for one ML method run.

    ``request_id`` is generated at construction time and remains stable across
    retries or repeated calls to :meth:`Session.run`.  Mapping fields are copied
    into :class:`RunConfig` when the request is dispatched, so later caller-side
    mutation cannot alter an in-flight run.
    """

    method: str = ""
    overrides: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)
    diag: bool = False
    seed: int = 42
    persist: bool = False
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self):
        # Freeze the request boundary from caller-side mapping mutation.  The
        # contained values are copied again in ``config`` for dispatch safety.
        object.__setattr__(self, "overrides", dict(self.overrides))
        object.__setattr__(self, "extras", dict(self.extras))

    def config(self) -> RunConfig:
        return RunConfig(
            overrides=dict(self.overrides),
            extras=dict(self.extras),
            diag=bool(self.diag),
            seed=int(self.seed),
        )


@dataclass(frozen=True)
class RunSnapshot:
    """Small serializable status object suitable for polling or callbacks."""

    request_id: str
    method: str
    state: RunState
    terminal_reason: TerminalReason | None = None
    run_id: str | None = None
    message: str = ""

    @property
    def terminal(self) -> bool:
        return self.state in {
            RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED,
            RunState.INTERRUPTED,
        }


class Session:
    """Headless session for data preparation, method execution and prediction.

    The session owns only bookkeeping for requests/records.  It does not own a
    second global registry: the existing registry remains the single source of
    method plugins.  Persistence is explicit and uses the existing
    ``core.persistence`` format until a versioned envelope is introduced.
    """

    def __init__(self, *, seed: int = 42, load_builtin: bool = True):
        self.seed = int(seed)
        self._lock = threading.RLock()
        self._snapshots: dict[str, RunSnapshot] = {}
        self._request_runs: dict[str, str] = {}
        self._records: dict[str, RunRecord] = {}
        if load_builtin:
            registry.load_builtin()

    def list_methods(self) -> list[dict[str, Any]]:
        """Return JSON-friendly method descriptors in stable name order."""
        registry.load_builtin()
        out = []
        for method in sorted(registry.all_methods(), key=lambda m: m.name):
            out.append({
                "name": method.name,
                "display_name": method.display_name,
                "family": method.family,
                "task": method.task,
                "target_kind": method.target_kind,
                "tags": list(method.tags),
                "purposes": list(method.purposes_of()),
                "supports_predict": hasattr(method, "predict"),
            })
        return out

    def prepare(self, dataset: Dataset, *, pipeline: Pipeline | None = None,
                diag: bool = False) -> DataSpec:
        """Fit a fresh pipeline view for ``dataset``.

        This is a compatibility wrapper around ``Pipeline.run``.  The returned
        ``DataSpec`` is still backed by the legacy step state; callers needing
        cross-run immutability should use a fresh pipeline per prepared view
        until ``FittedPipeline`` is introduced.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a ml_toolbox.core.Dataset")
        pipeline = pipeline or Pipeline.default()
        return pipeline.run(dataset, diag=diag)

    def run(self, spec: DataSpec, request: RunRequest | str, *,
            progress: Callable[[int, int, str], None] | None = None,
            saver: Callable[[RunRecord], object] | None = None) -> RunRecord:
        """Dispatch one request and return the existing ``RunRecord``.

        Method-level failures are returned in ``RunRecord.result.error`` as in
        the legacy runner.  Configuration/dispatch failures raise
        :class:`SessionError` after recording a terminal snapshot.
        """
        if isinstance(request, str):
            request = RunRequest(method=request, seed=self.seed)
        if not isinstance(request, RunRequest):
            raise TypeError("request must be RunRequest or method name")
        if not request.method:
            raise SessionError("method must be non-empty")

        with self._lock:
            self._transition(request, RunState.VALIDATING)
            try:
                method = registry.get(request.method)
                if not method.can_handle(spec):
                    raise SessionError(
                        f"method {request.method!r} cannot handle the prepared DataSpec")
                self._transition(request, RunState.RUNNING)
                record = runner.run_one(method, spec, request.config())
            except SessionError as exc:
                self._transition(request, RunState.FAILED,
                                 TerminalReason.CONFIG_ERROR, message=str(exc))
                raise
            except (KeyError, TypeError, ValueError) as exc:
                self._transition(request, RunState.FAILED,
                                 TerminalReason.CONFIG_ERROR, message=str(exc))
                raise SessionError(f"invalid request: {exc}") from exc
            except Exception as exc:
                self._transition(request, RunState.FAILED,
                                 TerminalReason.INTERRUPTED,
                                 message=f"{type(exc).__name__}: {exc}")
                raise SessionError(f"dispatch failed: {exc}") from exc

            self._records[record.run_id] = record
            self._request_runs[request.request_id] = record.run_id
            reason = None
            if record.result.ok:
                state, reason = RunState.SUCCEEDED, TerminalReason.COMPLETED
            else:
                error = record.result.error or ""
                state = RunState.FAILED
                reason = (TerminalReason.RESOURCE_LIMIT
                          if "资源预算超限" in error else TerminalReason.METHOD_ERROR)
            self._transition(request, state, reason, run_id=record.run_id,
                             message=record.result.error or "")
            if request.persist:
                self.save(record, saver=saver)
            return record

    def run_batch(self, spec: DataSpec, methods: Iterable[str], *,
                  request: RunRequest | None = None,
                  config: RunConfig | None = None,
                  persist: bool = False,
                  progress: Callable[[int, int, str], None] | None = None,
                  saver: Callable[[RunRecord], object] | None = None
                  ) -> list[RunRecord]:
        """Run methods sequentially with the same isolation semantics as runner.

        ``request`` supplies common overrides/extras/seed; each method gets a
        unique request id.  Unknown methods are represented as a legacy error
        record by the underlying runner and do not stop the remaining methods.
        """
        if request is None:
            base = request or RunRequest(method="", seed=self.seed)
        else:
            base = request
        if config is not None:
            base = RunRequest(method=base.method,
                              overrides=dict(config.overrides),
                              extras=dict(config.extras),
                              diag=config.diag, seed=config.seed,
                              persist=persist or base.persist)
        names = [str(m) for m in methods]
        records: list[RunRecord] = []
        for i, name in enumerate(names):
            if progress:
                progress(i, len(names), name)
            req = RunRequest(method=name, overrides=dict(base.overrides),
                             extras=dict(base.extras), diag=base.diag,
                             seed=base.seed, persist=persist or base.persist)
            try:
                records.append(self.run(spec, req, saver=saver))
            except SessionError:
                # Keep the batch moving; the request's snapshot is available.
                continue
        return records

    def save(self, record: RunRecord, *,
             saver: Callable[[RunRecord], object] | None = None):
        """Persist a record explicitly and update its serialize timing."""
        return runner.save_timed(record, saver=saver)

    def predict(self, record: RunRecord | str, frame):
        """Predict with the method that produced ``record``."""
        if isinstance(record, str):
            record = self._records.get(record)
            if record is None:
                raise KeyError(f"unknown run_id in this session: {record}")
        if not isinstance(record, RunRecord):
            raise TypeError("record must be RunRecord or known run_id")
        method = registry.get(record.method)
        return method.predict(frame, record.result)

    def state(self, request_id: str) -> RunSnapshot | None:
        with self._lock:
            return self._snapshots.get(request_id)

    def last_state(self) -> RunSnapshot | None:
        with self._lock:
            values = list(self._snapshots.values())
            return values[-1] if values else None

    def record(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._records.get(run_id)

    def history(self) -> list[RunRecord]:
        with self._lock:
            return list(self._records.values())

    def describe(self) -> dict[str, Any]:
        return {"api_version": API_VERSION, "seed": self.seed,
                "method_count": len(registry.names())}

    def _transition(self, request: RunRequest, state: RunState,
                    reason: TerminalReason | None = None, *,
                    run_id: str | None = None, message: str = "") -> None:
        with self._lock:
            self._snapshots[request.request_id] = RunSnapshot(
                request_id=request.request_id, method=request.method,
                state=state, terminal_reason=reason,
                run_id=run_id or self._request_runs.get(request.request_id),
                message=message,
            )


__all__ = [
    "API_VERSION", "RunState", "TerminalReason", "SessionError", "RunRequest",
    "RunSnapshot", "Session",
]
