"""The run report: everything a run did and measured, serialisable to JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from weightsmith.evaluation.verify import Status, Verdict
from weightsmith.evidence import FinderResult, Finding


@dataclass(slots=True)
class Event:
    """One executed operation, in order."""

    op: str
    status: str  # "ok" | "noop" | "stopped" | "error" | "dry-run"
    detail: str
    location: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "status": self.status,
            "detail": self.detail,
            "location": self.location,
            "data": self.data,
        }


@dataclass(slots=True)
class RunReport:
    program: str
    program_hash: str
    backend: str
    dry_run: bool
    outputs: tuple[str, ...] = ()
    status: str = "completed"  # "completed" | "stopped" | "error"
    stop_reason: str | None = None
    inputs: list[dict[str, Any]] = field(default_factory=list)
    target: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    inspections: list[dict[str, Any]] = field(default_factory=list)
    finder_results: list[FinderResult] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    changes: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_verdicts(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.status is Status.FAILED]

    @property
    def inconclusive_verdicts(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.status is Status.INCONCLUSIVE]

    def exit_code(self) -> int:
        """0 success, 2 runtime error or stop, 3 a contract item failed, 4 one was inconclusive."""
        if self.status != "completed":
            return 2
        if self.failed_verdicts:
            return 3
        if self.inconclusive_verdicts:
            return 4
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "weightsmith.report/v0",
            "program": self.program,
            "program_hash": self.program_hash,
            "backend": self.backend,
            "dry_run": self.dry_run,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "outputs": list(self.outputs),
            "inputs": self.inputs,
            "target": self.target,
            "events": [e.to_dict() for e in self.events],
            "inspections": self.inspections,
            "finders": [r.to_dict() for r in self.finder_results],
            "findings": [f.to_dict() for f in self.findings],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "changes": self.changes,
            "budgets": self.budgets,
            "artifacts": self.artifacts,
            "provenance": self.provenance,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
