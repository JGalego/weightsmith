"""Findings and evidence: what a program learned about a model, and how strongly it is supported.

Evidence levels have operational definitions (docs/semantics.md):

* ``hypothesised``: asserted (by a person or a reasoning provider) with no runtime evidence.
* ``observed``: a deterministic statistic of the weights. No forward pass was run.
* ``experimentally_supported``: an intervention on this model in this run changed a measured
  behaviour as the claim predicts.
* ``independently_verified``: reproduced by a second method or on held-out probes. Version 0.1
  never assigns this level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EvidenceLevel(StrEnum):
    HYPOTHESISED = "hypothesised"
    OBSERVED = "observed"
    EXPERIMENTALLY_SUPPORTED = "experimentally_supported"
    INDEPENDENTLY_VERIFIED = "independently_verified"

    @property
    def rank(self) -> int:
        return list(EvidenceLevel).index(self)


@dataclass(frozen=True, slots=True)
class Evidence:
    level: EvidenceLevel
    method: str
    description: str
    measurements: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "method": self.method,
            "description": self.description,
            "measurements": self.measurements,
        }


@dataclass(frozen=True, slots=True)
class Location:
    """A place in a weight set: a tensor, optionally restricted to some rows."""

    tensor: str
    rows: tuple[int, ...] | None = None

    def __str__(self) -> str:
        if self.rows is None:
            return self.tensor
        return f"{self.tensor}[rows {', '.join(map(str, self.rows))}]"


@dataclass(slots=True)
class Finding:
    """A discovered property of a specific weight set, identified by its content hash."""

    kind: str
    binding: str
    model_id: str
    summary: str
    locations: tuple[Location, ...]
    evidence: list[Evidence]
    limitations: tuple[str, ...] = ()

    @property
    def level(self) -> EvidenceLevel:
        return max(
            (e.level for e in self.evidence),
            key=lambda lvl: lvl.rank,
            default=EvidenceLevel.HYPOTHESISED,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "binding": self.binding,
            "model_id": self.model_id,
            "summary": self.summary,
            "level": self.level.value,
            "locations": [
                {"tensor": loc.tensor, "rows": list(loc.rows) if loc.rows is not None else None}
                for loc in self.locations
            ],
            "evidence": [e.to_dict() for e in self.evidence],
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class FinderResult:
    """Findings plus coverage, so that an absence of findings can be interpreted."""

    finder: str
    findings: tuple[Finding, ...]
    examined: int
    skipped: tuple[tuple[str, str], ...] = ()  # (tensor, reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finder": self.finder,
            "found": len(self.findings),
            "examined": self.examined,
            "skipped": [{"tensor": t, "reason": r} for t, r in self.skipped],
        }
