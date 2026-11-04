"""Verdict semantics for contract items. Pure functions, independent of any backend.

* ``verified``: the measurement ran within budget on at least ``min_n`` examples and the
  comparator held.
* ``failed``: the measurement ran and the comparator did not hold.
* ``inconclusive``: no trustworthy measurement exists: the budget ran out, the suite is smaller
  than ``min_n``, or the threshold is finer than the suite's resolution (1/n for accuracy).
* ``skipped``: the item was deliberately not evaluated (dry run).

A verdict is a statement about the declared probe suite, not a statistical guarantee about the
behaviour in general.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

_EPS = 1e-12


class Status(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class Verdict:
    clause: str
    subject: str
    status: Status
    reason: str
    measurements: dict[str, Any] = field(default_factory=dict)

    @property
    def item(self) -> str:
        return f"{self.clause} {self.subject}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause": self.clause,
            "subject": self.subject,
            "status": self.status.value,
            "reason": self.reason,
            "measurements": self.measurements,
        }


def judge_behavior(
    clause: str,
    subject: str,
    before: float,
    after: float,
    n: int,
    params: dict[str, Any],
    higher_is_better: bool = True,
) -> Verdict:
    """Judge a ``preserve`` or ``improve`` item from before/after measurements on n examples."""
    gain = (after - before) if higher_is_better else (before - after)
    measurements: dict[str, Any] = {
        "before": before,
        "after": after,
        "change": after - before,
        "n": n,
    }
    min_n = int(params.get("min_n", 20))
    if n < min_n:
        return Verdict(
            clause,
            subject,
            Status.INCONCLUSIVE,
            f"the probe suite has {n} examples, fewer than min_n = {min_n}",
            measurements,
        )
    resolution = 1.0 / n
    if clause == "improve":
        min_delta = float(params["min_delta"])
        measurements["min_delta"] = min_delta
        if min_delta < resolution:
            return Verdict(
                clause,
                subject,
                Status.INCONCLUSIVE,
                f"min_delta {min_delta:g} is finer than the suite's resolution {resolution:g}",
                measurements,
            )
        if gain + _EPS >= min_delta:
            return Verdict(
                clause,
                subject,
                Status.VERIFIED,
                f"improved by {gain:+.4f}, at least {min_delta:g}",
                measurements,
            )
        return Verdict(
            clause,
            subject,
            Status.FAILED,
            f"changed by {gain:+.4f}, less than the required {min_delta:g}",
            measurements,
        )
    if clause == "preserve":
        tolerance = float(params["tolerance"])
        measurements["tolerance"] = tolerance
        degradation = -gain
        if degradation <= tolerance + _EPS:
            return Verdict(
                clause,
                subject,
                Status.VERIFIED,
                f"changed by {gain:+.4f}; degradation within {tolerance:g}",
                measurements,
            )
        return Verdict(
            clause,
            subject,
            Status.FAILED,
            f"degraded by {degradation:.4f}, more than the tolerated {tolerance:g}",
            measurements,
        )
    raise ValueError(f"unknown behavioural clause '{clause}'")


def judge_finite(nonfinite: int) -> Verdict:
    if nonfinite == 0:
        return Verdict(
            "require", "finite", Status.VERIFIED, "no NaN or infinite parameters", {"nonfinite": 0}
        )
    return Verdict(
        "require",
        "finite",
        Status.FAILED,
        f"{nonfinite} NaN or infinite parameters",
        {"nonfinite": nonfinite},
    )


def judge_changed_params(changed: int, total: int, maximum: float) -> Verdict:
    fraction = changed / total if total else 0.0
    measurements = {"changed": changed, "total": total, "fraction": fraction, "max": maximum}
    if fraction <= maximum + _EPS:
        return Verdict(
            "require",
            "changed_params",
            Status.VERIFIED,
            f"{fraction:.4%} of parameters changed, at most {maximum:.4%}",
            measurements,
        )
    return Verdict(
        "require",
        "changed_params",
        Status.FAILED,
        f"{fraction:.4%} of parameters changed, more than {maximum:.4%}",
        measurements,
    )


def unavailable(clause: str, subject: str, status: Status, reason: str) -> Verdict:
    return Verdict(clause, subject, status, reason)
