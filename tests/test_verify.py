from __future__ import annotations

import pytest

from weightsmith.evaluation.verify import (
    Status,
    judge_behavior,
    judge_changed_params,
    judge_finite,
)


@pytest.mark.parametrize(
    ("clause", "before", "after", "n", "params", "status"),
    [
        ("improve", 0.5, 0.9, 100, {"min_delta": 0.3}, Status.VERIFIED),
        ("improve", 0.5, 0.8, 100, {"min_delta": 0.3}, Status.VERIFIED),  # exactly at threshold
        ("improve", 0.5, 0.7, 100, {"min_delta": 0.3}, Status.FAILED),
        ("improve", 0.5, 0.9, 10, {"min_delta": 0.3}, Status.INCONCLUSIVE),  # n < min_n
        ("improve", 0.5, 0.9, 100, {"min_delta": 0.001}, Status.INCONCLUSIVE),  # below 1/n
        ("preserve", 0.9, 0.9, 100, {"tolerance": 0.0}, Status.VERIFIED),
        ("preserve", 0.9, 0.95, 100, {"tolerance": 0.0}, Status.VERIFIED),  # improving is fine
        ("preserve", 0.9, 0.88, 100, {"tolerance": 0.02}, Status.VERIFIED),
        ("preserve", 0.9, 0.85, 100, {"tolerance": 0.02}, Status.FAILED),
        ("preserve", 0.9, 0.9, 5, {"tolerance": 0.02, "min_n": 10}, Status.INCONCLUSIVE),
    ],
)
def test_behaviour_verdicts(
    clause: str, before: float, after: float, n: int, params: dict[str, float], status: Status
) -> None:
    verdict = judge_behavior(clause, "b", before, after, n, params)
    assert verdict.status is status, verdict.reason
    assert verdict.measurements["before"] == before
    assert verdict.measurements["after"] == after


def test_lower_is_better_metrics_flip_the_comparison() -> None:
    verdict = judge_behavior(
        "improve", "loss", 1.0, 0.5, 100, {"min_delta": 0.3}, higher_is_better=False
    )
    assert verdict.status is Status.VERIFIED


def test_structural_requirements() -> None:
    assert judge_finite(0).status is Status.VERIFIED
    assert judge_finite(3).status is Status.FAILED
    assert judge_changed_params(5, 100, 0.05).status is Status.VERIFIED
    assert judge_changed_params(6, 100, 0.05).status is Status.FAILED
