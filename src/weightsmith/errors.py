"""Runtime error types. Compile-time problems are reported as diagnostics instead."""

from __future__ import annotations

from weightsmith.diagnostics import Span


class WeightsmithError(Exception):
    """Base class for runtime failures. Carries the span of the operation that failed."""

    def __init__(self, message: str, span: Span | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.span = span

    def __str__(self) -> str:
        return f"{self.span}: {self.message}" if self.span else self.message


class LoadError(WeightsmithError):
    """A model, weight file or probe suite could not be loaded or is malformed."""


class IncompatibleError(WeightsmithError):
    """Two weight sets, or a patch and a weight set, are structurally incompatible."""


class UnsupportedOperation(WeightsmithError):
    """The selected backend does not declare a capability the program needs."""


class PolicyViolation(WeightsmithError):
    """An operation would use an effect the active policy forbids."""


class BudgetExceeded(WeightsmithError):
    """An operation would exceed a resource budget. The operation is not performed."""

    def __init__(self, effect: str, budget: str, limit: float, attempted: float, span: Span | None):
        super().__init__(
            f"{effect} budget '{budget}' would be exceeded: limit {limit:g}, needed {attempted:g}",
            span,
        )
        self.effect = effect
        self.budget = budget
        self.limit = limit
        self.attempted = attempted


class PlanningError(WeightsmithError):
    """The planner could not find an applicable strategy for a declarative statement."""
