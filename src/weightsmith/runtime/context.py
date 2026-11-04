"""Run-time values, the working copy of the target, and budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from weightsmith.backends.base import Backend
from weightsmith.compiler.effects import Effect, Policy
from weightsmith.compiler.ir import IndexSpec, Region
from weightsmith.diagnostics import Span
from weightsmith.errors import BudgetExceeded
from weightsmith.evaluation.probes import ProbeSuite
from weightsmith.models.hf import Architecture
from weightsmith.models.profiles import Profile


@dataclass(slots=True)
class RunOptions:
    out_dir: Path | None = None
    dry_run: bool = False
    overwrite: bool = False
    seed: int = 0


@dataclass(slots=True)
class LoadedWeights:
    binding: str
    path: Path
    tensors: Any
    arch: Architecture | None
    model_id: str
    profile: Profile | None = None
    profile_error: str | None = None


@dataclass(slots=True)
class LoadedBehavior:
    binding: str
    path: Path
    suite: ProbeSuite
    metric: str


@dataclass(frozen=True, slots=True)
class SelectionValue:
    """Parameters of one weight set: named tensors with one slice applied to each, or explicit
    regions (from a role such as ``layers[1].attn.heads[2]``)."""

    binding: str
    names: tuple[str, ...]
    index: tuple[IndexSpec, ...] = ()
    regions: tuple[Region, ...] | None = None
    role: str | None = None

    def region_list(self) -> list[Region]:
        if self.regions is not None:
            return list(self.regions)
        return [Region(name, self.index) for name in self.names]

    @property
    def partial(self) -> bool:
        """Whether the selection covers parts of tensors."""
        return bool(self.index) or any(r.index for r in self.regions or ())


Value = LoadedWeights | LoadedBehavior | SelectionValue


@dataclass(slots=True)
class Budgets:
    """Tracks consumption against the policy and refuses work that would exceed a budget."""

    policy: Policy
    forward_passes: int = 0
    bytes_loaded: int = 0

    def reserve_forward_passes(self, count: int, span: Span | None) -> None:
        limit = self.policy.budget(Effect.PROBE, "max_forward_passes")
        if limit is not None and self.forward_passes + count > limit:
            raise BudgetExceeded(
                "probe", "max_forward_passes", limit, self.forward_passes + count, span
            )
        self.forward_passes += count

    def reserve_bytes(self, count: int, span: Span | None) -> None:
        limit = self.policy.budget(Effect.LOAD, "max_bytes")
        if limit is not None and self.bytes_loaded + count > limit:
            raise BudgetExceeded("load", "max_bytes", limit, self.bytes_loaded + count, span)
        self.bytes_loaded += count

    def check_changed_fraction(self, fraction: float, span: Span | None) -> None:
        limit = self.policy.budget(Effect.MUTATE, "max_changed_params")
        if limit is not None and fraction > limit + 1e-12:
            raise BudgetExceeded("mutate", "max_changed_params", limit, fraction, span)

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward_passes": {
                "used": self.forward_passes,
                "limit": self.policy.budget(Effect.PROBE, "max_forward_passes"),
            },
            "bytes_loaded": {
                "used": self.bytes_loaded,
                "limit": self.policy.budget(Effect.LOAD, "max_bytes"),
            },
            "changed_params": {"limit": self.policy.budget(Effect.MUTATE, "max_changed_params")},
        }


@dataclass(slots=True)
class WorkingCopy:
    """Copy-on-write view of the target. ``base`` is never modified; edits replace entries."""

    backend: Backend
    base: dict[str, Any]
    tensors: dict[str, Any] = field(default_factory=dict)
    mutations: list[str] = field(default_factory=list)  # kinds of the ops that changed it

    def __post_init__(self) -> None:
        self.tensors = dict(self.base)

    def changed(self) -> dict[str, int]:
        return self.backend.changed_params(self.base, self.tensors)

    @property
    def total(self) -> int:
        return self.backend.manifest(self.base).numel
