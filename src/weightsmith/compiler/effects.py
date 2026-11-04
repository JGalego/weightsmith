"""Effects, budgets and policies.

Every IR operation declares the effects it performs. A program's policy says which effects are
allowed and with which budgets. The compiler rejects programs that use effects their policy does not
allow; the runtime enforces budgets and stops before an operation would exceed one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from weightsmith.compiler.types import FRACTION, INT, Type
from weightsmith.diagnostics import DiagnosticSink, Span, Stage
from weightsmith.language import nodes as n

if TYPE_CHECKING:
    from weightsmith.compiler.ir import IRProgram


class Effect(StrEnum):
    LOAD = "load"  # read model files from disk
    INSPECT = "inspect"  # read parameter values
    PROBE = "probe"  # run forward passes
    MUTATE = "mutate"  # change parameters of the working copy
    TRAIN = "train"  # run an optimiser (no v0.1 operation does this)
    REASON = "reason"  # call an external reasoning provider (no v0.1 operation does this)
    SAVE = "save"  # write model or patch artifacts


BUDGETS: dict[Effect, dict[str, Type]] = {
    Effect.LOAD: {"max_bytes": INT},
    Effect.INSPECT: {},
    Effect.PROBE: {"max_forward_passes": INT},
    Effect.MUTATE: {"max_changed_params": FRACTION},
    Effect.TRAIN: {"max_steps": INT},
    Effect.REASON: {"max_calls": INT},
    Effect.SAVE: {},
}

DEFAULT_ALLOWED = frozenset({Effect.LOAD, Effect.INSPECT})
UNIMPLEMENTED = frozenset({Effect.TRAIN, Effect.REASON})


@dataclass(frozen=True, slots=True)
class Grant:
    allowed: bool
    budgets: tuple[tuple[str, float], ...] = ()

    def budget(self, name: str) -> float | None:
        return dict(self.budgets).get(name)


@dataclass(frozen=True, slots=True)
class Policy:
    grants: tuple[tuple[Effect, Grant], ...]
    origin: str = "default"  # "default" | "file" | "program"

    def grant(self, effect: Effect) -> Grant:
        return dict(self.grants)[effect]

    def allows(self, effect: Effect) -> bool:
        return self.grant(effect).allowed

    def budget(self, effect: Effect, name: str) -> float | None:
        return self.grant(effect).budget(name)

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "effects": {
                effect.value: {"allowed": grant.allowed, "budgets": dict(grant.budgets)}
                for effect, grant in self.grants
            },
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> Policy:
        effects = data["effects"]
        assert isinstance(effects, dict)
        grants = tuple(
            (
                Effect(name),
                Grant(bool(spec["allowed"]), tuple(sorted(dict(spec["budgets"]).items()))),
            )
            for name, spec in effects.items()
        )
        return Policy(grants, str(data.get("origin", "default")))


def default_policy() -> Policy:
    return Policy(tuple((e, Grant(e in DEFAULT_ALLOWED)) for e in Effect), "default")


@dataclass(slots=True)
class _Builder:
    grants: dict[Effect, Grant] = field(
        default_factory=lambda: {e: Grant(e in DEFAULT_ALLOWED) for e in Effect}
    )


def policy_from_block(block: n.PolicyBlock, origin: str, sink: DiagnosticSink) -> Policy:
    """Build a policy from a ``policy { ... }`` block, starting from the default policy."""
    builder = _Builder()
    seen: dict[Effect, Span] = {}
    for rule in block.rules:
        try:
            effect = Effect(rule.effect.name)
        except ValueError:
            sink.error(
                Stage.EFFECTS,
                "E401",
                f"unknown effect '{rule.effect.name}'",
                rule.effect.span,
                f"effects are: {', '.join(e.value for e in Effect)}",
            )
            continue
        if effect in seen:
            sink.error(
                Stage.EFFECTS,
                "E402",
                f"effect '{effect}' appears more than once in this policy",
                rule.span,
                f"first mentioned at {seen[effect]}",
            )
            continue
        seen[effect] = rule.span
        if rule.action == "forbid":
            if rule.kwargs:
                sink.error(
                    Stage.EFFECTS, "E403", "a forbidden effect cannot have budgets", rule.span
                )
            builder.grants[effect] = Grant(False)
            continue
        budgets: dict[str, float] = {}
        allowed_budgets = BUDGETS[effect]
        for kwarg in rule.kwargs:
            name = kwarg.name.name
            if name not in allowed_budgets:
                options = ", ".join(allowed_budgets) or "none"
                sink.error(
                    Stage.EFFECTS,
                    "E404",
                    f"effect '{effect}' has no budget '{name}'",
                    kwarg.name.span,
                    f"budgets for '{effect}': {options}",
                )
                continue
            value = _budget_value(kwarg, allowed_budgets[name], sink)
            if value is not None:
                budgets[name] = value
        if effect in UNIMPLEMENTED:
            sink.warning(
                Stage.EFFECTS, "W401", f"no operation in Weightsmith 0.1 uses '{effect}'", rule.span
            )
        builder.grants[effect] = Grant(True, tuple(sorted(budgets.items())))
    return Policy(tuple(builder.grants.items()), origin)


def _budget_value(kwarg: n.Kwarg, expected: Type, sink: DiagnosticSink) -> float | None:
    value = kwarg.value
    if expected == INT and isinstance(value, n.IntLit):
        if value.value < 0:
            sink.error(Stage.EFFECTS, "E405", "budgets cannot be negative", value.span)
            return None
        return float(value.value)
    if expected == FRACTION and isinstance(value, n.PercentLit | n.FloatLit):
        if not 0.0 <= value.value <= 1.0:
            sink.error(
                Stage.EFFECTS, "E405", "fraction budgets must lie between 0% and 100%", value.span
            )
            return None
        return float(value.value)
    sink.error(
        Stage.EFFECTS,
        "E406",
        f"budget '{kwarg.name.name}' expects {expected}",
        value.span,
        "write fractions as percentages, for example 5%",
    )
    return None


def effective_policy(
    file_policy: Policy | None, program_policy: Policy | None, span: Span, sink: DiagnosticSink
) -> Policy:
    """Combine a file-level policy with a program-level one. Inner policies may only tighten."""
    if program_policy is None:
        return file_policy or default_policy()
    if file_policy is None:
        return program_policy
    grants: list[tuple[Effect, Grant]] = []
    for effect in Effect:
        outer, inner = file_policy.grant(effect), program_policy.grant(effect)
        if inner.allowed and not outer.allowed:
            sink.error(
                Stage.EFFECTS,
                "E407",
                f"the program allows '{effect}' but the file policy does not",
                span,
                "a program policy may only tighten the file policy",
            )
            grants.append((effect, Grant(False)))
            continue
        if not inner.allowed:
            grants.append((effect, Grant(False)))
            continue
        budgets = dict(outer.budgets)
        for name, value in inner.budgets:
            if name in budgets and value > budgets[name]:
                sink.error(
                    Stage.EFFECTS,
                    "E408",
                    f"budget {effect}.{name} = {value:g} exceeds the "
                    f"file policy's {budgets[name]:g}",
                    span,
                )
                continue
            budgets[name] = value
        grants.append((effect, Grant(True, tuple(sorted(budgets.items())))))
    return Policy(tuple(grants), "program")


def check_effects(program: IRProgram, sink: DiagnosticSink) -> None:
    """Report every operation whose effects the program's policy does not allow."""
    policy = program.policy
    for op in program.ops:
        for effect in sorted(op.effects()):
            if not policy.allows(effect):
                sink.error(
                    Stage.EFFECTS,
                    "E410",
                    f"'{op.describe()}' needs the '{effect}' effect, "
                    f"which the policy of program '{program.name}' does not allow",
                    op.span,
                    f"add 'allow {effect}' to the program's policy block",
                )
    probe_limit = policy.budget(Effect.PROBE, "max_forward_passes")
    needed = program.estimated_forward_passes()
    if probe_limit is not None and needed is not None and needed > probe_limit:
        sink.warning(
            Stage.EFFECTS,
            "W402",
            f"verification needs about {needed} forward passes but "
            f"the probe budget is {probe_limit:g}",
            program.span,
            "behaviour checks that cannot run within the budget will be inconclusive",
        )
