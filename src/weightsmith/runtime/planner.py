"""Planning: capability checks before execution, and strategy selection for ``repair``."""

from __future__ import annotations

from dataclasses import dataclass

from weightsmith.backends.base import Backend, Capability
from weightsmith.compiler import ir
from weightsmith.errors import PlanningError, UnsupportedOperation
from weightsmith.evidence import Finding
from weightsmith.registry import STRATEGIES


def required_capabilities(program: ir.IRProgram) -> dict[Capability, list[ir.Op]]:
    needs: dict[Capability, list[ir.Op]] = {}

    def need(cap: Capability, op: ir.Op) -> None:
        needs.setdefault(cap, []).append(op)

    for op in program.ops:
        match op:
            case ir.Load(what="model"):
                need(Capability.LOAD_MODEL, op)
            case ir.Load(what="weights"):
                need(Capability.LOAD_WEIGHTS, op)
            case ir.Inspect():
                need(Capability.INSPECT, op)
            case ir.Find():
                need(Capability.FIND, op)
            case ir.Transform():
                need(Capability.EDIT, op)
            case ir.Compose():
                need(Capability.COMPOSE, op)
            case ir.Repair():
                need(Capability.REPAIR, op)
            case ir.Verify() if any(i.behavior is not None for i in op.items):
                need(Capability.FORWARD, op)
            case ir.Emit() if "patch" in op.outputs:
                need(Capability.PATCH, op)
            case ir.Emit() if "model" in op.outputs:
                need(Capability.SAVE, op)
    return needs


def check_backend(program: ir.IRProgram, backend: Backend) -> None:
    """Fail before any operation runs if the backend cannot execute the whole program."""
    problems: list[str] = []
    available = backend.capabilities()
    for cap, ops in required_capabilities(program).items():
        if cap not in available:
            problems.append(f"capability '{cap}' needed by {ops[0].describe()}")
    for op in program.ops:
        names: list[tuple[str, str]] = []
        match op:
            case ir.Find():
                names.append(("finder", op.finder))
                names += [("finder", k) for k in op.params.get("kinds", ())]
            case ir.Transform():
                names.append(("edit", op.op))
            case ir.Compose():
                names.append(("compose", op.op))
            case ir.Repair():
                names += [("strategy", s) for s in op.strategies]
            case ir.Load(what="behavior"):
                names.append(("metric", str(op.params.get("metric"))))
        problems += [f"{kind} '{name}'" for kind, name in names if not backend.supports(kind, name)]
    if problems:
        raise UnsupportedOperation(
            f"the '{backend.name}' backend cannot run program '{program.name}': "
            + "; ".join(problems)
        )


@dataclass(frozen=True, slots=True)
class RepairPlan:
    steps: tuple[tuple[Finding, str], ...]
    unhandled: tuple[Finding, ...]


def plan_repair(findings: list[Finding], strategies: tuple[str, ...]) -> RepairPlan:
    """Pair each finding with the first allowed strategy that handles its kind."""
    steps: list[tuple[Finding, str]] = []
    unhandled: list[Finding] = []
    for finding in findings:
        strategy = next((s for s in strategies if finding.kind in STRATEGIES[s].handles), None)
        if strategy is None:
            unhandled.append(finding)
        else:
            steps.append((finding, strategy))
    if findings and not steps:
        kinds = sorted({f.kind for f in findings})
        raise PlanningError(
            f"no applicable strategy: findings are of kind {', '.join(kinds)}, and none of "
            f"{', '.join(strategies)} handles them"
        )
    return RepairPlan(tuple(steps), tuple(unhandled))
