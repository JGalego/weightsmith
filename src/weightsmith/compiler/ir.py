"""The Weightsmith intermediate representation.

A program lowers to a linear sequence of operations over numbered values (``%0``, ``%1``, ...).
Loaded weight sets, behaviours and selections are immutable values. Mutating operations act on a
single implicit *working copy* of the program's target; the original stays untouched, which is what
makes before/after comparison, patches and rollback possible.

The IR is framework-independent: it names operations and parameters, never tensors. It serialises
to JSON so ``weightsmith compile`` output can be inspected, stored and executed later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from weightsmith.compiler.effects import Effect, Policy
from weightsmith.diagnostics import Span

IR_VERSION = "weightsmith.ir/v0"

Params = dict[str, Any]


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """One dimension of a slice: a single index, a half-open range, or everything."""

    kind: str  # "int" | "range" | "all"
    start: int = 0
    stop: int = 0

    def __str__(self) -> str:
        if self.kind == "int":
            return str(self.start)
        if self.kind == "range":
            return f"{self.start}..{self.stop}"
        return ":"


@dataclass(frozen=True, slots=True, kw_only=True)
class Op:
    kind: ClassVar[str] = "op"
    span: Span | None = None

    def effects(self) -> frozenset[Effect]:
        return frozenset()

    def describe(self) -> str:
        return self.kind


@dataclass(frozen=True, slots=True)
class Load(Op):
    kind: ClassVar[str] = "load"
    out: int
    binding: str
    what: str  # "model" | "weights" | "behavior"
    path: str
    params: Params = field(default_factory=dict)

    def effects(self) -> frozenset[Effect]:
        return frozenset({Effect.LOAD})

    def describe(self) -> str:
        return f"load {self.binding}"


@dataclass(frozen=True, slots=True)
class Select(Op):
    """Select parameters of a weight set by glob, or restrict a selection to index ranges."""

    kind: ClassVar[str] = "select"
    out: int
    source: int
    pattern: str | None = None
    index: tuple[IndexSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class Inspect(Op):
    kind: ClassVar[str] = "inspect"
    source: int

    def effects(self) -> frozenset[Effect]:
        return frozenset({Effect.INSPECT})


@dataclass(frozen=True, slots=True)
class Find(Op):
    kind: ClassVar[str] = "find"
    finder: str
    scope: int
    params: Params = field(default_factory=dict)

    def effects(self) -> frozenset[Effect]:
        return frozenset({Effect.INSPECT})

    def describe(self) -> str:
        return f"find {self.finder}"


@dataclass(frozen=True, slots=True)
class Transform(Op):
    """An explicit edit of the working copy (``edit { ... }``)."""

    kind: ClassVar[str] = "transform"
    op: str
    operand: int
    params: Params = field(default_factory=dict)

    def effects(self) -> frozenset[Effect]:
        return frozenset({Effect.MUTATE})

    def describe(self) -> str:
        return f"edit {self.op}"


@dataclass(frozen=True, slots=True)
class Compose(Op):
    """Combine the working copy with another weight set (``compose { ... }``)."""

    kind: ClassVar[str] = "compose"
    op: str
    operand: int
    params: Params = field(default_factory=dict)

    def effects(self) -> frozenset[Effect]:
        return frozenset({Effect.MUTATE})

    def describe(self) -> str:
        return f"compose {self.op}"


@dataclass(frozen=True, slots=True)
class Repair(Op):
    """Declarative repair: the planner chooses transformations from findings at run time."""

    kind: ClassVar[str] = "repair"
    strategies: tuple[str, ...]

    def effects(self) -> frozenset[Effect]:
        return frozenset({Effect.MUTATE})


@dataclass(frozen=True, slots=True)
class ContractItem:
    clause: str  # "preserve" | "improve" | "require"
    subject: str
    behavior: int | None
    params: Params
    span: Span | None = None

    def describe(self) -> str:
        return f"{self.clause} {self.subject}"


@dataclass(frozen=True, slots=True)
class Verify(Op):
    kind: ClassVar[str] = "verify"
    items: tuple[ContractItem, ...]

    def effects(self) -> frozenset[Effect]:
        needs_probe = any(item.behavior is not None for item in self.items)
        return frozenset({Effect.PROBE if needs_probe else Effect.INSPECT})


@dataclass(frozen=True, slots=True)
class Emit(Op):
    kind: ClassVar[str] = "emit"
    outputs: tuple[str, ...]

    def effects(self) -> frozenset[Effect]:
        return frozenset({Effect.SAVE}) if {"model", "patch"} & set(self.outputs) else frozenset()

    def describe(self) -> str:
        return "output"


OP_TYPES: dict[str, type[Op]] = {
    cls.kind: cls for cls in (Load, Select, Inspect, Find, Transform, Compose, Repair, Verify, Emit)
}


@dataclass(frozen=True, slots=True)
class IRProgram:
    name: str
    source: str
    target: int
    target_binding: str
    target_is_model: bool
    policy: Policy
    ops: tuple[Op, ...]
    probe_sizes: dict[int, int] = field(default_factory=dict)  # behaviour value -> suite size
    span: Span | None = None
    version: str = IR_VERSION

    # analysis ---------------------------------------------------------------
    def effects(self) -> frozenset[Effect]:
        return frozenset().union(*(op.effects() for op in self.ops))

    def estimated_forward_passes(self) -> int | None:
        """Forward passes verification needs: one baseline pass per suite, one per verify."""
        verifies = [op for op in self.ops if isinstance(op, Verify)]
        behaviors = {
            item.behavior for op in verifies for item in op.items if item.behavior is not None
        }
        if not behaviors:
            return 0
        if any(b not in self.probe_sizes for b in behaviors):
            return None
        total = sum(self.probe_sizes[b] for b in behaviors)
        total += sum(
            self.probe_sizes[b]
            for op in verifies
            for b in {i.behavior for i in op.items if i.behavior is not None}
        )
        return total

    # serialisation -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "target_binding": self.target_binding,
            "target_is_model": self.target_is_model,
            "policy": self.policy.to_dict(),
            "probe_sizes": {str(k): v for k, v in self.probe_sizes.items()},
            "span": self.span.to_dict() if self.span else None,
            "ops": [_op_to_dict(op) for op in self.ops],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> IRProgram:
        if data.get("version") != IR_VERSION:
            raise ValueError(
                f"unsupported IR version {data.get('version')!r}, expected {IR_VERSION}"
            )
        return IRProgram(
            name=data["name"],
            source=data["source"],
            target=int(data["target"]),
            target_binding=data["target_binding"],
            target_is_model=bool(data["target_is_model"]),
            policy=Policy.from_dict(data["policy"]),
            ops=tuple(_op_from_dict(op) for op in data["ops"]),
            probe_sizes={int(k): int(v) for k, v in data.get("probe_sizes", {}).items()},
            span=Span.from_dict(data["span"]) if data.get("span") else None,
        )

    @staticmethod
    def from_json(text: str) -> IRProgram:
        return IRProgram.from_dict(json.loads(text))

    def content_hash(self) -> str:
        """Hash of the program's semantics: ops and policy, without source locations."""
        body = self.to_dict()
        for op in body["ops"]:
            op.pop("span", None)
            for item in op.get("items", []):
                item.pop("span", None)
        body.pop("span", None)
        body.pop("source", None)
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    # pretty printing -----------------------------------------------------------
    def to_text(self) -> str:
        lines = [
            f"program {self.name}  ; {self.version}, target %{self.target} ({self.target_binding})"
        ]
        for op in self.ops:
            effects = ", ".join(sorted(e.value for e in op.effects())) or "pure"
            lines.append(f"  {_op_text(op):<72} ; {effects}")
        return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _op_to_dict(op: Op) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": op.kind}
    for f in fields(op):
        value = getattr(op, f.name)
        if f.name == "span":
            out["span"] = value.to_dict() if value else None
        elif f.name == "index":
            out["index"] = [{"kind": i.kind, "start": i.start, "stop": i.stop} for i in value]
        elif f.name == "items":
            out["items"] = [
                {
                    "clause": i.clause,
                    "subject": i.subject,
                    "behavior": i.behavior,
                    "params": _jsonable(i.params),
                    "span": i.span.to_dict() if i.span else None,
                }
                for i in value
            ]
        else:
            out[f.name] = _jsonable(value)
    return out


def _op_from_dict(data: dict[str, Any]) -> Op:
    data = dict(data)
    cls = OP_TYPES[data.pop("kind")]
    span = data.pop("span", None)
    kwargs: dict[str, Any] = {"span": Span.from_dict(span) if span else None}
    for name, value in data.items():
        if name == "index":
            kwargs[name] = tuple(IndexSpec(**i) for i in value)
        elif name == "items":
            kwargs[name] = tuple(
                ContractItem(
                    i["clause"],
                    i["subject"],
                    i["behavior"],
                    _detuple(i["params"]),
                    Span.from_dict(i["span"]) if i.get("span") else None,
                )
                for i in value
            )
        elif name in ("outputs", "strategies"):
            kwargs[name] = tuple(value)
        elif name == "params":
            kwargs[name] = _detuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _detuple(params: dict[str, Any]) -> Params:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}


def _fmt_params(params: Params) -> str:
    if not params:
        return ""
    parts = []
    for key, value in params.items():
        shown = "[" + ", ".join(map(str, value)) + "]" if isinstance(value, tuple | list) else value
        parts.append(f"{key}={shown}")
    return " (" + ", ".join(parts) + ")"


def _op_text(op: Op) -> str:
    match op:
        case Load():
            return f'%{op.out} = load {op.what} {op.binding} "{op.path}"{_fmt_params(op.params)}'
        case Select():
            what = f'["{op.pattern}"]' if op.pattern is not None else ""
            if op.index:
                what += "[" + ", ".join(map(str, op.index)) + "]"
            return f"%{op.out} = select %{op.source}{what}"
        case Inspect():
            return f"inspect %{op.source}"
        case Find():
            return f"find {op.finder} in %{op.scope}{_fmt_params(op.params)}"
        case Transform():
            return f"transform {op.op} %{op.operand}{_fmt_params(op.params)}"
        case Compose():
            return f"compose {op.op} %{op.operand}{_fmt_params(op.params)}"
        case Repair():
            return f"repair (strategies=[{', '.join(op.strategies)}])"
        case Verify():
            items = "; ".join(
                f"{i.clause} {i.subject}"
                + (f"(%{i.behavior})" if i.behavior is not None else "")
                + _fmt_params(i.params)
                for i in op.items
            )
            return f"verify {{{items}}}"
        case Emit():
            return f"emit {{{', '.join(op.outputs)}}}"
        case _:  # pragma: no cover
            return op.kind
