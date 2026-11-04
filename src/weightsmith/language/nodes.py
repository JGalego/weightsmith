"""AST node definitions.

Every node is an immutable dataclass carrying the source span it was parsed from. The module is
named ``nodes`` rather than ``ast`` so it cannot shadow the standard library module.
"""

from __future__ import annotations

from dataclasses import dataclass

from weightsmith.diagnostics import Span


@dataclass(frozen=True, slots=True, kw_only=True)
class Node:
    span: Span


@dataclass(frozen=True, slots=True)
class Ident(Node):
    name: str


# --------------------------------------------------------------------------- expressions


@dataclass(frozen=True, slots=True)
class Name(Node):
    ident: str


@dataclass(frozen=True, slots=True)
class Attr(Node):
    obj: Expr
    attr: Ident


@dataclass(frozen=True, slots=True)
class IntLit(Node):
    value: int


@dataclass(frozen=True, slots=True)
class FloatLit(Node):
    value: float


@dataclass(frozen=True, slots=True)
class PercentLit(Node):
    """``p%``. ``value`` holds the fraction ``p / 100``."""

    value: float
    text: str


@dataclass(frozen=True, slots=True)
class StrLit(Node):
    value: str


@dataclass(frozen=True, slots=True)
class BoolLit(Node):
    value: bool


@dataclass(frozen=True, slots=True)
class ListLit(Node):
    items: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class RangeIndex(Node):
    """``a..b``: the half-open index range [a, b)."""

    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class FullIndex(Node):
    """``:``: every index along a dimension."""


Index = IntLit | StrLit | RangeIndex | FullIndex


@dataclass(frozen=True, slots=True)
class Subscript(Node):
    obj: Expr
    indices: tuple[Index, ...]


Literal = IntLit | FloatLit | PercentLit | StrLit | BoolLit | ListLit
Expr = Name | Attr | Subscript | Literal


@dataclass(frozen=True, slots=True)
class Kwarg(Node):
    name: Ident
    value: Expr


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class LoadDecl(Node):
    """``model M = load("dir")`` or ``weights W = load("file.safetensors")``."""

    kind: str  # "model" | "weights"
    name: Ident
    path: StrLit


@dataclass(frozen=True, slots=True)
class BehaviorDecl(Node):
    """``behavior B = probes("suite.json", metric = accuracy)``."""

    name: Ident
    path: StrLit
    kwargs: tuple[Kwarg, ...]


@dataclass(frozen=True, slots=True)
class SelectionDecl(Node):
    name: Ident
    expr: Expr


@dataclass(frozen=True, slots=True)
class PolicyRule(Node):
    action: str  # "allow" | "forbid"
    effect: Ident
    kwargs: tuple[Kwarg, ...]


@dataclass(frozen=True, slots=True)
class PolicyBlock(Node):
    rules: tuple[PolicyRule, ...]


# --------------------------------------------------------------------------- statements


@dataclass(frozen=True, slots=True)
class LetStmt(Node):
    name: Ident
    expr: Expr


@dataclass(frozen=True, slots=True)
class InspectStmt(Node):
    subject: Expr | None


@dataclass(frozen=True, slots=True)
class FindStmt(Node):
    finder: Ident
    scope: Expr | None
    kwargs: tuple[Kwarg, ...]


@dataclass(frozen=True, slots=True)
class OpStmt(Node):
    """One operation inside an ``edit`` or ``compose`` block: ``op operand (k = v, ...)``."""

    op: Ident
    operand: Expr
    kwargs: tuple[Kwarg, ...]


@dataclass(frozen=True, slots=True)
class EditBlock(Node):
    ops: tuple[OpStmt, ...]


@dataclass(frozen=True, slots=True)
class ComposeBlock(Node):
    ops: tuple[OpStmt, ...]


@dataclass(frozen=True, slots=True)
class RepairStmt(Node):
    kwargs: tuple[Kwarg, ...]


@dataclass(frozen=True, slots=True)
class ContractItem(Node):
    clause: str  # "preserve" | "improve" | "require"
    subject: Ident
    kwargs: tuple[Kwarg, ...]


@dataclass(frozen=True, slots=True)
class ContractBlock(Node):
    items: tuple[ContractItem, ...]


@dataclass(frozen=True, slots=True)
class VerifyStmt(Node):
    pass


@dataclass(frozen=True, slots=True)
class OutputBlock(Node):
    items: tuple[Ident, ...]


Stmt = (
    LetStmt
    | InspectStmt
    | FindStmt
    | EditBlock
    | ComposeBlock
    | RepairStmt
    | ContractBlock
    | VerifyStmt
    | OutputBlock
    | PolicyBlock
)


@dataclass(frozen=True, slots=True)
class Program(Node):
    name: Ident
    target: Ident
    body: tuple[Stmt, ...]


Decl = LoadDecl | BehaviorDecl | SelectionDecl | PolicyBlock | Program


@dataclass(frozen=True, slots=True)
class SourceFile(Node):
    path: str
    decls: tuple[Decl, ...]

    @property
    def programs(self) -> list[Program]:
        return [d for d in self.decls if isinstance(d, Program)]
