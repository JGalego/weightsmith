"""Parse Weightsmith source text into an AST (:mod:`weightsmith.language.nodes`)."""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedInput, UnexpectedToken
from lark.tree import Meta

from weightsmith.diagnostics import DiagnosticSink, Span, Stage
from weightsmith.language import nodes as n


@cache
def _lark() -> Lark:
    grammar = files("weightsmith.language").joinpath("grammar.lark").read_text(encoding="utf-8")
    return Lark(grammar, parser="lalr", lexer="contextual", propagate_positions=True)


class _Kwargs(tuple[n.Kwarg, ...]):
    """Marker type so optional ``kwargs`` children can be told apart from expressions."""


def _span_of(file: str, item: Meta | Token) -> Span:
    if isinstance(item, Token):
        line = item.line or 1
        column = item.column or 1
        return Span(file, line, column, item.end_line or line, item.end_column or column + 1)
    if getattr(item, "empty", True):
        return Span(file, 1, 1, 1, 1)
    return Span(file, item.line, item.column, item.end_line, item.end_column)


@v_args(meta=True)
class _ToAst(Transformer[Token, n.SourceFile]):
    def __init__(self, file: str) -> None:
        super().__init__()
        self.file = file

    # helpers -----------------------------------------------------------------
    def _s(self, item: Meta | Token) -> Span:
        return _span_of(self.file, item)

    def _ident(self, token: Token) -> n.Ident:
        return n.Ident(str(token), span=self._s(token))

    def _string(self, token: Token) -> n.StrLit:
        return n.StrLit(json.loads(str(token)), span=self._s(token))

    @staticmethod
    def _split_optional(children: list[Any]) -> tuple[list[Any], tuple[n.Kwarg, ...]]:
        if children and isinstance(children[-1], _Kwargs):
            return children[:-1], tuple(children[-1])
        return children, ()

    # file and declarations -----------------------------------------------------
    def start(self, meta: Meta, children: list[Any]) -> n.SourceFile:
        return n.SourceFile(self.file, tuple(children), span=self._s(meta))

    def load_decl(self, meta: Meta, children: list[Any]) -> n.LoadDecl:
        kind, name, path = children
        return n.LoadDecl(str(kind), self._ident(name), self._string(path), span=self._s(meta))

    def behavior_decl(self, meta: Meta, children: list[Any]) -> n.BehaviorDecl:
        name, path, *kwargs = children
        return n.BehaviorDecl(
            self._ident(name), self._string(path), tuple(kwargs), span=self._s(meta)
        )

    def selection_decl(self, meta: Meta, children: list[Any]) -> n.SelectionDecl:
        name, expr = children
        return n.SelectionDecl(self._ident(name), expr, span=self._s(meta))

    def policy_block(self, meta: Meta, children: list[Any]) -> n.PolicyBlock:
        return n.PolicyBlock(tuple(children), span=self._s(meta))

    def policy_rule(self, meta: Meta, children: list[Any]) -> n.PolicyRule:
        rest, kwargs = self._split_optional(children)
        action, effect = rest
        return n.PolicyRule(str(action), self._ident(effect), kwargs, span=self._s(meta))

    def program(self, meta: Meta, children: list[Any]) -> n.Program:
        name, target, *body = children
        return n.Program(self._ident(name), self._ident(target), tuple(body), span=self._s(meta))

    # statements ------------------------------------------------------------------
    def let_stmt(self, meta: Meta, children: list[Any]) -> n.LetStmt:
        name, expr = children
        return n.LetStmt(self._ident(name), expr, span=self._s(meta))

    def inspect_stmt(self, meta: Meta, children: list[Any]) -> n.InspectStmt:
        return n.InspectStmt(children[0] if children else None, span=self._s(meta))

    def find_stmt(self, meta: Meta, children: list[Any]) -> n.FindStmt:
        rest, kwargs = self._split_optional(children)
        finder = self._ident(rest[0])
        scope = rest[1] if len(rest) > 1 else None
        return n.FindStmt(finder, scope, kwargs, span=self._s(meta))

    def edit_block(self, meta: Meta, children: list[Any]) -> n.EditBlock:
        return n.EditBlock(tuple(children), span=self._s(meta))

    def compose_block(self, meta: Meta, children: list[Any]) -> n.ComposeBlock:
        return n.ComposeBlock(tuple(children), span=self._s(meta))

    def op_stmt(self, meta: Meta, children: list[Any]) -> n.OpStmt:
        rest, kwargs = self._split_optional(children)
        op, operand = rest
        return n.OpStmt(self._ident(op), operand, kwargs, span=self._s(meta))

    def repair_stmt(self, meta: Meta, children: list[Any]) -> n.RepairStmt:
        _, kwargs = self._split_optional(children)
        return n.RepairStmt(kwargs, span=self._s(meta))

    def contract_block(self, meta: Meta, children: list[Any]) -> n.ContractBlock:
        return n.ContractBlock(tuple(children), span=self._s(meta))

    def contract_item(self, meta: Meta, children: list[Any]) -> n.ContractItem:
        rest, kwargs = self._split_optional(children)
        clause, subject = rest
        return n.ContractItem(str(clause), self._ident(subject), kwargs, span=self._s(meta))

    def verify_stmt(self, meta: Meta, children: list[Any]) -> n.VerifyStmt:
        return n.VerifyStmt(span=self._s(meta))

    def output_block(self, meta: Meta, children: list[Any]) -> n.OutputBlock:
        return n.OutputBlock(tuple(children), span=self._s(meta))

    def output_item(self, meta: Meta, children: list[Any]) -> n.Ident:
        return self._ident(children[0])

    # arguments and expressions -----------------------------------------------------
    def kwargs(self, meta: Meta, children: list[Any]) -> _Kwargs:
        return _Kwargs(children)

    def kwarg(self, meta: Meta, children: list[Any]) -> n.Kwarg:
        name, value = children
        return n.Kwarg(self._ident(name), value, span=self._s(meta))

    def attr(self, meta: Meta, children: list[Any]) -> n.Attr:
        obj, name = children
        return n.Attr(obj, self._ident(name), span=self._s(meta))

    def subscript(self, meta: Meta, children: list[Any]) -> n.Subscript:
        obj, *indices = children
        return n.Subscript(obj, tuple(indices), span=self._s(meta))

    def name(self, meta: Meta, children: list[Any]) -> n.Name:
        return n.Name(str(children[0]), span=self._s(meta))

    def int_lit(self, meta: Meta, children: list[Any]) -> n.IntLit:
        return n.IntLit(int(children[0]), span=self._s(meta))

    def float_lit(self, meta: Meta, children: list[Any]) -> n.FloatLit:
        return n.FloatLit(float(children[0]), span=self._s(meta))

    def percent_lit(self, meta: Meta, children: list[Any]) -> n.PercentLit:
        text = str(children[0])
        return n.PercentLit(float(text[:-1]) / 100.0, text, span=self._s(meta))

    def str_lit(self, meta: Meta, children: list[Any]) -> n.StrLit:
        return self._string(children[0])

    def bool_lit(self, meta: Meta, children: list[Any]) -> n.BoolLit:
        return n.BoolLit(str(children[0]) == "true", span=self._s(meta))

    def list_lit(self, meta: Meta, children: list[Any]) -> n.ListLit:
        return n.ListLit(tuple(children), span=self._s(meta))

    def range_index(self, meta: Meta, children: list[Any]) -> n.RangeIndex:
        start, stop = children
        return n.RangeIndex(int(start), int(stop), span=self._s(meta))

    def full_index(self, meta: Meta, children: list[Any]) -> n.FullIndex:
        return n.FullIndex(span=self._s(meta))


def _describe_terminal(name: str) -> str:
    try:
        pattern = _lark().get_terminal(name).pattern
    except KeyError:
        return name
    if pattern.type == "str":
        return f"'{pattern.value}'"
    return {
        "NAME": "a name",
        "STRING": "a string",
        "INT": "an integer",
        "FLOAT": "a number",
        "PERCENT": "a percentage",
        "BOOL": "true or false",
    }.get(name, name.lower())


def _expected(names: set[str] | frozenset[str]) -> str:
    described = sorted({_describe_terminal(name) for name in names})
    if len(described) > 8:
        described = [*described[:8], "..."]
    return ", ".join(described)


def parse(source: str, file: str, sink: DiagnosticSink) -> n.SourceFile | None:
    """Parse ``source``. Syntax errors are reported to ``sink`` and ``None`` is returned."""
    try:
        tree = _lark().parse(source)
    except UnexpectedInput as exc:
        line = getattr(exc, "line", 1) or 1
        column = getattr(exc, "column", 1) or 1
        span = Span(file, line, column, line, column + 1)
        if isinstance(exc, UnexpectedToken):
            token = exc.token
            if token.type == "$END":
                sink.error(
                    Stage.SYNTAX,
                    "E101",
                    "unexpected end of file",
                    span,
                    f"expected {_expected(exc.expected)}",
                )
            else:
                span = _span_of(file, token)
                what = _describe_terminal(token.type)
                if not what.startswith("'"):
                    what = f"{what.removeprefix('a ').removeprefix('an ')} '{token}'"
                sink.error(
                    Stage.SYNTAX,
                    "E102",
                    f"unexpected {what}",
                    span,
                    f"expected {_expected(exc.expected)}",
                )
        elif isinstance(exc, UnexpectedCharacters):
            char = source[exc.pos_in_stream] if exc.pos_in_stream < len(source) else "?"
            sink.error(Stage.SYNTAX, "E103", f"unexpected character {char!r}", span)
        elif isinstance(exc, UnexpectedEOF):
            sink.error(
                Stage.SYNTAX,
                "E101",
                "unexpected end of file",
                span,
                f"expected {_expected(set(exc.expected))}",
            )
        else:  # pragma: no cover - lark has no other public subclasses
            sink.error(Stage.SYNTAX, "E100", str(exc), span)
        return None
    try:
        return _ToAst(file).transform(tree)
    except json.JSONDecodeError as exc:  # malformed escape sequence inside a string literal
        sink.error(Stage.SYNTAX, "E104", f"invalid string literal: {exc.msg}", None)
        return None
