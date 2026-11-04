from __future__ import annotations

from pathlib import Path

import pytest

from weightsmith.diagnostics import DiagnosticSink, Stage
from weightsmith.language import nodes as n
from weightsmith.language.parser import parse

from .conftest import EXAMPLES


def _parse(source: str) -> tuple[n.SourceFile | None, DiagnosticSink]:
    sink = DiagnosticSink()
    return parse(source, "t.wsm", sink), sink


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.wsm")), ids=lambda p: p.name)
def test_examples_parse(path: Path) -> None:
    sink = DiagnosticSink()
    tree = parse(path.read_text(), str(path), sink)
    assert tree is not None, sink.items
    assert not sink.items
    assert len(tree.programs) == 1


def test_declarations_and_statements() -> None:
    tree, _ = _parse("""
        // comment
        model M = load("dir")
        weights W = load("w.safetensors")
        behavior b = probes("b.json", metric = accuracy)
        selection S = M.params["layers.*"]
        policy { allow inspect forbid save }
        program P {
            target M
            let s = M.params["a"][1..3, :]
            inspect
            inspect s
            find outlier_rows in M.params["x"] (threshold = 6.5)
            edit { zero s  scale M.params["y"] (factor = 0.5) }
            compose { interpolate W (alpha = 50%) }
            repair (strategies = [rescale_outlier_rows])
            contract { preserve b (tolerance = 0.01) require finite }
            verify
            output { model, patch }
        }
    """)
    assert tree is not None
    kinds = [type(d).__name__ for d in tree.decls]
    assert kinds == [
        "LoadDecl",
        "LoadDecl",
        "BehaviorDecl",
        "SelectionDecl",
        "PolicyBlock",
        "Program",
    ]
    program = tree.programs[0]
    let = program.body[0]
    assert isinstance(let, n.LetStmt)
    assert isinstance(let.expr, n.Subscript)
    assert let.expr.indices == (
        n.RangeIndex(1, 3, span=let.expr.indices[0].span),
        n.FullIndex(span=let.expr.indices[1].span),
    )
    compose = program.body[5]
    assert isinstance(compose, n.ComposeBlock)
    alpha = compose.ops[0].kwargs[0].value
    assert isinstance(alpha, n.PercentLit)
    assert alpha.value == pytest.approx(0.5)
    output = program.body[-1]
    assert isinstance(output, n.OutputBlock)
    assert [i.name for i in output.items] == ["model", "patch"]


def test_spans_point_at_source() -> None:
    tree, _ = _parse("program P {\n  target M\n  find nan_inf\n}\n")
    assert tree is not None
    find = tree.programs[0].body[0]
    assert isinstance(find, n.FindStmt)
    assert (find.span.line, find.span.column) == (3, 3)
    assert (find.finder.span.line, find.finder.span.column) == (3, 8)


def test_keywords_do_not_swallow_longer_names() -> None:
    tree, sink = _parse("program P { target M  inspect trueish  inspect models }")
    assert tree is not None, sink.items
    subjects = [s.subject for s in tree.programs[0].body]  # type: ignore[union-attr]
    assert [s.ident for s in subjects] == ["trueish", "models"]  # type: ignore[union-attr]


def test_number_literals() -> None:
    tree, _ = _parse('program P { target M  edit { fill M.params["w"][2..4] (value = 1.5e-3) } }')
    assert tree is not None
    block = tree.programs[0].body[0]
    assert isinstance(block, n.EditBlock)
    op = block.ops[0]
    assert isinstance(op.operand, n.Subscript)
    index = op.operand.indices[0]
    assert index == n.RangeIndex(2, 4, span=index.span)
    value = op.kwargs[0].value
    assert isinstance(value, n.FloatLit)
    assert value.value == pytest.approx(1.5e-3)


@pytest.mark.parametrize(
    ("source", "code", "line"),
    [
        ("program P {\n  target M\n  find\n}\n", "E102", 4),
        ("program P {\n  target M\n", "E101", 2),
        ("program P {\n  target M\n  inspect $\n}\n", "E103", 3),
        ("model M = load(dir)\n", "E102", 1),
    ],
)
def test_syntax_errors(source: str, code: str, line: int) -> None:
    tree, sink = _parse(source)
    assert tree is None
    [diag] = sink.items
    assert diag.code == code
    assert diag.stage is Stage.SYNTAX
    assert diag.span is not None
    assert diag.span.line == line


def test_syntax_error_lists_expected_tokens() -> None:
    _, sink = _parse("program P {\n  target M\n  find\n}\n")
    assert any("a name" in note for note in sink.items[0].notes)
