from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from weightsmith.compiler import ir
from weightsmith.compiler.driver import CompileResult, compile_file

from .conftest import EXAMPLES

Compile = Callable[[str], CompileResult]


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.wsm")), ids=lambda p: p.name)
def test_examples_lower_and_round_trip(path: Path) -> None:
    result = compile_file(path)
    assert result.ok, [d.message for d in result.sink.errors]
    program = result.program()
    again = ir.IRProgram.from_json(program.to_json())
    assert again.to_json() == program.to_json()
    assert again.content_hash() == program.content_hash()


def test_lowering_shape(workspace: Path) -> None:
    program = compile_file(workspace / "merge_mlps.wsm").program()
    kinds = [op.kind for op in program.ops]
    assert kinds == [
        "load",
        "load",
        "load",
        "load",
        "compose",
        "select",
        "compose",
        "verify",
        "emit",
    ]
    loads = [op for op in program.ops if isinstance(op, ir.Load)]
    assert program.target == loads[0].out
    assert [op.what for op in loads] == ["model", "behavior", "behavior", "model"]
    verify = next(op for op in program.ops if isinstance(op, ir.Verify))
    assert [i.describe() for i in verify.items] == [
        "preserve parity",
        "preserve majority",
        "require finite",
    ]
    assert program.probe_sizes == {1: 256, 2: 256}
    assert program.estimated_forward_passes() == 1024


def test_hash_ignores_source_locations(compile_text: Compile) -> None:
    body = (
        '\nweights W = load("fixtures/tiny_mlp/model.safetensors")\nprogram P { target W inspect }'
    )
    a = compile_text(body).program()
    b = compile_text("\n\n\n" + body).program()
    assert a.to_json() != b.to_json()
    assert a.content_hash() == b.content_hash()


def test_rejects_unknown_version(workspace: Path) -> None:
    data = json.loads(compile_file(workspace / "repair_mlp.wsm").program().to_json())
    data["version"] = "weightsmith.ir/v9"
    with pytest.raises(ValueError, match="unsupported IR version"):
        ir.IRProgram.from_dict(data)


def test_effects_per_op() -> None:
    from weightsmith.compiler.effects import Effect

    assert ir.Emit(("findings",)).effects() == frozenset()
    assert ir.Emit(("patch",)).effects() == {Effect.SAVE}
    assert ir.Verify((ir.ContractItem("require", "finite", None, {}),)).effects() == {
        Effect.INSPECT
    }
    assert ir.Verify((ir.ContractItem("preserve", "b", 1, {}),)).effects() == {Effect.PROBE}


def test_text_form_mentions_every_op(workspace: Path) -> None:
    text = compile_file(workspace / "repair_mlp.wsm").program().to_text()
    for word in ("load model Base", "find anomalies", "repair", "verify", "emit"):
        assert word in text
