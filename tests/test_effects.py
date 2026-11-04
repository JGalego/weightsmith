from __future__ import annotations

from collections.abc import Callable

import pytest

from weightsmith.compiler.driver import CompileResult
from weightsmith.compiler.effects import Effect

from .conftest import HEADER, codes

Compile = Callable[[str], CompileResult]

BODY = """
    contract { preserve parity (tolerance = 0.02) }
    edit { scale Base.params["head.weight"] (factor = 1.0) }
    verify
    output { patch }
"""


def test_default_policy_allows_only_load_and_inspect(compile_text: Compile) -> None:
    result = compile_text(f"{HEADER}\nprogram P {{ target Base {BODY} }}")
    errors = [d for d in result.sink.errors if d.code == "E410"]
    needed = {msg.split("'")[3] for msg in (d.message for d in errors)}
    assert needed == {"mutate", "probe", "save"}


def test_policy_allows_effects(compile_text: Compile) -> None:
    result = compile_text(
        f"{HEADER}\nprogram P {{ target Base\n policy {{ allow probe allow "
        f"mutate allow save }} {BODY} }}"
    )
    assert result.ok
    program = result.program()
    assert program.effects() == {Effect.LOAD, Effect.MUTATE, Effect.PROBE, Effect.SAVE}
    assert program.policy.allows(Effect.PROBE)
    assert not program.policy.allows(Effect.TRAIN)


@pytest.mark.parametrize(
    ("policy", "code"),
    [
        ("allow teleport", "E401"),
        ("allow probe forbid probe", "E402"),
        ("forbid probe(max_forward_passes = 3)", "E403"),
        ("allow probe(max_calls = 3)", "E404"),
        ("allow probe(max_forward_passes = 2.5)", "E406"),
        ("allow mutate(max_changed_params = 0.02)", None),
        ("allow mutate(max_changed_params = 5)", "E406"),
        ("allow mutate(max_changed_params = 500%)", "E405"),
    ],
)
def test_policy_rules(compile_text: Compile, policy: str, code: str | None) -> None:
    result = compile_text(f"{HEADER}\nprogram P {{ target Base policy {{ {policy} }} inspect }}")
    if code is None:
        assert result.ok
    else:
        assert code in codes(result)


def test_program_policy_may_only_tighten(compile_text: Compile) -> None:
    outer = "policy { allow probe(max_forward_passes = 100) }"
    looser = compile_text(
        f"{HEADER}\n{outer}\nprogram P {{ target Base policy {{ allow mutate }} inspect }}"
    )
    assert "E407" in codes(looser)
    bigger = compile_text(
        f"{HEADER}\n{outer}\nprogram P {{ target Base policy {{ "
        f"allow probe(max_forward_passes = 500) }} inspect }}"
    )
    assert "E408" in codes(bigger)
    tighter = compile_text(
        f"{HEADER}\n{outer}\nprogram P {{ target Base policy {{ "
        f"allow probe(max_forward_passes = 10) }} inspect }}"
    )
    assert tighter.ok
    assert tighter.program().policy.budget(Effect.PROBE, "max_forward_passes") == 10


def test_inherited_budget_from_file_policy(compile_text: Compile) -> None:
    result = compile_text(
        f"{HEADER}\npolicy {{ allow probe(max_forward_passes = 100) }}\n"
        "program P { target Base policy { allow probe } inspect }"
    )
    assert result.program().policy.budget(Effect.PROBE, "max_forward_passes") == 100


def test_flagship_budget_contradiction_is_flagged(compile_text: Compile) -> None:
    """A probe budget too small for the declared verification is reported before running."""
    result = compile_text(
        f"{HEADER}\nprogram P {{ target Base\n policy {{ allow probe("
        f"max_forward_passes = 50) allow mutate allow save }} {BODY} }}"
    )
    assert result.ok
    assert "W402" in [d.code for d in result.sink.warnings]
    assert result.program().estimated_forward_passes() == 512


def test_unimplemented_effects_warn(compile_text: Compile) -> None:
    result = compile_text(f"{HEADER}\nprogram P {{ target Base policy {{ allow train }} inspect }}")
    assert "W401" in [d.code for d in result.sink.warnings]
