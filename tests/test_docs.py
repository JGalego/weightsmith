"""The semantics document must describe every name the registries define."""

from __future__ import annotations

from pathlib import Path

import pytest

from weightsmith import registry
from weightsmith.compiler.effects import Effect
from weightsmith.compiler.ir import OP_TYPES

SEMANTICS = (Path(__file__).resolve().parent.parent / "docs" / "semantics.md").read_text()

NAMES = [
    *registry.FINDERS,
    *registry.EDIT_OPS,
    *registry.COMPOSE_OPS,
    *registry.STRATEGIES,
    *registry.METRICS,
    *registry.REQUIREMENTS,
    *registry.OUTPUTS,
    *registry.PROVIDERS,
    *registry.TOOLS,
    *(e.value for e in Effect),
]


@pytest.mark.parametrize("name", NAMES)
def test_registry_entry_is_documented(name: str) -> None:
    assert f"`{name}`" in SEMANTICS


@pytest.mark.parametrize("kind", sorted(OP_TYPES))
def test_ir_op_is_documented(kind: str) -> None:
    assert f"`{kind.capitalize()}`" in SEMANTICS
