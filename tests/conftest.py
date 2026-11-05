"""Shared fixtures. Synthetic models are built once and cached in the pytest cache directory."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from weightsmith.compiler.driver import CompileResult, compile_source

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="session")
def fixtures_dir(request: pytest.FixtureRequest) -> Path:
    from weightsmith.synthetic import build_all

    root = Path(request.config.cache.mkdir("weightsmith-fixtures"))
    build_all(root)
    return root


@pytest.fixture
def workspace(tmp_path: Path, fixtures_dir: Path) -> Path:
    """A directory holding copies of the example programs and a link to built fixtures."""
    for program in EXAMPLES.glob("*.wsm"):
        shutil.copy(program, tmp_path / program.name)
    shutil.copytree(EXAMPLES / "editors", tmp_path / "editors")
    (tmp_path / "fixtures").symlink_to(fixtures_dir, target_is_directory=True)
    return tmp_path


@pytest.fixture
def compile_text(workspace: Path) -> Callable[[str], CompileResult]:
    """Compile source text as if it lived in the workspace (so fixture paths resolve)."""

    def _compile(source: str) -> CompileResult:
        return compile_source(source, str(workspace / "test.wsm"), workspace)

    return _compile


def codes(result: CompileResult) -> list[str]:
    return [d.code for d in result.sink.errors]


HEADER = """
model Base = load("fixtures/tiny_mlp")
model Donor = load("fixtures/tiny_mlp_donor")
weights W = load("fixtures/tiny_mlp/model.safetensors")
behavior parity = probes("fixtures/tiny_mlp/parity.json", metric = accuracy)
behavior majority = probes("fixtures/tiny_mlp/majority.json", metric = accuracy)
"""
