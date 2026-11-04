from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from weightsmith import __version__
from weightsmith.cli.main import app

from .conftest import EXAMPLES

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.wsm")), ids=lambda p: p.name)
def test_check_examples(workspace: Path, path: Path) -> None:
    result = runner.invoke(app, ["check", str(workspace / path.name)])
    assert result.exit_code == 0, result.output
    assert "Valid" in result.output


def test_check_reports_errors(tmp_path: Path) -> None:
    source = tmp_path / "bad.wsm"
    source.write_text('weights W = load("w.safetensors")\nprogram P { target W find loops }\n')
    result = runner.invoke(app, ["check", str(source)])
    assert result.exit_code == 1
    assert "E205" in result.output
    assert "bad.wsm:2" in result.output


def test_check_does_not_import_torch(workspace: Path) -> None:
    code = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from weightsmith.cli.main import app\n"
        f"r = CliRunner().invoke(app, ['check', {str(workspace / 'repair_mlp.wsm')!r}])\n"
        "assert r.exit_code == 0, r.output\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_compile_then_run_ir(workspace: Path, tmp_path: Path) -> None:
    ir_file = tmp_path / "repair.ir.json"
    result = runner.invoke(app, ["compile", str(workspace / "repair_mlp.wsm"), "-o", str(ir_file)])
    assert result.exit_code == 0, result.output
    assert json.loads(ir_file.read_text())["version"] == "weightsmith.ir/v0"
    result = runner.invoke(app, ["run", str(ir_file), "--out", str(tmp_path / "out"), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert {v["status"] for v in report["verdicts"]} == {"verified"}


@pytest.mark.parametrize(
    ("name", "exit_code"),
    [
        ("inspect_transformer.wsm", 0),
        ("repair_mlp.wsm", 0),
        ("merge_mlps.wsm", 0),
        ("prune_mlp.wsm", 0),
        ("merge_unrelated.wsm", 3),
    ],
)
def test_run_examples(workspace: Path, tmp_path: Path, name: str, exit_code: int) -> None:
    result = runner.invoke(app, ["run", str(workspace / name), "--out", str(tmp_path / "out")])
    assert result.exit_code == exit_code, result.output


def test_inspect_diff_apply(workspace: Path, tmp_path: Path) -> None:
    defective = workspace / "fixtures" / "tiny_mlp_defective"
    result = runner.invoke(app, ["inspect", str(defective)])
    assert result.exit_code == 0, result.output
    assert "outlier_rows" in result.output

    run = runner.invoke(
        app, ["run", str(workspace / "repair_mlp.wsm"), "--out", str(tmp_path / "out")]
    )
    assert run.exit_code == 0, run.output
    repaired = tmp_path / "out" / "model"
    diff = runner.invoke(
        app, ["diff", str(defective), str(repaired), "--out", str(tmp_path / "diffpatch")]
    )
    assert diff.exit_code == 0, diff.output
    assert "130 of 4996" in diff.output

    applied = runner.invoke(
        app,
        [
            "apply",
            str(defective),
            str(tmp_path / "out" / "patch"),
            "--out",
            str(tmp_path / "applied"),
        ],
    )
    assert applied.exit_code == 0, applied.output
    assert (tmp_path / "applied" / "arch.json").is_file()
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["target"]["final_model_id"] in applied.output

    wrong = runner.invoke(
        app,
        [
            "apply",
            str(workspace / "fixtures" / "tiny_mlp"),
            str(tmp_path / "out" / "patch"),
            "--out",
            str(tmp_path / "w"),
        ],
    )
    assert wrong.exit_code == 2


def test_run_rejects_invalid_program(tmp_path: Path) -> None:
    source = tmp_path / "bad.wsm"
    source.write_text("program P { target Nothing inspect }\n")
    result = runner.invoke(app, ["run", str(source)])
    assert result.exit_code == 1
    assert "E201" in result.output
