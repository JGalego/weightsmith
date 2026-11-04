from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from weightsmith.backends.pytorch import PyTorchBackend
from weightsmith.compiler.driver import CompileResult, compile_file
from weightsmith.compiler.effects import Effect, Grant, Policy
from weightsmith.compiler.ir import IRProgram
from weightsmith.evaluation.report import RunReport
from weightsmith.evaluation.verify import Status
from weightsmith.runtime.context import RunOptions
from weightsmith.runtime.executor import execute

from .conftest import HEADER

Compile = Callable[[str], CompileResult]
BACKEND = PyTorchBackend(num_threads=None)


def run(program: IRProgram, out: Path | None, **kw: object) -> RunReport:
    return execute(program, BACKEND, RunOptions(out_dir=out, **kw))  # type: ignore[arg-type]


def statuses(report: RunReport) -> dict[str, Status]:
    return {v.item: v.status for v in report.verdicts}


def test_repair_end_to_end(workspace: Path, tmp_path: Path) -> None:
    program = compile_file(workspace / "repair_mlp.wsm").program()
    report = run(program, tmp_path / "out")
    assert report.status == "completed", report.stop_reason
    assert statuses(report) == {
        "improve parity": Status.VERIFIED,
        "preserve majority": Status.VERIFIED,
        "require finite": Status.VERIFIED,
        "require changed_params": Status.VERIFIED,
    }
    [finding] = report.findings
    assert finding.level.value == "experimentally_supported"
    assert (tmp_path / "out" / "model" / "model.safetensors").is_file()
    assert (tmp_path / "out" / "model" / "arch.json").is_file()
    assert (tmp_path / "out" / "patch" / "manifest.json").is_file()
    assert (tmp_path / "out" / "report.json").is_file()
    assert report.exit_code() == 0


def test_original_model_is_never_modified(workspace: Path, tmp_path: Path) -> None:
    source = workspace / "fixtures" / "tiny_mlp_defective" / "model.safetensors"
    before = source.read_bytes()
    run(compile_file(workspace / "repair_mlp.wsm").program(), tmp_path / "out")
    assert source.read_bytes() == before


def test_unrelated_merge_fails_verification(workspace: Path, tmp_path: Path) -> None:
    report = run(compile_file(workspace / "merge_unrelated.wsm").program(), tmp_path / "out")
    assert report.status == "completed"
    assert set(statuses(report).values()) == {Status.FAILED}
    assert report.exit_code() == 3


def test_related_merge_preserves_behaviour(workspace: Path, tmp_path: Path) -> None:
    report = run(compile_file(workspace / "merge_mlps.wsm").program(), tmp_path / "out")
    assert set(statuses(report).values()) == {Status.VERIFIED}


def test_dry_run_measures_and_writes_nothing(workspace: Path, tmp_path: Path) -> None:
    report = run(
        compile_file(workspace / "repair_mlp.wsm").program(), tmp_path / "out", dry_run=True
    )
    assert set(statuses(report).values()) == {Status.SKIPPED}
    assert report.budgets["forward_passes"]["used"] == 0
    assert report.changes["changed_params"] == 130
    assert not (tmp_path / "out").exists()


def test_mutate_budget_stops_before_exceeding(compile_text: Compile, tmp_path: Path) -> None:
    program = compile_text(
        HEADER
        + """
        program P {
            target Base
            policy { allow mutate(max_changed_params = 6%) }
            edit {
                zero Base.params["head.weight"]
                zero Base.params["layers.1.weight"]
            }
        }"""
    ).program()
    report = run(program, tmp_path / "out")
    assert report.status == "stopped"
    assert "max_changed_params" in (report.stop_reason or "")
    assert report.changes["changed_params"] == 256, "the first edit applied, the second did not"


def test_probe_budget_makes_items_inconclusive(compile_text: Compile, tmp_path: Path) -> None:
    program = compile_text(
        HEADER
        + """
        program P {
            target Base
            policy { allow probe(max_forward_passes = 600) allow mutate }
            contract { preserve parity (tolerance = 0.1) preserve majority (tolerance = 0.1) }
            edit { scale Base.params["head.weight"] (factor = 1.0) }
            verify
            inspect
        }"""
    ).program()
    report = run(program, tmp_path / "out")
    assert statuses(report) == {
        "preserve parity": Status.VERIFIED,
        "preserve majority": Status.INCONCLUSIVE,
    }
    assert report.status == "stopped"
    assert report.budgets["forward_passes"]["used"] == 512
    assert not report.inspections, "the run stops after the budget is exhausted"


def test_refuses_output_inside_input(workspace: Path) -> None:
    program = compile_file(workspace / "repair_mlp.wsm").program()
    report = run(program, workspace / "fixtures" / "tiny_mlp_defective" / "out")
    assert report.status == "error"
    assert "inside the input directory" in (report.stop_reason or "")


def test_refuses_non_empty_output(workspace: Path, tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("x")
    report = run(compile_file(workspace / "repair_mlp.wsm").program(), tmp_path / "out")
    assert report.status == "error"
    assert (tmp_path / "out" / "keep.txt").read_text() == "x"


def test_stale_findings_are_not_repaired(compile_text: Compile, tmp_path: Path) -> None:
    program = compile_text(
        HEADER.replace('fixtures/tiny_mlp")', 'fixtures/tiny_mlp_defective")', 1)
        + """
        program P {
            target Base
            policy { allow mutate }
            find outlier_rows
            edit { scale Base.params["head.weight"] (factor = 2.0) }
            repair (strategies = [rescale_outlier_rows])
        }"""
    ).program()
    report = run(program, tmp_path / "out")
    assert report.status == "error"
    assert "earlier state" in (report.stop_reason or "")


def test_repair_with_nothing_to_repair_is_a_noop(compile_text: Compile, tmp_path: Path) -> None:
    program = compile_text(
        HEADER
        + """
        program P {
            target Base
            policy { allow mutate }
            find outlier_rows
            repair (strategies = [rescale_outlier_rows])
        }"""
    ).program()
    report = run(program, tmp_path / "out")
    assert report.status == "completed"
    assert report.events[-1].status == "noop"


def test_runtime_rechecks_policy_of_ir_json(workspace: Path, tmp_path: Path) -> None:
    program = compile_file(workspace / "repair_mlp.wsm").program()
    forbid_all = Policy(tuple((e, Grant(e in (Effect.LOAD, Effect.INSPECT))) for e in Effect))
    tampered = IRProgram.from_dict({**program.to_dict(), "policy": forbid_all.to_dict()})
    report = run(tampered, tmp_path / "out")
    assert report.status == "error"
    assert "does not allow" in (report.stop_reason or "")


def test_ir_json_runs_like_source(workspace: Path, tmp_path: Path) -> None:
    program = compile_file(workspace / "repair_mlp.wsm").program()
    from_json = IRProgram.from_json(program.to_json())
    report = run(from_json, tmp_path / "out")
    assert set(statuses(report).values()) == {Status.VERIFIED}


@pytest.mark.parametrize("name", ["inspect_transformer.wsm", "prune_mlp.wsm"])
def test_other_examples_complete(workspace: Path, tmp_path: Path, name: str) -> None:
    report = run(compile_file(workspace / name).program(), tmp_path / "out")
    assert report.exit_code() == 0, report.stop_reason
