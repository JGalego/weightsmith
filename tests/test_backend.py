from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from weightsmith import registry
from weightsmith.backends.base import Capability
from weightsmith.backends.pytorch import PyTorchBackend, finders, ops
from weightsmith.compiler.ir import IndexSpec
from weightsmith.errors import IncompatibleError, UnsupportedOperation
from weightsmith.runtime.planner import check_backend
from weightsmith.synthetic import accuracy, plant_nan, plant_outlier_rows_for_test

BACKEND = PyTorchBackend()


def test_registry_and_backend_agree() -> None:
    assert set(finders.FINDERS) | {"anomalies"} == set(registry.FINDERS)
    assert set(ops.EDIT_OPS) == set(registry.EDIT_OPS)
    assert set(ops.COMPOSE_OPS) == set(registry.COMPOSE_OPS)
    assert set(ops.STRATEGIES) == set(registry.STRATEGIES)
    for name in registry.METRICS:
        assert BACKEND.supports("metric", name)


def _defective(fixtures_dir: Path) -> tuple[dict[str, torch.Tensor], list[int]]:
    tensors = BACKEND.load_weights(fixtures_dir / "tiny_mlp_defective")
    truth = json.loads((fixtures_dir / "tiny_mlp_defective" / "fixture.json").read_text())
    return tensors, truth["planted"][0]["rows"]


def test_outlier_rows_finds_exactly_the_planted_rows(fixtures_dir: Path) -> None:
    tensors, rows = _defective(fixtures_dir)
    result = BACKEND.find("outlier_rows", tensors, sorted(tensors), {"threshold": 8.0}, "M", "id")
    [finding] = result.findings
    assert finding.locations[0].tensor == "layers.1.weight"
    assert list(finding.locations[0].rows or ()) == rows
    assert finding.level.value == "observed"
    assert ("head.weight", "4 rows; robust statistics need at least 8") in result.skipped


def test_outlier_rows_quiet_on_healthy_model(fixtures_dir: Path) -> None:
    tensors = BACKEND.load_weights(fixtures_dir / "tiny_mlp")
    result = BACKEND.find("outlier_rows", tensors, sorted(tensors), {"threshold": 8.0}, "M", "id")
    assert result.findings == ()
    assert result.examined == 2


def test_structural_finders_locate_planted_transformer_defects(fixtures_dir: Path) -> None:
    tensors = BACKEND.load_weights(fixtures_dir / "tiny_transformer")
    truth = json.loads((fixtures_dir / "tiny_transformer" / "fixture.json").read_text())
    names = sorted(tensors)
    found = {}
    for kind in ("nan_inf", "duplicate_tensors", "dead_rows", "zero_tensors"):
        found[kind] = BACKEND.find(kind, tensors, names, {}, "T", "id").findings
    assert [f.locations[0].tensor for f in found["nan_inf"]] == [truth["planted"][0]["tensor"]]
    assert [sorted(loc.tensor for loc in f.locations) for f in found["duplicate_tensors"]] == [
        truth["planted"][1]["tensors"]
    ]
    assert [(f.locations[0].tensor, f.locations[0].rows) for f in found["dead_rows"]] == [
        (truth["planted"][2]["tensor"], (truth["planted"][2]["row"],))
    ]
    assert found["zero_tensors"] == ()


def test_low_effective_rank() -> None:
    low = torch.outer(torch.arange(1.0, 33.0), torch.ones(16))
    tensors = {"low": low, "full": torch.eye(16)}
    result = BACKEND.find(
        "low_effective_rank", tensors, ["full", "low"], {"threshold": 0.1}, "T", "id"
    )
    assert [f.locations[0].tensor for f in result.findings] == ["low"]


def test_edit_ops_respect_slices() -> None:
    w = torch.arange(12.0).reshape(3, 4)
    changed = BACKEND.edit("zero", {"w": w}, ["w"], (IndexSpec("int", 1, 2), IndexSpec("all")), {})
    assert changed["w"][1].eq(0).all()
    assert torch.equal(changed["w"][0], w[0])
    assert torch.equal(w, torch.arange(12.0).reshape(3, 4)), "inputs must not be modified"


def test_prune_magnitude_prunes_exact_fraction() -> None:
    w = torch.randn(10, 10, generator=torch.Generator().manual_seed(0))
    pruned = BACKEND.edit("prune_magnitude", {"w": w}, ["w"], (), {"fraction": 0.3})["w"]
    assert int((pruned == 0).sum()) == 30
    kept = pruned[pruned != 0].abs().min()
    assert kept >= w.abs().flatten().sort().values[29]


def test_low_rank_reduces_rank() -> None:
    w = torch.randn(8, 6, generator=torch.Generator().manual_seed(1))
    approx = BACKEND.edit("low_rank", {"w": w}, ["w"], (), {"rank": 2})["w"]
    assert int(torch.linalg.matrix_rank(approx)) == 2


def test_compose_rejects_shape_mismatch() -> None:
    with pytest.raises(IncompatibleError):
        BACKEND.compose(
            "interpolate", {"w": torch.zeros(2, 2)}, {"w": torch.zeros(3, 2)}, ["w"], {"alpha": 0.5}
        )


def test_interpolate_endpoints() -> None:
    a, b = {"w": torch.zeros(2)}, {"w": torch.ones(2)}
    assert torch.equal(BACKEND.compose("interpolate", a, b, ["w"], {"alpha": 0.0})["w"], a["w"])
    assert torch.equal(BACKEND.compose("interpolate", a, b, ["w"], {"alpha": 1.0})["w"], b["w"])


def test_forward_matches_reference_training_code(fixtures_dir: Path) -> None:
    from weightsmith.evaluation.probes import load_probe_suite

    tensors, arch = BACKEND.load_model(fixtures_dir / "tiny_mlp_defective")
    reference = accuracy(tensors)
    for task in ("parity", "majority"):
        suite = load_probe_suite(fixtures_dir / "tiny_mlp" / f"{task}.json")
        assert BACKEND.evaluate(arch, tensors, suite, "accuracy") == pytest.approx(reference[task])


def test_repair_strategy_restores_planted_rows(fixtures_dir: Path) -> None:
    tensors, rows = _defective(fixtures_dir)
    result = BACKEND.find("outlier_rows", tensors, sorted(tensors), {}, "M", "id")
    changed = BACKEND.repair("rescale_outlier_rows", tensors, result.findings[0])
    assert set(changed) == {"layers.1.weight", "layers.1.bias"}
    norms = changed["layers.1.weight"].norm(dim=1)
    assert norms[rows].max() <= norms.median() * 1.0001
    assert accuracy({**tensors, **changed}) == {"parity": 1.0, "majority": 1.0}


def test_nonfinite_counting() -> None:
    tensors = plant_nan({"w": torch.zeros(3, 3)}, "w", (1, 1))
    assert BACKEND.nonfinite(tensors) == 1


def test_plant_helper_is_detectable() -> None:
    tensors = plant_outlier_rows_for_test()
    result = BACKEND.find("outlier_rows", tensors, ["layer.weight"], {}, "M", "id")
    assert list(result.findings[0].locations[0].rows or ()) == [3]


class _LimitedBackend(PyTorchBackend):
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.LOAD_MODEL, Capability.LOAD_WEIGHTS, Capability.INSPECT})


def test_planner_refuses_backends_without_capabilities(workspace: Path) -> None:
    from weightsmith.compiler.driver import compile_file

    program = compile_file(workspace / "repair_mlp.wsm").program()
    with pytest.raises(UnsupportedOperation, match="capability 'find'") as info:
        check_backend(program, _LimitedBackend(num_threads=None))
    assert "capability 'repair'" in str(info.value)
    assert "capability 'forward'" in str(info.value)
