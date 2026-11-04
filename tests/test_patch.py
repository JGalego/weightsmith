from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from weightsmith.backends.pytorch import PyTorchBackend
from weightsmith.backends.pytorch.weights import content_hash
from weightsmith.errors import IncompatibleError

BACKEND = PyTorchBackend(num_threads=None)


def _pair() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(0)
    before = {
        "a": torch.randn(20, 20, generator=g),
        "b": torch.randn(10, generator=g),
        "c": torch.randn(4, 4, generator=g),
    }
    after = dict(before)
    sparse = before["a"].clone()
    sparse[3, 4] = 7.0
    after["a"] = sparse
    after["b"] = before["b"] * 2  # dense change
    return before, after


def test_round_trip_and_reverse(tmp_path: Path) -> None:
    before, after = _pair()
    manifest = BACKEND.write_patch(before, after, tmp_path / "p", {"test": True})
    assert manifest.base_model_id == content_hash(before)
    assert manifest.result_model_id == content_hash(after)
    assert {e.name: e.encoding for e in manifest.entries} == {"a": "sparse", "b": "dense"}
    assert manifest.changed_params == 1 + 10
    applied = BACKEND.apply_patch(before, tmp_path / "p")
    assert content_hash(applied) == content_hash(after)
    reverted = BACKEND.apply_patch(applied, tmp_path / "p", reverse=True)
    assert content_hash(reverted) == content_hash(before)


def test_refuses_wrong_base(tmp_path: Path) -> None:
    before, after = _pair()
    BACKEND.write_patch(before, after, tmp_path / "p", {})
    with pytest.raises(IncompatibleError, match="not this patch's base"):
        BACKEND.apply_patch(after, tmp_path / "p")


def test_changes_are_bitwise(tmp_path: Path) -> None:
    """-0.0 == 0.0 numerically, but a patch must reproduce exact bytes; NaN must compare equal."""
    before = {"w": torch.tensor([0.0, float("nan"), 1.0])}
    after = {"w": torch.tensor([-0.0, float("nan"), 1.0])}
    assert BACKEND.changed_params(before, after) == {"w": 1}
    BACKEND.write_patch(before, after, tmp_path / "p", {})
    applied = BACKEND.apply_patch(before, tmp_path / "p")
    assert content_hash(applied) == content_hash(after)


def test_empty_patch(tmp_path: Path) -> None:
    before, _ = _pair()
    manifest = BACKEND.write_patch(before, dict(before), tmp_path / "p", {})
    assert manifest.entries == []
    assert content_hash(BACKEND.apply_patch(before, tmp_path / "p")) == content_hash(before)


def test_manifest_is_documented_json(tmp_path: Path) -> None:
    before, after = _pair()
    BACKEND.write_patch(before, after, tmp_path / "p", {"program": "X"})
    data = json.loads((tmp_path / "p" / "manifest.json").read_text())
    assert data["format"] == "weightsmith.patch/v0"
    assert data["provenance"] == {"program": "X"}
