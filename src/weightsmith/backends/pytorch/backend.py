"""The PyTorch backend, reference implementation of :class:`~weightsmith.backends.base.Backend`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from weightsmith.artifacts.patch import PatchManifest
from weightsmith.backends.base import Capability
from weightsmith.backends.pytorch import finders, ops, patch_io
from weightsmith.backends.pytorch.forward import forward
from weightsmith.backends.pytorch.weights import (
    content_hash,
    load_model,
    load_weights,
    save_weights,
)
from weightsmith.compiler.ir import IndexSpec
from weightsmith.errors import UnsupportedOperation
from weightsmith.evaluation.probes import ProbeSuite
from weightsmith.evidence import FinderResult, Finding
from weightsmith.models.archspec import ArchSpec
from weightsmith.models.manifest import Manifest

Tensors = dict[str, torch.Tensor]


class PyTorchBackend:
    name = "pytorch"

    def __init__(self, num_threads: int | None = 1) -> None:
        # Version 0.1 targets small models on CPU, where extra threads only add contention.
        if num_threads is not None:
            torch.set_num_threads(num_threads)

    def capabilities(self) -> frozenset[Capability]:
        return frozenset(Capability)

    def supports(self, kind: str, name: str) -> bool:
        table: dict[str, dict[str, Any]] = {
            "finder": finders.FINDERS,
            "edit": ops.EDIT_OPS,
            "compose": ops.COMPOSE_OPS,
            "strategy": ops.STRATEGIES,
            "metric": {"accuracy": None},
        }
        return name in table.get(kind, {}) or (kind == "finder" and name == "anomalies")

    # loading and identity ---------------------------------------------------------
    def load_weights(self, path: Path) -> Tensors:
        return load_weights(path).tensors

    def load_model(self, path: Path) -> tuple[Tensors, ArchSpec]:
        model = load_model(path)
        return model.weights.tensors, model.arch

    def manifest(self, tensors: Tensors) -> Manifest:
        from weightsmith.backends.pytorch.weights import WeightSet

        return WeightSet(tensors).manifest()

    def content_hash(self, tensors: Tensors) -> str:
        return content_hash(tensors)

    # analysis ---------------------------------------------------------------------------
    def stats(
        self, tensors: Tensors, names: list[str], index: tuple[IndexSpec, ...]
    ) -> list[dict[str, Any]]:
        rows = []
        for name in names:
            t = tensors[name]
            region = t[ops.region_mask(t, index)] if index else t.reshape(-1)
            values = region.double()
            finite = values[torch.isfinite(values)]
            entry: dict[str, Any] = {
                "name": name,
                "shape": list(t.shape),
                "dtype": str(t.dtype).removeprefix("torch."),
                "numel": int(region.numel()),
                "nonfinite": int(values.numel() - finite.numel()),
                "zeros": int((values == 0).sum()),
            }
            if finite.numel():
                entry.update(
                    mean=finite.mean().item(),
                    std=finite.std().item() if finite.numel() > 1 else 0.0,
                    min=finite.min().item(),
                    max=finite.max().item(),
                    l2=finite.norm().item(),
                )
            rows.append(entry)
        return rows

    def find(
        self,
        finder: str,
        tensors: Tensors,
        names: list[str],
        params: dict[str, Any],
        binding: str,
        model_id: str,
    ) -> FinderResult:
        fn = finders.FINDERS.get(finder)
        if fn is None:
            raise UnsupportedOperation(f"the pytorch backend has no finder '{finder}'")
        return fn(tensors, names, params, binding, model_id)

    # transformations ----------------------------------------------------------------------
    def edit(
        self,
        op: str,
        tensors: Tensors,
        names: list[str],
        index: tuple[IndexSpec, ...],
        params: dict[str, Any],
    ) -> Tensors:
        fn = ops.EDIT_OPS.get(op)
        if fn is None:
            raise UnsupportedOperation(f"the pytorch backend has no edit operation '{op}'")
        changed: Tensors = {}
        for name in names:
            t = tensors[name]
            new = fn(t, ops.region_mask(t, index), params)
            if new is not t:
                changed[name] = new
        return changed

    def compose(
        self, op: str, tensors: Tensors, donor: Tensors, names: list[str], params: dict[str, Any]
    ) -> Tensors:
        fn = ops.COMPOSE_OPS.get(op)
        if fn is None:
            raise UnsupportedOperation(f"the pytorch backend has no compose operation '{op}'")
        return fn(tensors, donor, names, params)

    def repair(self, strategy: str, tensors: Tensors, finding: Finding) -> Tensors:
        fn = ops.STRATEGIES.get(strategy)
        if fn is None:
            raise UnsupportedOperation(f"the pytorch backend has no repair strategy '{strategy}'")
        return fn(tensors, finding)

    # measurement ----------------------------------------------------------------------------
    def evaluate(self, arch: ArchSpec, tensors: Tensors, suite: ProbeSuite, metric: str) -> float:
        if metric != "accuracy":
            raise UnsupportedOperation(f"the pytorch backend has no metric '{metric}'")
        x = torch.tensor([e.input for e in suite.examples], dtype=torch.float32)
        labels = torch.tensor([e.label for e in suite.examples])
        logits = forward(arch, tensors, x)
        if suite.output_slice is not None:
            logits = logits[:, suite.output_slice[0] : suite.output_slice[1]]
        predictions = logits.argmax(dim=1)
        return float((predictions == labels).double().mean())

    def changed_params(self, before: Tensors, after: Tensors) -> dict[str, int]:
        return patch_io.changed_counts(before, after)

    def nonfinite(self, tensors: Tensors) -> int:
        return sum(
            int((~torch.isfinite(t)).sum()) for t in tensors.values() if t.is_floating_point()
        )

    # artifacts --------------------------------------------------------------------------------
    def save_weights(self, tensors: Tensors, path: Path, metadata: dict[str, str]) -> None:
        save_weights(tensors, path, metadata)

    def write_patch(
        self, before: Tensors, after: Tensors, directory: Path, provenance: dict[str, Any]
    ) -> PatchManifest:
        return patch_io.write_patch(before, after, directory, provenance)

    def apply_patch(self, tensors: Tensors, directory: Path, reverse: bool = False) -> Tensors:
        return patch_io.apply_patch(tensors, directory, reverse)
