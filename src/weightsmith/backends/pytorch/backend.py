"""The PyTorch backend, reference implementation of :class:`~weightsmith.backends.base.Backend`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from weightsmith.artifacts.patch import PatchManifest
from weightsmith.backends.base import Capability
from weightsmith.backends.pytorch import finders, hf, ops, patch_io
from weightsmith.backends.pytorch.forward import forward
from weightsmith.backends.pytorch.weights import (
    WeightSet,
    content_hash,
    load_model,
    load_weights,
    save_weights,
)
from weightsmith.compiler.ir import Region
from weightsmith.errors import IncompatibleError, UnsupportedOperation
from weightsmith.evaluation.probes import ProbeSuite
from weightsmith.evidence import FinderResult, Finding
from weightsmith.models.hf import Architecture, HFArch, model_format, read_hf_arch
from weightsmith.models.manifest import Manifest

Tensors = dict[str, torch.Tensor]


class PyTorchBackend:
    name = "pytorch"

    def __init__(self, num_threads: int | None = 1) -> None:
        # Version 0.1 targets small models on CPU, where extra threads only add contention.
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        self._runners: dict[str, hf.HFRunner] = {}

    def capabilities(self) -> frozenset[Capability]:
        return frozenset(Capability)

    def supports(self, kind: str, name: str) -> bool:
        table: dict[str, dict[str, Any]] = {
            "finder": {**finders.FINDERS, **finders.UNIT_FINDERS},
            "edit": ops.EDIT_OPS,
            "compose": ops.COMPOSE_OPS,
            "strategy": ops.STRATEGIES,
            "metric": {"accuracy": None, "success": None},
            "format": {"weightsmith": None, **({"hf": None} if hf.available() else {})},
        }
        return name in table.get(kind, {}) or (kind == "finder" and name == "anomalies")

    # loading and identity ---------------------------------------------------------
    def load_weights(self, path: Path) -> Tensors:
        return load_weights(path).tensors

    def load_model(self, path: Path) -> tuple[Tensors, Architecture]:
        if model_format(path) == "hf":
            weights = load_weights(path)
            return weights.tensors, read_hf_arch(path, weights.manifest())
        model = load_model(path)
        return model.weights.tensors, model.arch

    def manifest(self, tensors: Tensors) -> Manifest:
        return WeightSet(tensors).manifest()

    def content_hash(self, tensors: Tensors) -> str:
        return content_hash(tensors)

    # analysis ---------------------------------------------------------------------------
    def stats(self, tensors: Tensors, regions: list[Region]) -> list[dict[str, Any]]:
        rows = []
        for name, mask in ops.region_masks(tensors, regions).items():
            t = tensors[name]
            values = t[mask].double()
            finite = values[torch.isfinite(values)]
            entry: dict[str, Any] = {
                "name": name,
                "shape": list(t.shape),
                "dtype": str(t.dtype).removeprefix("torch."),
                "numel": int(values.numel()),
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

    def find_units(
        self,
        finder: str,
        tensors: Tensors,
        groups: list[list[tuple[str, tuple[Region, ...]]]],
        params: dict[str, Any],
        binding: str,
        model_id: str,
    ) -> FinderResult:
        fn = finders.UNIT_FINDERS.get(finder)
        if fn is None:
            raise UnsupportedOperation(f"the pytorch backend has no unit finder '{finder}'")
        return fn(tensors, groups, params, binding, model_id)

    # transformations ----------------------------------------------------------------------
    def edit(
        self, op: str, tensors: Tensors, regions: list[Region], params: dict[str, Any]
    ) -> Tensors:
        fn = ops.EDIT_OPS.get(op)
        if fn is None:
            raise UnsupportedOperation(f"the pytorch backend has no edit operation '{op}'")
        changed: Tensors = {}
        for name, mask in ops.region_masks(tensors, regions).items():
            t = tensors[name]
            new = fn(t, mask, params)
            if new is not t:
                changed[name] = new
        return changed

    def compose(
        self,
        op: str,
        tensors: Tensors,
        donor: Tensors,
        regions: list[Region],
        params: dict[str, Any],
    ) -> Tensors:
        fn = ops.COMPOSE_OPS.get(op)
        if fn is None:
            raise UnsupportedOperation(f"the pytorch backend has no compose operation '{op}'")
        missing = [r.tensor for r in regions if r.tensor not in tensors]
        if missing:
            raise IncompatibleError(f"the target has no parameter '{missing[0]}'")
        return fn(tensors, donor, ops.region_masks(donor, regions), params)

    def repair(self, strategy: str, tensors: Tensors, finding: Finding) -> Tensors:
        fn = ops.STRATEGIES.get(strategy)
        if fn is None:
            raise UnsupportedOperation(f"the pytorch backend has no repair strategy '{strategy}'")
        return fn(tensors, finding)

    # measurement ----------------------------------------------------------------------------
    def evaluate(
        self, arch: Architecture, tensors: Tensors, suite: ProbeSuite, metric: str
    ) -> float:
        if suite.rollouts is not None:
            if metric != "success":
                raise IncompatibleError(f"a 'rollouts' suite is measured by success, not {metric}")
            from weightsmith.evaluation.rollouts import run_rollouts

            arrays = {k: t.detach().cpu().numpy() for k, t in tensors.items()}
            result = run_rollouts(suite.rollouts, arrays)
            return float(result["success"]) / suite.rollouts.episodes
        if metric != "accuracy":
            raise UnsupportedOperation(f"the pytorch backend has no metric '{metric}' here")
        if isinstance(arch, HFArch):
            if suite.kind != "tokens":
                raise IncompatibleError(
                    f"probe suite '{suite.name}' holds input vectors; a language model needs a "
                    "'tokens' suite"
                )
            return self.runner(arch, tensors).accuracy(tensors, suite)
        if suite.kind != "vectors":
            raise IncompatibleError(
                f"probe suite '{suite.name}' is a token suite, but the model is not a "
                "language model"
            )
        x = torch.tensor([e.input for e in suite.vector_examples], dtype=torch.float32)
        labels = torch.tensor([e.label for e in suite.vector_examples])
        logits = forward(arch, tensors, x)
        if suite.output_slice is not None:
            logits = logits[:, suite.output_slice[0] : suite.output_slice[1]]
        predictions = logits.argmax(dim=1)
        return float((predictions == labels).double().mean())

    def examples(
        self, arch: Architecture, tensors: Tensors, suite: ProbeSuite
    ) -> list[dict[str, Any]]:
        """Each probe example with its prediction, for showing an editor where a model fails."""
        if suite.rollouts is not None:
            raise IncompatibleError(f"probe suite '{suite.name}' has episodes, not examples")
        if isinstance(arch, HFArch):
            run = self.runner(arch, tensors)
            predicted, targets = run.predict(tensors, suite)
            rows = []
            for example, p, t in zip(suite.token_examples, predicted, targets, strict=True):
                rows.append(
                    {
                        "prompt": example.prompt
                        if example.prompt is not None
                        else example.prompt_ids,
                        "target": run.decode(t),
                        "predicted": run.decode(p),
                        "correct": p == t,
                    }
                )
            return rows
        x = torch.tensor([e.input for e in suite.vector_examples], dtype=torch.float32)
        logits = forward(arch, tensors, x)
        if suite.output_slice is not None:
            logits = logits[:, suite.output_slice[0] : suite.output_slice[1]]
        predictions = logits.argmax(dim=1).tolist()
        return [
            {"input": e.input, "label": e.label, "predicted": p, "correct": p == e.label}
            for e, p in zip(suite.vector_examples, predictions, strict=True)
        ]

    def run_inputs(
        self,
        arch: Architecture,
        tensors: Tensors,
        prompts: list[str | list[int]] | None = None,
        inputs: list[list[float]] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Outputs for inputs an editor wrote: next-token candidates, or output vectors."""
        if isinstance(arch, HFArch):
            if not prompts:
                raise IncompatibleError("a language model takes 'prompts'")
            run = self.runner(arch, tensors)
            return [
                {
                    "prompt": prompt,
                    "next": [
                        {"token": run.decode(token), "id": token, "p": round(p, 4)}
                        for token, p in top
                    ],
                }
                for prompt, top in zip(
                    prompts, run.top_tokens(tensors, prompts, top_k), strict=True
                )
            ]
        if not inputs:
            raise IncompatibleError("this model takes 'inputs' (lists of numbers)")
        outputs = forward(arch, tensors, torch.tensor(inputs, dtype=torch.float32))
        return [
            {"input": i, "output": [round(v, 5) for v in o]}
            for i, o in zip(inputs, outputs.tolist(), strict=True)
        ]

    def arrays(self, tensors: Tensors, names: list[str]) -> dict[str, Any]:
        return {name: tensors[name].detach().double().cpu().numpy() for name in names}

    def region_values(self, tensors: Tensors, regions: list[Region]) -> list[dict[str, Any]]:
        """The values of each region, as rows labelled with their index in the tensor."""
        blocks = []
        for region in regions:
            t = tensors[region.tensor]
            slices = ops.region_slices(t, region.index)
            sub = t[slices].detach().double()
            first = slices[0].start or 0 if slices else 0
            if sub.dim() == 0:
                rows: list[tuple[int | None, list[float]]] = [(None, [sub.item()])]
            elif sub.dim() == 1:
                rows = [(None, sub.tolist())]
            else:
                flat = sub.reshape(sub.shape[0], -1)
                rows = [(first + i, row) for i, row in enumerate(flat.tolist())]
            blocks.append(
                {
                    "region": str(region),
                    "shape": list(sub.shape),
                    "dtype": str(t.dtype).removeprefix("torch."),
                    "numel": int(sub.numel()),
                    "rows": rows,
                }
            )
        return blocks

    def assign(self, tensors: Tensors, regions: list[Region], values: list[float]) -> Tensors:
        """Write ``values`` into the regions, in order, each region in row-major order."""
        changed: Tensors = {}
        offset = 0
        union = ops.region_masks(tensors, regions)
        total = sum(int(m.sum()) for m in union.values())
        sizes = [int(ops.region_mask(tensors[r.tensor], r.index).sum()) for r in regions]
        if sum(sizes) != total:
            raise IncompatibleError("the selection's regions overlap; write them one at a time")
        if len(values) != total:
            raise IncompatibleError(
                f"the selection has {total} entries but {len(values)} values were given"
            )
        for region, size in zip(regions, sizes, strict=True):
            current = changed.get(region.tensor, tensors[region.tensor])
            mask = ops.region_mask(current, region.index)
            new = current.clone()
            new[mask] = torch.tensor(values[offset : offset + size], dtype=current.dtype)
            changed[region.tensor] = new
            offset += size
        return changed

    def runner(self, arch: HFArch, tensors: Tensors) -> hf.HFRunner:
        key = str(arch.directory.resolve())
        if key not in self._runners:
            self._runners[key] = hf.HFRunner(arch, tensors)
        return self._runners[key]

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
