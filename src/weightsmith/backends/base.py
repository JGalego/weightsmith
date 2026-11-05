"""The backend interface.

The runtime never touches tensors. It asks a backend to load, select, inspect, search, transform,
compose, run and serialise. Handles returned by a backend (weight sets, models, tensor maps) are
opaque to the runtime. A backend declares the capabilities it implements; the planner refuses to
run a program that needs a capability the backend lacks, before any operation executes.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from weightsmith.artifacts.patch import PatchManifest
from weightsmith.compiler.ir import Region
from weightsmith.evaluation.probes import ProbeSuite
from weightsmith.evidence import FinderResult, Finding
from weightsmith.models.hf import Architecture
from weightsmith.models.manifest import Manifest

Tensors = Any  # backend-specific mapping from parameter names to tensors


class Capability(StrEnum):
    LOAD_WEIGHTS = "load_weights"
    LOAD_MODEL = "load_model"
    INSPECT = "inspect"
    FIND = "find"
    EDIT = "edit"
    COMPOSE = "compose"
    REPAIR = "repair"
    FORWARD = "forward"
    PATCH = "patch"
    SAVE = "save"


class Backend(Protocol):
    name: str

    def capabilities(self) -> frozenset[Capability]: ...
    def supports(self, kind: str, name: str) -> bool:
        """Whether the backend implements a registry entry (kind: finder, edit, compose,
        strategy, metric) or a model format (kind: format; name: weightsmith or hf)."""
        ...

    # loading and identity
    def load_weights(self, path: Path) -> Tensors: ...
    def load_model(self, path: Path) -> tuple[Tensors, Architecture]: ...
    def manifest(self, tensors: Tensors) -> Manifest: ...
    def content_hash(self, tensors: Tensors) -> str: ...

    # analysis; a region is a tensor name plus an optional slice
    def stats(self, tensors: Tensors, regions: list[Region]) -> list[dict[str, Any]]: ...
    def find(
        self,
        finder: str,
        tensors: Tensors,
        names: list[str],
        params: dict[str, Any],
        binding: str,
        model_id: str,
    ) -> FinderResult: ...
    def find_units(
        self,
        finder: str,
        tensors: Tensors,
        groups: list[list[tuple[str, tuple[Region, ...]]]],
        params: dict[str, Any],
        binding: str,
        model_id: str,
    ) -> FinderResult:
        """Compare architectural units (heads, neurons) within each group of (role, regions)."""
        ...

    # transformations: each returns only the tensors it changed
    def edit(
        self, op: str, tensors: Tensors, regions: list[Region], params: dict[str, Any]
    ) -> Tensors: ...
    def compose(
        self,
        op: str,
        tensors: Tensors,
        donor: Tensors,
        regions: list[Region],
        params: dict[str, Any],
    ) -> Tensors: ...
    def repair(self, strategy: str, tensors: Tensors, finding: Finding) -> Tensors: ...

    # measurement
    def evaluate(
        self, arch: Architecture, tensors: Tensors, suite: ProbeSuite, metric: str
    ) -> float: ...
    def changed_params(self, before: Tensors, after: Tensors) -> dict[str, int]: ...
    def nonfinite(self, tensors: Tensors) -> int: ...

    # for editors (the delegate statement)
    def examples(
        self, arch: Architecture, tensors: Tensors, suite: ProbeSuite
    ) -> list[dict[str, Any]]:
        """Each probe example with the model's prediction and whether it is correct."""
        ...

    def run_inputs(
        self,
        arch: Architecture,
        tensors: Tensors,
        prompts: list[str | list[int]] | None = None,
        inputs: list[list[float]] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Outputs for inputs an editor wrote."""
        ...

    def region_values(self, tensors: Tensors, regions: list[Region]) -> list[dict[str, Any]]:
        """Values of each region as labelled rows, for reading weights as text."""
        ...

    def arrays(self, tensors: Tensors, names: list[str]) -> dict[str, Any]:
        """The named tensors as float64 NumPy arrays (for rendering them as text)."""
        ...

    def assign(self, tensors: Tensors, regions: list[Region], values: list[float]) -> Tensors:
        """Write values into regions in row-major order; return the changed tensors."""
        ...

    # artifacts
    def save_weights(self, tensors: Tensors, path: Path, metadata: dict[str, str]) -> None: ...
    def write_patch(
        self, before: Tensors, after: Tensors, directory: Path, provenance: dict[str, Any]
    ) -> PatchManifest: ...
    def apply_patch(self, tensors: Tensors, directory: Path, reverse: bool = False) -> Tensors: ...
