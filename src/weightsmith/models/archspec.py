"""Architecture specifications: what turns a weight set into a runnable model.

A ``.safetensors`` file holds tensors only. To run a forward pass, Weightsmith needs an
architecture. Version 0.1 supports a small JSON format describing a sequential network; it is
enough for the synthetic fixtures and for simple MLPs. Architecture profiles for Transformers and
Hugging Face ``config.json`` files are on the roadmap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from weightsmith.errors import LoadError
from weightsmith.models.manifest import ARCH_FILE, Manifest

ARCH_FORMAT = "weightsmith.arch/v0"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LinearLayer(_Strict):
    type: Literal["linear"]
    name: str
    in_features: int = Field(gt=0)
    out_features: int = Field(gt=0)
    bias: bool = True


class LayerNormLayer(_Strict):
    type: Literal["layernorm"]
    name: str
    features: int = Field(gt=0)
    eps: float = 1e-5


class Activation(_Strict):
    type: Literal["relu", "gelu", "tanh", "sigmoid"]


Layer = Annotated[LinearLayer | LayerNormLayer | Activation, Field(discriminator="type")]


class ArchSpec(_Strict):
    """A sequential network: ``inputs`` features in, applied layer by layer."""

    format: Literal["weightsmith.arch/v0"] = ARCH_FORMAT
    kind: Literal["sequential"] = "sequential"
    inputs: int = Field(gt=0)
    layers: list[Layer]

    @property
    def outputs(self) -> int:
        width = self.inputs
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                width = layer.out_features
        return width

    def expected_parameters(self) -> dict[str, tuple[int, ...]]:
        params: dict[str, tuple[int, ...]] = {}
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                params[f"{layer.name}.weight"] = (layer.out_features, layer.in_features)
                if layer.bias:
                    params[f"{layer.name}.bias"] = (layer.out_features,)
            elif isinstance(layer, LayerNormLayer):
                params[f"{layer.name}.weight"] = (layer.features,)
                params[f"{layer.name}.bias"] = (layer.features,)
        return params

    def problems(self) -> list[str]:
        """Internal consistency problems (layer widths that do not chain)."""
        issues: list[str] = []
        width = self.inputs
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                if layer.in_features != width:
                    issues.append(
                        f"layer '{layer.name}' expects {layer.in_features} inputs "
                        f"but receives {width}"
                    )
                width = layer.out_features
            elif isinstance(layer, LayerNormLayer) and layer.features != width:
                issues.append(
                    f"layer '{layer.name}' normalises {layer.features} features "
                    f"but receives {width}"
                )
        return issues

    def mismatches(self, manifest: Manifest) -> list[str]:
        """Differences between the parameters this spec needs and those in ``manifest``."""
        expected = self.expected_parameters()
        actual = manifest.shapes()
        issues = [f"missing parameter '{name}'" for name in expected if name not in actual]
        issues += [f"unexpected parameter '{name}'" for name in actual if name not in expected]
        issues += [
            f"parameter '{name}' has shape {list(actual[name])}, expected {list(shape)}"
            for name, shape in expected.items()
            if name in actual and actual[name] != shape
        ]
        return issues


def arch_file(model_dir: Path) -> Path:
    return model_dir / ARCH_FILE


def read_archspec(model_dir: Path) -> ArchSpec:
    path = arch_file(model_dir)
    if not path.is_file():
        raise LoadError(
            f"{model_dir} has no {ARCH_FILE}; a 'model' needs an architecture "
            "(use 'weights' to load tensors only)"
        )
    try:
        return ArchSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LoadError(f"{path} is not a valid architecture spec: {exc}") from exc
