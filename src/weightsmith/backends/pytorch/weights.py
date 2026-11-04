"""Weight sets and models held as PyTorch tensors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from weightsmith.errors import LoadError
from weightsmith.models.archspec import ArchSpec, read_archspec
from weightsmith.models.manifest import Manifest, TensorInfo, weight_files

Tensors = dict[str, torch.Tensor]


def _dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float64: "F64",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }.get(dtype, str(dtype))


def content_hash(tensors: Tensors) -> str:
    """Content address of a tensor set: sha256 over sorted names, dtypes, shapes and bytes."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous().reshape(-1)
        digest.update(name.encode())
        digest.update(b"\0" + _dtype_name(tensor.dtype).encode())
        digest.update(b"\0" + ",".join(map(str, tensors[name].shape)).encode() + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


@dataclass(slots=True)
class WeightSet:
    """An ordered mapping from parameter names to tensors. Tensors are never modified in place."""

    tensors: Tensors
    source: Path | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def manifest(self) -> Manifest:
        return Manifest(
            tuple(
                TensorInfo(name, _dtype_name(t.dtype), tuple(t.shape))
                for name, t in sorted(self.tensors.items())
            )
        )

    @property
    def numel(self) -> int:
        return sum(t.numel() for t in self.tensors.values())

    def model_id(self) -> str:
        return content_hash(self.tensors)

    def with_tensors(self, tensors: Tensors) -> WeightSet:
        return WeightSet(dict(sorted(tensors.items())), self.source, dict(self.metadata))


@dataclass(slots=True)
class Model:
    """A weight set plus the architecture needed to run it."""

    weights: WeightSet
    arch: ArchSpec


def load_weights(path: Path) -> WeightSet:
    tensors: Tensors = {}
    for file in weight_files(path):
        try:
            tensors.update(load_file(str(file), device="cpu"))
        except (SafetensorError, OSError) as exc:
            raise LoadError(f"cannot load {file}: {exc}") from exc
    return WeightSet(dict(sorted(tensors.items())), path)


def load_model(path: Path) -> Model:
    if not path.is_dir():
        raise LoadError(f"a model is a directory with model.safetensors and arch.json; got {path}")
    weights = load_weights(path)
    arch = read_archspec(path)
    issues = arch.problems() + arch.mismatches(weights.manifest())
    if issues:
        raise LoadError(f"{path}: architecture and weights disagree: " + "; ".join(issues))
    return Model(weights, arch)


def save_weights(tensors: Tensors, file: Path, metadata: dict[str, str] | None = None) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    contiguous = {name: t.detach().contiguous() for name, t in tensors.items()}
    save_file(contiguous, str(file), metadata=metadata)
