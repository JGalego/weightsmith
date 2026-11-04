"""Read tensor manifests (names, dtypes, shapes) from safetensors headers without loading tensors.

A safetensors file starts with an 8-byte little-endian header length followed by a JSON header, so
the compiler can check parameter paths and shapes statically without importing PyTorch.
"""

from __future__ import annotations

import fnmatch
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from weightsmith.errors import LoadError

WEIGHTS_FILE = "model.safetensors"
ARCH_FILE = "arch.json"
_MAX_HEADER = 100 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


@dataclass(frozen=True, slots=True)
class Manifest:
    """Ordered description of the tensors in a weight set."""

    tensors: tuple[TensorInfo, ...]

    def names(self) -> list[str]:
        return [t.name for t in self.tensors]

    def get(self, name: str) -> TensorInfo | None:
        return next((t for t in self.tensors if t.name == name), None)

    def match(self, pattern: str) -> list[TensorInfo]:
        """Tensors whose names match a glob pattern (``*``, ``?``, ``[...]``)."""
        return [t for t in self.tensors if fnmatch.fnmatchcase(t.name, pattern)]

    @property
    def numel(self) -> int:
        return sum(t.numel for t in self.tensors)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {t.name: t.shape for t in self.tensors}


def weights_file(path: Path) -> Path:
    """Resolve a ``load`` path to the safetensors file it designates."""
    if path.is_dir():
        candidate = path / WEIGHTS_FILE
        if not candidate.is_file():
            raise LoadError(f"{path} is a directory without a {WEIGHTS_FILE}")
        return candidate
    if path.suffix != ".safetensors":
        raise LoadError(f"{path} is not a .safetensors file or a model directory")
    if not path.is_file():
        raise LoadError(f"{path} does not exist")
    return path


def read_manifest(path: Path) -> Manifest:
    """Read the manifest of a safetensors file (or of a model directory's weights file)."""
    file = weights_file(path)
    with file.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise LoadError(f"{file} is too short to be a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        if length > _MAX_HEADER:
            raise LoadError(f"{file} declares an implausible header size ({length} bytes)")
        try:
            header = json.loads(handle.read(length))
        except json.JSONDecodeError as exc:
            raise LoadError(f"{file} has a malformed safetensors header: {exc.msg}") from exc
    infos = [
        TensorInfo(name, str(meta["dtype"]), tuple(int(d) for d in meta["shape"]))
        for name, meta in header.items()
        if name != "__metadata__"
    ]
    infos.sort(key=lambda t: t.name)
    return Manifest(tuple(infos))
