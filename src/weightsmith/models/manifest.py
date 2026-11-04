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
SHARD_INDEX = "model.safetensors.index.json"
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


def weight_files(path: Path) -> list[Path]:
    """Resolve a ``load`` path to the safetensors file(s) it designates.

    A directory holds either one ``model.safetensors`` or, for sharded Hugging Face checkpoints,
    a ``model.safetensors.index.json`` naming the shards.
    """
    if path.is_dir():
        single = path / WEIGHTS_FILE
        if single.is_file():
            return [single]
        index = path / SHARD_INDEX
        if index.is_file():
            try:
                weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise LoadError(f"{index} is not a valid shard index: {exc}") from exc
            shards = [path / name for name in sorted(set(weight_map.values()))]
            missing = [s.name for s in shards if not s.is_file()]
            if missing:
                raise LoadError(f"{path} is missing shard(s): {', '.join(missing)}")
            return shards
        raise LoadError(f"{path} is a directory without {WEIGHTS_FILE} or {SHARD_INDEX}")
    if path.suffix != ".safetensors":
        raise LoadError(f"{path} is not a .safetensors file or a model directory")
    if not path.is_file():
        raise LoadError(f"{path} does not exist")
    return [path]


def weights_size(path: Path) -> int:
    """Total size in bytes of the safetensors file(s) behind a ``load`` path."""
    return sum(f.stat().st_size for f in weight_files(path))


def _read_header(file: Path) -> dict[str, object]:
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
    if not isinstance(header, dict):
        raise LoadError(f"{file} has a malformed safetensors header")
    return header


def read_manifest(path: Path) -> Manifest:
    """Read the manifest of a safetensors file, or of a model directory's weights (all shards)."""
    infos: dict[str, TensorInfo] = {}
    for file in weight_files(path):
        for name, meta in _read_header(file).items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            if name in infos:
                raise LoadError(f"tensor '{name}' appears in more than one shard of {path}")
            infos[name] = TensorInfo(name, str(meta["dtype"]), tuple(int(d) for d in meta["shape"]))
    return Manifest(tuple(sorted(infos.values(), key=lambda t: t.name)))
