"""Exact, invertible patches stored as safetensors plus a JSON manifest.

A parameter counts as changed when its stored bits differ, so applying a patch reproduces the
result byte for byte and its content hash matches the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from weightsmith.artifacts.patch import (
    PATCH_MANIFEST,
    PATCH_TENSORS,
    SPARSE_THRESHOLD,
    PatchEntry,
    PatchManifest,
)
from weightsmith.backends.pytorch.weights import _dtype_name, content_hash
from weightsmith.errors import IncompatibleError, LoadError

Tensors = dict[str, torch.Tensor]

_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bits(t: torch.Tensor) -> torch.Tensor:
    flat = t.detach().contiguous().reshape(-1)
    if flat.dtype == torch.bool:
        return flat.to(torch.uint8)
    return flat.view(_INT_VIEW[flat.element_size()])


def changed_mask(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Flat boolean mask of entries whose stored bits differ."""
    if before.shape != after.shape or before.dtype != after.dtype:
        raise IncompatibleError("cannot compare tensors of different shape or dtype")
    return _bits(before) != _bits(after)


def changed_counts(before: Tensors, after: Tensors) -> dict[str, int]:
    if set(before) != set(after):
        raise IncompatibleError("the two weight sets have different parameter names")
    counts = {}
    for name in before:
        if before[name] is after[name]:
            continue
        count = int(changed_mask(before[name], after[name]).sum())
        if count:
            counts[name] = count
    return counts


def write_patch(
    before: Tensors, after: Tensors, directory: Path, provenance: dict[str, Any]
) -> PatchManifest:
    counts = changed_counts(before, after)
    payload: Tensors = {}
    entries = []
    for name in sorted(counts):
        old = before[name].detach().contiguous().reshape(-1)
        new = after[name].detach().contiguous().reshape(-1)
        numel = old.numel()
        if counts[name] / numel < SPARSE_THRESHOLD:
            indices = changed_mask(before[name], after[name]).nonzero().flatten().to(torch.int64)
            payload[f"{name}::indices"] = indices
            payload[f"{name}::old"] = old[indices].clone()
            payload[f"{name}::new"] = new[indices].clone()
            encoding = "sparse"
        else:
            payload[f"{name}::old"] = old.clone()
            payload[f"{name}::new"] = new.clone()
            encoding = "dense"
        entries.append(
            PatchEntry(
                name=name,
                shape=list(before[name].shape),
                dtype=_dtype_name(before[name].dtype),
                encoding=encoding,
                changed=counts[name],
                numel=numel,
            )
        )
    manifest = PatchManifest(
        base_model_id=content_hash(before),
        result_model_id=content_hash(after),
        changed_params=sum(counts.values()),
        total_params=sum(t.numel() for t in before.values()),
        entries=entries,
        provenance=provenance,
    )
    directory.mkdir(parents=True, exist_ok=True)
    if payload:
        save_file(payload, str(directory / PATCH_TENSORS))
    (directory / PATCH_MANIFEST).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def read_manifest(directory: Path) -> PatchManifest:
    path = directory / PATCH_MANIFEST
    if not path.is_file():
        raise LoadError(f"{directory} is not a patch directory (no {PATCH_MANIFEST})")
    try:
        return PatchManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise LoadError(f"{path} is not a valid patch manifest: {exc}") from exc


def apply_patch(tensors: Tensors, directory: Path, reverse: bool = False) -> Tensors:
    """Apply a patch (or undo it with ``reverse``). Refuses if the input is not the patch's base."""
    manifest = read_manifest(directory)
    expected = manifest.result_model_id if reverse else manifest.base_model_id
    actual = content_hash(tensors)
    if actual != expected:
        side = "result" if reverse else "base"
        raise IncompatibleError(
            f"the weights are not this patch's {side}: expected {expected}, got {actual}"
        )
    payload = load_file(str(directory / PATCH_TENSORS)) if manifest.entries else {}
    source, target = ("new", "old") if reverse else ("old", "new")
    out = dict(tensors)
    for entry in manifest.entries:
        current = tensors[entry.name]
        if list(current.shape) != entry.shape:
            raise IncompatibleError(
                f"'{entry.name}' has shape {list(current.shape)}, the patch expects {entry.shape}"
            )
        flat = current.detach().contiguous().reshape(-1).clone()
        values = payload[f"{entry.name}::{target}"]
        if entry.encoding == "sparse":
            indices = payload[f"{entry.name}::indices"]
            if not torch.equal(_bits(flat[indices]), _bits(payload[f"{entry.name}::{source}"])):
                raise IncompatibleError(
                    f"'{entry.name}' does not hold the values the patch expects"
                )
            flat[indices] = values
        else:
            flat = values.clone()
        out[entry.name] = flat.reshape(current.shape)
    result_expected = manifest.base_model_id if reverse else manifest.result_model_id
    if content_hash(out) != result_expected:
        raise IncompatibleError("applying the patch did not reproduce the recorded result hash")
    return out
