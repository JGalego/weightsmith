"""The patch manifest schema (see docs/patch-format.md). Tensor payloads live in the backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PATCH_FORMAT = "weightsmith.patch/v0"
PATCH_TENSORS = "patch.safetensors"
PATCH_MANIFEST = "manifest.json"
SPARSE_THRESHOLD = 0.25  # store sparsely when fewer than this fraction of entries change


class PatchEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    shape: list[int]
    dtype: str
    encoding: Literal["sparse", "dense"]
    changed: int = Field(ge=0)
    numel: int = Field(ge=0)


class PatchManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["weightsmith.patch/v0"] = PATCH_FORMAT
    base_model_id: str
    result_model_id: str
    changed_params: int
    total_params: int
    entries: list[PatchEntry]
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return self.changed_params / self.total_params if self.total_params else 0.0
