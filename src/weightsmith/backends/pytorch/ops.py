"""Edit operations, composition operations and repair strategies on PyTorch tensors.

Every function is pure: it returns new tensors for the names it changes and never writes into its
inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from weightsmith.compiler.ir import IndexSpec, Region
from weightsmith.errors import IncompatibleError
from weightsmith.evidence import Finding

Tensors = dict[str, torch.Tensor]


def region_mask(tensor: torch.Tensor, index: tuple[IndexSpec, ...]) -> torch.Tensor:
    """A boolean mask selecting the region described by ``index`` (everything if empty)."""
    mask = torch.zeros(tensor.shape, dtype=torch.bool)
    mask[region_slices(tensor, index)] = True
    return mask


def region_slices(tensor: torch.Tensor, index: tuple[IndexSpec, ...]) -> tuple[slice, ...]:
    """The region described by ``index`` as slices; every dimension is kept."""
    if len(index) > tensor.dim():
        raise IncompatibleError(f"slice has {len(index)} dimensions but tensor has {tensor.dim()}")
    slices: list[slice] = []
    for dim, spec in enumerate(index):
        size = tensor.shape[dim]
        if spec.kind == "all":
            slices.append(slice(None))
            continue
        if not 0 <= spec.start < spec.stop <= size:
            raise IncompatibleError(
                f"index {spec} is out of range for dimension {dim} (size {size})"
            )
        slices.append(slice(spec.start, spec.stop))
    return tuple(slices)


def region_masks(tensors: Tensors, regions: list[Region]) -> dict[str, torch.Tensor]:
    """One mask per tensor: the union of the regions that fall in it, in first-seen order."""
    masks: dict[str, torch.Tensor] = {}
    for region in regions:
        if region.tensor not in tensors:
            raise IncompatibleError(f"there is no parameter '{region.tensor}'")
        mask = region_mask(tensors[region.tensor], region.index)
        masks[region.tensor] = masks[region.tensor] | mask if region.tensor in masks else mask
    return masks


EditFn = Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], torch.Tensor]


def _zero(t: torch.Tensor, mask: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    return torch.where(mask, torch.zeros((), dtype=t.dtype), t)


def _scale(t: torch.Tensor, mask: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    return torch.where(mask, t * float(params["factor"]), t)


def _fill(t: torch.Tensor, mask: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    return torch.where(mask, torch.tensor(float(params["value"]), dtype=t.dtype), t)


def _prune_magnitude(t: torch.Tensor, mask: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    region = t[mask]
    k = int(float(params["fraction"]) * region.numel())
    if k == 0:
        return t
    order = region.abs().argsort(stable=True)[:k]
    pruned = region.clone()
    pruned[order] = 0
    out = t.clone()
    out[mask] = pruned
    return out


def _low_rank(t: torch.Tensor, mask: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    if t.dim() != 2:
        raise IncompatibleError(f"low_rank needs a 2-D tensor, got shape {list(t.shape)}")
    rank = int(params["rank"])
    if rank >= min(t.shape):
        return t
    u, s, vh = torch.linalg.svd(t.double(), full_matrices=False)
    approx = (u[:, :rank] * s[:rank]) @ vh[:rank]
    return approx.to(t.dtype)


def _replace_nonfinite(t: torch.Tensor, mask: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    value = torch.tensor(float(params.get("value", 0.0)), dtype=t.dtype)
    return torch.where(mask & ~torch.isfinite(t), value, t)


EDIT_OPS: dict[str, EditFn] = {
    "zero": _zero,
    "scale": _scale,
    "fill": _fill,
    "prune_magnitude": _prune_magnitude,
    "low_rank": _low_rank,
    "replace_nonfinite": _replace_nonfinite,
}

Masks = dict[str, torch.Tensor]
ComposeFn = Callable[[Tensors, Tensors, Masks, dict[str, Any]], Tensors]


def _check_compatible(tensors: Tensors, donor: Tensors, masks: Masks) -> None:
    for name in masks:
        if name not in tensors:
            raise IncompatibleError(f"the target has no parameter '{name}'")
        if tensors[name].shape != donor[name].shape:
            raise IncompatibleError(
                f"'{name}' has shape {list(donor[name].shape)} in the donor "
                f"but {list(tensors[name].shape)} in the target"
            )


def _interpolate(tensors: Tensors, donor: Tensors, masks: Masks, params: dict[str, Any]) -> Tensors:
    _check_compatible(tensors, donor, masks)
    alpha = float(params["alpha"])
    out: Tensors = {}
    for name, mask in masks.items():
        mine = tensors[name]
        mixed = ((1 - alpha) * mine.double() + alpha * donor[name].double()).to(mine.dtype)
        out[name] = torch.where(mask, mixed, mine)
    return out


def _transplant(tensors: Tensors, donor: Tensors, masks: Masks, params: dict[str, Any]) -> Tensors:
    _check_compatible(tensors, donor, masks)
    return {
        name: torch.where(mask, donor[name].to(tensors[name].dtype), tensors[name])
        for name, mask in masks.items()
    }


COMPOSE_OPS: dict[str, ComposeFn] = {"interpolate": _interpolate, "transplant": _transplant}

StrategyFn = Callable[[Tensors, Finding], Tensors]


def _rescale_outlier_rows(tensors: Tensors, finding: Finding) -> Tensors:
    changed: Tensors = {}
    for location in finding.locations:
        if location.rows is None:
            continue
        weight = tensors[location.tensor]
        norms = weight.double().norm(dim=1)
        median = norms.median()
        factors = torch.ones(weight.shape[0], dtype=torch.float64)
        for row in location.rows:
            if norms[row] > 0:
                factors[row] = median / norms[row]
        changed[location.tensor] = (weight.double() * factors[:, None]).to(weight.dtype)
        if location.tensor.endswith(".weight"):
            bias_name = location.tensor[: -len(".weight")] + ".bias"
            bias = tensors.get(bias_name)
            if bias is not None and bias.dim() == 1 and bias.shape[0] == weight.shape[0]:
                changed[bias_name] = (bias.double() * factors).to(bias.dtype)
    return changed


def _repair_nonfinite(tensors: Tensors, finding: Finding) -> Tensors:
    out: Tensors = {}
    for location in finding.locations:
        t = tensors[location.tensor]
        out[location.tensor] = torch.where(torch.isfinite(t), t, torch.zeros((), dtype=t.dtype))
    return out


def _rescale_outlier_units(tensors: Tensors, finding: Finding) -> Tensors:
    """Scale each outlying part of an outlying unit down to its peers' median norm.

    A part (for example a neuron's up-projection row) is outlying when its norm is at least
    ``min_ratio`` times the median norm of the same part across the compared units.
    """
    measured = finding.evidence[0].measurements if finding.evidence else {}
    parts = measured.get("parts", [])
    min_ratio = float(measured.get("min_ratio", 2.0))
    changed: Tensors = {}
    for location in finding.locations:
        for region, part in zip(location.regions, parts, strict=False):
            current = changed.get(region.tensor, tensors[region.tensor])
            mask = region_mask(current, region.index)
            norm = current[mask].double().norm()
            median = float(part["median_norm"])
            if median > 0 and norm >= min_ratio * median:
                factor = median / norm
                scaled = (current.double() * factor).to(current.dtype)
                changed[region.tensor] = torch.where(mask, scaled, current)
    return changed


STRATEGIES: dict[str, StrategyFn] = {
    "rescale_outlier_rows": _rescale_outlier_rows,
    "rescale_outlier_units": _rescale_outlier_units,
    "replace_nonfinite": _repair_nonfinite,
}
