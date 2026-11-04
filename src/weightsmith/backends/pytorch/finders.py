"""Structural finders. Each examines tensors and returns findings at evidence level 'observed'."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from weightsmith.evidence import Evidence, EvidenceLevel, FinderResult, Finding, Location

Tensors = dict[str, torch.Tensor]
FinderFn = Callable[[Tensors, list[str], dict[str, Any], str, str], FinderResult]

MIN_ROWS = 8  # robust row statistics need at least this many rows
MIN_SIBLINGS = 3

_STRUCTURAL = (
    "This is a statistical observation about the weights. It does not establish that the "
    "location causes any behaviour."
)


def _observed(method: str, description: str, **measurements: Any) -> list[Evidence]:
    return [Evidence(EvidenceLevel.OBSERVED, method, description, measurements)]


def _robust_z(values: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    values = values.double()
    median = values.median()
    mad = (values - median).abs().median()
    scale = 1.4826 * mad
    if scale <= 0:
        scale = torch.tensor(torch.finfo(torch.float64).eps) * max(1.0, abs(median.item()))
    return (values - median) / scale, median.item(), mad.item()


def nan_inf(
    tensors: Tensors, names: list[str], params: dict[str, Any], binding: str, model_id: str
) -> FinderResult:
    findings = []
    examined = 0
    skipped = []
    for name in names:
        t = tensors[name]
        if not t.is_floating_point():
            skipped.append((name, "not a floating-point tensor"))
            continue
        examined += 1
        nan = int(torch.isnan(t).sum())
        inf = int(torch.isinf(t).sum())
        if nan or inf:
            findings.append(
                Finding(
                    "nan_inf",
                    binding,
                    model_id,
                    f"{name} contains {nan} NaN and {inf} infinite values",
                    (Location(name),),
                    _observed(
                        "count_nonfinite",
                        "Counted non-finite entries.",
                        nan=nan,
                        inf=inf,
                        numel=t.numel(),
                    ),
                    ("Non-finite values propagate through any forward pass that reads them.",),
                )
            )
    return FinderResult("nan_inf", tuple(findings), examined, tuple(skipped))


def zero_tensors(
    tensors: Tensors, names: list[str], params: dict[str, Any], binding: str, model_id: str
) -> FinderResult:
    findings = []
    for name in names:
        t = tensors[name]
        if t.numel() and bool((t == 0).all()):
            findings.append(
                Finding(
                    "zero_tensors",
                    binding,
                    model_id,
                    f"{name} is entirely zero",
                    (Location(name),),
                    _observed("all_zero", "Every entry is exactly zero.", numel=t.numel()),
                    ("Some tensors are legitimately initialised to zero (for example biases).",),
                )
            )
    return FinderResult("zero_tensors", tuple(findings), len(names))


def dead_rows(
    tensors: Tensors, names: list[str], params: dict[str, Any], binding: str, model_id: str
) -> FinderResult:
    findings, skipped, examined = [], [], 0
    for name in names:
        t = tensors[name]
        if t.dim() != 2:
            skipped.append((name, f"{t.dim()}-D tensor; rows need 2-D"))
            continue
        examined += 1
        rows = (t == 0).all(dim=1).nonzero().flatten().tolist()
        if rows and len(rows) < t.shape[0]:
            findings.append(
                Finding(
                    "dead_rows",
                    binding,
                    model_id,
                    f"{name} has {len(rows)} all-zero row(s)",
                    (Location(name, tuple(rows)),),
                    _observed(
                        "zero_rows",
                        "Rows whose entries are all exactly zero.",
                        rows=rows,
                        row_count=t.shape[0],
                    ),
                    (_STRUCTURAL,),
                )
            )
    return FinderResult("dead_rows", tuple(findings), examined, tuple(skipped))


def outlier_rows(
    tensors: Tensors, names: list[str], params: dict[str, Any], binding: str, model_id: str
) -> FinderResult:
    threshold = float(params.get("threshold", 8.0))
    findings, skipped, examined = [], [], 0
    for name in names:
        t = tensors[name]
        if t.dim() != 2:
            skipped.append((name, f"{t.dim()}-D tensor; rows need 2-D"))
            continue
        if t.shape[0] < MIN_ROWS:
            skipped.append((name, f"{t.shape[0]} rows; robust statistics need at least {MIN_ROWS}"))
            continue
        if not bool(torch.isfinite(t).all()):
            skipped.append((name, "contains non-finite values"))
            continue
        examined += 1
        norms = t.double().norm(dim=1)
        z, median, mad = _robust_z(norms)
        rows = (z > threshold).nonzero().flatten().tolist()
        if rows:
            findings.append(
                Finding(
                    "outlier_rows",
                    binding,
                    model_id,
                    f"{len(rows)} row(s) of {name} have outlying norms "
                    f"(up to {norms[rows].max().item() / median:.1f}x the median)",
                    (Location(name, tuple(rows)),),
                    _observed(
                        "row_norm_robust_z",
                        "Row L2 norms compared with the tensor's median using a MAD-based z-score.",
                        rows=rows,
                        norms=[round(norms[r].item(), 6) for r in rows],
                        z_scores=[round(z[r].item(), 2) for r in rows],
                        median_norm=round(median, 6),
                        mad=round(mad, 6),
                        threshold=threshold,
                    ),
                    (_STRUCTURAL,),
                )
            )
    return FinderResult("outlier_rows", tuple(findings), examined, tuple(skipped))


def outlier_tensors(
    tensors: Tensors, names: list[str], params: dict[str, Any], binding: str, model_id: str
) -> FinderResult:
    threshold = float(params.get("threshold", 8.0))
    groups: dict[tuple[int, ...], list[str]] = {}
    for name in names:
        groups.setdefault(tuple(tensors[name].shape), []).append(name)
    findings, skipped, examined = [], [], 0
    for shape, members in groups.items():
        if len(members) < MIN_SIBLINGS:
            skipped += [
                (m, f"fewer than {MIN_SIBLINGS} tensors of shape {list(shape)}") for m in members
            ]
            continue
        examined += len(members)
        norms = torch.stack([tensors[m].double().norm() for m in members])
        z, median, _ = _robust_z(norms)
        for i, member in enumerate(members):
            if z[i] > threshold:
                findings.append(
                    Finding(
                        "outlier_tensors",
                        binding,
                        model_id,
                        f"{member} has an outlying norm among {len(members)} tensors of shape "
                        f"{list(shape)}",
                        (Location(member),),
                        _observed(
                            "tensor_norm_robust_z",
                            "Tensor norm compared with same-shape siblings.",
                            norm=norms[i].item(),
                            median_norm=median,
                            z_score=round(z[i].item(), 2),
                            threshold=threshold,
                        ),
                        (_STRUCTURAL,),
                    )
                )
    return FinderResult("outlier_tensors", tuple(findings), examined, tuple(skipped))


def duplicate_tensors(
    tensors: Tensors, names: list[str], params: dict[str, Any], binding: str, model_id: str
) -> FinderResult:
    groups: dict[tuple[Any, ...], list[str]] = {}
    skipped = []
    for name in names:
        t = tensors[name]
        flat = t.contiguous().view(-1)
        if flat.numel() > 1 and bool((flat == flat[0]).all()):
            skipped.append((name, "constant tensor (identical constants are not informative)"))
            continue
        key = (tuple(t.shape), t.dtype, flat.view(torch.uint8).numpy().tobytes())
        groups.setdefault(key, []).append(name)
    findings = []
    for members in groups.values():
        if len(members) > 1:
            findings.append(
                Finding(
                    "duplicate_tensors",
                    binding,
                    model_id,
                    f"{len(members)} tensors are bit-for-bit identical: {', '.join(members)}",
                    tuple(Location(m) for m in members),
                    _observed(
                        "bitwise_equal",
                        "Tensors with identical shape, dtype and bytes.",
                        tensors=members,
                    ),
                    ("Tied weights are identical by design in some architectures.",),
                )
            )
    return FinderResult(
        "duplicate_tensors", tuple(findings), len(names) - len(skipped), tuple(skipped)
    )


def low_effective_rank(
    tensors: Tensors, names: list[str], params: dict[str, Any], binding: str, model_id: str
) -> FinderResult:
    threshold = float(params.get("threshold", 0.1))
    findings, skipped, examined = [], [], 0
    for name in names:
        t = tensors[name]
        if t.dim() != 2 or min(t.shape) < 2:
            skipped.append((name, "not a 2-D tensor with at least 2 rows and columns"))
            continue
        if not bool(torch.isfinite(t).all()):
            skipped.append((name, "contains non-finite values"))
            continue
        examined += 1
        s = torch.linalg.svdvals(t.double())
        if s.sum() <= 0:
            continue
        p = s / s.sum()
        p = p[p > 0]
        erank = float(torch.exp(-(p * p.log()).sum()))
        max_rank = min(t.shape)
        ratio = erank / max_rank
        if ratio < threshold:
            findings.append(
                Finding(
                    "low_effective_rank",
                    binding,
                    model_id,
                    f"{name} has effective rank {erank:.2f} of a possible {max_rank}",
                    (Location(name),),
                    _observed(
                        "entropy_effective_rank",
                        "Exponential of the entropy of the normalised "
                        "singular values (Roy and Vetterli, 2007).",
                        effective_rank=erank,
                        max_rank=max_rank,
                        ratio=ratio,
                        threshold=threshold,
                    ),
                    (_STRUCTURAL,),
                )
            )
    return FinderResult("low_effective_rank", tuple(findings), examined, tuple(skipped))


FINDERS: dict[str, FinderFn] = {
    "nan_inf": nan_inf,
    "zero_tensors": zero_tensors,
    "dead_rows": dead_rows,
    "outlier_rows": outlier_rows,
    "outlier_tensors": outlier_tensors,
    "duplicate_tensors": duplicate_tensors,
    "low_effective_rank": low_effective_rank,
}
