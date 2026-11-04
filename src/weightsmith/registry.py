"""Registries of the operations a Weightsmith program can name.

This module is deliberately free of tensor libraries: the compiler uses it to validate names and
arguments without importing PyTorch. Backends provide implementations keyed by the same names, and
a test asserts the two stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from weightsmith.compiler.types import (
    BOOL,
    FLOAT,
    FRACTION,
    INT,
    STRING,
    ListType,
    SymbolType,
    Type,
)


@dataclass(frozen=True, slots=True)
class ParamSpec:
    name: str
    type: Type
    doc: str
    required: bool = False
    default: object = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True, slots=True)
class Entry:
    """Common shape of every registry entry."""

    name: str
    doc: str
    params: tuple[ParamSpec, ...] = ()

    def param(self, name: str) -> ParamSpec | None:
        return next((p for p in self.params if p.name == name), None)


@dataclass(frozen=True, slots=True)
class FinderSpec(Entry):
    """A finder. ``unit`` finders compare architectural units (heads, neurons) and need the
    architecture profile of the weights they examine."""

    unit: str | None = None


@dataclass(frozen=True, slots=True)
class OpSpec(Entry):
    """An operation inside an ``edit`` or ``compose`` block."""

    block: str = "edit"  # "edit" | "compose"
    allows_slicing: bool = True
    requires_2d: bool = False


@dataclass(frozen=True, slots=True)
class StrategySpec(Entry):
    """A repair strategy: turns findings of the kinds it ``handles`` into a transformation."""

    handles: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class MetricSpec(Entry):
    higher_is_better: bool = True


@dataclass(frozen=True, slots=True)
class RequirementSpec(Entry):
    """A structural contract item, written ``require <name> (...)``."""


def _p(name: str, type_: Type, doc: str, **kw: object) -> ParamSpec:
    return ParamSpec(name, type_, doc, **kw)  # type: ignore[arg-type]


# Shared by the norm-outlier finders. A robust z-score alone is unstable when the compared items
# are few or nearly equal (the MAD is then tiny), so an outlier must also be min_ratio times the
# median norm.
_OUTLIER_PARAMS = (
    _p("threshold", FLOAT, "Robust z-score threshold.", default=8.0, minimum=0.0),
    _p("min_ratio", FLOAT, "Smallest norm, as a multiple of the median.", default=2.0, minimum=1.0),
)


FINDERS: dict[str, FinderSpec] = {
    f.name: f
    for f in [
        FinderSpec("nan_inf", "Tensors containing NaN or infinite values."),
        FinderSpec("zero_tensors", "Tensors whose entries are all exactly zero."),
        FinderSpec("dead_rows", "Rows of 2-D tensors whose entries are all exactly zero."),
        FinderSpec(
            "outlier_rows",
            "Rows of 2-D tensors whose L2 norm is a robust outlier within the tensor: a "
            "median/MAD z-score above the threshold and at least min_ratio times the median.",
            _OUTLIER_PARAMS,
        ),
        FinderSpec(
            "outlier_tensors",
            "Tensors whose norm is a robust outlier among at least three same-shape tensors.",
            _OUTLIER_PARAMS,
        ),
        FinderSpec(
            "duplicate_tensors",
            "Groups of non-constant tensors with identical shape, dtype and contents.",
        ),
        FinderSpec(
            "low_effective_rank",
            "2-D tensors whose entropy-based effective rank is below a fraction of min(shape).",
            (
                _p(
                    "threshold",
                    FRACTION,
                    "Fraction of the maximum possible rank.",
                    default=0.1,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
        ),
        FinderSpec(
            "outlier_heads",
            "Attention heads whose parameter norm is a robust outlier among comparable heads "
            "(the same layer when it has at least 8 heads, otherwise every examined layer). "
            "Needs an architecture profile.",
            _OUTLIER_PARAMS,
            unit="heads",
        ),
        FinderSpec(
            "outlier_neurons",
            "MLP neurons whose parameter norm is a robust outlier among the neurons of their "
            "layer. Needs an architecture profile.",
            _OUTLIER_PARAMS,
            unit="neurons",
        ),
    ]
}
STRUCTURAL_FINDERS = tuple(sorted(n for n, f in FINDERS.items() if f.unit is None))
FINDERS["anomalies"] = FinderSpec(
    "anomalies",
    "Runs several finders with their default parameters. The default runs every finder that "
    "does not need an architecture profile.",
    (
        _p(
            "kinds",
            ListType(SymbolType("finder")),
            "Finders to run.",
            default=STRUCTURAL_FINDERS,
        ),
    ),
)

EDIT_OPS: dict[str, OpSpec] = {
    o.name: o
    for o in [
        OpSpec("zero", "Set the selected entries to zero."),
        OpSpec(
            "scale",
            "Multiply the selected entries by a factor.",
            (_p("factor", FLOAT, "Multiplier.", required=True),),
        ),
        OpSpec(
            "fill",
            "Set the selected entries to a constant.",
            (_p("value", FLOAT, "Value to write.", required=True),),
        ),
        OpSpec(
            "prune_magnitude",
            "Zero the given fraction of selected entries with the smallest magnitude, per tensor.",
            (
                _p(
                    "fraction",
                    FRACTION,
                    "Fraction of entries to zero.",
                    required=True,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
        ),
        OpSpec(
            "low_rank",
            "Replace each selected 2-D tensor with its best rank-k approximation (truncated SVD).",
            (_p("rank", INT, "Target rank.", required=True, minimum=1),),
            allows_slicing=False,
            requires_2d=True,
        ),
        OpSpec(
            "replace_nonfinite",
            "Replace NaN and infinite entries with a constant.",
            (_p("value", FLOAT, "Replacement value.", default=0.0),),
        ),
    ]
}

COMPOSE_OPS: dict[str, OpSpec] = {
    o.name: o
    for o in [
        OpSpec(
            "interpolate",
            "Linear interpolation towards a donor: w <- (1 - alpha) * w + alpha * donor, over the "
            "operand's tensors or regions. Shapes must match.",
            (
                _p(
                    "alpha",
                    FLOAT,
                    "Interpolation weight of the donor.",
                    required=True,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
            block="compose",
        ),
        OpSpec(
            "transplant",
            "Copy the selected donor tensors or regions (for example one attention head) into "
            "the target. Names and shapes must match.",
            block="compose",
        ),
    ]
}

STRATEGIES: dict[str, StrategySpec] = {
    s.name: s
    for s in [
        StrategySpec(
            "rescale_outlier_rows",
            "Rescale each outlier row to the median row norm of its tensor. If a sibling "
            "'<prefix>.bias' has one entry per row, that entry is scaled by the same factor.",
            handles=frozenset({"outlier_rows"}),
        ),
        StrategySpec(
            "rescale_outlier_units",
            "For each outlying head or neuron, rescale each outlying part of it (for example "
            "the up-projection row or the down-projection column of a neuron) to the median "
            "norm of that part across the compared units. A part is outlying when its norm is "
            "at least the finder's min_ratio times that median; other parts are left alone.",
            handles=frozenset({"outlier_heads", "outlier_neurons"}),
        ),
        StrategySpec(
            "replace_nonfinite",
            "Replace NaN and infinite entries with zero.",
            handles=frozenset({"nan_inf"}),
        ),
    ]
}

METRICS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec(
        "accuracy",
        "Fraction of examples predicted correctly: for a 'vectors' suite, the argmax over the "
        "output slice equals the label; for a 'tokens' suite, the most likely next token "
        "(among the candidates, if given) is the target.",
        higher_is_better=True,
    ),
}

REQUIREMENTS: dict[str, RequirementSpec] = {
    r.name: r
    for r in [
        RequirementSpec("finite", "No parameter of the working model is NaN or infinite."),
        RequirementSpec(
            "changed_params",
            "The fraction of parameters that differ from the original is at most 'max'.",
            (
                _p(
                    "max",
                    FRACTION,
                    "Largest permitted fraction.",
                    required=True,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
        ),
    ]
}

CLAUSES: dict[str, Entry] = {
    "preserve": Entry(
        "preserve",
        "The behaviour's metric may get worse by at most 'tolerance'.",
        (
            _p("tolerance", FLOAT, "Largest permitted degradation.", required=True, minimum=0.0),
            _p(
                "min_n",
                INT,
                "Smallest probe suite that can support a verdict.",
                default=20,
                minimum=1,
            ),
        ),
    ),
    "improve": Entry(
        "improve",
        "The behaviour's metric must improve by at least 'min_delta'.",
        (
            _p("min_delta", FLOAT, "Smallest improvement that counts.", required=True, minimum=0.0),
            _p(
                "min_n",
                INT,
                "Smallest probe suite that can support a verdict.",
                default=20,
                minimum=1,
            ),
        ),
    ),
}

BEHAVIOR_PARAMS: tuple[ParamSpec, ...] = (
    _p("metric", SymbolType("metric"), "Metric computed over the probe suite.", required=True),
)

SYMBOL_REGISTRIES: dict[str, dict[str, Entry]] = {
    "finder": FINDERS,  # type: ignore[dict-item]
    "metric": METRICS,  # type: ignore[dict-item]
    "strategy": STRATEGIES,  # type: ignore[dict-item]
}

REPAIR_PARAMS: tuple[ParamSpec, ...] = (
    _p(
        "strategies",
        ListType(SymbolType("strategy")),
        "Strategies the planner may use, in order of preference.",
        required=True,
    ),
)

OUTPUTS: dict[str, str] = {
    "model": "Save the working model (weights, plus the architecture spec for a model).",
    "patch": "Save the patch from the original to the working model.",
    "findings": "Print findings.",
    "evidence": "Print the evidence attached to each finding.",
    "evaluation": "Print verification verdicts.",
    "summary": "Print parameter statistics from inspect statements.",
}

__all__ = [
    "BEHAVIOR_PARAMS",
    "BOOL",
    "CLAUSES",
    "COMPOSE_OPS",
    "EDIT_OPS",
    "FINDERS",
    "METRICS",
    "OUTPUTS",
    "REPAIR_PARAMS",
    "REQUIREMENTS",
    "STRATEGIES",
    "STRING",
    "STRUCTURAL_FINDERS",
    "SYMBOL_REGISTRIES",
]
