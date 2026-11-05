"""Registries of the operations a Weightsmith program can name.

This module is deliberately free of tensor libraries: the compiler uses it to validate names and
arguments without importing PyTorch. Backends provide implementations keyed by the same names, and
a test asserts the two stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from weightsmith.compiler.effects import Effect
from weightsmith.compiler.types import (
    BOOL,
    FLOAT,
    FRACTION,
    INT,
    STRING,
    BehaviorType,
    EditorType,
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
    choices: tuple[str, ...] | None = None  # for String parameters with a fixed set of values
    allow_empty: bool = False  # for list parameters: whether [] is a valid value


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
    "success": MetricSpec(
        "success",
        "Fraction of episodes that end in success, for a 'rollouts' suite.",
        higher_is_better=True,
    ),
}

REQUIREMENTS: dict[str, RequirementSpec] = {
    r.name: r
    for r in [
        RequirementSpec("finite", "No parameter of the working model is NaN or infinite."),
        RequirementSpec(
            "answer",
            "The editor's last reply has a 'field' line whose value matches the regular "
            "expression 'matches'. The pattern stays in the program; the editor never sees it.",
            (
                _p("editor", EditorType(), "The editor that was asked.", required=True),
                _p("field", STRING, "The field name, as in 'Game: ...'.", required=True),
                _p("matches", STRING, "A regular expression the answer must match.", required=True),
            ),
        ),
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


@dataclass(frozen=True, slots=True)
class ProviderSpec(Entry):
    """A provider of editor models. ``extra`` names the optional dependency it needs."""

    extra: str | None = None


PROVIDERS: dict[str, ProviderSpec] = {
    p.name: p
    for p in [
        ProviderSpec(
            "anthropic",
            "Claude models through the Anthropic API. Credentials come from the environment "
            "(ANTHROPIC_API_KEY or an 'ant auth login' profile), never from the program.",
            (
                _p("model", STRING, "Model ID.", default="claude-opus-5-5"),
                _p(
                    "effort",
                    STRING,
                    "Reasoning effort.",
                    default="high",
                    choices=("low", "medium", "high", "xhigh", "max"),
                ),
                _p("max_tokens", INT, "Output token limit per call.", default=64000, minimum=1),
            ),
            extra="anthropic",
        ),
        ProviderSpec(
            "openai",
            "OpenAI models through the Responses API, or any server that speaks Chat Completions, "
            "including local ones (base_url). Credentials come from the environment "
            "(OPENAI_API_KEY).",
            (
                _p("model", STRING, "Model ID.", required=True),
                _p("base_url", STRING, "API endpoint, for example a local server."),
                _p(
                    "api",
                    STRING,
                    "Which API to call (default: responses, or chat when base_url is set).",
                    choices=("responses", "chat"),
                ),
                _p(
                    "effort",
                    STRING,
                    "Reasoning effort, for models that support it.",
                    choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
                ),
                _p("max_tokens", INT, "Output token limit per call.", default=64000, minimum=1),
            ),
            extra="openai",
        ),
        ProviderSpec(
            "scripted",
            "A deterministic editor that replays a JSON script of tool calls. For tests and for "
            "re-applying a published sequence of edits without a model.",
            (_p("script", STRING, "Path of the script, relative to the program.", required=True),),
        ),
    ]
}


@dataclass(frozen=True, slots=True)
class ToolSpec(Entry):
    """A tool an editor can call during ``delegate``. ``effects`` are what the tool performs."""

    effects: frozenset[Effect] = field(default_factory=frozenset)


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec(
            "describe",
            "The target's tensors, architecture profile, remaining budgets and the practice "
            "behaviours.",
            effects=frozenset({Effect.INSPECT}),
        ),
        ToolSpec(
            "read_weights",
            "Values of a selection as text, up to max_read numbers per call.",
            effects=frozenset({Effect.INSPECT}),
        ),
        ToolSpec(
            "stats",
            "Per-tensor statistics of a selection.",
            effects=frozenset({Effect.INSPECT}),
        ),
        ToolSpec(
            "find",
            "Run a finder on the working copy and return its findings.",
            effects=frozenset({Effect.INSPECT}),
        ),
        ToolSpec(
            "run_probes",
            "Measure a practice behaviour on the working copy, with some failing examples. "
            "Costs one forward pass per example.",
            effects=frozenset({Effect.PROBE}),
        ),
        ToolSpec(
            "forward",
            "Run the working copy on inputs the editor writes: the most likely next tokens for "
            "a language model, the outputs for a vector model. Costs one forward pass per input.",
            effects=frozenset({Effect.PROBE}),
        ),
        ToolSpec(
            "edit",
            "Apply an edit operation (zero, scale, fill, ...) to a selection.",
            effects=frozenset({Effect.MUTATE}),
        ),
        ToolSpec(
            "set_weights",
            "Write explicit values into a selection.",
            effects=frozenset({Effect.MUTATE}),
        ),
        ToolSpec(
            "undo",
            "Revert the editor's most recent change.",
            effects=frozenset({Effect.MUTATE}),
        ),
        ToolSpec(
            "python",
            "Run Python (numpy, torch) on a copy of the weights in a separate process. Changed "
            "arrays are proposed as edits and need the mutate effect. Not a security boundary.",
            effects=frozenset({Effect.EXECUTE}),
        ),
        ToolSpec(
            "edits",
            'Write weight edits as numbers, in an edits block (protocol "blocks" only): '
            "statements such as 'actor0.W[6, 0:3] = [1, 2, 3]', '*=', '+=', '-=' and "
            "'ZERO t', on the tensors as the weight view names and shows them.",
            effects=frozenset({Effect.MUTATE}),
        ),
    ]
}

# Tools an editor can use under protocol = "blocks", by writing a fenced block of that name.
BLOCK_TOOLS: frozenset[str] = frozenset({"edits"})

DELEGATE_PARAMS: tuple[ParamSpec, ...] = (
    _p("task", STRING, "What the editor should do."),
    _p("task_file", STRING, "A text file with the task, relative to the program."),
    _p(
        "message_file",
        STRING,
        "A text file that is sent, as it is, as the first message instead of Weightsmith's "
        "briefing. It may contain {{weights}}, which is replaced by every weight as text.",
    ),
    _p(
        "tools",
        ListType(SymbolType("tool")),
        "Tools the editor may call; [] for none.",
        default=("describe", "read_weights", "stats"),
        allow_empty=True,
    ),
    _p(
        "protocol",
        STRING,
        "How the editor acts: by calling tools, or by writing fenced blocks in its replies.",
        default="tools",
        choices=("tools", "blocks"),
    ),
    _p("system", STRING, "System instructions that replace Weightsmith's own."),
    _p("system_file", STRING, "A text file with the system instructions."),
    _p(
        "practice",
        ListType(BehaviorType()),
        "Behaviours the editor may measure. Keep them apart from the ones the contract verifies.",
        default=(),
    ),
    _p("max_steps", INT, "Most editor turns.", default=20, minimum=1),
    _p("max_read", INT, "Most numbers one read returns.", default=20000, minimum=1),
    _p(
        "include_weights",
        BOOL,
        "Put every weight, as text, in the first message (for small networks).",
        default=False,
    ),
    _p(
        "weights_format",
        STRING,
        "How weights are written as text: decimals(n), or int_scale(n) for integers times one "
        "scale per tensor with n significant digits.",
    ),
)

BEHAVIOR_PARAMS: tuple[ParamSpec, ...] = (
    _p("metric", SymbolType("metric"), "Metric computed over the probe suite.", required=True),
)

SYMBOL_REGISTRIES: dict[str, dict[str, Entry]] = {
    "finder": FINDERS,  # type: ignore[dict-item]
    "metric": METRICS,  # type: ignore[dict-item]
    "strategy": STRATEGIES,  # type: ignore[dict-item]
    "tool": TOOLS,  # type: ignore[dict-item]
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
    "BLOCK_TOOLS",
    "BOOL",
    "CLAUSES",
    "COMPOSE_OPS",
    "DELEGATE_PARAMS",
    "EDIT_OPS",
    "FINDERS",
    "METRICS",
    "OUTPUTS",
    "PROVIDERS",
    "REPAIR_PARAMS",
    "REQUIREMENTS",
    "STRATEGIES",
    "STRING",
    "STRUCTURAL_FINDERS",
    "SYMBOL_REGISTRIES",
    "TOOLS",
]
