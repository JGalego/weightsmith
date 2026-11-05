"""Name resolution and type checking.

Besides ordinary typing, the checker uses whatever it can read without a tensor library: tensor
manifests from safetensors headers, architecture specs and probe suites. When those files exist,
parameter globs, slice bounds, shape compatibility for composition and probe/architecture widths
are checked statically. When they do not exist yet, the checks are deferred to run time and a
warning says so.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from weightsmith import registry as reg
from weightsmith.compiler.effects import Policy, effective_policy, policy_from_block
from weightsmith.compiler.ir import IndexSpec, Region
from weightsmith.compiler.types import (
    BOOL,
    FLOAT,
    FRACTION,
    INT,
    STRING,
    BehaviorType,
    EditorType,
    ListType,
    ParamsType,
    Prim,
    SelectionType,
    SymbolType,
    Type,
    WeightsType,
    is_assignable,
)
from weightsmith.diagnostics import DiagnosticSink, Span, Stage
from weightsmith.errors import LoadError
from weightsmith.evaluation.probes import ProbeSuite, load_probe_suite
from weightsmith.language import nodes as n
from weightsmith.models.archspec import ArchSpec, read_archspec
from weightsmith.models.hf import HFArch, model_format, read_hf_arch
from weightsmith.models.manifest import Manifest, read_manifest
from weightsmith.models.profiles import (
    COMPONENTS,
    ROLE_ROOTS,
    UNITS,
    Profile,
    RoleError,
    RoleRef,
)

# --------------------------------------------------------------------------- results


@dataclass(slots=True)
class Binding:
    name: str
    kind: str  # "model" | "weights" | "behavior" | "selection" | "editor"
    type: Type
    span: Span
    path: Path | None = None
    manifest: Manifest | None = None
    arch: ArchSpec | None = None
    hf: HFArch | None = None
    profile: Profile | None = None
    profile_error: str | None = None
    suite: ProbeSuite | None = None
    params: dict[str, Any] = field(default_factory=dict)
    expr: n.Expr | None = None  # for selections


@dataclass(frozen=True, slots=True)
class SelectionInfo:
    """What a selection expression denotes, as far as is known statically."""

    binding: str
    pattern: str | None  # None means every parameter
    index: tuple[n.Index, ...]
    names: tuple[str, ...] | None  # resolved parameter names when the manifest is known
    role: RoleRef | None = None
    regions: tuple[Region, ...] | None = None  # resolved role regions when the profile is known

    @property
    def sliced(self) -> bool:
        """Whether the selection covers parts of tensors rather than whole tensors."""
        return bool(self.index) or (self.role is not None and self.role.is_unit)


@dataclass(slots=True)
class CheckedProgram:
    ast: n.Program
    target: Binding
    policy: Policy
    contract: n.ContractBlock | None
    locals: dict[str, Binding] = field(default_factory=dict)
    kwargs: dict[int, dict[str, Any]] = field(default_factory=dict)  # id(node) -> values


@dataclass(slots=True)
class CheckedFile:
    ast: n.SourceFile
    base_dir: Path
    bindings: dict[str, Binding]
    file_policy: Policy | None
    programs: dict[str, CheckedProgram]
    roles: dict[int, tuple[str, RoleRef]] = field(default_factory=dict)  # id(expr) -> role


# --------------------------------------------------------------------------- helpers


def _suggest(name: str, options: list[str]) -> list[str]:
    close = difflib.get_close_matches(name, options, n=3, cutoff=0.6)
    return [f"did you mean {', '.join(repr(c) for c in close)}?"] if close else []


def _available(kind: str, options: list[str] | dict[str, Any]) -> str:
    return f"available {kind}: {', '.join(sorted(options))}"


def _literal_type(expr: n.Expr) -> Type | None:
    match expr:
        case n.IntLit():
            return INT
        case n.FloatLit():
            return FLOAT
        case n.PercentLit():
            return FRACTION
        case n.StrLit():
            return STRING
        case n.BoolLit():
            return BOOL
        case _:
            return None


class Checker:
    def __init__(self, file: n.SourceFile, base_dir: Path, sink: DiagnosticSink) -> None:
        self.file = file
        self.base_dir = base_dir
        self.sink = sink
        self.bindings: dict[str, Binding] = {}
        self._selections: dict[int, SelectionInfo | None] = {}
        self._roles: dict[int, _RoleResult] = {}

    # diagnostics shortcuts --------------------------------------------------------
    def name_error(self, code: str, msg: str, span: Span | None, *notes: str) -> None:
        self.sink.error(Stage.NAMES, code, msg, span, *notes)

    def type_error(self, code: str, msg: str, span: Span | None, *notes: str) -> None:
        self.sink.error(Stage.TYPES, code, msg, span, *notes)

    def warn(self, code: str, msg: str, span: Span | None, *notes: str) -> None:
        self.sink.warning(Stage.TYPES, code, msg, span, *notes)

    # top level -----------------------------------------------------------------------
    def check(self) -> CheckedFile:
        decls = self.file.decls
        for decl in decls:
            if isinstance(decl, n.LoadDecl | n.BehaviorDecl | n.SelectionDecl | n.EditorDecl):
                self._declare(decl)
        for decl in decls:
            match decl:
                case n.LoadDecl():
                    self._check_load(decl)
                case n.BehaviorDecl():
                    self._check_behavior(decl)
                case n.EditorDecl():
                    self._check_editor(decl)
        for decl in decls:
            if isinstance(decl, n.SelectionDecl):
                binding = self.bindings[decl.name.name]
                if binding.span != decl.name.span:
                    continue  # duplicate, already reported
                typ = self.type_of(decl.expr, self.bindings)
                if typ is not None and not isinstance(typ, SelectionType):
                    self.type_error(
                        "E302",
                        f"'{decl.name.name}' must be a selection, not {typ}",
                        decl.expr.span,
                        'write M.params["pattern"]',
                    )
                elif typ is not None:
                    binding.type = typ

        file_policies = [d for d in decls if isinstance(d, n.PolicyBlock)]
        for extra in file_policies[1:]:
            self.type_error("E303", "a file can have at most one policy block", extra.span)
        file_policy = (
            policy_from_block(file_policies[0], "file", self.sink) if file_policies else None
        )

        programs: dict[str, CheckedProgram] = {}
        for program in self.file.programs:
            if program.name.name in programs:
                self.name_error(
                    "E202", f"program '{program.name.name}' is defined twice", program.name.span
                )
                continue
            if program.name.name in self.bindings:
                self.name_error(
                    "E202", f"'{program.name.name}' is already declared", program.name.span
                )
            checked = self._check_program(program, file_policy)
            if checked is not None:
                programs[program.name.name] = checked
        if not programs and not self.sink.has_errors():
            self.warn("W301", "this file declares no program", None)
        roles = {k: (r.binding, r.ref) for k, r in self._roles.items() if isinstance(r, _Role)}
        return CheckedFile(self.file, self.base_dir, self.bindings, file_policy, programs, roles)

    def _declare(self, decl: n.LoadDecl | n.BehaviorDecl | n.SelectionDecl | n.EditorDecl) -> None:
        name = decl.name
        if name.name in self.bindings:
            self.name_error(
                "E202",
                f"'{name.name}' is declared twice",
                name.span,
                f"first declared at {self.bindings[name.name].span}",
            )
            return
        match decl:
            case n.LoadDecl():
                typ: Type = WeightsType(name.name, is_model=decl.kind == "model")
                kind = decl.kind
            case n.BehaviorDecl():
                typ, kind = BehaviorType(), "behavior"
            case n.SelectionDecl():
                typ, kind = SelectionType("?", sliced=False), "selection"
            case n.EditorDecl():
                typ, kind = EditorType(), "editor"
        self.bindings[name.name] = Binding(
            name.name, kind, typ, name.span, expr=getattr(decl, "expr", None)
        )

    def _resolve_path(self, lit: n.StrLit) -> Path:
        path = Path(lit.value)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    def _check_load(self, decl: n.LoadDecl) -> None:
        binding = self.bindings.get(decl.name.name)
        if binding is None or binding.span != decl.name.span:
            return
        path = self._resolve_path(decl.path)
        binding.path = path
        if not path.exists():
            self.warn(
                "W201",
                f"'{decl.path.value}' does not exist yet",
                decl.path.span,
                f"resolved to {path}",
                "parameter names and shapes will be checked at run time",
            )
            return
        fmt = model_format(path)
        if decl.kind == "model" and fmt is None:
            self.type_error(
                "E304",
                "a model is loaded from a directory containing its weights and either arch.json "
                "(a Weightsmith architecture) or config.json (a Hugging Face model)",
                decl.path.span,
                "use 'weights' to load tensors without an architecture",
            )
            return
        try:
            binding.manifest = read_manifest(path)
        except LoadError as exc:
            self.type_error("E305", f"cannot read weights: {exc}", decl.path.span)
            return
        if fmt == "hf":
            try:
                hf = read_hf_arch(path, binding.manifest)
            except LoadError as exc:
                self.type_error("E306", str(exc), decl.path.span)
                return
            binding.profile, binding.profile_error = hf.profile, hf.profile_error
            if decl.kind == "model":
                binding.hf = hf
            return
        binding.profile_error = "it has no Hugging Face config.json"
        if decl.kind == "model":
            try:
                arch = read_archspec(path)
            except LoadError as exc:
                self.type_error("E306", str(exc), decl.path.span)
                return
            issues = arch.problems() + arch.mismatches(binding.manifest)
            for issue in issues:
                self.type_error(
                    "E307", f"architecture and weights disagree: {issue}", decl.path.span
                )
            if not issues:
                binding.arch = arch

    def _check_behavior(self, decl: n.BehaviorDecl) -> None:
        binding = self.bindings.get(decl.name.name)
        if binding is None or binding.span != decl.name.span:
            return
        binding.params = self.check_kwargs(decl.kwargs, reg.BEHAVIOR_PARAMS, "probes", decl.span)
        path = self._resolve_path(decl.path)
        binding.path = path
        if not path.exists():
            self.warn(
                "W202",
                f"probe suite '{decl.path.value}' does not exist yet",
                decl.path.span,
                f"resolved to {path}",
            )
            return
        try:
            binding.suite = load_probe_suite(path)
        except LoadError as exc:
            self.type_error("E308", str(exc), decl.path.span)

    def _check_editor(self, decl: n.EditorDecl) -> None:
        binding = self.bindings.get(decl.name.name)
        if binding is None or binding.span != decl.name.span:
            return
        name = decl.provider.value
        spec = reg.PROVIDERS.get(name)
        if spec is None:
            self.name_error(
                "E210",
                f"unknown provider '{name}'",
                decl.provider.span,
                *_suggest(name, list(reg.PROVIDERS)),
                _available("providers", reg.PROVIDERS),
            )
            return
        for kwarg in decl.kwargs:
            if "key" in kwarg.name.name or "token" in kwarg.name.name:
                self.type_error(
                    "E364",
                    "credentials do not belong in a program",
                    kwarg.span,
                    "providers read them from the environment, for example ANTHROPIC_API_KEY",
                )
                return
        values = self.check_kwargs(decl.kwargs, spec.params, f"provider {name}", decl.span)
        if name == "scripted" and isinstance(values.get("script"), str):
            script = self._resolve_path(n.StrLit(values["script"], span=decl.span))
            values["script"] = str(script)
            if not script.is_file():
                self.warn("W204", f"script '{script}' does not exist yet", decl.provider.span)
        binding.params = {"provider": name, **values}

    # expressions --------------------------------------------------------------------------
    def lookup(self, name: str, span: Span, env: dict[str, Binding]) -> Binding | None:
        binding = env.get(name)
        if binding is None:
            self.name_error("E201", f"unknown name '{name}'", span, *_suggest(name, list(env)))
        return binding

    def type_of(self, expr: n.Expr, env: dict[str, Binding]) -> Type | None:
        if isinstance(expr, n.Attr | n.Subscript):
            role = self.role_of(expr, env)
            if isinstance(role, _Role):
                return SelectionType(role.binding, sliced=role.ref.is_unit)
            if role is _INVALID:
                return None
        match expr:
            case n.Name():
                binding = self.lookup(expr.ident, expr.span, env)
                return binding.type if binding else None
            case n.Attr():
                obj = self.type_of(expr.obj, env)
                if obj is None:
                    return None
                if isinstance(obj, WeightsType) and expr.attr.name == "params" and obj.binding:
                    return ParamsType(obj.binding)
                notes = ["weight sets have the attribute params"]
                if isinstance(obj, WeightsType):
                    notes.append(
                        "with an architecture profile they also have the roles "
                        + ", ".join(ROLE_ROOTS)
                    )
                self.type_error(
                    "E310", f"{obj} has no attribute '{expr.attr.name}'", expr.attr.span, *notes
                )
                return None
            case n.Subscript():
                return self._type_subscript(expr, env)
            case n.ListLit():
                self.type_error("E311", "a list is not valid here", expr.span)
                return None
            case _:
                return _literal_type(expr)

    def _type_subscript(self, expr: n.Subscript, env: dict[str, Binding]) -> Type | None:
        obj = self.type_of(expr.obj, env)
        if obj is None:
            return None
        if isinstance(obj, ParamsType):
            if len(expr.indices) != 1 or not isinstance(expr.indices[0], n.StrLit):
                self.type_error(
                    "E312",
                    "parameters are selected with one glob pattern string",
                    expr.span,
                    'for example M.params["layers.*.weight"]',
                )
                return None
            info = self.selection(expr, env)
            return SelectionType(obj.binding, sliced=False) if info else None
        if isinstance(obj, SelectionType):
            if obj.sliced:
                self.type_error("E313", "this selection is already sliced", expr.span)
                return None
            if any(isinstance(i, n.StrLit) for i in expr.indices):
                self.type_error(
                    "E314", "slice indices must be integers, ranges (a..b) or ':'", expr.span
                )
                return None
            info = self.selection(expr, env)
            return SelectionType(obj.binding, sliced=True) if info else None
        self.type_error(
            "E315",
            f"{obj} cannot be indexed",
            expr.span,
            "index M.params with a pattern, then optionally slice the selection",
        )
        return None

    # architectural roles ------------------------------------------------------------------
    def role_of(self, expr: n.Attr | n.Subscript, env: dict[str, Binding]) -> _RoleResult:
        """Whether ``expr`` is a whole role path such as ``M.layers[1].attn.heads[2]``.

        Returns the role, ``_NOT_ROLE`` when the expression is something else (``M.params[...]``,
        or a slice applied to a role), or ``_INVALID`` after reporting an error. Memoised so each
        problem is reported once.
        """
        key = id(expr)
        if key not in self._roles:
            self._roles[key] = self._role_of(expr, env)
        return self._roles[key]

    def _role_of(self, expr: n.Attr | n.Subscript, env: dict[str, Binding]) -> _RoleResult:
        steps: list[n.Ident | n.Subscript] = []
        node: n.Expr = expr
        while isinstance(node, n.Attr | n.Subscript):
            steps.append(node.attr if isinstance(node, n.Attr) else node)
            node = node.obj
        steps.reverse()
        if not isinstance(node, n.Name) or not steps or not isinstance(steps[0], n.Ident):
            return _NOT_ROLE
        if steps[0].name not in ROLE_ROOTS:
            return _NOT_ROLE
        binding = self.lookup(node.ident, node.span, env)
        if binding is None:
            return _INVALID
        if binding.kind not in ("model", "weights"):
            self.type_error(
                "E310", f"{binding.type} has no attribute '{steps[0].name}'", steps[0].span
            )
            return _INVALID
        ref = self._parse_role(steps, expr)
        if ref is None or isinstance(ref, _NotRole):
            return _NOT_ROLE if isinstance(ref, _NotRole) else _INVALID
        if binding.manifest is not None and binding.profile is None:
            self.type_error(
                "E316",
                f"'{binding.name}' has no architecture profile, so it has no role '{ref.root}'",
                steps[0].span,
                f"reason: {binding.profile_error}",
                f'select parameters by name instead: {binding.name}.params["pattern"]',
            )
            return _INVALID
        regions: tuple[Region, ...] | None = None
        if binding.profile is not None:
            try:
                regions = binding.profile.resolve(ref)
            except RoleError as exc:
                self.type_error("E510", f"{binding.name}.{ref}: {exc}", expr.span)
                return _INVALID
            if not regions:
                self.type_error("E511", f"{binding.name}.{ref} selects no parameters", expr.span)
                return _INVALID
            if ref.root == "unembed" and binding.profile.tied:
                self.warn(
                    "W305",
                    f"'{binding.name}' ties its output projection to its token embeddings; "
                    "unembed selects the embeddings, so editing it also changes the input side",
                    expr.span,
                )
        return _Role(binding.name, ref, regions)

    def _parse_role(
        self, steps: list[n.Ident | n.Subscript], expr: n.Expr
    ) -> RoleRef | _NotRole | None:
        """Consume role steps. A subscript after a complete role is a slice, not part of it."""
        root = steps[0]
        assert isinstance(root, n.Ident)
        rest = steps[1:]
        if root.name != "layers":
            if rest and isinstance(rest[0], n.Ident):
                self.type_error("E317", f"'{root.name}' has no part '{rest[0].name}'", rest[0].span)
                return None
            return RoleRef(root.name) if not rest else _NotRole()
        layers = self._role_index(rest, "layers", root.span)
        if layers is None:
            return None
        rest = rest[1:]
        if not rest:
            return RoleRef("layers", layers)
        if isinstance(rest[0], n.Subscript):
            return _NotRole()
        component = rest[0]
        if component.name not in COMPONENTS:
            self.type_error(
                "E317",
                f"a layer has no part '{component.name}'",
                component.span,
                f"parts: {', '.join(COMPONENTS)}",
            )
            return None
        rest = rest[1:]
        if not rest:
            return RoleRef("layers", layers, component.name)
        if isinstance(rest[0], n.Subscript):
            return _NotRole()
        unit = rest[0]
        units = UNITS.get(component.name, {})
        if unit.name not in units:
            options = f"units: {', '.join(units)}" if units else "it has no indexed units"
            self.type_error(
                "E317", f"'{component.name}' has no part '{unit.name}'", unit.span, options
            )
            return None
        index = self._role_index(rest[1:], unit.name, unit.span)
        if index is None:
            return None
        if len(rest) > 2:
            follow = rest[2]
            if isinstance(follow, n.Ident):
                self.type_error("E317", f"'{unit.name}' has no part '{follow.name}'", follow.span)
            else:
                self.type_error(
                    "E313", "a head or neuron selection is already a slice", follow.span
                )
            return None
        return RoleRef("layers", layers, component.name, unit.name, index)

    def _role_index(
        self, rest: list[n.Ident | n.Subscript], what: str, span: Span
    ) -> IndexSpec | None:
        example = f"{what}[3], {what}[0..4] or {what}[:]"
        if not rest or not isinstance(rest[0], n.Subscript):
            self.type_error("E318", f"'{what}' needs an index", span, f"for example {example}")
            return None
        sub = rest[0]
        if len(sub.indices) != 1:
            self.type_error("E318", f"'{what}' takes one index", sub.span, f"for example {example}")
            return None
        match sub.indices[0]:
            case n.IntLit() as i:
                return IndexSpec("int", i.value, i.value + 1)
            case n.RangeIndex() as r:
                if r.start >= r.stop:
                    self.type_error("E318", f"empty range {r.start}..{r.stop}", r.span)
                    return None
                return IndexSpec("range", r.start, r.stop)
            case n.FullIndex():
                return IndexSpec("all")
            case other:
                self.type_error(
                    "E318", f"'{what}' is indexed by integers", other.span, f"for example {example}"
                )
                return None

    def selection(self, expr: n.Expr, env: dict[str, Binding]) -> SelectionInfo | None:
        """Describe a selection-typed (or weights-typed) expression and check it statically.

        Results are memoised per node so each problem is reported once.
        """
        key = id(expr)
        if key not in self._selections:
            self._selections[key] = self._selection(expr, env)
        return self._selections[key]

    def _selection(self, expr: n.Expr, env: dict[str, Binding]) -> SelectionInfo | None:
        if isinstance(expr, n.Attr | n.Subscript):
            role = self.role_of(expr, env)
            if isinstance(role, _Role):
                names = (
                    tuple(dict.fromkeys(r.tensor for r in role.regions))
                    if role.regions is not None
                    else None
                )
                return SelectionInfo(role.binding, None, (), names, role.ref, role.regions)
            if role is _INVALID:
                return None
        match expr:
            case n.Name():
                binding = env.get(expr.ident)
                if binding is None:
                    return None
                if binding.kind in ("model", "weights"):
                    names = tuple(binding.manifest.names()) if binding.manifest else None
                    return SelectionInfo(binding.name, None, (), names)
                if binding.kind == "selection" and binding.expr is not None:
                    return self.selection(binding.expr, env)
                return None
            case n.Subscript(obj=n.Attr(obj=n.Name() as root, attr=attr)) if attr.name == "params":
                pattern_lit = expr.indices[0]
                assert isinstance(pattern_lit, n.StrLit)
                binding = env.get(root.ident)
                if binding is None:
                    return None
                names: tuple[str, ...] | None = None
                if binding.manifest is not None:
                    matched = binding.manifest.match(pattern_lit.value)
                    if not matched:
                        self.type_error(
                            "E501",
                            f'pattern "{pattern_lit.value}" matches no '
                            f"parameter of '{binding.name}'",
                            pattern_lit.span,
                            *_suggest(pattern_lit.value, binding.manifest.names()),
                            f"parameters: {', '.join(binding.manifest.names()[:12])}",
                        )
                        return None
                    names = tuple(t.name for t in matched)
                return SelectionInfo(binding.name, pattern_lit.value, (), names)
            case n.Subscript():
                base = self.selection(expr.obj, env)
                if base is None:
                    return None
                if base.sliced:
                    return None  # reported by the type check of this expression
                self._check_slice(expr, base, env)
                return SelectionInfo(
                    base.binding, base.pattern, expr.indices, base.names, base.role, base.regions
                )
            case _:
                return None

    def _check_slice(self, expr: n.Subscript, base: SelectionInfo, env: dict[str, Binding]) -> None:
        binding = env.get(base.binding)
        if binding is None or binding.manifest is None or base.names is None:
            return
        for name in base.names:
            info = binding.manifest.get(name)
            if info is None:
                continue
            if len(expr.indices) > len(info.shape):
                self.type_error(
                    "E502",
                    f"'{name}' has {len(info.shape)} dimension(s) but the slice "
                    f"has {len(expr.indices)}",
                    expr.span,
                )
                return
            for dim, index in enumerate(expr.indices):
                size = info.shape[dim]
                if isinstance(index, n.IntLit) and not 0 <= index.value < size:
                    self.type_error(
                        "E503",
                        f"index {index.value} is out of range for dimension "
                        f"{dim} of '{name}' (size {size})",
                        index.span,
                    )
                    return
                if isinstance(index, n.RangeIndex) and not 0 <= index.start < index.stop <= size:
                    self.type_error(
                        "E503",
                        f"range {index.start}..{index.stop} is out of range for "
                        f"dimension {dim} of '{name}' (size {size})",
                        index.span,
                    )
                    return

    # keyword arguments ---------------------------------------------------------------------
    def check_kwargs(
        self,
        kwargs: tuple[n.Kwarg, ...],
        specs: tuple[reg.ParamSpec, ...],
        where: str,
        span: Span,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        given: set[str] = set()
        by_name = {s.name: s for s in specs}
        for kwarg in kwargs:
            name = kwarg.name.name
            if name in given:
                self.type_error("E320", f"argument '{name}' is given twice", kwarg.name.span)
                continue
            given.add(name)
            spec = by_name.get(name)
            if spec is None:
                options = ", ".join(by_name) or "none"
                self.type_error(
                    "E321",
                    f"'{where}' has no argument '{name}'",
                    kwarg.name.span,
                    f"arguments: {options}",
                    *_suggest(name, list(by_name)),
                )
                continue
            value = self._kwarg_value(kwarg.value, spec, where)
            if value is not None:
                values[name] = value
        for spec in specs:
            if spec.name in given:
                continue
            if spec.required:
                self.type_error(
                    "E322",
                    f"'{where}' requires argument '{spec.name}' ({spec.type})",
                    span,
                    spec.doc,
                )
            elif spec.default is not None:
                values[spec.name] = spec.default
        return values

    def _kwarg_value(self, expr: n.Expr, spec: reg.ParamSpec, where: str) -> Any:
        expected = spec.type
        if isinstance(expected, ListType) and isinstance(expected.elem, BehaviorType):
            if not isinstance(expr, n.ListLit):
                self.type_error("E323", f"'{spec.name}' expects a list of behaviours", expr.span)
                return None
            names: list[str] = []
            for item in expr.items:
                if not isinstance(item, n.Name):
                    self.type_error("E328", "expected a behaviour name", item.span)
                    return None
                binding = self.lookup(item.ident, item.span, self.bindings)
                if binding is None:
                    return None
                if binding.kind != "behavior":
                    self.type_error(
                        "E351", f"'{item.ident}' is {binding.type}, not a behaviour", item.span
                    )
                    return None
                names.append(item.ident)
            return tuple(names)
        if isinstance(expected, SymbolType):
            return self._symbol(expr, expected.registry)
        if isinstance(expected, ListType) and isinstance(expected.elem, SymbolType):
            if not isinstance(expr, n.ListLit) or not expr.items:
                self.type_error(
                    "E323",
                    f"'{spec.name}' expects a non-empty list of {expected.elem.registry} names",
                    expr.span,
                )
                return None
            items = [self._symbol(item, expected.elem.registry) for item in expr.items]
            if any(item is None for item in items):
                return None
            if where == "anomalies" and "anomalies" in items:
                self.type_error("E324", "'anomalies' cannot run itself", expr.span)
                return None
            return tuple(items)
        actual = _literal_type(expr)
        if actual is None:
            self.type_error("E325", f"'{spec.name}' expects a {expected} literal", expr.span)
            return None
        value: Any = expr.value  # type: ignore[union-attr]
        ok = is_assignable(actual, expected)
        if expected == FRACTION and isinstance(expr, n.FloatLit | n.IntLit) and 0 <= value <= 1:
            ok = True  # a plain number between 0 and 1 is accepted as a fraction
        if not ok:
            hint = (
                ["write fractions as percentages, for example 5%"] if expected == FRACTION else []
            )
            self.type_error(
                "E326", f"'{spec.name}' expects {expected}, got {actual}", expr.span, *hint
            )
            return None
        if isinstance(value, int | float) and not isinstance(value, bool):
            if spec.minimum is not None and value < spec.minimum:
                self.type_error(
                    "E327", f"'{spec.name}' must be at least {spec.minimum:g}", expr.span
                )
                return None
            if spec.maximum is not None and value > spec.maximum:
                self.type_error(
                    "E327", f"'{spec.name}' must be at most {spec.maximum:g}", expr.span
                )
                return None
            if expected in (FLOAT, FRACTION):
                value = float(value)
        if spec.choices is not None and value not in spec.choices:
            self.type_error(
                "E329",
                f"'{spec.name}' must be one of {', '.join(spec.choices)}",
                expr.span,
            )
            return None
        return value

    def _symbol(self, expr: n.Expr, registry: str) -> str | None:
        options = reg.SYMBOL_REGISTRIES[registry]
        if not isinstance(expr, n.Name):
            self.type_error(
                "E328",
                f"expected a {registry} name",
                expr.span,
                _available(f"{registry}s", options),
            )
            return None
        if expr.ident not in options:
            self.name_error(
                "E203",
                f"unknown {registry} '{expr.ident}'",
                expr.span,
                *_suggest(expr.ident, list(options)),
                _available(f"{registry}s", options),
            )
            return None
        return expr.ident

    # programs ------------------------------------------------------------------------------
    def _check_program(
        self, program: n.Program, file_policy: Policy | None
    ) -> CheckedProgram | None:
        target = self.bindings.get(program.target.name)
        if target is None:
            self.name_error(
                "E201",
                f"unknown target '{program.target.name}'",
                program.target.span,
                *_suggest(program.target.name, list(self.bindings)),
            )
            return None
        if target.kind not in ("model", "weights"):
            self.type_error(
                "E330",
                f"the target must be a model or weight set, not {target.type}",
                program.target.span,
            )
            return None
        state = _ProgramState(target=target)
        env = dict(self.bindings)
        checked = CheckedProgram(program, target, file_policy or Policy((), "default"), None)

        policy_blocks = [s for s in program.body if isinstance(s, n.PolicyBlock)]
        for extra in policy_blocks[1:]:
            self.type_error("E303", "a program can have at most one policy block", extra.span)
        contracts = [s for s in program.body if isinstance(s, n.ContractBlock)]
        for extra in contracts[1:]:
            self.type_error("E331", "a program can have at most one contract block", extra.span)
        if contracts:
            checked.contract = contracts[0]
            self._check_contract(contracts[0], target, checked)
        outputs = [s for s in program.body if isinstance(s, n.OutputBlock)]
        if (outputs and program.body[-1] is not outputs[-1]) or len(outputs) > 1:
            self.type_error(
                "E332",
                "a program has at most one output block, and it comes last",
                outputs[-1].span,
            )

        for stmt in program.body:
            self._check_stmt(stmt, env, state, checked)

        program_policy = (
            policy_from_block(policy_blocks[0], "program", self.sink) if policy_blocks else None
        )
        span = policy_blocks[0].span if policy_blocks else program.span
        checked.policy = effective_policy(file_policy, program_policy, span, self.sink)
        return checked

    def _check_stmt(
        self, stmt: n.Stmt, env: dict[str, Binding], state: _ProgramState, checked: CheckedProgram
    ) -> None:
        target = state.target
        match stmt:
            case n.LetStmt():
                if stmt.name.name in env:
                    self.name_error(
                        "E204",
                        f"'{stmt.name.name}' is already bound",
                        stmt.name.span,
                        "Weightsmith does not allow shadowing",
                    )
                    return
                typ = self.type_of(stmt.expr, env)
                if typ is not None and not isinstance(typ, SelectionType):
                    self.type_error("E333", f"let binds selections; got {typ}", stmt.expr.span)
                    return
                if typ is not None:
                    binding = Binding(
                        stmt.name.name, "selection", typ, stmt.name.span, expr=stmt.expr
                    )
                    env[stmt.name.name] = binding
                    checked.locals[stmt.name.name] = binding
            case n.InspectStmt():
                state.inspected = True
                if stmt.subject is not None:
                    typ = self.type_of(stmt.subject, env)
                    if typ is not None and not isinstance(typ, SelectionType | WeightsType):
                        self.type_error("E334", f"cannot inspect {typ}", stmt.subject.span)
            case n.FindStmt():
                self._check_find(stmt, env, state, checked)
            case n.EditBlock():
                for op in stmt.ops:
                    self._check_edit_op(op, env, state, checked)
            case n.ComposeBlock():
                for op in stmt.ops:
                    self._check_compose_op(op, env, state, checked)
            case n.DelegateStmt():
                self._check_delegate(stmt, state, checked)
            case n.RepairStmt():
                values = self.check_kwargs(stmt.kwargs, reg.REPAIR_PARAMS, "repair", stmt.span)
                checked.kwargs[id(stmt)] = values
                state.mutated = True
                if not state.target_finders:
                    self.type_error(
                        "E340",
                        "repair needs findings about the target; add a find "
                        f"statement on '{target.name}' before it",
                        stmt.span,
                    )
                    return
                strategies = values.get("strategies", ())
                handled = {k for s in strategies for k in reg.STRATEGIES[s].handles}
                if strategies and not handled & state.target_finders:
                    self.type_error(
                        "E341",
                        "none of the repair strategies handles the findings this program looks for",
                        stmt.span,
                        f"findings requested: {', '.join(sorted(state.target_finders))}",
                        f"strategies handle: {', '.join(sorted(handled))}",
                    )
            case n.VerifyStmt():
                state.verified = True
                if checked.contract is None:
                    self.type_error(
                        "E342", "verify needs a contract block in the same program", stmt.span
                    )
            case n.OutputBlock():
                self._check_output(stmt, state)
            case n.ContractBlock() | n.PolicyBlock():
                pass  # checked at program level

    def _check_delegate(
        self, stmt: n.DelegateStmt, state: _ProgramState, checked: CheckedProgram
    ) -> None:
        editor = self.lookup(stmt.editor.name, stmt.editor.span, self.bindings)
        values = self.check_kwargs(stmt.kwargs, reg.DELEGATE_PARAMS, "delegate", stmt.span)
        checked.kwargs[id(stmt)] = values
        if editor is not None and editor.kind != "editor":
            self.type_error(
                "E360",
                f"delegate takes an editor, but '{editor.name}' is {editor.type}",
                stmt.editor.span,
                'declare one with: editor E = provider("anthropic")',
            )
        if ("task" in values) == ("task_file" in values):
            self.type_error("E361", "give exactly one of 'task' and 'task_file'", stmt.span)
        elif "task_file" in values:
            path = self._resolve_path(n.StrLit(values.pop("task_file"), span=stmt.span))
            try:
                values["task"] = path.read_text(encoding="utf-8")
            except OSError as exc:
                self.type_error("E363", f"cannot read the task file: {exc}", stmt.span)
        tools = set(values.get("tools", ()))
        target = state.target
        runs = tools & {"run_probes", "forward"} or values.get("practice")
        if runs and not target.type.is_model:  # type: ignore[union-attr]
            self.type_error(
                "E362",
                f"running the network needs a model, but '{target.name}' was loaded with 'weights'",
                stmt.span,
            )
        practice = values.get("practice", ())
        if practice and "run_probes" not in tools:
            self.warn("W307", "practice behaviours are only measured through run_probes", stmt.span)
        for name in practice:
            behavior = self.bindings[name]
            if target.type.is_model:  # type: ignore[union-attr]
                self._check_suite_fits(behavior, target, stmt.span, {"min_n": 1})
        verified = {
            item.subject.name
            for item in (checked.contract.items if checked.contract else ())
            if item.clause in ("improve", "preserve")
        }
        for name in sorted(verified & set(practice)):
            self.warn(
                "W306",
                f"the editor practises on '{name}', which the contract also verifies; the "
                "verdict will not be independent of the editor's search",
                stmt.span,
                "verify on a held-out probe suite instead",
            )
        if tools & {"edit", "set_weights", "undo", "python"}:
            state.mutated = True

    def _check_find(
        self,
        stmt: n.FindStmt,
        env: dict[str, Binding],
        state: _ProgramState,
        checked: CheckedProgram,
    ) -> None:
        state.found = True
        spec = reg.FINDERS.get(stmt.finder.name)
        if spec is None:
            self.name_error(
                "E205",
                f"unknown finder '{stmt.finder.name}'",
                stmt.finder.span,
                *_suggest(stmt.finder.name, list(reg.FINDERS)),
                _available("finders", reg.FINDERS),
                "behavioural finders such as circuits or features are not in 0.1",
            )
            return
        root = state.target.name
        if stmt.scope is not None:
            typ = self.type_of(stmt.scope, env)
            if typ is None:
                return
            if isinstance(typ, SelectionType) and typ.sliced:
                self.type_error(
                    "E343",
                    "finders search whole tensors or layers, not a slice, head or neuron",
                    stmt.scope.span,
                )
                return
            if not isinstance(typ, SelectionType | WeightsType):
                self.type_error("E344", f"cannot search {typ}", stmt.scope.span)
                return
            root = typ.binding or root
        values = self.check_kwargs(stmt.kwargs, spec.params, spec.name, stmt.span)
        checked.kwargs[id(stmt)] = values
        kinds = values.get("kinds", ()) if spec.name == "anomalies" else (spec.name,)
        unit_finders = [k for k in kinds if k in reg.FINDERS and reg.FINDERS[k].unit]
        if unit_finders:
            self._check_unit_scope(stmt, unit_finders, env.get(root), env)
        if root == state.target.name:
            kinds = values.get("kinds", ()) if spec.name == "anomalies" else (spec.name,)
            state.target_finders.update(kinds)

    def _check_unit_scope(
        self, stmt: n.FindStmt, finders: list[str], binding: Binding | None, env: dict[str, Binding]
    ) -> None:
        """Unit finders compare heads or neurons: they need a profile and whole layers."""
        what = ", ".join(finders)
        if binding is not None and binding.manifest is not None and binding.profile is None:
            self.type_error(
                "E319",
                f"{what} needs an architecture profile, and '{binding.name}' has none",
                stmt.finder.span,
                f"reason: {binding.profile_error}",
            )
            return
        if stmt.scope is None or isinstance(stmt.scope, n.Name):
            if isinstance(stmt.scope, n.Name) and env.get(stmt.scope.ident) is not None:
                scoped = env[stmt.scope.ident]
                if scoped.kind == "selection":
                    info = self.selection(stmt.scope, env)
                    if info is None or info.role is None or info.role.root != "layers":
                        self.type_error(
                            "E319",
                            f"{what} searches whole layers: use the model or a role such as "
                            "M.layers[0..4], M.layers[1].attn or M.layers[1].mlp",
                            stmt.scope.span,
                        )
            return
        info = self.selection(stmt.scope, env)
        if info is None:
            return
        if info.role is None or info.role.root != "layers" or info.index:
            self.type_error(
                "E319",
                f"{what} searches whole layers: use the model or a role such as "
                "M.layers[0..4], M.layers[1].attn or M.layers[1].mlp",
                stmt.scope.span,
            )
            return
        for finder in finders:
            needed = "attn" if reg.FINDERS[finder].unit in ("heads", "kv_heads") else "mlp"
            if info.role.component not in (None, needed):
                self.type_error(
                    "E319",
                    f"{finder} looks in '{needed}', but the scope is '{info.role.component}'",
                    stmt.scope.span,
                )

    def _operand(self, op: n.OpStmt, env: dict[str, Binding]) -> tuple[Type, SelectionInfo] | None:
        typ = self.type_of(op.operand, env)
        if typ is None:
            return None
        if not isinstance(typ, SelectionType | WeightsType):
            self.type_error(
                "E345", f"'{op.op.name}' operates on parameters, not {typ}", op.operand.span
            )
            return None
        info = self.selection(op.operand, env)
        return (typ, info) if info else None

    def _check_edit_op(
        self, op: n.OpStmt, env: dict[str, Binding], state: _ProgramState, checked: CheckedProgram
    ) -> None:
        state.mutated = True
        spec = reg.EDIT_OPS.get(op.op.name)
        if spec is None:
            notes = ["it is a compose operation"] if op.op.name in reg.COMPOSE_OPS else []
            self.name_error(
                "E206",
                f"unknown edit operation '{op.op.name}'",
                op.op.span,
                *notes,
                *_suggest(op.op.name, list(reg.EDIT_OPS)),
                _available("edit operations", reg.EDIT_OPS),
            )
            return
        checked.kwargs[id(op)] = self.check_kwargs(op.kwargs, spec.params, spec.name, op.span)
        operand = self._operand(op, env)
        if operand is None:
            return
        _, info = operand
        if info.binding != state.target.name:
            self.type_error(
                "E346",
                f"edit changes the target '{state.target.name}', but the operand "
                f"belongs to '{info.binding}'",
                op.operand.span,
                "use compose to bring parameters in from another model",
            )
            return
        if info.sliced and not spec.allows_slicing:
            self.type_error(
                "E347",
                f"'{spec.name}' works on whole tensors; it cannot take a slice, a head or a neuron",
                op.operand.span,
            )
        if spec.requires_2d:
            self._require_2d(op, info, state.target)

    def _require_2d(self, op: n.OpStmt, info: SelectionInfo, binding: Binding) -> None:
        if binding.manifest is None or info.names is None:
            return
        for name in info.names:
            tensor = binding.manifest.get(name)
            if tensor is not None and len(tensor.shape) != 2:
                self.type_error(
                    "E504",
                    f"'{op.op.name}' needs 2-D tensors; '{name}' has shape {list(tensor.shape)}",
                    op.operand.span,
                )
                return

    def _check_compose_op(
        self, op: n.OpStmt, env: dict[str, Binding], state: _ProgramState, checked: CheckedProgram
    ) -> None:
        state.mutated = True
        spec = reg.COMPOSE_OPS.get(op.op.name)
        if spec is None:
            notes = ["it is an edit operation"] if op.op.name in reg.EDIT_OPS else []
            self.name_error(
                "E207",
                f"unknown compose operation '{op.op.name}'",
                op.op.span,
                *notes,
                *_suggest(op.op.name, list(reg.COMPOSE_OPS)),
                _available("compose operations", reg.COMPOSE_OPS),
            )
            return
        checked.kwargs[id(op)] = self.check_kwargs(op.kwargs, spec.params, spec.name, op.span)
        operand = self._operand(op, env)
        if operand is None:
            return
        typ, info = operand
        target = state.target
        if info.binding == target.name:
            self.type_error(
                "E348",
                "compose combines the target with another weight set; the "
                f"operand belongs to the target '{target.name}'",
                op.operand.span,
            )
            return
        if info.sliced and not spec.allows_slicing:
            self.type_error(
                "E347", f"'{spec.name}' works on whole tensors; remove the slice", op.operand.span
            )
            return
        donor = env[info.binding]
        if target.manifest is None or donor.manifest is None or info.names is None:
            return
        if spec.name == "interpolate" and isinstance(typ, WeightsType):
            if donor.manifest.shapes() != target.manifest.shapes():
                self.type_error(
                    "E505",
                    f"cannot interpolate: '{donor.name}' and '{target.name}' "
                    "have different parameter names or shapes",
                    op.operand.span,
                    "select the compatible parameters explicitly",
                )
            return
        for name in info.names:
            mine, theirs = target.manifest.get(name), donor.manifest.get(name)
            if mine is None:
                self.type_error(
                    "E506", f"'{target.name}' has no parameter '{name}'", op.operand.span
                )
                return
            if theirs is not None and mine.shape != theirs.shape:
                self.type_error(
                    "E507",
                    f"'{name}' has shape {list(theirs.shape)} in "
                    f"'{donor.name}' but {list(mine.shape)} in '{target.name}'",
                    op.operand.span,
                )
                return

    def _check_contract(
        self, block: n.ContractBlock, target: Binding, checked: CheckedProgram
    ) -> None:
        subjects: set[tuple[str, str]] = set()
        for item in block.items:
            key = (item.clause, item.subject.name)
            if key in subjects:
                self.type_error(
                    "E350", f"'{item.clause} {item.subject.name}' appears twice", item.span
                )
            subjects.add(key)
            if item.clause == "require":
                spec = reg.REQUIREMENTS.get(item.subject.name)
                if spec is None:
                    self.name_error(
                        "E208",
                        f"unknown requirement '{item.subject.name}'",
                        item.subject.span,
                        _available("requirements", reg.REQUIREMENTS),
                    )
                    continue
                checked.kwargs[id(item)] = self.check_kwargs(
                    item.kwargs, spec.params, spec.name, item.span
                )
                continue
            clause = reg.CLAUSES[item.clause]
            values = self.check_kwargs(item.kwargs, clause.params, item.clause, item.span)
            checked.kwargs[id(item)] = values
            binding = self.lookup(item.subject.name, item.subject.span, self.bindings)
            if binding is None:
                continue
            if binding.kind != "behavior":
                self.type_error(
                    "E351",
                    f"'{item.clause}' takes a behaviour, but '{binding.name}' is {binding.type}",
                    item.subject.span,
                    'declare one with: behavior name = probes("suite.json", metric = accuracy)',
                )
                continue
            if not target.type.is_model:  # type: ignore[union-attr]
                self.type_error(
                    "E352",
                    f"behaviour checks need a runnable model, but '{target.name}' "
                    "was loaded with 'weights'",
                    item.subject.span,
                    "load it with 'model' from a directory that contains arch.json",
                )
                continue
            self._check_suite_fits(binding, target, item.subject.span, values)

    def _check_suite_fits(
        self, behavior: Binding, target: Binding, span: Span, values: dict[str, Any]
    ) -> None:
        suite = behavior.suite
        if suite is None:
            return
        if suite.kind == "tokens" and target.arch is not None:
            self.type_error(
                "E512",
                f"probe suite '{behavior.name}' is a token suite, but '{target.name}' is not a "
                "language model",
                span,
            )
        if suite.kind == "vectors" and target.hf is not None:
            self.type_error(
                "E512",
                f"probe suite '{behavior.name}' holds input vectors, but '{target.name}' is a "
                "language model; use a 'tokens' suite",
                span,
            )
        if suite.kind == "tokens" and target.hf is not None:
            if suite.needs_tokenizer and not target.hf.has_tokenizer:
                self.type_error(
                    "E513",
                    f"probe suite '{behavior.name}' contains text, but '{target.name}' has no "
                    "tokenizer files",
                    span,
                    "give prompt_ids and target_id instead, or add tokenizer.json",
                )
            top = suite.max_token_id()
            if top is not None and top >= target.hf.vocab_size:
                self.type_error(
                    "E514",
                    f"probe suite '{behavior.name}' uses token id {top}, but '{target.name}' has "
                    f"a vocabulary of {target.hf.vocab_size}",
                    span,
                )
        if target.arch is not None and suite.kind == "vectors":
            if suite.input_width != target.arch.inputs:
                self.type_error(
                    "E508",
                    f"probe suite '{behavior.name}' has inputs of width "
                    f"{suite.input_width} but '{target.name}' takes {target.arch.inputs}",
                    span,
                )
            if suite.output_slice and suite.output_slice[1] > target.arch.outputs:
                self.type_error(
                    "E509",
                    f"probe suite '{behavior.name}' reads outputs "
                    f"{suite.output_slice[0]}..{suite.output_slice[1]} but "
                    f"'{target.name}' has {target.arch.outputs}",
                    span,
                )
        min_n = int(values.get("min_n", 20))
        if suite.n < min_n:
            self.warn(
                "W302",
                f"probe suite '{behavior.name}' has {suite.n} examples, fewer than "
                f"min_n = {min_n}; its verdicts will be inconclusive",
                span,
            )
        delta = values.get("min_delta")
        if isinstance(delta, float) and delta < 1.0 / suite.n:
            self.warn(
                "W303",
                f"min_delta {delta:g} is finer than the resolution 1/{suite.n} of "
                f"probe suite '{behavior.name}'; the verdict will be inconclusive",
                span,
            )

    def _check_output(self, block: n.OutputBlock, state: _ProgramState) -> None:
        seen: set[str] = set()
        for item in block.items:
            name = item.name
            if name not in reg.OUTPUTS:
                self.name_error(
                    "E209",
                    f"unknown output '{name}'",
                    item.span,
                    _available("outputs", reg.OUTPUTS),
                )
                continue
            if name in seen:
                self.warn("W304", f"output '{name}' is listed twice", item.span)
            seen.add(name)
            needs = {
                "model": (state.mutated, "a mutating statement (edit, compose or repair)"),
                "patch": (state.mutated, "a mutating statement (edit, compose or repair)"),
                "evaluation": (state.verified, "a verify statement"),
                "findings": (state.found, "a find statement"),
                "evidence": (state.found, "a find statement"),
                "summary": (state.inspected, "an inspect statement"),
            }
            ok, what = needs[name]
            if not ok:
                self.type_error(
                    "E353", f"output '{name}' needs {what} earlier in the program", item.span
                )


@dataclass(frozen=True, slots=True)
class _Role:
    binding: str
    ref: RoleRef
    regions: tuple[Region, ...] | None


class _NotRole:
    """Marker: the expression is not a role path."""


class _Invalid:
    """Marker: the expression was a role path with an error, already reported."""


_NOT_ROLE = _NotRole()
_INVALID = _Invalid()
_RoleResult = _Role | _NotRole | _Invalid


@dataclass(slots=True)
class _ProgramState:
    target: Binding
    mutated: bool = False
    verified: bool = False
    found: bool = False
    inspected: bool = False
    target_finders: set[str] = field(default_factory=set)


def check_file(file: n.SourceFile, base_dir: Path, sink: DiagnosticSink) -> CheckedFile:
    return Checker(file, base_dir, sink).check()


__all__ = ["Binding", "CheckedFile", "CheckedProgram", "Prim", "SelectionInfo", "check_file"]
