"""The tools an editor uses during ``delegate``, and the loop that drives the conversation.

Every tool acts on the program's working copy through the executor, so an editor's changes obey
the same budgets, appear in the same patch and are verified by the same contract as any other
mutation. A tool that fails, or would exceed a budget, returns an error to the editor instead of
stopping the run; the editor can then try something else.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from weightsmith.compiler import ir
from weightsmith.compiler.effects import Effect
from weightsmith.compiler.ir import IndexSpec, Region
from weightsmith.errors import BudgetExceeded, WeightsmithError
from weightsmith.models.hf import HFArch
from weightsmith.models.profiles import RoleError, parse_index, parse_role
from weightsmith.models.view import View, default_view, parse_format, read_view, render
from weightsmith.reasoning.blocks import Block
from weightsmith.reasoning.edit_language import apply_edits
from weightsmith.reasoning.provider import ToolCall, ToolResult, ToolSchema
from weightsmith.registry import EDIT_OPS, FINDERS, TOOLS
from weightsmith.runtime.context import LoadedBehavior, LoadedWeights

if TYPE_CHECKING:
    from weightsmith.runtime.executor import Executor

DEFAULT_PYTHON_SECONDS = 600.0
MAX_TENSORS_LISTED = 300

SYSTEM_PROMPT = """\
You are the editor in a Weightsmith run. Weightsmith is a language for programming neural \
networks at the level of their weights. A program has handed you the working copy of a model and \
a task. You act on the model only through the tools you are given. Each tool call is checked \
against the program's policy and budgets, and every change you make is recorded in a patch.

Selections name parts of the model. Use one of these forms:
- "all" for every parameter;
- params["glob"] for parameters whose names match a glob, optionally followed by a slice such \
as params["layers.0.weight"][0..4, :];
- a role path, when the model has an architecture profile: embed, unembed, final_norm, \
layers[3], layers[0..2].mlp, layers[1].attn, layers[1].norms, layers[1].attn.heads[2], \
layers[1].attn.kv_heads[0], layers[0].mlp.neurons[17]. Whole-tensor roles accept a slice too.
Indices are integers, half-open ranges a..b, or ":".

Budgets are shared with the rest of the program. When a tool reports that a budget is exhausted, \
plan with what remains. After you finish, the program verifies its contract on its own probe \
suites, which you cannot see, so aim for the behaviour rather than the practice examples.

When you are done, reply without calling a tool. Say what you changed and why, and what you \
believe about how the model works. Your account is recorded as a hypothesis; only measurements \
made by the program count as evidence."""

_SELECT = {"type": "string", "description": "A selection, as described in the instructions."}

SCHEMAS: dict[str, dict[str, Any]] = {
    "describe": {"type": "object", "properties": {}},
    "read_weights": {
        "type": "object",
        "properties": {
            "select": _SELECT,
            "precision": {"type": "integer", "description": "Significant digits (default 4)."},
        },
        "required": ["select"],
    },
    "stats": {"type": "object", "properties": {"select": _SELECT}, "required": ["select"]},
    "find": {
        "type": "object",
        "properties": {
            "finder": {"type": "string", "description": "Finder name, see describe."},
            "select": {**_SELECT, "description": "Where to search (default: all)."},
            "params": {"type": "object", "description": "Finder parameters, e.g. threshold."},
        },
        "required": ["finder"],
    },
    "run_probes": {
        "type": "object",
        "properties": {
            "behavior": {"type": "string", "description": "A practice behaviour name."},
            "show_failures": {"type": "integer", "description": "Failing examples to show (5)."},
        },
        "required": ["behavior"],
    },
    "forward": {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Language models: prompts, tokenized as the model would see them.",
            },
            "prompt_ids": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "Language models: prompts as token ids.",
            },
            "inputs": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "Vector models: input vectors.",
            },
            "top_k": {"type": "integer", "description": "Next tokens to show (default 5)."},
        },
    },
    "edit": {
        "type": "object",
        "properties": {
            "op": {"type": "string", "description": "zero, scale, fill, prune_magnitude, ..."},
            "select": _SELECT,
            "params": {"type": "object", "description": 'For example {"factor": 0.5}.'},
        },
        "required": ["op", "select"],
    },
    "set_weights": {
        "type": "object",
        "properties": {
            "select": _SELECT,
            "values": {
                "type": "array",
                "description": "Numbers in row-major order, flat or nested. Regions follow the "
                "order read_weights shows them in.",
            },
        },
        "required": ["select", "values"],
    },
    "undo": {"type": "object", "properties": {}},
    "python": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python. W maps parameter names to writable NumPy arrays and np is "
                "NumPy. Modify W in place to propose edits. Print what you want to see. Each call "
                "is a fresh process; files in the working directory persist.",
            }
        },
        "required": ["code"],
    },
}

_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _validate(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """A small check of required keys and top-level types (inputs may arrive truncated)."""
    for key in schema.get("required", []):
        if key not in arguments:
            return f"missing argument '{key}'"
    for key, value in arguments.items():
        spec = schema.get("properties", {}).get(key)
        if spec is None:
            return f"unknown argument '{key}'"
        expected = _TYPES.get(spec.get("type", ""))
        if expected is not None and (
            not isinstance(value, expected) or (expected is int and isinstance(value, bool))
        ):
            return f"argument '{key}' must be of type {spec['type']}"
    return None


def _parse_slice(text: str) -> tuple[IndexSpec, ...]:
    parts = [p.strip() for p in text.split(",")]
    out = []
    for part in parts:
        if re.fullmatch(r"\d+:\d+", part):
            part = part.replace(":", "..")
        try:
            out.append(parse_index(part))
        except ValueError as exc:
            raise WeightsmithError(f"'{part}' is not an index, a range a..b or ':'") from exc
    return tuple(out)


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        return [v for item in values for v in _flatten(item)]
    if isinstance(values, bool) or not isinstance(values, int | float):
        raise WeightsmithError("values must be numbers")
    return [float(values)]


def _fmt(value: float, precision: int) -> str:
    return f"{value:.{precision}g}"


@dataclass(slots=True)
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class EditorSession:
    executor: Executor
    op: ir.Delegate
    workdir: Path
    log: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    version: int = 0
    python_seconds: float = 0.0
    _sandbox: Any = None

    # context ---------------------------------------------------------------------------
    @property
    def target(self) -> LoadedWeights:
        return self.executor.target_value()

    @property
    def tensors(self) -> dict[str, Any]:
        working = self.executor.working
        assert working is not None
        return working.tensors

    def practice(self) -> dict[str, LoadedBehavior]:
        out = {}
        for value_id in self.op.practice:
            value = self.executor.values[value_id]
            assert isinstance(value, LoadedBehavior)
            out[value.binding] = value
        return out

    @property
    def blocks(self) -> bool:
        return self.op.params.get("protocol") == "blocks"

    def schemas(self) -> list[ToolSchema]:
        if self.blocks:
            return []  # the editor acts by writing blocks, see run_block
        return [ToolSchema(name, TOOLS[name].doc, SCHEMAS[name]) for name in self.op.tools]

    def run_block(self, block: Block) -> str:
        """Run a block the editor wrote; the result is sent back as the next message."""
        entry: dict[str, Any] = {"block": block.kind, "body": _short(block.body)}
        self.log.append(entry)
        try:
            if block.kind != "edits":
                raise WeightsmithError(f"no block runner for '{block.kind}'")
            text, entry["report"] = self._block_edits(block.body)
        except WeightsmithError as exc:
            entry["error"] = exc.message
            raise
        return text

    def _block_edits(self, body: str) -> tuple[str, dict[str, Any]]:
        """Apply an edits block to the tensors as the view shows them (see edit_language)."""
        import numpy as np

        view = self.view()
        backend = self.executor.backend
        arrays = backend.arrays(self.tensors, [t.param for t in view.tensors])
        shown = {
            t.label: (arrays[t.param].T if t.transpose else arrays[t.param]) for t in view.tensors
        }
        edited = {label: array.copy() for label, array in shown.items()}
        report = apply_edits(body, edited)
        regions, values = [], []
        for t in view.tensors:
            if not np.array_equal(edited[t.label], shown[t.label]):
                regions.append(Region(t.param))
                values += (edited[t.label].T if t.transpose else edited[t.label]).ravel().tolist()
        changes = backend.assign(self.tensors, regions, values) if regions else {}
        return f"{report.summary()}\n{self._commit(changes, 'edits')}", report.to_dict()

    def system_prompt(self) -> str:
        return str(self.op.params.get("system") or SYSTEM_PROMPT)

    def first_message(self) -> str:
        """The program's own message, {{weights}} filled in, if it gave one; else a briefing."""
        message = self.op.params.get("message")
        if message is None:
            return self.briefing()
        return re.sub(r"\{\{\s*weights\s*\}\}", lambda _: self.weights_text(), str(message))

    def view(self) -> View:
        target = self.target
        view = read_view(target.path) if target.path.is_dir() else None
        return view or default_view(self.executor.backend.manifest(target.tensors).names())

    def weights_text(self) -> str:
        """Every weight in the view, written in the delegation's weight format."""
        view = self.view()
        fmt = parse_format(str(self.op.params.get("weights_format") or "decimals(4)"))
        arrays = self.executor.backend.arrays(self.tensors, [t.param for t in view.tensors])
        return render(view, fmt, arrays)

    def briefing(self) -> str:
        target = self.target
        manifest = self.executor.backend.manifest(target.tensors)
        kind = (
            "language model"
            if isinstance(target.arch, HFArch)
            else ("runnable model" if target.arch is not None else "weight set that cannot run")
        )
        lines = [
            "Task:",
            self.op.task.strip(),
            "",
            f"Target: '{target.binding}', a {kind} with {manifest.numel} parameters in "
            f"{len(manifest.tensors)} tensors.",
        ]
        if target.profile is not None:
            summary = ", ".join(f"{k} {v}" for k, v in target.profile.summary().items())
            lines.append(f"Architecture profile: {summary}.")
        lines.append(f"Tools: {', '.join(self.op.tools)}. Call describe for details.")
        lines.append(f"Budgets: {self._budget_text()}")
        practice = self.practice()
        if practice:
            lines.append("Practice behaviours (measure them with run_probes):")
            lines += [
                f"- {name}: {b.suite.n} examples. {b.suite.description}".rstrip()
                for name, b in practice.items()
            ]
        goals = self.executor.contract_summary()
        if goals:
            lines.append(f"After you finish, the program will verify: {goals}.")
        if self.op.params.get("include_weights"):
            if self.op.params.get("weights_format"):
                lines += ["", "Weights:", self.weights_text()]
            else:
                regions = [Region(n) for n in manifest.names()]
                try:
                    lines += ["", "Weights:", self._read(regions, 4)]
                except WeightsmithError as exc:
                    lines += ["", f"(The weights are not included: {exc})"]
        return "\n".join(lines)

    def _budget_text(self) -> str:
        policy = self.executor.program.policy
        budgets = self.executor.budgets
        parts = []
        passes = policy.budget(Effect.PROBE, "max_forward_passes")
        if Effect.PROBE in self._granted():
            left = "unlimited" if passes is None else int(passes - budgets.forward_passes)
            parts.append(f"forward passes left {left}")
        changed = policy.budget(Effect.MUTATE, "max_changed_params")
        if Effect.MUTATE in self._granted():
            parts.append(
                "changed parameters "
                + ("unlimited" if changed is None else f"at most {changed:.4%} of all")
            )
        calls = policy.budget(Effect.REASON, "max_calls")
        if calls is not None:
            parts.append(f"editor turns left {int(calls - budgets.editor_calls)}")
        seconds = policy.budget(Effect.EXECUTE, "max_seconds")
        if "python" in self.op.tools:
            # The limit, not what is left: anything derived from wall-clock time would make
            # requests differ between a run and its replay.
            total = DEFAULT_PYTHON_SECONDS if seconds is None else seconds
            parts.append(f"python time {total:.0f} s in total")
        return "; ".join(parts) or "none"

    def _granted(self) -> frozenset[Effect]:
        return self.op.effects()

    # selections ------------------------------------------------------------------------
    def regions(self, text: str) -> list[Region]:
        text = text.strip()
        names = sorted(self.tensors)
        if text in ("all", "*"):
            return [Region(n) for n in names]
        match = re.fullmatch(r"params\[\s*([\"'])(.+?)\1\s*\](?:\[(.*)\])?", text)
        if match:
            matched = [n for n in names if fnmatch.fnmatchcase(n, match[2])]
            if not matched:
                raise WeightsmithError(f'pattern "{match[2]}" matches no parameter')
            index = _parse_slice(match[3]) if match[3] else ()
            return [Region(n, index) for n in matched]
        profile = self.target.profile
        try:
            role, sliced = parse_role(text), None
        except RoleError:
            match = re.fullmatch(r"(.+)\[([^\[\]]*)\]", text)
            if match is None:
                raise WeightsmithError(
                    f"'{text}' is not a selection; use all, params[\"glob\"] or a role path"
                ) from None
            try:
                role, sliced = parse_role(match[1]), match[2]
            except RoleError as exc:
                raise WeightsmithError(str(exc)) from None
        if profile is None:
            raise WeightsmithError(
                f"this model has no architecture profile ({self.target.profile_error}); "
                'select parameters with params["glob"]'
            )
        try:
            regions = list(profile.resolve(role))
        except RoleError as exc:
            raise WeightsmithError(str(exc)) from None
        if sliced is not None:
            if role.is_unit:
                raise WeightsmithError("a head or neuron selection cannot be sliced further")
            index = _parse_slice(sliced)
            regions = [Region(r.tensor, index) for r in regions]
        return regions

    # tools -------------------------------------------------------------------------------
    def call(self, call: ToolCall) -> ToolResult:
        entry: dict[str, Any] = {"tool": call.name, "arguments": _short(call.arguments)}
        self.log.append(entry)
        if call.error is not None or call.arguments is None:
            entry["error"] = call.error or "no arguments"
            raw = json.dumps({"INVALID_JSON": call.error or "unparseable"})
            return ToolResult(call.id, raw, is_error=True)
        if call.name not in self.op.tools:
            entry["error"] = "not granted"
            return ToolResult(
                call.id, f"the tool '{call.name}' is not available in this run", is_error=True
            )
        problem = _validate(call.arguments, SCHEMAS[call.name])
        if problem is not None:
            entry["error"] = problem
            return ToolResult(call.id, problem, is_error=True)
        try:
            content = getattr(self, f"_tool_{call.name}")(**call.arguments)
        except BudgetExceeded as exc:
            entry["error"] = f"budget: {exc.message}"
            return ToolResult(call.id, f"budget exhausted: {exc.message}", is_error=True)
        except (WeightsmithError, ValueError, KeyError, TypeError, IndexError) as exc:
            message = exc.message if isinstance(exc, WeightsmithError) else str(exc)
            entry["error"] = message
            return ToolResult(call.id, message, is_error=True)
        # A preview for the report; the full result is in the transcript.
        entry["result"] = (
            content if len(content) <= 400 else f"{content[:250]} ... {content[-150:]}"
        )
        return ToolResult(call.id, content)

    def _tool_describe(self) -> str:
        target = self.target
        manifest = self.executor.backend.manifest(target.tensors)
        tensors = [[t.name, list(t.shape), t.dtype] for t in manifest.tensors]
        practice = self.practice()
        data: dict[str, Any] = {
            "target": target.binding,
            "runnable": target.arch is not None,
            "language_model": isinstance(target.arch, HFArch),
            "parameters": manifest.numel,
            "profile": target.profile.summary() if target.profile is not None else None,
            "profile_note": target.profile_error if target.profile is None else None,
            "tensors": tensors[:MAX_TENSORS_LISTED],
            "tensors_not_listed": max(0, len(tensors) - MAX_TENSORS_LISTED),
            "changed_parameters": sum(self.executor.working_changes().values()),
            "budgets": self._budget_text(),
            "practice": {
                name: {"examples": b.suite.n, "kind": b.suite.kind, "about": b.suite.description}
                for name, b in practice.items()
            },
            "verified_afterwards": self.executor.contract_summary(),
            "tools": list(self.op.tools),
            "edit_ops": {name: spec.doc for name, spec in EDIT_OPS.items()},
            "finders": sorted(n for n in FINDERS if n != "anomalies"),
        }
        return json.dumps(data)

    def _read(self, regions: list[Region], precision: int) -> str:
        backend = self.executor.backend
        values = backend.region_values(self.tensors, regions)
        total = sum(v["numel"] for v in values)
        limit = int(self.op.params.get("max_read", 20000))
        if total > limit:
            raise WeightsmithError(
                f"the selection has {total} values, more than max_read = {limit}; select less"
            )
        lines = []
        for block in values:
            lines.append(f"{block['region']} shape {block['shape']} {block['dtype']}")
            rows = block["rows"]
            for label, row in rows:
                text = " ".join(_fmt(v, precision) for v in row)
                lines.append(f"  {label}: {text}" if label is not None else f"  {text}")
        return "\n".join(lines)

    def _tool_read_weights(self, select: str, precision: int = 4) -> str:
        return self._read(self.regions(select), max(1, min(precision, 12)))

    def _tool_stats(self, select: str) -> str:
        rows = self.executor.backend.stats(self.tensors, self.regions(select))
        return json.dumps(
            [{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()} for r in rows]
        )

    def _tool_find(
        self, finder: str, select: str = "all", params: dict[str, Any] | None = None
    ) -> str:
        spec = FINDERS.get(finder)
        if spec is None or finder == "anomalies":
            raise WeightsmithError(f"unknown finder '{finder}'")
        values = {p.name: p.default for p in spec.params}
        for key, value in (params or {}).items():
            if spec.param(key) is None:
                raise WeightsmithError(f"'{finder}' has no parameter '{key}'")
            values[key] = value
        backend = self.executor.backend
        target = self.target
        model_id = backend.content_hash(self.tensors)
        if spec.unit is not None:
            if target.profile is None:
                raise WeightsmithError(f"'{finder}' needs an architecture profile")
            role = None if select.strip() in ("all", "*") else parse_role(select.strip())
            layers = target.profile.layers_of(role)
            groups = target.profile.unit_groups(spec.unit, layers)
            result = backend.find_units(
                finder, self.tensors, groups, values, target.binding, model_id
            )
        else:
            regions = self.regions(select)
            if any(r.index for r in regions):
                raise WeightsmithError("finders search whole tensors; remove the slice")
            names = list(dict.fromkeys(r.tensor for r in regions))
            result = backend.find(finder, self.tensors, names, values, target.binding, model_id)
        return json.dumps(
            {
                "examined": result.examined,
                "unit": result.noun,
                "skipped": [list(s) for s in result.skipped[:20]],
                "findings": [
                    {
                        "summary": f.summary,
                        "locations": [str(loc) for loc in f.locations],
                        "measurements": _short(f.evidence[0].measurements if f.evidence else {}),
                    }
                    for f in result.findings
                ],
            }
        )

    def _tool_run_probes(self, behavior: str, show_failures: int = 5) -> str:
        practice = self.practice()
        if behavior not in practice:
            raise WeightsmithError(
                f"'{behavior}' is not a practice behaviour; available: {', '.join(practice)}"
            )
        target = self.target
        if target.arch is None:
            raise WeightsmithError("the target cannot run")
        loaded = practice[behavior]
        self.executor.budgets.reserve_forward_passes(loaded.suite.n, self.op.span)
        rows = self.executor.backend.examples(target.arch, self.tensors, loaded.suite)
        correct = sum(1 for r in rows if r["correct"])
        failures = [r for r in rows if not r["correct"]][: max(0, show_failures)]
        return json.dumps(
            {
                "behavior": behavior,
                "accuracy": round(correct / len(rows), 4),
                "correct": correct,
                "examples": len(rows),
                "failures_shown": failures,
                "budgets": self._budget_text(),
            }
        )

    def _tool_forward(
        self,
        prompts: list[str] | None = None,
        prompt_ids: list[list[int]] | None = None,
        inputs: list[list[float]] | None = None,
        top_k: int = 5,
    ) -> str:
        target = self.target
        if target.arch is None:
            raise WeightsmithError("the target cannot run")
        texts: list[str | list[int]] = [*(prompts or []), *(prompt_ids or [])]
        count = len(texts) or len(inputs or [])
        if count == 0:
            raise WeightsmithError("give prompts, prompt_ids or inputs")
        self.executor.budgets.reserve_forward_passes(count, self.op.span)
        outputs = self.executor.backend.run_inputs(
            target.arch, self.tensors, texts or None, inputs, max(1, min(top_k, 50))
        )
        return json.dumps({"outputs": outputs, "budgets": self._budget_text()})

    def _commit(self, changes: dict[str, Any], detail: str) -> str:
        if not changes:
            return f"{detail}: nothing changed"
        snapshot = dict(self.tensors)
        fraction = self.executor.apply_changes(self.op, changes)
        self.history.append(snapshot)
        self.version += 1
        return (
            f"{detail}: changed {', '.join(sorted(changes))}; {fraction:.4%} of parameters now "
            "differ from the original"
        )

    def _tool_edit(self, op: str, select: str, params: dict[str, Any] | None = None) -> str:
        spec = EDIT_OPS.get(op)
        if spec is None:
            raise WeightsmithError(f"unknown edit operation '{op}'; see describe")
        values = {p.name: p.default for p in spec.params if p.default is not None}
        for key, value in (params or {}).items():
            param = spec.param(key)
            if param is None:
                raise WeightsmithError(f"'{op}' has no parameter '{key}'")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise WeightsmithError(f"'{key}' must be a number")
            if param.minimum is not None and value < param.minimum:
                raise WeightsmithError(f"'{key}' must be at least {param.minimum:g}")
            if param.maximum is not None and value > param.maximum:
                raise WeightsmithError(f"'{key}' must be at most {param.maximum:g}")
            values[key] = value
        missing = [p.name for p in spec.params if p.required and p.name not in values]
        if missing:
            raise WeightsmithError(f"'{op}' requires {', '.join(missing)}")
        regions = self.regions(select)
        if not spec.allows_slicing and any(r.index for r in regions):
            raise WeightsmithError(f"'{op}' works on whole tensors")
        changes = self.executor.backend.edit(op, self.tensors, regions, values)
        return self._commit(changes, f"{op} on {select}")

    def _tool_set_weights(self, select: str, values: list[Any]) -> str:
        regions = self.regions(select)
        changes = self.executor.backend.assign(self.tensors, regions, _flatten(values))
        return self._commit(changes, f"set {select}")

    def _tool_undo(self) -> str:
        if not self.history:
            raise WeightsmithError("there is nothing of yours to undo")
        working = self.executor.working
        assert working is not None
        working.tensors = self.history.pop()
        self.version += 1
        changed = sum(self.executor.working_changes().values())
        return f"reverted your last change; {changed} parameters now differ from the original"

    def _tool_python(self, code: str) -> str:
        from weightsmith.runtime.sandbox import Sandbox

        limit = self.executor.program.policy.budget(Effect.EXECUTE, "max_seconds")
        left = (DEFAULT_PYTHON_SECONDS if limit is None else limit) - self.python_seconds
        if left <= 0:
            raise BudgetExceeded("execute", "max_seconds", limit or 0, self.python_seconds, None)
        if self._sandbox is None:
            self._sandbox = Sandbox(self.workdir)
        result = self._sandbox.run(code, self.tensors, self.version, left)
        self.python_seconds += result.seconds
        parts = []
        if result.stdout:
            parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            parts.append(f"stderr:\n{result.stderr}")
        if result.timed_out:
            parts.append(f"timed out after {result.seconds:.0f} s")
        elif result.returncode != 0:
            parts.append(f"exit code {result.returncode}; no changes were taken")
        if result.changed:
            if not self.executor.program.policy.allows(Effect.MUTATE):
                parts.append(
                    f"changes to {', '.join(sorted(result.changed))} were refused: the policy "
                    "does not allow mutate"
                )
            else:
                try:
                    parts.append(self._commit(result.changed, "python"))
                except BudgetExceeded as exc:
                    parts.append(f"changes refused: {exc.message}")
        return "\n".join(parts) or "(no output, no changes)"


def _short(value: Any, limit: int = 400) -> Any:
    """Arguments and measurements as logged: long values are abbreviated."""
    text = json.dumps(value, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + f"... ({len(text)} characters)"
