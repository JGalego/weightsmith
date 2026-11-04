"""Execute an IR program on a backend and produce a :class:`RunReport`.

Execution order follows the IR. The default flow the semantics guarantee:

1. load the original target, which is never modified;
2. analyse it and transform a copy-on-write working copy;
3. check every mutation against the mutate budget before it is applied;
4. verify the contract on before/after measurements;
5. write artifacts to a separate output directory, never over an input.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from weightsmith.artifacts.provenance import collect
from weightsmith.backends.base import Backend
from weightsmith.compiler import ir
from weightsmith.compiler.effects import Effect
from weightsmith.diagnostics import Span
from weightsmith.errors import (
    BudgetExceeded,
    IncompatibleError,
    PolicyViolation,
    WeightsmithError,
)
from weightsmith.evaluation import verify as v
from weightsmith.evaluation.probes import load_probe_suite
from weightsmith.evaluation.report import Event, RunReport
from weightsmith.evidence import Evidence, EvidenceLevel, Finding
from weightsmith.models.manifest import ARCH_FILE, WEIGHTS_FILE, weights_file
from weightsmith.registry import FINDERS, METRICS
from weightsmith.runtime import planner
from weightsmith.runtime.context import (
    Budgets,
    LoadedBehavior,
    LoadedWeights,
    RunOptions,
    SelectionValue,
    Value,
    WorkingCopy,
)


def default_out_dir(program: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("out") / f"{program}-{stamp}"


def _where(span: Span | None) -> str | None:
    return str(span) if span else None


class Executor:
    def __init__(self, program: ir.IRProgram, backend: Backend, options: RunOptions) -> None:
        self.program = program
        self.backend = backend
        self.options = options
        self.values: dict[int, Value] = {}
        self.budgets = Budgets(program.policy)
        self.working: WorkingCopy | None = None
        self.baselines: dict[int, float] = {}
        self.repaired: list[tuple[Finding, str]] = []
        self.outputs = next((op.outputs for op in program.ops if isinstance(op, ir.Emit)), ())
        self.report = RunReport(
            program.name, program.content_hash(), backend.name, options.dry_run, self.outputs
        )
        self.out_dir: Path | None = None

    # entry point --------------------------------------------------------------------------
    def run(self) -> RunReport:
        report = self.report
        report.provenance = collect(
            seed=self.options.seed, program_hash=report.program_hash, source=self.program.source
        )
        try:
            planner.check_backend(self.program, self.backend)
            self._check_policy()
            self._prepare_out_dir()
            for op in self.program.ops:
                stop = self._execute(op)
                if stop:
                    break
        except BudgetExceeded as exc:
            report.status, report.stop_reason = "stopped", str(exc)
            report.events.append(Event("budget", "stopped", exc.message, _where(exc.span)))
        except WeightsmithError as exc:
            report.status, report.stop_reason = "error", str(exc)
            report.events.append(Event("error", "error", exc.message, _where(exc.span)))
        self._finish()
        return report

    def _check_policy(self) -> None:
        """Re-check effects at run time; IR loaded from JSON has not been through the compiler."""
        for op in self.program.ops:
            for effect in op.effects():
                if not self.program.policy.allows(effect):
                    raise PolicyViolation(
                        f"'{op.describe()}' needs the '{effect}' effect, which "
                        "the policy does not allow",
                        op.span,
                    )

    def _prepare_out_dir(self) -> None:
        if self.options.dry_run:
            return
        out = (self.options.out_dir or default_out_dir(self.program.name)).resolve()
        for op in self.program.ops:
            if isinstance(op, ir.Load):
                source = Path(op.path).resolve()
                source_dir = source if source.is_dir() else source.parent
                if out == source_dir or source_dir in out.parents:
                    raise WeightsmithError(
                        f"refusing to write output into {out}: it is inside the "
                        f"input directory {source_dir}",
                        op.span,
                    )
        if out.exists() and any(out.iterdir()) and not self.options.overwrite:
            raise WeightsmithError(
                f"output directory {out} is not empty; choose another or pass --overwrite"
            )
        out.mkdir(parents=True, exist_ok=True)
        self.out_dir = out

    def _finish(self) -> None:
        report = self.report
        if self.working is not None:
            changed = self.working.changed()
            total = self.working.total
            report.changes = {
                "changed_params": sum(changed.values()),
                "total_params": total,
                "fraction": sum(changed.values()) / total if total else 0.0,
                "tensors": changed,
            }
            report.target["final_model_id"] = self.backend.content_hash(self.working.tensors)
        report.budgets = self.budgets.to_dict()
        if self.out_dir is not None:
            (self.out_dir / "report.json").write_text(report.to_json() + "\n", encoding="utf-8")
            report.artifacts["report"] = str(self.out_dir / "report.json")

    # dispatch --------------------------------------------------------------------------------
    def _execute(self, op: ir.Op) -> bool:
        """Run one operation. Returns True when the run must stop."""
        match op:
            case ir.Load():
                self._load(op)
            case ir.Select():
                self._select(op)
            case ir.Inspect():
                self._inspect(op)
            case ir.Find():
                self._find(op)
            case ir.Transform():
                self._transform(op)
            case ir.Compose():
                self._compose(op)
            case ir.Repair():
                self._repair(op)
            case ir.Verify():
                return self._verify(op)
            case ir.Emit():
                self._emit(op)
            case _:  # pragma: no cover
                raise WeightsmithError(f"unknown operation {op.kind}", op.span)
        return False

    def _event(self, op: ir.Op, status: str, detail: str, **data: Any) -> None:
        self.report.events.append(Event(op.describe(), status, detail, _where(op.span), data))

    # values ------------------------------------------------------------------------------------
    def _load(self, op: ir.Load) -> None:
        path = Path(op.path)
        if op.what == "behavior":
            self.budgets.reserve_bytes(path.stat().st_size if path.is_file() else 0, op.span)
            suite = load_probe_suite(path)
            metric = str(op.params.get("metric"))
            if metric not in METRICS:
                raise WeightsmithError(f"unknown metric '{metric}'", op.span)
            self.values[op.out] = LoadedBehavior(op.binding, path, suite, metric)
            self._event(op, "ok", f"{suite.n} probes from {path.name}", n=suite.n)
            return
        try:
            self.budgets.reserve_bytes(weights_file(path).stat().st_size, op.span)
            if op.what == "model":
                tensors, arch = self.backend.load_model(path)
            else:
                tensors, arch = self.backend.load_weights(path), None
        except WeightsmithError as exc:
            exc.span = exc.span or op.span
            raise
        model_id = self.backend.content_hash(tensors)
        value = LoadedWeights(op.binding, path, tensors, arch, model_id)
        self.values[op.out] = value
        manifest = self.backend.manifest(tensors)
        record = {
            "binding": op.binding,
            "kind": op.what,
            "path": str(path),
            "model_id": model_id,
            "tensors": len(manifest.tensors),
            "params": manifest.numel,
        }
        self.report.inputs.append(record)
        if op.out == self.program.target:
            self.working = WorkingCopy(self.backend, tensors)
            self.report.target = {"binding": op.binding, "original_model_id": model_id}
        self._event(
            op,
            "ok",
            f"{len(manifest.tensors)} tensors, {manifest.numel} parameters",
            model_id=model_id,
        )

    def _weights(self, value_id: int) -> LoadedWeights:
        value = self.values[value_id]
        assert isinstance(value, LoadedWeights)
        return value

    def _tensors_of(self, binding: str) -> Any:
        if binding == self.program.target_binding:
            assert self.working is not None
            return self.working.tensors
        for value in self.values.values():
            if isinstance(value, LoadedWeights) and value.binding == binding:
                return value.tensors
        raise WeightsmithError(f"'{binding}' is not loaded")

    def _selection(self, value_id: int) -> SelectionValue:
        value = self.values[value_id]
        if isinstance(value, LoadedWeights):
            names = tuple(self.backend.manifest(value.tensors).names())
            return SelectionValue(value.binding, names)
        assert isinstance(value, SelectionValue)
        return value

    def _select(self, op: ir.Select) -> None:
        source = self.values[op.source]
        if isinstance(source, LoadedWeights):
            manifest = self.backend.manifest(source.tensors)
            infos = manifest.match(op.pattern) if op.pattern is not None else list(manifest.tensors)
            if not infos:
                raise IncompatibleError(
                    f"pattern \"{op.pattern}\" matches no parameter of '{source.binding}'", op.span
                )
            self.values[op.out] = SelectionValue(source.binding, tuple(t.name for t in infos))
            return
        assert isinstance(source, SelectionValue)
        self.values[op.out] = SelectionValue(source.binding, source.names, op.index)

    # analysis ------------------------------------------------------------------------------------
    def _inspect(self, op: ir.Inspect) -> None:
        selection = self._selection(op.source)
        tensors = self._tensors_of(selection.binding)
        rows = self.backend.stats(tensors, list(selection.names), selection.index)
        total = sum(r["numel"] for r in rows)
        self.report.inspections.append(
            {
                "binding": selection.binding,
                "slice": [str(i) for i in selection.index],
                "tensors": rows,
                "total_params": total,
                "nonfinite": sum(r["nonfinite"] for r in rows),
            }
        )
        self._event(op, "ok", f"{len(rows)} tensors, {total} parameters in '{selection.binding}'")

    def _find(self, op: ir.Find) -> None:
        selection = self._selection(op.scope)
        tensors = self._tensors_of(selection.binding)
        model_id = self.backend.content_hash(tensors)
        kinds = list(op.params.get("kinds", ())) if op.finder == "anomalies" else [op.finder]
        found = 0
        for kind in kinds:
            params = (
                op.params
                if kind == op.finder
                else {p.name: p.default for p in FINDERS[kind].params}
            )
            result = self.backend.find(
                kind, tensors, list(selection.names), params, selection.binding, model_id
            )
            self.report.finder_results.append(result)
            self.report.findings.extend(result.findings)
            found += len(result.findings)
        self._event(op, "ok", f"{found} finding(s) in '{selection.binding}'", finders=kinds)

    # mutation ------------------------------------------------------------------------------------
    def _commit(self, op: ir.Op, changes: dict[str, Any], detail: str) -> None:
        working = self.working
        assert working is not None
        if not changes:
            self._event(op, "noop", f"{detail}: nothing changed")
            return
        prospective = {**working.tensors, **changes}
        counts = self.backend.changed_params(working.base, prospective)
        fraction = sum(counts.values()) / working.total if working.total else 0.0
        self.budgets.check_changed_fraction(fraction, op.span)
        working.tensors = prospective
        working.mutations.append(op.kind)
        self._event(
            op,
            "dry-run" if self.options.dry_run else "ok",
            f"{detail}; {fraction:.4%} of parameters now differ from the original",
            tensors=sorted(changes),
            changed_fraction=fraction,
        )

    def _transform(self, op: ir.Transform) -> None:
        selection = self._selection(op.operand)
        assert self.working is not None
        changes = self.backend.edit(
            op.op, self.working.tensors, list(selection.names), selection.index, op.params
        )
        self._commit(op, changes, f"{op.op} on {len(selection.names)} tensor(s)")

    def _compose(self, op: ir.Compose) -> None:
        selection = self._selection(op.operand)
        assert self.working is not None
        donor = self._tensors_of(selection.binding)
        changes = self.backend.compose(
            op.op, self.working.tensors, donor, list(selection.names), op.params
        )
        self._commit(
            op, changes, f"{op.op} with '{selection.binding}' over {len(selection.names)} tensor(s)"
        )

    def _repair(self, op: ir.Repair) -> None:
        assert self.working is not None
        current = self.backend.content_hash(self.working.tensors)
        target = self.program.target_binding
        about_target = [f for f in self.report.findings if f.binding == target]
        stale = [f for f in about_target if f.model_id != current]
        fresh = [f for f in about_target if f.model_id == current]
        if stale and not fresh:
            raise WeightsmithError(
                "the findings describe an earlier state of the working copy; run "
                "find again after the last edit",
                op.span,
            )
        plan = planner.plan_repair(fresh, op.strategies)
        if not plan.steps:
            self._event(op, "noop", "no findings to repair")
            return
        changes: dict[str, Any] = {}
        for finding, strategy in plan.steps:
            view = {**self.working.tensors, **changes}
            changes.update(self.backend.repair(strategy, view, finding))
        self.repaired.extend(plan.steps)
        detail = "; ".join(f"{s} on {', '.join(map(str, f.locations))}" for f, s in plan.steps)
        if plan.unhandled:
            detail += (
                f"; {len(plan.unhandled)} finding(s) left unrepaired: no strategy handles "
                + ", ".join(sorted({f.kind for f in plan.unhandled}))
            )
        self._commit(op, changes, detail)

    # verification --------------------------------------------------------------------------------
    def _verify(self, op: ir.Verify) -> bool:
        assert self.working is not None
        target = self._weights(self.program.target)
        exhausted: BudgetExceeded | None = None
        verdicts: list[v.Verdict] = []
        for item in op.items:
            if self.options.dry_run:
                verdicts.append(
                    v.unavailable(
                        item.clause, item.subject, v.Status.SKIPPED, "dry run: nothing was measured"
                    )
                )
                continue
            if item.clause == "require":
                verdicts.append(self._requirement(item))
                continue
            assert item.behavior is not None
            behavior = self.values[item.behavior]
            assert isinstance(behavior, LoadedBehavior)
            if target.arch is None:
                verdicts.append(
                    v.unavailable(
                        item.clause,
                        item.subject,
                        v.Status.INCONCLUSIVE,
                        "the target has no architecture, so it cannot run",
                    )
                )
                continue
            try:
                before = self._baseline(item.behavior, behavior, target, item.span)
                self.budgets.reserve_forward_passes(behavior.suite.n, item.span)
            except BudgetExceeded as exc:
                exhausted = exc
                verdicts.append(
                    v.unavailable(
                        item.clause,
                        item.subject,
                        v.Status.INCONCLUSIVE,
                        f"budget exhausted: {exc.message}",
                    )
                )
                continue
            after = self.backend.evaluate(
                target.arch, self.working.tensors, behavior.suite, behavior.metric
            )
            verdicts.append(
                v.judge_behavior(
                    item.clause,
                    item.subject,
                    before,
                    after,
                    behavior.suite.n,
                    item.params,
                    METRICS[behavior.metric].higher_is_better,
                )
            )
        self.report.verdicts.extend(verdicts)
        self._attach_intervention_evidence(verdicts)
        summary = ", ".join(f"{vd.item}: {vd.status}" for vd in verdicts)
        self._event(op, "stopped" if exhausted else "ok", summary)
        if exhausted is not None:
            self.report.status = "stopped"
            self.report.stop_reason = str(exhausted)
            return True
        return False

    def _baseline(
        self, behavior_id: int, behavior: LoadedBehavior, target: LoadedWeights, span: Span | None
    ) -> float:
        if behavior_id not in self.baselines:
            assert target.arch is not None
            self.budgets.reserve_forward_passes(behavior.suite.n, span)
            self.baselines[behavior_id] = self.backend.evaluate(
                target.arch, target.tensors, behavior.suite, behavior.metric
            )
        return self.baselines[behavior_id]

    def _requirement(self, item: ir.ContractItem) -> v.Verdict:
        assert self.working is not None
        if item.subject == "finite":
            return v.judge_finite(self.backend.nonfinite(self.working.tensors))
        if item.subject == "changed_params":
            changed = sum(self.working.changed().values())
            return v.judge_changed_params(changed, self.working.total, float(item.params["max"]))
        raise WeightsmithError(f"unknown requirement '{item.subject}'", item.span)

    def _attach_intervention_evidence(self, verdicts: list[v.Verdict]) -> None:
        """Record that repairing a finding was followed by a verified improvement.

        Only when every mutation in the run came from repair, so the improvement can be attributed
        to the repaired locations (jointly, if there were several).
        """
        assert self.working is not None
        if not self.repaired or any(kind != "repair" for kind in self.working.mutations):
            return
        improved = [
            vd for vd in verdicts if vd.clause == "improve" and vd.status is v.Status.VERIFIED
        ]
        for finding, strategy in self.repaired:
            for verdict in improved:
                others = len(self.repaired) - 1
                joint = f" jointly with {others} other repaired finding(s)" if others else ""
                finding.evidence.append(
                    Evidence(
                        EvidenceLevel.EXPERIMENTALLY_SUPPORTED,
                        f"intervention:{strategy}",
                        f"Applying {strategy} here{joint} was followed by a verified "
                        "improvement of "
                        f"'{verdict.subject}' from {verdict.measurements['before']:.4f} to "
                        f"{verdict.measurements['after']:.4f}.",
                        dict(verdict.measurements),
                    )
                )

    # output ------------------------------------------------------------------------------
    def _emit(self, op: ir.Emit) -> None:
        if self.options.dry_run or self.out_dir is None:
            self._event(op, "dry-run", "dry run: no artifacts written")
            return
        assert self.working is not None
        if not self.program.policy.allows(Effect.SAVE) and {"model", "patch"} & set(op.outputs):
            raise PolicyViolation("saving needs the 'save' effect", op.span)
        target = self._weights(self.program.target)
        provenance = dict(
            self.report.provenance, program=self.program.name, base_model_id=target.model_id
        )
        if "model" in op.outputs:
            model_dir = self.out_dir / "model"
            metadata = {
                "weightsmith.program": self.program.name,
                "weightsmith.base_model_id": target.model_id,
                "weightsmith.program_hash": self.report.program_hash,
            }
            self.backend.save_weights(self.working.tensors, model_dir / WEIGHTS_FILE, metadata)
            if target.arch is not None:
                (model_dir / ARCH_FILE).write_text(
                    target.arch.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
            self.report.artifacts["model"] = str(model_dir)
        if "patch" in op.outputs:
            patch_dir = self.out_dir / "patch"
            manifest = self.backend.write_patch(
                target.tensors, self.working.tensors, patch_dir, provenance
            )
            self.report.artifacts["patch"] = str(patch_dir)
            self.report.target["patch_changed_params"] = manifest.changed_params
        written = ", ".join(sorted(set(self.report.artifacts) & {"model", "patch"})) or "reports"
        self._event(op, "ok", f"wrote {written} to {self.out_dir}")


def execute(
    program: ir.IRProgram, backend: Backend, options: RunOptions | None = None
) -> RunReport:
    return Executor(program, backend, options or RunOptions()).run()
