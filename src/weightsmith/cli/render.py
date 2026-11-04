"""Console rendering of diagnostics, IR and run reports."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from weightsmith.compiler.driver import CompileResult
from weightsmith.diagnostics import Severity, Stage, render
from weightsmith.evaluation.report import RunReport
from weightsmith.evaluation.verify import Status

STATUS_STYLE = {
    Status.VERIFIED: "green",
    Status.FAILED: "red",
    Status.INCONCLUSIVE: "yellow",
    Status.SKIPPED: "dim",
}


def diagnostics(console: Console, result: CompileResult) -> None:
    for diag in result.sink.items:
        style = "red" if diag.severity is Severity.ERROR else "yellow"
        text = render(diag, result.source)
        head, _, rest = text.partition("\n")
        console.print(f"[bold {style}]{escape(head)}[/]")
        if rest:
            console.print(escape(rest), highlight=False)


def check_summary(console: Console, result: CompileResult) -> None:
    console.print(f"[bold]weightsmith check[/] {escape(result.path)}")
    for stage in Stage:
        if stage not in result.stages_run:
            console.print(f"  [dim]-[/] {stage.value} [dim](not run: earlier errors)[/]")
        elif result.sink.has_errors(stage):
            count = sum(1 for d in result.sink.errors if d.stage is stage)
            console.print(f"  [red]x[/] {stage.value} [red]({count} error(s))[/]")
        else:
            console.print(f"  [green]✓[/] {stage.value}")
    if result.ok:
        names = ", ".join(result.programs) or "no programs"
        warnings = len(result.sink.warnings)
        suffix = f" with {warnings} warning(s)" if warnings else ""
        console.print(f"[green]Valid[/]{suffix}: {escape(names)}")
    else:
        console.print(f"[red]Invalid[/]: {len(result.sink.errors)} error(s)")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def report(console: Console, rep: RunReport) -> None:
    shown = set(rep.outputs) or {"findings", "evidence", "evaluation", "summary"}
    title = f"[bold]{escape(rep.program)}[/] on backend {rep.backend}"
    if rep.dry_run:
        title += " [yellow](dry run)[/]"
    console.print(title)

    events = Table(title="Operations", show_lines=False, title_justify="left")
    events.add_column("operation")
    events.add_column("status")
    events.add_column("detail", overflow="fold")
    for event in rep.events:
        style = {"ok": "green", "noop": "dim", "dry-run": "yellow"}.get(event.status, "red")
        events.add_row(escape(event.op), f"[{style}]{event.status}[/]", escape(event.detail))
    console.print(events)

    if "summary" in shown:
        for inspection in rep.inspections:
            table = Table(
                title=f"Parameters of {inspection['binding']}"
                + (f" [{', '.join(inspection['slice'])}]" if inspection["slice"] else ""),
                title_justify="left",
            )
            for column in (
                "tensor",
                "shape",
                "numel",
                "mean",
                "std",
                "min",
                "max",
                "l2",
                "zeros",
                "non-finite",
            ):
                table.add_column(column, justify="left" if column == "tensor" else "right")
            for row in inspection["tensors"]:
                table.add_row(
                    escape(row["name"]),
                    "x".join(map(str, row["shape"])) or "scalar",
                    str(row["numel"]),
                    *(_fmt(row.get(k, "-")) for k in ("mean", "std", "min", "max", "l2")),
                    str(row["zeros"]),
                    str(row["nonfinite"]),
                )
            console.print(table)
            console.print(
                f"  {inspection['total_params']} parameters, {inspection['nonfinite']} non-finite"
            )

    if "findings" in shown or "evidence" in shown:
        console.print("[bold]Findings[/]")
        for result in rep.finder_results:
            skipped = f", {len(result.skipped)} skipped" if result.skipped else ""
            console.print(
                f"  [dim]{result.finder}: examined {result.examined} tensor(s)"
                f"{skipped}, {len(result.findings)} finding(s)[/]"
            )
            if "evidence" in shown:
                for tensor, reason in result.skipped:
                    console.print(f"    [dim]skipped {escape(tensor)}: {escape(reason)}[/]")
        if not rep.findings:
            console.print("  none")
        for finding in rep.findings:
            console.print(
                f"  [bold]{finding.kind}[/] [{finding.level.value}] {escape(finding.summary)}"
            )
            if "evidence" in shown:
                for ev in finding.evidence:
                    console.print(
                        f"    evidence ({ev.level.value}, {escape(ev.method)}): "
                        f"{escape(ev.description)}"
                    )
                    if ev.measurements:
                        items = ", ".join(f"{k}={_fmt(val)}" for k, val in ev.measurements.items())
                        console.print(f"      [dim]{escape(items)}[/]")
                for limitation in finding.limitations:
                    console.print(f"    [dim]limitation: {escape(limitation)}[/]")

    if "evaluation" in shown and rep.verdicts:
        table = Table(title="Verification", title_justify="left")
        table.add_column("contract item")
        table.add_column("verdict")
        table.add_column("reason", overflow="fold")
        for verdict in rep.verdicts:
            style = STATUS_STYLE[verdict.status]
            table.add_row(
                escape(verdict.item), f"[{style}]{verdict.status.value}[/]", escape(verdict.reason)
            )
        console.print(table)

    if rep.changes and rep.changes.get("changed_params"):
        console.print(
            f"Changed parameters: {rep.changes['changed_params']} of "
            f"{rep.changes['total_params']} ({rep.changes['fraction']:.4%}) in "
            f"{len(rep.changes['tensors'])} tensor(s)"
        )
    budget = rep.budgets.get("forward_passes", {})
    if budget.get("used"):
        limit = budget.get("limit")
        console.print(
            f"Forward passes: {budget['used']}"
            + (f" of {int(limit)} allowed" if limit is not None else "")
        )
    for name, path in rep.artifacts.items():
        console.print(f"Wrote {name}: {escape(path)}")

    code = rep.exit_code()
    if rep.status != "completed":
        console.print(f"[red]Run {rep.status}[/]: {escape(rep.stop_reason or '')}")
    elif code == 3:
        console.print(
            f"[red]Verification failed[/] for {len(rep.failed_verdicts)} contract item(s)"
        )
    elif code == 4:
        console.print("[yellow]Completed, but some contract items are inconclusive[/]")
    else:
        console.print("[green]Completed[/]")
