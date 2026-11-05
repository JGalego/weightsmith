"""Entry point of the ``weightsmith`` command.

Static commands (``check``, ``compile``) never import a tensor library. Commands that touch
weights import the PyTorch backend lazily.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from weightsmith import __version__
from weightsmith.cli import render
from weightsmith.compiler.driver import CompileResult, compile_file
from weightsmith.compiler.ir import IRProgram
from weightsmith.errors import WeightsmithError

app = typer.Typer(
    name="weightsmith",
    help="Weightsmith: a programming language for weight-space programming.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console(highlight=False)
err = Console(stderr=True, highlight=False)


def _version(value: bool) -> None:
    if value:
        console.print(f"weightsmith {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Inspect, transform, compose and verify neural networks in weight space."""


def _compile(path: Path) -> CompileResult:
    if not path.is_file():
        err.print(f"[red]error[/]: {escape(str(path))} does not exist")
        raise typer.Exit(1)
    return compile_file(path)


@app.command()
def check(program: Annotated[Path, typer.Argument(help="A .wsm source file.")]) -> None:
    """Parse and statically check a program without running it."""
    result = _compile(program)
    render.diagnostics(console, result)
    render.check_summary(console, result)
    raise typer.Exit(0 if result.ok else 1)


@app.command("compile")
def compile_(
    program: Annotated[Path, typer.Argument(help="A .wsm source file.")],
    output: Annotated[
        Path | None, typer.Option("-o", "--output", help="Write IR JSON here.")
    ] = None,
    name: Annotated[str | None, typer.Option("--program", help="Program to compile.")] = None,
    text: Annotated[
        bool, typer.Option("--text", help="Print readable IR instead of JSON.")
    ] = False,
) -> None:
    """Compile a program to Weightsmith IR (JSON)."""
    result = _compile(program)
    if not result.ok:
        render.diagnostics(console, result)
        raise typer.Exit(1)
    try:
        ir = result.program(name)
    except KeyError as exc:
        err.print(f"[red]error[/]: {escape(str(exc.args[0]))}")
        raise typer.Exit(1) from exc
    rendered = ir.to_text() if text else ir.to_json()
    if output is None:
        typer.echo(rendered)
    else:
        output.write_text(rendered + "\n", encoding="utf-8")
        console.print(f"Wrote {escape(str(output))}")


@app.command()
def run(
    program: Annotated[Path, typer.Argument(help="A .wsm source file or compiled IR (.json).")],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output directory (default: out/<program>-<timestamp>)."),
    ] = None,
    name: Annotated[str | None, typer.Option("--program", help="Program to run.")] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Plan and compute in memory; measure and write nothing."),
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Allow a non-empty output directory.")
    ] = False,
    seed: Annotated[int, typer.Option("--seed", help="Seed recorded in provenance.")] = 0,
    replay: Annotated[
        Path | None,
        typer.Option(
            "--replay",
            help="Replay editors from the transcripts in this directory instead of calling them.",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Run a program. Exit codes: 0 ok, 1 invalid program, 2 runtime error or budget stop,
    3 a contract item failed, 4 a contract item was inconclusive."""
    if program.suffix == ".json":
        try:
            ir = IRProgram.from_json(program.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as exc:
            err.print(f"[red]error[/]: cannot read IR from {escape(str(program))}: {exc}")
            raise typer.Exit(1) from exc
    else:
        result = _compile(program)
        if not result.ok:
            render.diagnostics(console, result)
            raise typer.Exit(1)
        try:
            ir = result.program(name)
        except KeyError as exc:
            err.print(f"[red]error[/]: {escape(str(exc.args[0]))}")
            raise typer.Exit(1) from exc

    from weightsmith.backends.pytorch import PyTorchBackend
    from weightsmith.runtime.context import RunOptions
    from weightsmith.runtime.executor import execute

    options = RunOptions(
        out_dir=out, replay=replay, dry_run=dry_run, overwrite=overwrite, seed=seed
    )
    rep = execute(ir, PyTorchBackend(), options)
    if as_json:
        typer.echo(rep.to_json())
    else:
        render.report(console, rep)
    raise typer.Exit(rep.exit_code())


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(help="A .safetensors file or a model directory.")],
    find: Annotated[
        bool, typer.Option("--find/--no-find", help="Run every structural finder.")
    ] = True,
) -> None:
    """Show parameter statistics and structural findings for a weight file."""
    from weightsmith.backends.pytorch import PyTorchBackend
    from weightsmith.compiler.ir import Region
    from weightsmith.models.hf import read_profile
    from weightsmith.registry import FINDERS

    backend = PyTorchBackend()
    try:
        tensors = backend.load_weights(path)
    except WeightsmithError as exc:
        err.print(f"[red]error[/]: {escape(str(exc))}")
        raise typer.Exit(2) from exc
    manifest = backend.manifest(tensors)
    model_id = backend.content_hash(tensors)
    table = Table(title=f"{path}  ({model_id[:19]}...)", title_justify="left")
    for column in ("tensor", "dtype", "shape", "mean", "std", "min", "max", "non-finite"):
        table.add_column(column, justify="left" if column in ("tensor", "dtype") else "right")
    for row in backend.stats(tensors, [Region(name) for name in manifest.names()]):
        table.add_row(
            escape(row["name"]),
            row["dtype"],
            "x".join(map(str, row["shape"])),
            *(f"{row[k]:.4g}" if k in row else "-" for k in ("mean", "std", "min", "max")),
            str(row["nonfinite"]),
        )
    console.print(table)
    console.print(f"{len(manifest.tensors)} tensors, {manifest.numel} parameters")
    profile, reason = read_profile(path, manifest)
    if profile is not None:
        summary = ", ".join(f"{k} {v}" for k, v in profile.summary().items())
        console.print(f"Architecture profile: {escape(summary)}")
    elif path.is_dir() and (path / "config.json").is_file():
        console.print(f"No architecture profile: {escape(str(reason))}")
    if not find:
        return
    console.print("[bold]Structural findings[/] (evidence level: observed)")
    total = 0
    for finder, spec in FINDERS.items():
        if finder == "anomalies" or (spec.unit is not None and profile is None):
            continue
        params = {p.name: p.default for p in spec.params}
        if spec.unit is not None:
            assert profile is not None
            groups = profile.unit_groups(spec.unit, list(range(profile.n_layers)))
            result = backend.find_units(finder, tensors, groups, params, str(path), model_id)
        else:
            result = backend.find(finder, tensors, manifest.names(), params, str(path), model_id)
        for finding in result.findings:
            total += 1
            console.print(f"  [bold]{finder}[/]: {escape(finding.summary)}")
    if not total:
        console.print("  none")


@app.command()
def diff(
    a: Annotated[Path, typer.Argument(help="Original weights (file or model directory).")],
    b: Annotated[Path, typer.Argument(help="Modified weights (file or model directory).")],
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the difference as a patch directory.")
    ] = None,
) -> None:
    """Compare two weight sets and optionally save the difference as a patch."""
    import torch

    from weightsmith.artifacts.provenance import collect
    from weightsmith.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    try:
        before, after = backend.load_weights(a), backend.load_weights(b)
    except WeightsmithError as exc:
        err.print(f"[red]error[/]: {escape(str(exc))}")
        raise typer.Exit(2) from exc
    only_a, only_b = sorted(set(before) - set(after)), sorted(set(after) - set(before))
    reshaped = sorted(
        n
        for n in set(before) & set(after)
        if before[n].shape != after[n].shape or before[n].dtype != after[n].dtype
    )
    if only_a or only_b or reshaped:
        for label, names in (
            ("only in A", only_a),
            ("only in B", only_b),
            ("shape or dtype differs", reshaped),
        ):
            for n in names:
                console.print(f"  {label}: {escape(n)}")
        console.print("[red]The weight sets are structurally different; no patch is possible.[/]")
        raise typer.Exit(1)
    counts = backend.changed_params(before, after)
    table = Table(title="Changed tensors", title_justify="left")
    for column in ("tensor", "changed", "of", "max |delta|", "relative L2 change"):
        table.add_column(column, justify="left" if column == "tensor" else "right")
    for name, count in sorted(counts.items()):
        delta = after[name].double() - before[name].double()
        finite = delta[torch.isfinite(delta)]
        max_delta = f"{finite.abs().max().item():.4g}" if finite.numel() else "nan"
        norm = before[name].double().norm().item()
        rel = f"{delta.norm().item() / norm:.4g}" if norm else "-"
        table.add_row(escape(name), str(count), str(before[name].numel()), max_delta, rel)
    console.print(table)
    total = sum(t.numel() for t in before.values())
    changed = sum(counts.values())
    console.print(f"{changed} of {total} parameters differ ({changed / total:.4%})")
    if out is not None:
        manifest = backend.write_patch(
            before, after, out, collect(source_a=str(a), source_b=str(b))
        )
        console.print(f"Wrote patch with {len(manifest.entries)} entries to {escape(str(out))}")


@app.command()
def apply(
    base: Annotated[Path, typer.Argument(help="Weights the patch was made against.")],
    patch: Annotated[Path, typer.Argument(help="A patch directory.")],
    out: Annotated[Path, typer.Option("--out", help="Directory for the patched weights.")],
    reverse: Annotated[bool, typer.Option("--reverse", help="Undo the patch instead.")] = False,
) -> None:
    """Apply a patch (or undo it) and write the result to a new directory."""
    import shutil

    from weightsmith.backends.pytorch import PyTorchBackend
    from weightsmith.models.hf import SIDE_FILES
    from weightsmith.models.manifest import ARCH_FILE, WEIGHTS_FILE

    backend = PyTorchBackend()
    if out.exists() and any(out.iterdir()):
        err.print(f"[red]error[/]: {escape(str(out))} is not empty")
        raise typer.Exit(1)
    try:
        tensors = backend.load_weights(base)
        result = backend.apply_patch(tensors, patch, reverse=reverse)
    except WeightsmithError as exc:
        err.print(f"[red]error[/]: {escape(str(exc))}")
        raise typer.Exit(2) from exc
    backend.save_weights(result, out / WEIGHTS_FILE, {"weightsmith.patch": str(patch)})
    for name in (ARCH_FILE, *SIDE_FILES) if base.is_dir() else ():
        if (base / name).is_file():
            shutil.copy(base / name, out / name)
    console.print(f"Wrote {escape(str(out))} ({backend.content_hash(result)})")


@app.command()
def fixtures(
    directory: Annotated[Path, typer.Argument(help="Where to build the fixtures.")] = Path(
        "examples/fixtures"
    ),
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if up to date.")] = False,
) -> None:
    """Build the synthetic models with planted defects used by the examples and tests."""
    from weightsmith.synthetic import build_all

    summary = build_all(directory, force=force)
    console.print(f"Fixtures in {escape(str(directory))}:")
    for name, acc in summary.items():
        console.print(f"  {name}: " + ", ".join(f"{k} accuracy {v:.3f}" for k, v in acc.items()))
    ground_truth = directory / "tiny_mlp_defective" / "fixture.json"
    planted = json.loads(ground_truth.read_text())["planted"]
    console.print(f"  planted in tiny_mlp_defective: {planted}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
