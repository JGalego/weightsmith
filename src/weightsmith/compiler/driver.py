"""The static pipeline: parse, resolve and type-check, lower, check effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from weightsmith.compiler.effects import check_effects
from weightsmith.compiler.ir import IRProgram
from weightsmith.compiler.lower import lower_program
from weightsmith.compiler.typecheck import CheckedFile, check_file
from weightsmith.diagnostics import DiagnosticSink, Stage
from weightsmith.language.nodes import SourceFile
from weightsmith.language.parser import parse


@dataclass(slots=True)
class CompileResult:
    path: str
    source: str
    sink: DiagnosticSink
    ast: SourceFile | None = None
    checked: CheckedFile | None = None
    programs: dict[str, IRProgram] = field(default_factory=dict)
    stages_run: list[Stage] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.sink.has_errors()

    def program(self, name: str | None = None) -> IRProgram:
        """Return the named program, or the only program when ``name`` is None."""
        if name is not None:
            if name not in self.programs:
                raise KeyError(
                    f"no program '{name}' in {self.path}; programs: "
                    f"{', '.join(self.programs) or 'none'}"
                )
            return self.programs[name]
        if len(self.programs) != 1:
            raise KeyError(
                f"{self.path} defines {len(self.programs)} programs; choose one with "
                f"--program ({', '.join(self.programs)})"
            )
        return next(iter(self.programs.values()))


def compile_source(source: str, path: str, base_dir: Path | None = None) -> CompileResult:
    """Run every static stage that the previous stages allow, collecting diagnostics."""
    sink = DiagnosticSink()
    result = CompileResult(path, source, sink)
    result.stages_run.append(Stage.SYNTAX)
    result.ast = parse(source, path, sink)
    if result.ast is None:
        return result
    result.stages_run += [Stage.NAMES, Stage.TYPES]
    base = base_dir if base_dir is not None else Path(path).resolve().parent
    result.checked = check_file(result.ast, base, sink)
    if sink.has_errors():
        return result
    result.stages_run.append(Stage.EFFECTS)
    for name, checked in result.checked.programs.items():
        program = lower_program(result.checked, checked)
        check_effects(program, sink)
        result.programs[name] = program
    if sink.has_errors():
        result.programs = {}
    return result


def compile_file(path: Path) -> CompileResult:
    return compile_source(path.read_text(encoding="utf-8"), str(path))
