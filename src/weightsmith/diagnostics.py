"""Source locations and compiler diagnostics shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open region of a source file. Lines and columns are 1-based."""

    file: str
    line: int
    column: int
    end_line: int
    end_column: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"

    def to_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> Span:
        return Span(
            file=str(data["file"]),
            line=int(data["line"]),  # type: ignore[arg-type]
            column=int(data["column"]),  # type: ignore[arg-type]
            end_line=int(data["end_line"]),  # type: ignore[arg-type]
            end_column=int(data["end_column"]),  # type: ignore[arg-type]
        )


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Stage(StrEnum):
    """The pipeline stage that produced a diagnostic."""

    SYNTAX = "syntax"
    NAMES = "names"
    TYPES = "types"
    EFFECTS = "effects"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    severity: Severity
    stage: Stage
    code: str
    message: str
    span: Span | None = None
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class DiagnosticSink:
    """Collects diagnostics. Stages report into a sink instead of raising on the first problem."""

    items: list[Diagnostic] = field(default_factory=list)

    def error(self, stage: Stage, code: str, message: str, span: Span | None, *notes: str) -> None:
        self.items.append(Diagnostic(Severity.ERROR, stage, code, message, span, tuple(notes)))

    def warning(
        self, stage: Stage, code: str, message: str, span: Span | None, *notes: str
    ) -> None:
        self.items.append(Diagnostic(Severity.WARNING, stage, code, message, span, tuple(notes)))

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.items if d.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.items if d.severity is Severity.WARNING]

    def has_errors(self, stage: Stage | None = None) -> bool:
        return any(d.stage is stage or stage is None for d in self.errors)


def render(diagnostic: Diagnostic, source: str | None = None) -> str:
    """Render a diagnostic as plain text with a source excerpt and caret underline."""
    lines = [f"{diagnostic.severity.value}[{diagnostic.code}]: {diagnostic.message}"]
    span = diagnostic.span
    if span is not None:
        lines.append(f"  --> {span}")
        text = source
        if text is None:
            try:
                text = Path(span.file).read_text(encoding="utf-8")
            except OSError:
                text = None
        if text is not None:
            source_lines = text.splitlines()
            if 0 < span.line <= len(source_lines):
                line_text = source_lines[span.line - 1]
                gutter = len(str(span.line))
                end = span.end_column if span.end_line == span.line else len(line_text) + 1
                width = max(1, end - span.column)
                lines.append(f"{' ' * gutter} |")
                lines.append(f"{span.line} | {line_text}")
                lines.append(f"{' ' * gutter} | {' ' * (span.column - 1)}{'^' * width}")
    lines.extend(f"  = note: {note}" for note in diagnostic.notes)
    return "\n".join(lines)
