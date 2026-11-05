"""The edit language of ``edits`` blocks: weight edits written as numbers, with no tools.

An editor that cannot run code writes its edits in a fenced block, one statement per line::

    actor_head.b[0:3] = [0.5, 2.0, -1.0]   # explicit values along a range
    actor_head.W[7, 0:3] = 0               # fill a region with one value
    embed.W[26, 5] = 1.25                  # one entry
    actor0.W[:, 3] *= 0.5                  # scale a region (also += and -=)
    ZERO actor4.W                          # set a whole tensor to 0

Tensors are named, and indexed, as the weight view shows them (``view.json``). An index is an
integer, a half-open range ``a:b`` (either end may be left out) or ``:`` for the whole axis;
missing trailing indices mean whole axes. A list gives one value per entry of the region in
row-major order and may continue over the following lines. ``#`` starts a comment. Values are
plain decimal numbers with an absolute value of at most 1e6. Statements apply in order, so a later
one overwrites an earlier one, and a statement that cannot be read is skipped and reported while
the others still apply. This is the language of the "Sparks of weight-space programming"
experiments (Takehi, 2026).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAX_ABS_VALUE = 1e6

_LABEL = r"[A-Za-z_][A-Za-z0-9_.]*"
_ASSIGN = re.compile(rf"^({_LABEL})\s*(?:\[([^\]]*)\])?\s*(=|\*=|\+=|-=)\s*(.+)$", re.DOTALL)
_ZERO = re.compile(rf"^ZERO\s+({_LABEL})$")
_INT = re.compile(r"^-?\d+$")


@dataclass(slots=True)
class EditReport:
    statements: int = 0
    applied: int = 0
    entries_written: int = 0
    errors: list[str] = field(default_factory=list)
    tensors_touched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "statements": self.statements,
            "applied": self.applied,
            "entries_written": self.entries_written,
            "errors": list(self.errors),
            "tensors_touched": sorted(self.tensors_touched),
        }

    def summary(self) -> str:
        text = (
            f"{self.applied} of {self.statements} statements applied, "
            f"{self.entries_written} entries written"
        )
        if self.errors:
            text += "; skipped:\n" + "\n".join(f"- {e}" for e in self.errors)
        return text


class EditError(ValueError):
    pass


def statements(text: str) -> list[tuple[int, str]]:
    """The statements of an edits block with their line numbers, comments removed and continued
    lists joined."""
    out: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if pending is not None:
            line = f"{pending[1]} {line}".strip()
            number = pending[0]
            pending = None
        if not line:
            continue
        if line.count("[") > line.count("]"):
            pending = (number, line)
            continue
        out.append((number, line))
    if pending is not None:
        out.append(pending)
    return out


def apply_edits(text: str, arrays: dict[str, np.ndarray]) -> EditReport:
    """Apply an edits block to ``arrays`` (float64, keyed by view label), in place."""
    report = EditReport()
    touched: set[str] = set()
    for number, statement in statements(text):
        report.statements += 1
        try:
            label, written = _apply(statement, arrays)
        except EditError as exc:
            report.errors.append(f"line {number}: {exc} ({_clip(statement)})")
            continue
        report.applied += 1
        report.entries_written += written
        touched.add(label)
    report.tensors_touched = sorted(touched)
    return report


def _apply(statement: str, arrays: dict[str, np.ndarray]) -> tuple[str, int]:
    zero = _ZERO.match(statement)
    if zero:
        label = zero.group(1)
        array = _tensor(arrays, label)
        array[...] = 0.0
        return label, int(array.size)
    match = _ASSIGN.match(statement)
    if match is None:
        raise EditError("not a statement")
    label, index_text, op, value_text = match.groups()
    array = _tensor(arrays, label)
    index = _index(index_text, array.shape) if index_text is not None else ()
    region = array[index]
    size = int(np.size(region))
    if size == 0:
        raise EditError("the region is empty")
    value = _values(value_text.strip(), size)
    shaped = value if np.isscalar(value) else np.reshape(value, np.shape(region))
    if op == "=":
        array[index] = shaped
    elif op == "*=":
        array[index] = region * shaped
    elif op == "+=":
        array[index] = region + shaped
    else:
        array[index] = region - shaped
    return label, size


def _tensor(arrays: dict[str, np.ndarray], label: str) -> np.ndarray:
    if label not in arrays:
        raise EditError(f"no tensor '{label}'")
    return arrays[label]


def _index(text: str, shape: tuple[int, ...]) -> tuple[int | slice, ...]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) > len(shape):
        raise EditError(f"{len(parts)} indices for a tensor with {len(shape)} dimension(s)")
    out: list[int | slice] = []
    for axis, part in enumerate(parts):
        size = shape[axis]
        if _INT.match(part):
            i = int(part)
            if not -size <= i < size:
                raise EditError(f"index {i} is out of range for an axis of {size}")
            out.append(i)
            continue
        ends = part.split(":")
        if len(ends) != 2 or not all(e.strip() == "" or _INT.match(e.strip()) for e in ends):
            raise EditError(f"bad index '{part}'")
        start, stop = (int(e) if e.strip() else None for e in ends)
        out.append(slice(start, stop))
    return tuple(out)


def _values(text: str, size: int) -> float | np.ndarray:
    if text.startswith("["):
        if not text.endswith("]"):
            raise EditError("unclosed list")
        items = [v for v in re.split(r"[\s,\[\]]+", text) if v]
        values = np.array([_number(v) for v in items], dtype=np.float64)
        if values.size != size:
            raise EditError(f"{values.size} values for a region of {size} entries")
        return values
    return _number(text)


def _number(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise EditError(f"'{_clip(text)}' is not a number") from None
    if not np.isfinite(value) or abs(value) > MAX_ABS_VALUE:
        raise EditError(f"{text} is not a finite number of absolute value at most 1e6")
    return value


def _clip(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = ["MAX_ABS_VALUE", "EditError", "EditReport", "apply_edits", "statements"]
