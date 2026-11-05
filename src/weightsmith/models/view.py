"""How a model's weights are shown to an editor as text, and how its edits name them.

A *view* is optional metadata in a model directory (``view.json``). It lists the tensors an editor
sees, in order, each with the name the editor knows it by (``W1``, ``actor0.W``), whether it is
shown transposed, and the header line above it. Edits the editor writes against those names, such
as ``actor0.W[6, 122] = 80``, are mapped back through the same view, so what is shown and what is
edited always agree. Without a view, every parameter is shown under its own name, unchanged.

A *weight format* decides how the numbers are written:

* ``decimals(n)``: every value with ``n`` decimals (``-0.000`` is written ``0.000``).
* ``int_scale(n)``: each tensor as integers times one power-of-ten scale, chosen so that the
  tensor's largest entry keeps ``n`` significant digits.

Both are lossy. What an editor reads is the rounded network, so a program that wants the editor's
edits applied to exactly what it saw should load the rounded network as its target.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from weightsmith.errors import LoadError

if TYPE_CHECKING:  # NumPy is imported where it is used, so `check` can parse formats cheaply
    import numpy as np

VIEW_FILE = "view.json"
VIEW_FORMAT = "weightsmith.view/v0"
FORMATS = ("decimals", "int_scale")


@dataclass(frozen=True, slots=True)
class ViewTensor:
    label: str  # the name the editor sees
    param: str  # the parameter in the weight set
    transpose: bool = False
    header: str | None = None  # template; see :func:`render`


@dataclass(frozen=True, slots=True)
class View:
    tensors: tuple[ViewTensor, ...]

    def get(self, label: str) -> ViewTensor | None:
        return next((t for t in self.tensors if t.label == label), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": VIEW_FORMAT,
            "tensors": [
                {
                    "label": t.label,
                    "param": t.param,
                    **({"transpose": True} if t.transpose else {}),
                    **({"header": t.header} if t.header is not None else {}),
                }
                for t in self.tensors
            ],
        }


def default_view(names: list[str]) -> View:
    return View(tuple(ViewTensor(name, name) for name in names))


def read_view(model_dir: Path) -> View | None:
    path = model_dir / VIEW_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != VIEW_FORMAT:
            raise ValueError(f"not a {VIEW_FORMAT} file")
        tensors = tuple(
            ViewTensor(
                str(t["label"]),
                str(t["param"]),
                bool(t.get("transpose", False)),
                None if t.get("header") is None else str(t["header"]),
            )
            for t in data["tensors"]
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LoadError(f"cannot read {path}: {exc}") from exc
    labels = [t.label for t in tensors]
    if len(set(labels)) != len(labels):
        raise LoadError(f"{path}: a label appears twice")
    return View(tensors)


@dataclass(frozen=True, slots=True)
class WeightFormat:
    kind: str
    digits: int

    def __str__(self) -> str:
        return f"{self.kind}({self.digits})"


_FORMAT = re.compile(r"^\s*([a-z_]+)\s*\(\s*(\d+)\s*\)\s*$")


def parse_format(text: str) -> WeightFormat:
    """Parse ``decimals(3)`` or ``int_scale(3)``; raise ``ValueError`` otherwise."""
    match = _FORMAT.match(text)
    if match is None or match.group(1) not in FORMATS:
        raise ValueError(f"'{text}' is not a weight format; use decimals(n) or int_scale(n)")
    digits = int(match.group(2))
    if not 1 <= digits <= 8:
        raise ValueError("the number of digits must be between 1 and 8")
    return WeightFormat(match.group(1), digits)


def shown(array: np.ndarray, tensor: ViewTensor) -> np.ndarray:
    """The array as the editor sees it (transposed if the view says so)."""
    return array.T if tensor.transpose and array.ndim == 2 else array


def scale_exponent(array: np.ndarray, digits: int) -> int:
    """The power of ten that keeps ``digits`` significant digits of the largest entry."""
    import numpy as np

    top = float(np.max(np.abs(array))) if array.size else 0.0
    if top == 0.0 or not math.isfinite(top):
        return 0
    return math.floor(math.log10(top)) - (digits - 1)


def _decimals(values: np.ndarray, digits: int) -> list[str]:
    zero = "-" + format(0.0, f".{digits}f")
    return [s[1:] if s == zero else s for s in (format(float(v), f".{digits}f") for v in values)]


def render(view: View, fmt: WeightFormat, arrays: Mapping[str, np.ndarray]) -> str:
    """Every tensor of ``view`` as text, separated by blank lines.

    A header template may use ``{label}``, ``{shape}`` (``500 x 32``), ``{rows}``, ``{cols}`` and,
    for ``int_scale``, ``{scale}`` (``1e-3``). Matrices are written one line per row; with
    ``int_scale`` each row starts with its index (``7: ...``).
    """
    import numpy as np

    blocks = []
    for tensor in view.tensors:
        array = shown(np.asarray(arrays[tensor.param], dtype=np.float64), tensor)
        matrix = array.reshape(1, -1) if array.ndim <= 1 else array.reshape(array.shape[0], -1)
        fields: dict[str, Any] = {
            "label": tensor.label,
            "shape": " x ".join(str(d) for d in array.shape),
            "rows": array.shape[0] if array.ndim else 1,
            "cols": array.shape[1] if array.ndim > 1 else (array.shape[0] if array.ndim else 1),
        }
        if fmt.kind == "decimals":
            header = tensor.header or "{label} ({shape}):"
            lines = [" ".join(_decimals(row, fmt.digits)) for row in matrix]
        else:
            exponent = scale_exponent(array, fmt.digits)
            fields["scale"] = f"1e{exponent}"
            header = tensor.header or "{label} ({shape}), scale {scale}:"
            ints = np.rint(matrix / 10.0**exponent).astype(np.int64)
            lines = [" ".join(str(int(v)) for v in row) for row in ints]
            if array.ndim > 1:
                lines = [f"{i}: {line}" for i, line in enumerate(lines)]
        blocks.append("\n".join([header.format(**fields), *lines]))
    return "\n\n".join(blocks)


def round_like_shown(array: np.ndarray, fmt: WeightFormat) -> np.ndarray:
    """The values an editor reads in format ``fmt``, as numbers (orientation unchanged). A value
    that rounds to zero is +0, as the text shows it, never -0."""
    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    if fmt.kind == "decimals":
        return np.round(values, fmt.digits) + 0.0
    exponent = scale_exponent(values, fmt.digits)
    return np.rint(values / 10.0**exponent) * 10.0**exponent + 0.0
