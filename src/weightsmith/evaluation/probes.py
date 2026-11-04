"""Probe suites: the declared, finite evidence base for a behaviour.

A behaviour in Weightsmith is *operational*: a probe suite plus a metric. Declaring
``behavior parity = probes("parity.json", metric = accuracy)`` says nothing about where or how the
model computes parity; it only fixes how parity is measured.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from weightsmith.errors import LoadError

PROBE_FORMAT = "weightsmith.probes/v0"


class ProbeExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: list[float]
    label: int = Field(ge=0)


class ProbeSuite(BaseModel):
    """A classification probe suite. ``output_slice`` selects the logits the metric reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: str = PROBE_FORMAT
    name: str
    description: str = ""
    output_slice: tuple[int, int] | None = None
    examples: list[ProbeExample] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> ProbeSuite:
        if self.format != PROBE_FORMAT:
            raise ValueError(f"unsupported probe format '{self.format}', expected {PROBE_FORMAT}")
        widths = {len(e.input) for e in self.examples}
        if len(widths) != 1:
            raise ValueError("all examples must have the same input width")
        if self.output_slice is not None:
            start, stop = self.output_slice
            if not 0 <= start < stop:
                raise ValueError("output_slice must satisfy 0 <= start < stop")
            classes = stop - start
            if any(e.label >= classes for e in self.examples):
                raise ValueError(f"a label is outside the {classes} classes of output_slice")
        return self

    @property
    def input_width(self) -> int:
        return len(self.examples[0].input)

    @property
    def n(self) -> int:
        return len(self.examples)


def load_probe_suite(path: Path) -> ProbeSuite:
    if not path.is_file():
        raise LoadError(f"probe suite {path} does not exist")
    try:
        return ProbeSuite.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LoadError(f"{path} is not a valid probe suite: {exc}") from exc
