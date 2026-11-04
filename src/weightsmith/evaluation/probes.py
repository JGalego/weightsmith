"""Probe suites: the declared, finite evidence base for a behaviour.

A behaviour in Weightsmith is *operational*: a probe suite plus a metric. Declaring
``behavior parity = probes("parity.json", metric = accuracy)`` says nothing about where or how the
model computes parity; it only fixes how parity is measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from weightsmith.errors import LoadError

PROBE_FORMAT = "weightsmith.probes/v0"


class ProbeExample(BaseModel):
    """A vector example for a ``kind: "vectors"`` suite: an input and a class label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: list[float]
    label: int = Field(ge=0)


class TokenExample(BaseModel):
    """A next-token example for a ``kind: "tokens"`` suite.

    The prompt is text (tokenized with the model's tokenizer, special tokens included) or token
    ids. The target is the single token the model should predict next: text that tokenizes to
    exactly one token (without special tokens), or a token id.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str | None = None
    prompt_ids: list[int] | None = Field(default=None, min_length=1)
    target: str | None = None
    target_id: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _one_of_each(self) -> TokenExample:
        if (self.prompt is None) == (self.prompt_ids is None):
            raise ValueError("give exactly one of 'prompt' and 'prompt_ids'")
        if (self.target is None) == (self.target_id is None):
            raise ValueError("give exactly one of 'target' and 'target_id'")
        if self.prompt_ids is not None and min(self.prompt_ids) < 0:
            raise ValueError("token ids are non-negative")
        return self

    @property
    def needs_tokenizer(self) -> bool:
        return self.prompt is not None or self.target is not None


class ProbeSuite(BaseModel):
    """A probe suite: the finite, declared evidence base of a behaviour.

    ``kind: "vectors"`` (the default) is classification over input vectors; ``output_slice``
    selects the logits the metric reads. ``kind: "tokens"`` is next-token prediction for language
    models; ``candidates`` (or ``candidate_ids``) optionally restricts the prediction to a set of
    tokens, which makes the suite a multiple-choice test.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: str = PROBE_FORMAT
    kind: Literal["vectors", "tokens"] = "vectors"
    name: str
    description: str = ""
    output_slice: tuple[int, int] | None = None
    candidates: list[str] | None = Field(default=None, min_length=2)
    candidate_ids: list[int] | None = Field(default=None, min_length=2)
    examples: list[ProbeExample | TokenExample] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> ProbeSuite:
        if self.format != PROBE_FORMAT:
            raise ValueError(f"unsupported probe format '{self.format}', expected {PROBE_FORMAT}")
        wanted = ProbeExample if self.kind == "vectors" else TokenExample
        if any(not isinstance(e, wanted) for e in self.examples):
            fields = "'input' and 'label'" if self.kind == "vectors" else "a prompt and a target"
            raise ValueError(f"every example of a '{self.kind}' suite needs {fields}")
        if self.kind == "tokens":
            if self.output_slice is not None:
                raise ValueError("output_slice applies to 'vectors' suites only")
            if self.candidates is not None and self.candidate_ids is not None:
                raise ValueError("give at most one of 'candidates' and 'candidate_ids'")
            return self
        if self.candidates is not None or self.candidate_ids is not None:
            raise ValueError("candidates apply to 'tokens' suites only")
        widths = {len(e.input) for e in self.vector_examples}
        if len(widths) != 1:
            raise ValueError("all examples must have the same input width")
        if self.output_slice is not None:
            start, stop = self.output_slice
            if not 0 <= start < stop:
                raise ValueError("output_slice must satisfy 0 <= start < stop")
            classes = stop - start
            if any(e.label >= classes for e in self.vector_examples):
                raise ValueError(f"a label is outside the {classes} classes of output_slice")
        return self

    @property
    def vector_examples(self) -> list[ProbeExample]:
        return [e for e in self.examples if isinstance(e, ProbeExample)]

    @property
    def token_examples(self) -> list[TokenExample]:
        return [e for e in self.examples if isinstance(e, TokenExample)]

    @property
    def needs_tokenizer(self) -> bool:
        return self.candidates is not None or any(e.needs_tokenizer for e in self.token_examples)

    def max_token_id(self) -> int | None:
        """The largest token id written literally in the suite, if any."""
        ids = [i for e in self.token_examples for i in (e.prompt_ids or [])]
        ids += [e.target_id for e in self.token_examples if e.target_id is not None]
        ids += self.candidate_ids or []
        return max(ids) if ids else None

    @property
    def input_width(self) -> int:
        return len(self.vector_examples[0].input) if self.kind == "vectors" else 0

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
