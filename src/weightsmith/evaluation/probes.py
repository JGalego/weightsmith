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


class Rollouts(BaseModel):
    """Episodes of a policy network in a simulated environment, for a ``kind: "rollouts"`` suite.

    The episodes are run by a worker in its own environment (``weightsmith.evaluation.rollouts``).
    ``seed`` fixes the stream of episodes, so two suites with different seeds measure on different
    episodes. ``mode`` is how actions are chosen: sampled from the policy, or the most likely.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    env: Literal["kinetix"]
    level: str
    episodes: int = Field(ge=1)
    seed: int = 0
    mode: Literal["sample", "argmax"] = "sample"


# Problems per benchmark: the test split each job reads (``workers/hf_bench_job.py``).
BENCHMARK_PROBLEMS = {"math500": 500, "gsm8k": 1319}


class Benchmark(BaseModel):
    """A generative benchmark of a language model, for a ``kind: "benchmark"`` suite.

    The model answers every problem ``samples`` times and each answer is graded; the score is the
    share of correct answers. The problems are split into ``shards`` (problems i, i + k, ...),
    each one a job on Hugging Face Jobs with GPU ``flavor`` that may run for at most
    ``timeout_hours``; ``only_shards`` runs some of them only, as a pilot does. At most
    ``parallel`` jobs run at once. ``protocol`` fixes the prompt and the sampling
    (``weightsmith.evaluation.benchmarks``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["math500", "gsm8k"]
    samples: int = Field(ge=1)
    protocol: Literal["think35"] = "think35"
    shards: int = Field(default=1, ge=1)
    only_shards: list[int] | None = Field(default=None, min_length=1)
    flavor: str = "a100-large"
    timeout_hours: float = Field(default=2.0, gt=0, le=24)
    parallel: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _shards_exist(self) -> Benchmark:
        if self.shards > BENCHMARK_PROBLEMS[self.name]:
            raise ValueError(f"{self.name} has fewer problems than {self.shards} shards")
        if self.only_shards is not None:
            if len(set(self.only_shards)) != len(self.only_shards):
                raise ValueError("only_shards lists a shard twice")
            if not all(0 <= i < self.shards for i in self.only_shards):
                raise ValueError(f"only_shards must lie in 0..{self.shards - 1}")
        return self

    @property
    def selected(self) -> list[int]:
        return (
            sorted(self.only_shards) if self.only_shards is not None else list(range(self.shards))
        )

    @property
    def problems(self) -> int:
        """Problems in the selected shards."""
        total = BENCHMARK_PROBLEMS[self.name]
        return sum(len(range(i, total, self.shards)) for i in self.selected)

    @property
    def gpu_hours(self) -> float:
        """The most GPU time the evaluation can take: every job running to its timeout."""
        return len(self.selected) * self.timeout_hours


class ProbeSuite(BaseModel):
    """A probe suite: the finite, declared evidence base of a behaviour.

    ``kind: "vectors"`` (the default) is classification over input vectors; ``output_slice``
    selects the logits the metric reads. ``kind: "tokens"`` is next-token prediction for language
    models; ``candidates`` (or ``candidate_ids``) optionally restricts the prediction to a set of
    tokens, which makes the suite a multiple-choice test. ``kind: "rollouts"`` has no examples:
    it is a number of episodes of a policy network in an environment, each one a probe.
    ``kind: "benchmark"`` has none either: it is a generative benchmark of a language model, each
    sampled answer a probe.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: str = PROBE_FORMAT
    kind: Literal["vectors", "tokens", "rollouts", "benchmark"] = "vectors"
    name: str
    description: str = ""
    output_slice: tuple[int, int] | None = None
    candidates: list[str] | None = Field(default=None, min_length=2)
    candidate_ids: list[int] | None = Field(default=None, min_length=2)
    examples: list[ProbeExample | TokenExample] = Field(default_factory=list)
    rollouts: Rollouts | None = None
    benchmark: Benchmark | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ProbeSuite:
        if self.format != PROBE_FORMAT:
            raise ValueError(f"unsupported probe format '{self.format}', expected {PROBE_FORMAT}")
        for kind in ("rollouts", "benchmark"):
            entry = getattr(self, kind)
            if self.kind == kind:
                if entry is None or self.examples:
                    raise ValueError(f"a '{kind}' suite has a '{kind}' entry and no examples")
                if self.output_slice is not None or self.candidates or self.candidate_ids:
                    raise ValueError(f"output_slice and candidates do not apply to '{kind}' suites")
            elif entry is not None:
                raise ValueError(f"'{kind}' applies to '{kind}' suites only")
        if self.kind in ("rollouts", "benchmark"):
            return self
        if not self.examples:
            raise ValueError("a suite needs at least one example")
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
        if self.rollouts is not None:
            return self.rollouts.episodes
        if self.benchmark is not None:
            return self.benchmark.problems * self.benchmark.samples
        return len(self.examples)

    @property
    def gpu_hours(self) -> float:
        """The most GPU time one evaluation of the suite can take (0 when it runs here)."""
        return self.benchmark.gpu_hours if self.benchmark is not None else 0.0


def load_probe_suite(path: Path) -> ProbeSuite:
    if not path.is_file():
        raise LoadError(f"probe suite {path} does not exist")
    try:
        return ProbeSuite.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LoadError(f"{path} is not a valid probe suite: {exc}") from exc
