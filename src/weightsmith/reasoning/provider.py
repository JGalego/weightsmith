"""The reasoning-provider interface (no implementation ships in 0.1).

A reasoning provider, such as an LLM, may *propose* hypotheses about a model: candidate locations
for a behaviour, or candidate transformations. It never executes anything. Its contract:

* Programs never depend on which provider is configured. A provider is selected by the runtime
  and gated by the ``reason`` effect, which is forbidden by default because it may send model
  data off the machine.
* Every response is recorded as a :class:`Hypothesis` artifact. A run can be replayed from the
  recording without contacting the provider, which keeps runs deterministic and auditable.
* Proposals enter the program at evidence level ``hypothesised``. Only experiments run by the
  runtime can raise that level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from weightsmith.evidence import EvidenceLevel, Location


@dataclass(frozen=True, slots=True)
class HypothesisRequest:
    question: str
    model_id: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    request: HypothesisRequest
    provider: str
    claim: str
    locations: tuple[Location, ...] = ()
    proposed_operations: tuple[dict[str, Any], ...] = ()
    raw_response: str = ""
    level: EvidenceLevel = EvidenceLevel.HYPOTHESISED


class ReasoningProvider(Protocol):
    name: str

    def propose(self, request: HypothesisRequest) -> Hypothesis:
        """Return a hypothesis. Must not modify any model."""
        ...
