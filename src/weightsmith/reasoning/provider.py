"""Editors: language models (or scripts) that act on a model's weights through tools.

A ``delegate`` statement opens a :class:`Conversation` with an editor. The runtime sends a
briefing, the editor answers with tool calls, the runtime executes them against the working copy
and sends the results back, until the editor stops calling tools or a budget runs out.

Every provider adapter turns a conversation step into a request *body* (a JSON-compatible dict)
and hands it to a :data:`Transport`, which returns the response as a dict. That single boundary is
where runs are recorded and replayed:

* :class:`RecordingTransport` stores every exchange, keyed by the SHA-256 of the canonical request.
* :class:`ReplayTransport` answers from a recording without contacting the provider. If a request
  differs from the recorded one (because the weights, the tools or the program changed), it raises
  :class:`ReplayDivergence` instead of answering, so a replay is exact or it fails.

What an editor says enters the run at evidence level ``hypothesised``. Only the runtime's own
measurements can raise it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from weightsmith.errors import WeightsmithError

TRANSCRIPT_FORMAT = "weightsmith.transcript/v0"

Body = dict[str, Any]
Transport = Callable[[Body], Body]


class ProviderError(WeightsmithError):
    """The provider could not be reached or used (missing SDK, credentials, API error)."""


class ReplayDivergence(WeightsmithError):
    """A replayed run sent a request the recording does not contain."""


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments object


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | None
    error: str | None = None  # set when the arguments could not be parsed


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Turn:
    """One editor response."""

    text: str
    calls: tuple[ToolCall, ...]
    stop: str  # "end" | "tool_use" | "max_tokens" | "refusal"
    input_tokens: int = 0
    output_tokens: int = 0
    detail: str = ""  # for example the refusal category


class Conversation(Protocol):
    def send(self, text: str | None = None, results: tuple[ToolResult, ...] = ()) -> Turn:
        """Send a user message, or the results of the previous turn's tool calls."""
        ...


def request_hash(body: Body) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(slots=True)
class Transcript:
    """Every exchange of one delegation, written as ``weightsmith.transcript/v0`` JSON."""

    editor: str
    provider: str
    settings: dict[str, Any]
    exchanges: list[dict[str, Any]] = field(default_factory=list)
    last_request: Body | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": TRANSCRIPT_FORMAT,
            "editor": self.editor,
            "provider": self.provider,
            "settings": self.settings,
            "exchanges": self.exchanges,
            "last_request": self.last_request,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, default=str) + "\n", encoding="utf-8")

    @staticmethod
    def read(path: Path) -> Transcript:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayDivergence(f"cannot read transcript {path}: {exc}") from exc
        if data.get("format") != TRANSCRIPT_FORMAT:
            raise ReplayDivergence(f"{path} is not a {TRANSCRIPT_FORMAT} transcript")
        return Transcript(data["editor"], data["provider"], data["settings"], data["exchanges"])


class RecordingTransport:
    """Forward requests to ``inner`` and record each exchange, writing after every call."""

    def __init__(self, inner: Transport, transcript: Transcript, path: Path | None) -> None:
        self.inner = inner
        self.transcript = transcript
        self.path = path

    def __call__(self, body: Body) -> Body:
        response = self.inner(body)
        self.transcript.exchanges.append(
            {"request_sha256": request_hash(body), "response": response}
        )
        self.transcript.last_request = body
        if self.path is not None:
            self.transcript.write(self.path)
        return response


class ReplayTransport:
    """Answer from a recording. Diverging requests raise :class:`ReplayDivergence`."""

    def __init__(self, recorded: Transcript) -> None:
        self.recorded = recorded
        self.position = 0

    def __call__(self, body: Body) -> Body:
        exchanges = self.recorded.exchanges
        if self.position >= len(exchanges):
            raise ReplayDivergence(
                f"the run made more editor calls than the {len(exchanges)} recorded"
            )
        exchange = exchanges[self.position]
        digest = request_hash(body)
        if digest != exchange["request_sha256"]:
            raise ReplayDivergence(
                f"editor call {self.position + 1} differs from the recording "
                f"({digest[:12]} vs {exchange['request_sha256'][:12]}): the weights, tools, "
                "budgets or program changed since it was recorded"
            )
        self.position += 1
        return exchange["response"]
