"""A deterministic editor that plays back a script of tool calls.

Script format (``weightsmith.script/v0``)::

    {"format": "weightsmith.script/v0",
     "turns": [{"text": "Looking first.", "calls": [{"name": "describe", "arguments": {}}]},
               {"text": "Done: rescaled two neurons."}]}

Each request is answered with the next turn, whatever the tool results were. When the script
runs out, the editor stops. Requests and responses go through the same transport as a live
provider, so scripted runs are recorded and replayed like any other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weightsmith.reasoning.provider import (
    Body,
    ProviderError,
    ToolCall,
    ToolResult,
    ToolSchema,
    Transport,
    Turn,
)

SCRIPT_FORMAT = "weightsmith.script/v0"


def live_transport(script: Path) -> Transport:
    try:
        data = json.loads(script.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderError(f"cannot read script {script}: {exc}") from exc
    if data.get("format") != SCRIPT_FORMAT or not isinstance(data.get("turns"), list):
        raise ProviderError(f"{script} is not a {SCRIPT_FORMAT} script")
    turns: list[dict[str, Any]] = data["turns"]

    def send(body: Body) -> Body:
        index = int(body["turn"])
        if index >= len(turns):
            return {"text": "(the script has ended)", "calls": []}
        return turns[index]

    return send


class ScriptedConversation:
    def __init__(
        self,
        settings: dict[str, Any],
        system: str,
        tools: list[ToolSchema],
        transport: Transport,
    ) -> None:
        self.transport = transport
        self.turn = 0

    def send(self, text: str | None = None, results: tuple[ToolResult, ...] = ()) -> Turn:
        body: Body = {
            "turn": self.turn,
            "user": text,
            "results": [
                {"call_id": r.call_id, "content": r.content, "is_error": r.is_error}
                for r in results
            ],
        }
        response = self.transport(body)
        calls = tuple(
            ToolCall(f"call_{self.turn}_{i}", str(c["name"]), dict(c.get("arguments") or {}))
            for i, c in enumerate(response.get("calls") or [])
        )
        self.turn += 1
        return Turn(str(response.get("text", "")), calls, "tool_use" if calls else "end")
