"""Models behind the OpenAI Chat Completions API (optional extra ``openai``).

``base_url`` points the adapter at any server that speaks the same API, such as a local
inference server, so open models can act as editors too.
"""

from __future__ import annotations

import json
import os
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

_STOPS = {
    "stop": "end",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def live_transport(base_url: str | None = None) -> Transport:
    try:
        import openai
    except ImportError as exc:
        raise ProviderError(
            "the openai provider needs the 'openai' extra: pip install 'weightsmith[openai]'"
        ) from exc
    # A local server usually ignores the key, but the SDK insists on one.
    key = os.environ.get("OPENAI_API_KEY") or ("not-needed" if base_url else None)
    client: Any = openai.OpenAI(base_url=base_url, api_key=key)  # see the anthropic adapter

    def send(body: Body) -> Body:
        try:
            return client.chat.completions.create(**body).model_dump(mode="json", exclude_none=True)
        except openai.APIStatusError as exc:
            raise ProviderError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(f"cannot reach the OpenAI API: {exc}") from exc

    return send


class OpenAIConversation:
    def __init__(
        self,
        settings: dict[str, Any],
        system: str,
        tools: list[ToolSchema],
        transport: Transport,
    ) -> None:
        self.transport = transport
        self.base: Body = {
            "model": str(settings["model"]),
            "max_completion_tokens": int(settings.get("max_tokens", 16000)),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ],
        }
        if settings.get("reasoning_effort"):
            self.base["reasoning_effort"] = settings["reasoning_effort"]
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    def send(self, text: str | None = None, results: tuple[ToolResult, ...] = ()) -> Turn:
        for r in results:
            content = f"ERROR: {r.content}" if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": content})
        if text is not None:
            self.messages.append({"role": "user", "content": text})
        response = self.transport({**self.base, "messages": list(self.messages)})
        choice = (response.get("choices") or [{}])[0]
        message = dict(choice.get("message") or {})
        message.setdefault("role", "assistant")
        self.messages.append(message)
        return parse_response(response)


def parse_response(response: Body) -> Turn:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
            error = None if isinstance(arguments, dict) else "the arguments are not a JSON object"
        except json.JSONDecodeError as exc:
            arguments, error = None, f"invalid JSON arguments: {exc.msg}"
        calls.append(
            ToolCall(
                str(call.get("id")),
                str(function.get("name")),
                arguments if isinstance(arguments, dict) else None,
                error,
            )
        )
    usage = response.get("usage") or {}
    return Turn(
        str(message.get("content") or "").strip(),
        tuple(calls),
        _STOPS.get(str(choice.get("finish_reason")), "end"),
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )
