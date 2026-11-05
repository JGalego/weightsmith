"""OpenAI models as editors (optional extra ``openai``).

Two APIs are supported:

* ``responses`` (the default for api.openai.com): the Responses API. Reasoning models need it to
  call tools while reasoning, and it is the only one that accepts every effort level. Requests are
  sent with ``store: false`` and ask for the encrypted reasoning items, which are passed back on the
  next turn; reasoning carries over between turns without any state on the server, so every
  request is complete in itself and a recording replays exactly.
* ``chat``: Chat Completions (the default when ``base_url`` is set), which any compatible server
  speaks, so open models served locally can act as editors too.
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

_CHAT_STOPS = {
    "stop": "end",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}

# A request can reason for a long time before its first output token; wait for it.
REQUEST_TIMEOUT = 3600.0


def api_of(settings: dict[str, Any]) -> str:
    return str(settings.get("api") or ("chat" if settings.get("base_url") else "responses"))


def live_transport(settings: dict[str, Any]) -> Transport:
    try:
        import openai
    except ImportError as exc:
        raise ProviderError(
            "the openai provider needs the 'openai' extra: pip install 'weightsmith[openai]'"
        ) from exc
    base_url = settings.get("base_url")
    # A local server usually ignores the key, but the SDK insists on one.
    key = os.environ.get("OPENAI_API_KEY") or ("not-needed" if base_url else None)
    client: Any = openai.OpenAI(  # see the anthropic adapter
        base_url=base_url, api_key=key, timeout=REQUEST_TIMEOUT
    )
    responses = api_of(settings) == "responses"

    def send(body: Body) -> Body:
        try:
            if responses:
                # Streamed, so that a long silence while the model reasons does not look like a
                # dropped connection.
                with client.responses.stream(**body) as stream:
                    final = stream.get_final_response()
                return final.model_dump(mode="json", exclude_none=True)
            return client.chat.completions.create(**body).model_dump(mode="json", exclude_none=True)
        except openai.APIStatusError as exc:
            raise ProviderError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(f"cannot reach the OpenAI API: {exc}") from exc

    return send


def open_openai(
    settings: dict[str, Any], system: str, tools: list[ToolSchema], transport: Transport
) -> OpenAIConversation | ResponsesConversation:
    if api_of(settings) == "responses":
        return ResponsesConversation(settings, system, tools, transport)
    return OpenAIConversation(settings, system, tools, transport)


class ResponsesConversation:
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
            "instructions": system,
            "max_output_tokens": int(settings.get("max_tokens", 64000)),
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if tools:
            self.base["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ]
        if settings.get("effort"):
            self.base["reasoning"] = {"effort": settings["effort"]}
        self.items: list[dict[str, Any]] = []

    def send(self, text: str | None = None, results: tuple[ToolResult, ...] = ()) -> Turn:
        for r in results:
            output = f"ERROR: {r.content}" if r.is_error else r.content
            self.items.append(
                {"type": "function_call_output", "call_id": r.call_id, "output": output}
            )
        if text is not None:
            self.items.append({"role": "user", "content": text})
        response = self.transport({**self.base, "input": list(self.items)})
        # Everything the model produced, reasoning included, goes back with the next request.
        self.items.extend(dict(item) for item in response.get("output") or [])
        return parse_responses(response)


def parse_responses(response: Body) -> Turn:
    texts: list[str] = []
    calls: list[ToolCall] = []
    refusal = ""
    for item in response.get("output") or []:
        kind = item.get("type")
        if kind == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(str(part.get("text") or ""))
                elif part.get("type") == "refusal":
                    refusal = str(part.get("refusal") or "refused")
        elif kind == "function_call":
            calls.append(_call(str(item.get("call_id")), item.get("name"), item.get("arguments")))
    usage = response.get("usage") or {}
    reason = (response.get("incomplete_details") or {}).get("reason")
    if refusal or reason == "content_filter":
        stop, detail = "refusal", refusal or "content_filter"
    elif reason == "max_output_tokens":
        stop, detail = "max_tokens", ""
    else:
        stop, detail = ("tool_use" if calls else "end"), ""
    return Turn(
        "\n".join(t for t in texts if t).strip(),
        tuple(calls),
        stop,
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
        detail,
    )


class OpenAIConversation:
    """Chat Completions."""

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
            "max_completion_tokens": int(settings.get("max_tokens", 64000)),
        }
        if tools:
            self.base["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        if settings.get("effort"):
            self.base["reasoning_effort"] = settings["effort"]
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
    calls = [
        _call(
            str(call.get("id")),
            (call.get("function") or {}).get("name"),
            (call.get("function") or {}).get("arguments"),
        )
        for call in message.get("tool_calls") or []
    ]
    usage = response.get("usage") or {}
    return Turn(
        str(message.get("content") or "").strip(),
        tuple(calls),
        _CHAT_STOPS.get(str(choice.get("finish_reason")), "end"),
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def _call(call_id: str, name: Any, arguments: Any) -> ToolCall:
    try:
        parsed = json.loads(arguments or "{}")
        error = None if isinstance(parsed, dict) else "the arguments are not a JSON object"
    except json.JSONDecodeError as exc:
        parsed, error = None, f"invalid JSON arguments: {exc.msg}"
    return ToolCall(call_id, str(name), parsed if isinstance(parsed, dict) else None, error)
