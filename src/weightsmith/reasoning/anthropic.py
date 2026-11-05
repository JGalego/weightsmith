"""Claude models through the Anthropic Messages API (optional extra ``anthropic``).

The conversation is append-only: each assistant response is stored exactly as the API returned
it (thinking blocks included) and sent back unchanged, which keeps the prompt cache warm and
thinking blocks valid. Requests stream (long tool inputs such as weight values or code would
otherwise sit behind a long silent gap), with ``eager_input_streaming`` on every tool; the tool
layer validates every input before running it. A refusal ends the delegation; no other model is
substituted.
"""

from __future__ import annotations

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
    "end_turn": "end",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


def live_transport() -> Transport:
    """Send requests with the official SDK; credentials resolve from the environment."""
    try:
        import anthropic
    except ImportError as exc:
        raise ProviderError(
            "the anthropic provider needs the 'anthropic' extra: "
            "pip install 'weightsmith[anthropic]'"
        ) from exc
    # Request bodies are plain dicts (they are what gets recorded), so the typed overloads of
    # the SDK add nothing here; Any keeps type checking fast.
    client: Any = anthropic.Anthropic()

    def send(body: Body) -> Body:
        for attempt in range(3):
            try:
                with client.messages.stream(**body) as stream:
                    return stream.get_final_message().to_dict(mode="json")
            except ValueError as exc:  # tool input JSON the SDK could not parse at all
                if attempt == 2:
                    raise ProviderError(f"unparseable tool input from the model: {exc}") from exc
            except anthropic.APIStatusError as exc:
                raise ProviderError(
                    f"Anthropic API error {exc.status_code}: {exc.message}"
                ) from exc
            except anthropic.APIConnectionError as exc:
                raise ProviderError(f"cannot reach the Anthropic API: {exc}") from exc
            except anthropic.APIError as exc:  # e.g. an error event in the middle of a stream
                raise ProviderError(f"Anthropic API error: {exc.message}") from exc
        raise AssertionError("unreachable")

    return send


class AnthropicConversation:
    def __init__(
        self,
        settings: dict[str, Any],
        system: str,
        tools: list[ToolSchema],
        transport: Transport,
    ) -> None:
        model = str(settings.get("model", "claude-opus-5-5"))
        self.transport = transport
        self.base: Body = {
            "model": model,
            "max_tokens": int(settings.get("max_tokens", 64000)),
            "system": system,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                    "eager_input_streaming": True,
                }
                for t in tools
            ],
            "output_config": {"effort": str(settings.get("effort", "high"))},
            "cache_control": {"type": "ephemeral"},
        }
        self.messages: list[dict[str, Any]] = []

    def send(self, text: str | None = None, results: tuple[ToolResult, ...] = ()) -> Turn:
        content: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": r.content,
                **({"is_error": True} if r.is_error else {}),
            }
            for r in results
        ]
        if text is not None:
            content.append({"type": "text", "text": text})
        self.messages.append({"role": "user", "content": content})
        response = self.transport({**self.base, "messages": list(self.messages)})
        blocks = response.get("content", [])
        self.messages.append({"role": "assistant", "content": blocks})
        return parse_response(response)


def parse_response(response: Body) -> Turn:
    blocks = response.get("content", [])
    text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
    calls = tuple(
        ToolCall(
            b["id"],
            b["name"],
            b.get("input") if isinstance(b.get("input"), dict) else None,
            None if isinstance(b.get("input"), dict) else "the input is not a JSON object",
        )
        for b in blocks
        if b.get("type") == "tool_use"
    )
    usage = response.get("usage") or {}
    input_tokens = sum(
        int(usage.get(k) or 0)
        for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    stop = _STOPS.get(str(response.get("stop_reason")), "end")
    details = response.get("stop_details") or {}
    return Turn(
        text,
        calls,
        stop,
        input_tokens,
        int(usage.get("output_tokens") or 0),
        str(details.get("category") or "") if stop == "refusal" else "",
    )
