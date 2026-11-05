"""Editors: language models (or scripts) that act on weights through tools.

See :mod:`weightsmith.reasoning.provider` for the conversation and recording model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from weightsmith.reasoning.provider import (
    Conversation,
    ProviderError,
    ToolSchema,
    Transport,
)
from weightsmith.registry import PROVIDERS


def missing_extra(provider: str) -> str | None:
    """The extra a live provider needs, if it is not installed."""
    extra = PROVIDERS[provider].extra
    if extra is not None and importlib.util.find_spec(extra) is None:
        return extra
    return None


def live_transport(provider: str, settings: dict[str, Any]) -> Transport:
    if provider == "anthropic":
        from weightsmith.reasoning import anthropic

        return anthropic.live_transport()
    if provider == "openai":
        from weightsmith.reasoning import openai

        return openai.live_transport(settings)
    if provider == "scripted":
        from weightsmith.reasoning import scripted

        return scripted.live_transport(Path(str(settings["script"])))
    raise ProviderError(f"unknown provider '{provider}'")


def open_conversation(
    provider: str,
    settings: dict[str, Any],
    system: str,
    tools: list[ToolSchema],
    transport: Transport,
) -> Conversation:
    if provider == "anthropic":
        from weightsmith.reasoning.anthropic import AnthropicConversation

        return AnthropicConversation(settings, system, tools, transport)
    if provider == "openai":
        from weightsmith.reasoning.openai import open_openai

        return open_openai(settings, system, tools, transport)
    if provider == "scripted":
        from weightsmith.reasoning.scripted import ScriptedConversation

        return ScriptedConversation(settings, system, tools, transport)
    raise ProviderError(f"unknown provider '{provider}'")
