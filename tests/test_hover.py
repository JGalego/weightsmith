"""Hover documentation: what the language server shows for each kind of word."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from weightsmith import registry
from weightsmith.compiler.effects import BUDGETS, Effect
from weightsmith.language import highlight
from weightsmith.language.hover import (
    BUDGET_DOCS,
    EFFECT_DOCS,
    FUNCTION_DOCS,
    KEYWORD_DOCS,
    hover,
)

SOURCE = """// a comment about the model
model Base = load("fixtures/tiny_mlp")
behavior parity = probes("parity.json", metric = accuracy)
editor S = provider("scripted", script = "s.json")
selection hidden = Base.params["0.*"]
policy {
    allow mutate(max_changed_params = 0.1)
    allow reason
}
program P {
    target Base
    find outlier_rows in hidden (threshold = 4.0)
    edit {
        scale hidden[0, :] (factor = 0.5)
    }
    repair (strategies = [rescale_outlier_rows])
    delegate S (task = "fix", tools = [find, python], practice = [parity])
    contract {
        improve parity (min_delta = 0.1)
        require changed_params (max = 0.2)
    }
    verify
    output { patch, evaluation }
}
"""


def at(needle: str) -> str | None:
    return hover(SOURCE, SOURCE.index(needle))


@pytest.mark.parametrize(
    ("needle", "expected"),
    [
        ("model Base", "**model** (keyword)"),
        ("load(", "**load** (built-in)"),
        ("metric =", "*Argument of `probes`.*"),
        ("accuracy", "**accuracy** (metric)"),
        ('"scripted"', "**scripted** (provider)"),
        ("script =", 'Argument of `provider("scripted")`'),
        ("mutate", "**mutate** (effect)"),
        ("max_changed_params", "budget of `mutate`"),
        ("outlier_rows", "**outlier_rows** (finder)"),
        ("threshold", "*Argument of `outlier_rows`.*"),
        ("scale", "**scale** (edit operation)"),
        ("factor", "*Argument of `scale`.*"),
        ("rescale_outlier_rows", "(strategy)"),
        ("tools", "*Argument of `delegate`.*"),
        ("find, python", "**find** (tool)"),
        ("python]", "**python** (tool)"),
        ("improve", "**improve** (clause)"),
        ("min_delta", "*Argument of `improve`.*"),
        ("changed_params (", "**changed_params** (contract item)"),
        ("max =", "*Argument of `changed_params`.*"),
        ("patch", "**patch** (output)"),
        ("hidden (", 'selection hidden = Base.params["0.*"]'),
        ("parity (", 'behavior parity = probes("parity.json", metric = accuracy)'),
    ],
)
def test_hover_explains_each_kind_of_word(needle: str, expected: str) -> None:
    text = at(needle)
    assert text is not None
    assert expected in text


def test_nothing_is_shown_for_comments_and_punctuation() -> None:
    assert at("a comment") is None
    assert at("{\n    allow") is None


def test_hover_works_in_a_program_that_does_not_parse() -> None:
    broken = "program P {\n  target Base\n  delegate S (max_steps = "
    assert "Most editor turns." in (hover(broken, broken.index("max_steps")) or "")


def test_every_keyword_function_effect_and_budget_is_documented() -> None:
    clauses = set(registry.CLAUSES)
    assert {k for k, v in KEYWORD_DOCS.items() if v} | clauses >= set(highlight.KEYWORDS)
    assert set(FUNCTION_DOCS) == set(highlight.FUNCTIONS)
    assert set(EFFECT_DOCS) == set(Effect)
    assert set(BUDGET_DOCS) == {b for budgets in BUDGETS.values() for b in budgets}


def _message(payload: dict[str, object]) -> bytes:
    body = json.dumps(payload).encode()
    return f"Content-Length: {len(body)}\r\n\r\n".encode() + body


def _read(stream: object) -> dict[str, object]:
    length = 0
    while (line := stream.readline().strip()) != b"":  # type: ignore[attr-defined]
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":")[1])
    return json.loads(stream.read(length))  # type: ignore[attr-defined]


def test_the_language_server_answers_hover_requests() -> None:
    pytest.importorskip("pygls")
    uri = "file:///tmp/p.wsm"
    line = SOURCE.splitlines().index("    verify")
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}},
        {"jsonrpc": "2.0", "method": "initialized", "params": {}},
        {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": uri,
                    "languageId": "weightsmith",
                    "version": 1,
                    "text": SOURCE,
                }
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "textDocument/hover",
            "params": {"textDocument": {"uri": uri}, "position": {"line": line, "character": 6}},
        },
    ]
    process = subprocess.Popen(
        [sys.executable, "-m", "weightsmith", "lsp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        process.stdin.write(b"".join(_message(r) for r in requests))
        process.stdin.flush()
        replies = {}
        while 2 not in replies:
            reply = _read(process.stdout)
            if "id" in reply:
                replies[reply["id"]] = reply
        assert replies[1]["result"]["capabilities"]["hoverProvider"]  # type: ignore[index]
        contents = replies[2]["result"]["contents"]  # type: ignore[index]
        assert contents["kind"] == "markdown"
        assert contents["value"].startswith("**verify** (keyword)")
    finally:
        process.kill()
        process.wait()
