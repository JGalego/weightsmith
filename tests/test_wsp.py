"""What the "Sparks of weight-space programming" replication needs: weight views and formats,
program-supplied messages, the blocks protocol, the answer requirement and the Responses API."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from weightsmith.backends.pytorch import PyTorchBackend
from weightsmith.cli.main import _load_env, app
from weightsmith.compiler.driver import CompileResult
from weightsmith.evaluation.verify import Status, answer_field, judge_answer
from weightsmith.models.view import (
    View,
    ViewTensor,
    parse_format,
    read_view,
    render,
    round_like_shown,
)
from weightsmith.reasoning.blocks import last_block
from weightsmith.reasoning.openai import ResponsesConversation, parse_responses
from weightsmith.reasoning.provider import ToolResult, ToolSchema
from weightsmith.runtime.context import RunOptions
from weightsmith.runtime.executor import execute

from .conftest import codes

Compile = Callable[[str], CompileResult]
WSP = Path(__file__).resolve().parent.parent / "examples" / "wsp"
runner = CliRunner()


# --------------------------------------------------------------------------- views and formats


def test_decimals_format_transposes_and_drops_negative_zero() -> None:
    view = View((ViewTensor("W", "w", True, "{label} ({shape}): {cols} per line"),))
    text = render(view, parse_format("decimals(2)"), {"w": np.array([[1.234, -0.001, 5.0]])})
    assert text == "W (3 x 1): 1 per line\n1.23\n0.00\n5.00"


def test_int_scale_format_keeps_significant_digits() -> None:
    view = View((ViewTensor("m", "m"), ViewTensor("b", "b")))
    arrays = {"m": np.array([[0.1234, -0.05], [0.0049, 0.0]]), "b": np.array([12.5, -3.0])}
    text = render(view, parse_format("int_scale(3)"), arrays)
    assert text == "m (2 x 2), scale 1e-3:\n0: 123 -50\n1: 5 0\n\nb (2), scale 1e-1:\n125 -30"
    rounded = {k: round_like_shown(v, parse_format("int_scale(3)")) for k, v in arrays.items()}
    assert render(view, parse_format("int_scale(3)"), rounded) == text


@pytest.mark.parametrize("text", ["decimals", "decimals(0)", "hex(3)", "int_scale(x)"])
def test_bad_weight_formats_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match=r"format|digits"):
        parse_format(text)


def test_views_round_trip_and_reject_duplicate_labels(tmp_path: Path) -> None:
    view = View((ViewTensor("W1", "a.weight", True, "W1 {shape}"), ViewTensor("b1", "a.bias")))
    (tmp_path / "view.json").write_text(json.dumps(view.to_dict()))
    assert read_view(tmp_path) == view
    data = view.to_dict()
    data["tensors"][1]["label"] = "W1"
    (tmp_path / "view.json").write_text(json.dumps(data))
    with pytest.raises(Exception, match="twice"):
        read_view(tmp_path)


# --------------------------------------------------------------------------- blocks and answers


def test_only_the_last_block_of_a_granted_kind_counts() -> None:
    text = "```edits\na = 1\n```\nthen\n```python\nx\n```\n```edits\nb = 2\n```\nend"
    block = last_block(text, {"edits"})
    assert block is not None
    assert block.body == "b = 2\n"
    assert last_block(text, {"forward"}) is None


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("Game: Breakout, a variant\nConfidence: 99%", "Breakout, a variant"),
        ("## Game\n\nPong\n\n## Description", "Pong"),
        ("**Game:** Pac-Man", "Pac-Man"),
        ("Game: draft\n...\nGame: Freeway", "Freeway"),
        ("No such line", None),
    ],
)
def test_answer_fields_are_found_through_markdown(text: str, value: str | None) -> None:
    assert answer_field(text, "Game") == value


def test_answer_verdicts() -> None:
    assert judge_answer("A", "Game: Breakout", "Game", "(?i)breakout").status is Status.VERIFIED
    assert judge_answer("A", "Game: Pong", "Game", "(?i)breakout").status is Status.FAILED
    assert judge_answer("A", "no answer", "Game", "x").status is Status.INCONCLUSIVE
    assert judge_answer("A", None, "Game", "x").status is Status.INCONCLUSIVE


# --------------------------------------------------------------------------- Responses API


def test_responses_conversation_sends_back_everything_the_model_produced() -> None:
    sent: list[dict[str, Any]] = []
    replies = [
        {
            "output": [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "SECRET", "summary": []},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "add",
                    "arguments": '{"a": 2, "b": 3}',
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "5"}]}],
            "usage": {"input_tokens": 20, "output_tokens": 1},
        },
    ]

    def transport(body: dict[str, Any]) -> dict[str, Any]:
        sent.append(json.loads(json.dumps(body)))
        return replies[len(sent) - 1]

    tool = ToolSchema("add", "Add.", {"type": "object", "properties": {}})
    conversation = ResponsesConversation({"model": "m", "effort": "max"}, "SYS", [tool], transport)
    first = conversation.send("add 2 and 3")
    assert first.stop == "tool_use"
    assert first.calls[0].arguments == {"a": 2, "b": 3}
    second = conversation.send(None, (ToolResult("c1", "5"),))
    assert second.text == "5"
    assert second.stop == "end"
    assert sent[0]["instructions"] == "SYS"
    assert sent[0]["reasoning"] == {"effort": "max"}
    assert sent[0]["store"] is False
    assert sent[0]["include"] == ["reasoning.encrypted_content"]
    assert [i.get("type") for i in sent[1]["input"]] == [
        None,
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert sent[1]["input"][1]["encrypted_content"] == "SECRET"


def test_responses_stops() -> None:
    refused = {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}
    assert parse_responses(refused).stop == "refusal"
    cut = {"output": [], "incomplete_details": {"reason": "max_output_tokens"}}
    assert parse_responses(cut).stop == "max_tokens"


# --------------------------------------------------------------------------- programs

HEADER = """
model Base = load("fixtures/tiny_mlp")
editor S = provider("scripted", script = "script.json")
"""


def ask(extra: str = "", contract: str = "", message: str = "message.txt") -> str:
    contract = contract or 'require answer (editor = S, field = "Game", matches = "(?i)parity")'
    return (
        f"{HEADER}\nprogram P {{\n target Base\n policy {{ allow reason }}\n"
        f' delegate S (message_file = "{message}", protocol = "blocks", tools = [],'
        f" max_steps = 1{extra})\n contract {{ {contract} }}\n verify\n}}\n"
    )


def run_answer(
    workspace: Path, compile_text: Compile, answer: str, source: str, out: str = "out"
) -> Any:
    (workspace / "script.json").write_text(
        json.dumps({"format": "weightsmith.script/v0", "turns": [{"text": answer}]})
    )
    result = compile_text(source)
    assert result.ok, [d.message for d in result.sink.items]
    return execute(result.program("P"), PyTorchBackend(), RunOptions(out_dir=workspace / out))


def test_a_message_file_is_sent_as_written_with_the_weights_filled_in(
    workspace: Path, compile_text: Compile
) -> None:
    (workspace / "message.txt").write_text("Which task?\n\n{{weights}}\nAnswer with Game: <x>")
    report = run_answer(
        workspace,
        compile_text,
        "Observations: ...\nGame: parity\nConfidence: 80%",
        ask(', weights_format = "int_scale(2)", system = "Answer."'),
    )
    transcript = json.loads((workspace / "out" / "transcripts" / "00-S.json").read_text())
    message = transcript["last_request"]["user"]
    assert message.startswith("Which task?\n\n")
    assert message.endswith("\nAnswer with Game: <x>")
    assert "{{" not in message
    assert "scale 1e-" in message
    assert [v["status"] for v in report.to_dict()["verdicts"]] == ["verified"]
    assert report.delegations[0]["stop"] == "finished"


def test_a_wrong_answer_fails_and_a_missing_one_is_inconclusive(
    workspace: Path, compile_text: Compile
) -> None:
    (workspace / "message.txt").write_text("Which task?")
    wrong = run_answer(workspace, compile_text, "Game: majority", ask())
    assert [v["status"] for v in wrong.to_dict()["verdicts"]] == ["failed"]
    silent = run_answer(workspace, compile_text, "I cannot tell.", ask(), out="out2")
    assert [v["status"] for v in silent.to_dict()["verdicts"]] == ["inconclusive"]


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (ask(', task = "x"'), "E361"),
        (ask(', weights_format = "hex(3)"'), "E366"),
        (ask(message="bad_message.txt"), "E367"),
        (ask(", tools = [stats]").replace("tools = [],", ""), "E368"),
        (ask(contract='require answer (editor = S, field = "G", matches = "(")'), "E369"),
        (ask(', system = "a", system_file = "message.txt"'), "E365"),
        (ask(contract='require answer (editor = Base, field = "G", matches = "x")'), "E360"),
    ],
)
def test_delegation_messages_are_checked(
    workspace: Path, compile_text: Compile, source: str, code: str
) -> None:
    (workspace / "message.txt").write_text("Which task?")
    (workspace / "bad_message.txt").write_text("Here: {{weightz}}")
    assert code in codes(compile_text(source))


def test_env_files_fill_but_never_override_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text("# keys\nWSM_TEST_A='one'\nexport WSM_TEST_B=two\nWSM_TEST_C=three\n")
    monkeypatch.setenv("WSM_TEST_C", "kept")
    monkeypatch.delenv("WSM_TEST_A", raising=False)
    monkeypatch.delenv("WSM_TEST_B", raising=False)
    _load_env(env)
    assert (os.environ["WSM_TEST_A"], os.environ["WSM_TEST_B"]) == ("one", "two")
    assert os.environ["WSM_TEST_C"] == "kept"
    monkeypatch.delenv("WSM_TEST_A")
    monkeypatch.delenv("WSM_TEST_B")


# --------------------------------------------------------------------------- the article itself

ARTICLE = {"Breakout": "a9f535cc", "Pong": "dfc6cb65", "PacMan": "bdf99279", "Freeway": "0bbcc3ea"}
needs_article = pytest.mark.skipif(
    not (WSP / "fixtures" / "logs" / ".commit").is_file(),
    reason="run examples/wsp/fetch.py to get the article's logs",
)


@needs_article
@pytest.mark.parametrize("program_name", list(ARTICLE))
def test_which_game_sends_the_articles_prompt_and_scores_its_answers(
    tmp_path: Path, program_name: str
) -> None:
    net = ARTICLE[program_name]
    source = (WSP / "which_game.wsm").read_text()
    source = source.replace(
        'provider("openai", model = "gpt-6-astra", effort = "high")',
        f'provider("scripted", script = "fixtures/which_game/article/{net}.json")',
    )
    program = WSP / f"_test_{program_name}.wsm"
    program.write_text(source)
    try:
        result = runner.invoke(
            app,
            ["run", str(program), "--program", program_name, "--out", str(tmp_path / "out")],
        )
    finally:
        program.unlink()
    assert result.exit_code == 0, result.output
    transcript = json.loads((tmp_path / "out" / "transcripts" / "00-Astra.json").read_text())
    game = {"Breakout": "breakout", "Pong": "pong", "PacMan": "pacman", "Freeway": "freeway"}
    article = WSP / "fixtures" / "logs" / "exp1_which_game" / "runs" / game[program_name]
    assert transcript["last_request"]["user"] == (article / "prompt.txt").read_text()
