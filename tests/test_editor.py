"""Editors: the delegate statement, its tools, providers, recording and replay."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from weightsmith.backends.pytorch import PyTorchBackend
from weightsmith.cli.main import app
from weightsmith.compiler.driver import CompileResult
from weightsmith.compiler.effects import Effect
from weightsmith.compiler.ir import Delegate, IRProgram
from weightsmith.evaluation.report import RunReport
from weightsmith.reasoning.anthropic import AnthropicConversation
from weightsmith.reasoning.openai import OpenAIConversation
from weightsmith.reasoning.provider import (
    RecordingTransport,
    ReplayDivergence,
    ReplayTransport,
    ToolResult,
    ToolSchema,
    Transcript,
)
from weightsmith.runtime.context import RunOptions
from weightsmith.runtime.executor import execute

from .conftest import codes

pytest.importorskip("transformers")

runner = CliRunner()
Compile = Callable[[str], CompileResult]

HEADER = """
model LM = load("fixtures/tiny_llama_defective")
weights LW = load("fixtures/tiny_llama")
behavior count_train = probes("fixtures/tiny_llama/count_train.json", metric = accuracy)
behavior count_test = probes("fixtures/tiny_llama/count_test.json", metric = accuracy)
editor S = provider("scripted", script = "script.json")
"""
ALL_TOOLS = (
    "[describe, read_weights, stats, find, run_probes, forward, edit, set_weights, undo, python]"
)


def program(body: str, policy: str = "", target: str = "LM") -> str:
    policy = policy or (
        "allow probe allow mutate(max_changed_params = 1%) allow reason allow execute allow save"
    )
    return f"{HEADER}\nprogram P {{\n target {target}\n policy {{ {policy} }}\n {body}\n}}\n"


def delegate(tools: str = ALL_TOOLS, extra: str = "") -> str:
    return f'delegate S (task = "fix it", tools = {tools}, practice = [count_train]{extra})'


def run_script(
    workspace: Path, compile_text: Compile, turns: list[dict[str, Any]], body: str = "", **kw: Any
) -> RunReport:
    (workspace / "script.json").write_text(
        json.dumps({"format": "weightsmith.script/v0", "turns": turns})
    )
    result = compile_text(program(body or delegate(), **kw))
    assert result.ok, [d.message for d in result.sink.items]
    return execute(result.program("P"), PyTorchBackend(), RunOptions(out_dir=workspace / "out"))


def tool_results(workspace: Path) -> list[dict[str, Any]]:
    transcript = json.loads((workspace / "out" / "transcripts" / "00-S.json").read_text())
    return [r for e in [transcript["last_request"]] for r in e["results"]]


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments}


# --------------------------------------------------------------------------- checking


def test_delegate_program_checks_and_lowers(compile_text: Compile, workspace: Path) -> None:
    (workspace / "task.txt").write_text("Fix counting.\nDo not train.")
    result = compile_text(
        program(
            'delegate S (task_file = "task.txt", tools = [describe, python], '
            "practice = [count_train], max_steps = 3)"
        )
    )
    assert result.ok, [d.message for d in result.sink.items]
    [op] = [o for o in result.program("P").ops if isinstance(o, Delegate)]
    assert op.task == "Fix counting.\nDo not train."
    assert op.provider == "scripted"
    assert op.settings["script"] == str(workspace / "script.json")
    assert op.effects() == {Effect.REASON, Effect.INSPECT, Effect.EXECUTE}
    ir = result.program("P")
    assert IRProgram.from_json(ir.to_json()) == ir


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ('editor X = provider("gpt", model = "m")', "E210"),
        ('editor X = provider("anthropic", api_key = "sk-...")', "E364"),
        ('editor X = provider("anthropic", effort = "huge")', "E329"),
        ('editor X = provider("openai")', "E322"),
    ],
)
def test_editor_declaration_errors(compile_text: Compile, source: str, code: str) -> None:
    result = compile_text(source + "\n" + program(""))
    assert code in codes(result), [d.message for d in result.sink.items]


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ('delegate LM (task = "x")', "E360"),
        ("delegate S (tools = [describe])", "E361"),
        ('delegate S (task = "x", task_file = "t.txt")', "E361"),
        ('delegate S (task = "x", tools = [teleport])', "E203"),
        ('delegate S (task = "x", practice = [LM])', "E351"),
    ],
)
def test_delegate_errors(compile_text: Compile, body: str, code: str) -> None:
    result = compile_text(program(body))
    assert code in codes(result), [d.message for d in result.sink.items]


def test_running_tools_need_a_model(compile_text: Compile) -> None:
    result = compile_text(program('delegate S (task = "x", tools = [forward])', target="LW"))
    assert "E362" in codes(result)


def test_effects_of_tools_are_checked(compile_text: Compile) -> None:
    result = compile_text(program(delegate(), policy="allow probe allow mutate allow reason"))
    assert "E410" in codes(result)  # python needs execute
    result = compile_text(program(delegate("[describe]"), policy="allow probe"))
    assert "E410" in codes(result)  # every delegation needs reason
    result = compile_text(
        program(delegate("[python, run_probes]"), policy="allow probe allow reason allow execute")
    )
    assert result.ok
    assert "W403" in [d.code for d in result.sink.items]


def test_practising_on_verified_behaviour_warns(compile_text: Compile) -> None:
    body = (
        delegate() + " contract { improve count_train (min_delta = 0.1)"
        " preserve count_test (tolerance = 0.1) } verify"
    )
    result = compile_text(program(body))
    warnings = [d for d in result.sink.items if d.code == "W306"]
    assert len(warnings) == 1
    assert "count_train" in warnings[0].message


# --------------------------------------------------------------------------- tools


def test_tools_end_to_end(workspace: Path, compile_text: Compile, fixtures_dir: Path) -> None:
    truth = json.loads((fixtures_dir / "tiny_llama_defective" / "fixture.json").read_text())
    neuron = truth["planted"][0]["units"][0]
    report = run_script(
        workspace,
        compile_text,
        [
            {"calls": [call("describe"), call("read_weights", select=neuron, precision=3)]},
            {"calls": [call("find", finder="outlier_neurons", select="layers[0]")]},
            {"calls": [call("forward", prompts=["count : 3 4 5"], top_k=2)]},
            {"calls": [call("run_probes", behavior="count_train", show_failures=1)]},
            {"calls": [call("edit", op="scale", select=neuron, params={"factor": 0.1})]},
            {"calls": [call("undo")]},
            {
                "calls": [
                    call("set_weights", select='params["model.norm.weight"][0..2]', values=[1, 2])
                ]
            },
            {"text": "done"},
        ],
    )
    assert report.status == "completed", report.stop_reason
    [record] = report.delegations
    assert record["stop"] == "finished"
    assert record["tool_calls"] == 8
    assert record["failed_tool_calls"] == 0, record["log"]
    transcript = json.loads((workspace / "out" / "transcripts" / "00-S.json").read_text())
    responses = [e for e in transcript["exchanges"]]
    assert len(responses) == 8
    # Exactly the two set entries changed: scale was undone.
    assert report.changes["changed_params"] == 2


def test_tool_errors_do_not_stop_the_run(workspace: Path, compile_text: Compile) -> None:
    report = run_script(
        workspace,
        compile_text,
        [
            {
                "calls": [
                    call("read_weights", select="all"),  # more than max_read
                    call("run_probes", behavior="count_test"),  # not a practice behaviour
                    call("edit", op="zero", select="layers[:]"),  # beyond the 1% budget
                    call("edit", op="explode", select="embed"),
                    call("read_weights", select="layers[9]"),
                    call("undo"),
                    call("stats"),  # missing argument
                ]
            },
            {"text": "giving up"},
        ],
    )
    assert report.status == "completed"
    [record] = report.delegations
    errors = [e["error"] for e in record["log"]]
    assert len(errors) == 7
    assert "max_read" in errors[0]
    assert "not a practice behaviour" in errors[1]
    assert errors[2].startswith("budget")
    assert "unknown edit operation" in errors[3]
    assert "out of range" in errors[4]
    assert "nothing of yours to undo" in errors[5]
    assert "missing argument 'select'" in errors[6]
    assert report.changes.get("changed_params", 0) == 0


def test_ungranted_tool_is_refused(workspace: Path, compile_text: Compile) -> None:
    report = run_script(
        workspace,
        compile_text,
        [{"calls": [call("edit", op="zero", select="embed")]}, {"text": "ok"}],
        body=delegate("[describe]"),
    )
    assert report.delegations[0]["log"][0]["error"] == "not granted"


def test_python_tool(workspace: Path, compile_text: Compile) -> None:
    report = run_script(
        workspace,
        compile_text,
        [
            {"calls": [call("python", code='W["model.norm.weight"][:3] = 0.5\nprint("hi")')]},
            {"calls": [call("python", code='W["new"] = np.zeros(3)')]},
            {"calls": [call("python", code="raise SystemExit(3)")]},
            {
                "calls": [
                    call("python", code="import os\nprint(os.environ.get('ANTHROPIC_API_KEY'))")
                ]
            },
            {"text": "done"},
        ],
    )
    assert report.status == "completed", report.stop_reason
    assert report.changes["changed_params"] == 3
    results = [e["result"] for e in report.delegations[0]["log"]]
    assert "hi" in results[0]
    assert "python: changed model.norm.weight" in results[0]
    assert "new tensors cannot be added" in results[1]
    assert "SystemExit: 3" in results[2]
    assert "no changes were taken" in results[2]
    assert "stdout:\nNone" in results[3]  # credentials do not reach editor code


def test_python_changes_refused_without_mutate(workspace: Path, compile_text: Compile) -> None:
    report = run_script(
        workspace,
        compile_text,
        [{"calls": [call("python", code='W["model.norm.weight"][:] = 0')]}, {"text": "done"}],
        body=delegate("[python]"),
        policy="allow probe allow reason allow execute",
    )
    assert report.changes.get("changed_params", 0) == 0


def test_reason_budget_stops_the_editor_not_the_run(workspace: Path, compile_text: Compile) -> None:
    turns = [{"calls": [call("describe")]} for _ in range(5)]
    report = run_script(
        workspace,
        compile_text,
        turns,
        policy="allow probe allow mutate allow reason(max_calls = 2) allow execute allow save",
    )
    assert report.status == "completed"
    assert report.delegations[0]["turns"] == 2
    assert report.delegations[0]["stop"].startswith("budget")


def test_dry_run_does_not_call_the_editor(workspace: Path, compile_text: Compile) -> None:
    (workspace / "script.json").write_text("not even json")
    result = compile_text(program(delegate()))
    report = execute(result.program("P"), PyTorchBackend(), RunOptions(dry_run=True))
    assert report.status == "completed"
    assert report.delegations == []


# --------------------------------------------------------------------------- replay


def test_example_runs_and_replays_exactly(workspace: Path) -> None:
    out, again = workspace / "out", workspace / "again"
    first = runner.invoke(
        app, ["run", str(workspace / "repair_with_editor.wsm"), "--out", str(out)]
    )
    assert first.exit_code == 0, first.output
    second = runner.invoke(
        app,
        [
            "run",
            str(workspace / "repair_with_editor.wsm"),
            "--out",
            str(again),
            "--replay",
            str(out / "transcripts"),
        ],
    )
    assert second.exit_code == 0, second.output
    a = json.loads((out / "report.json").read_text())
    b = json.loads((again / "report.json").read_text())
    assert a["target"]["final_model_id"] == b["target"]["final_model_id"]
    weights = "model/model.safetensors"
    assert (out / weights).read_bytes() == (again / weights).read_bytes()
    assert b["delegations"][0]["replayed"]
    assert {v["status"] for v in a["verdicts"]} == {"verified"}


def test_replay_detects_divergence(workspace: Path) -> None:
    out = workspace / "out"
    assert (
        runner.invoke(
            app, ["run", str(workspace / "repair_with_editor.wsm"), "--out", str(out)]
        ).exit_code
        == 0
    )
    changed = workspace / "changed.wsm"
    changed.write_text(
        (workspace / "repair_with_editor.wsm").read_text().replace("Counting", "Counting really")
    )
    result = runner.invoke(
        app,
        ["run", str(changed), "--out", str(workspace / "x"), "--replay", str(out / "transcripts")],
    )
    assert result.exit_code == 2
    assert "differs from the recording" in result.output


def test_recording_and_replay_transports(tmp_path: Path) -> None:
    transcript = Transcript("E", "fake", {})
    recorded = RecordingTransport(lambda body: {"echo": body["n"]}, transcript, tmp_path / "t.json")
    assert recorded({"n": 1}) == {"echo": 1}
    replay = ReplayTransport(Transcript.read(tmp_path / "t.json"))
    assert replay({"n": 1}) == {"echo": 1}
    with pytest.raises(ReplayDivergence, match="more editor calls"):
        replay({"n": 2})
    with pytest.raises(ReplayDivergence, match="differs"):
        ReplayTransport(Transcript.read(tmp_path / "t.json"))({"n": 2})


def test_missing_provider_extra_fails_before_running(
    workspace: Path, compile_text: Compile, monkeypatch: pytest.MonkeyPatch
) -> None:
    import weightsmith.reasoning as reasoning

    monkeypatch.setattr(reasoning, "missing_extra", lambda provider: "anthropic")
    (workspace / "script.json").write_text("{}")
    result = compile_text(program(delegate()))
    report = execute(result.program("P"), PyTorchBackend(), RunOptions(out_dir=workspace / "o"))
    assert report.status == "error"
    assert "weightsmith[anthropic]" in str(report.stop_reason)
    assert all(e.op != "load LM" for e in report.events)


# --------------------------------------------------------------------------- adapters

TOOLS = [ToolSchema("edit", "Edit.", {"type": "object", "properties": {"op": {"type": "string"}}})]


def test_anthropic_request_and_response_shapes() -> None:
    sent: list[dict[str, Any]] = []
    responses = [
        {
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "Trying."},
                {"type": "tool_use", "id": "toolu_1", "name": "edit", "input": {"op": "zero"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 5},
        },
        {
            "content": [],
            "stop_reason": "refusal",
            "stop_details": {"type": "refusal", "category": "cyber"},
            "usage": {"input_tokens": 1, "output_tokens": 0},
        },
    ]

    def transport(body: dict[str, Any]) -> dict[str, Any]:
        sent.append(json.loads(json.dumps(body)))
        return responses[len(sent) - 1]

    conversation = AnthropicConversation({"model": "claude-opus-5-5"}, "SYS", TOOLS, transport)
    turn = conversation.send("brief")
    assert turn.stop == "tool_use"
    assert turn.calls[0].arguments == {"op": "zero"}
    assert (turn.input_tokens, turn.output_tokens) == (100, 5)
    first = sent[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["tools"][0]["eager_input_streaming"] is True
    assert first["tools"][0]["input_schema"] == TOOLS[0].parameters
    assert first["output_config"] == {"effort": "high"}
    assert first["cache_control"] == {"type": "ephemeral"}
    assert "fallbacks" not in first
    assert "betas" not in first
    assert "thinking" not in first
    assert "temperature" not in first
    turn = conversation.send(results=(ToolResult("toolu_1", "bad", is_error=True),))
    assert turn.stop == "refusal"
    assert turn.detail == "cyber"
    second = sent[1]["messages"]
    assert second[1] == {"role": "assistant", "content": responses[0]["content"]}  # unchanged
    assert second[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "bad", "is_error": True}
    ]


def test_openai_request_and_response_shapes() -> None:
    sent: list[dict[str, Any]] = []

    def transport(body: dict[str, Any]) -> dict[str, Any]:
        sent.append(json.loads(json.dumps(body)))
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "edit", "arguments": '{"op": "zero"}'},
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "edit", "arguments": "{oops"},
                            },
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    conversation = OpenAIConversation({"model": "local-model"}, "SYS", TOOLS, transport)
    turn = conversation.send("brief")
    assert [c.arguments for c in turn.calls] == [{"op": "zero"}, None]
    assert turn.calls[1].error is not None
    assert (turn.stop, turn.input_tokens, turn.output_tokens) == ("tool_use", 7, 3)
    assert sent[0]["messages"][0] == {"role": "system", "content": "SYS"}
    assert sent[0]["tools"][0]["function"]["name"] == "edit"
    conversation.send(results=(ToolResult("c1", "done"), ToolResult("c2", "bad", True)))
    tail = sent[1]["messages"][-2:]
    assert tail == [
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
        {"role": "tool", "tool_call_id": "c2", "content": "ERROR: bad"},
    ]
