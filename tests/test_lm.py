"""Language models in Hugging Face format: roles, token probes, unit finders, repair, artifacts."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from weightsmith.backends.pytorch import PyTorchBackend
from weightsmith.cli.main import app
from weightsmith.compiler.driver import CompileResult
from weightsmith.compiler.ir import IRProgram, Select
from weightsmith.evaluation.probes import ProbeSuite, load_probe_suite
from weightsmith.models.hf import HFArch
from weightsmith.models.profiles import parse_role

from .conftest import codes

transformers = pytest.importorskip("transformers")

BACKEND = PyTorchBackend()
runner = CliRunner()
Compile = Callable[[str], CompileResult]

LM_HEADER = """
model LM = load("fixtures/tiny_llama")
model Bad = load("fixtures/tiny_llama_defective")
model Donor = load("fixtures/tiny_llama_donor")
weights LW = load("fixtures/tiny_llama")
behavior count = probes("fixtures/tiny_llama/count.json", metric = accuracy)
behavior parity = probes("fixtures/tiny_mlp/parity.json", metric = accuracy)
model Base = load("fixtures/tiny_mlp")
"""


def lm_program(body: str, target: str = "LM") -> str:
    policy = "policy { allow probe allow mutate allow save }"
    return f"{LM_HEADER}\nprogram P {{\n target {target}\n {policy}\n {body}\n}}\n"


def planted(fixtures_dir: Path) -> dict:
    truth = json.loads((fixtures_dir / "tiny_llama_defective" / "fixture.json").read_text())
    return truth["planted"][0]


# --------------------------------------------------------------------------- checking


def test_role_program_checks(compile_text: Compile) -> None:
    result = compile_text(
        lm_program("""
        let head = LM.layers[1].attn.heads[3]
        inspect LM.embed[0..4, :]
        contract { preserve count (tolerance = 0.05) }
        find outlier_neurons in LM.layers[0].mlp
        edit { zero head  scale LM.layers[0].mlp.neurons[3..5] (factor = 0.5) }
        compose { transplant Donor.layers[0].attn.kv_heads[1] }
        verify
        output { model, patch, evaluation, summary }
    """)
    )
    assert result.ok, [d.message for d in result.sink.items]
    ir = result.program("P")
    roles = [op.role for op in ir.ops if isinstance(op, Select) and op.role]
    assert roles == [
        "layers[1].attn.heads[3]",
        "embed",
        "layers[0].mlp",
        "layers[0].mlp.neurons[3..5]",
        "layers[0].attn.kv_heads[1]",
    ]
    assert IRProgram.from_json(ir.to_json()) == ir


@pytest.mark.parametrize(
    ("body", "target", "code"),
    [
        ("inspect Base.layers[0]", "Base", "E316"),
        ("inspect LM.layers[0].ffn", "LM", "E317"),
        ("inspect LM.layers[0].mlp.heads[0]", "LM", "E317"),
        ("inspect LM.layers.attn", "LM", "E318"),
        ("inspect LM.layers[0].attn.heads", "LM", "E318"),
        ('inspect LM.layers["0"]', "LM", "E318"),
        ("inspect LM.layers[2]", "LM", "E510"),
        ("inspect LM.layers[0].attn.heads[4]", "LM", "E510"),
        ("inspect LM.layers[0].attn.heads[1][0..2]", "LM", "E313"),
        ("find outlier_neurons in LM.layers[0].attn", "LM", "E319"),
        ('find outlier_neurons in LM.params["model.layers.0.*"]', "LM", "E319"),
        ("find outlier_heads", "Base", "E319"),
        ("find outlier_rows in LM.layers[0].attn.heads[0]", "LM", "E343"),
        ("edit { low_rank LM.layers[0].attn.heads[0] (rank = 2) }", "LM", "E347"),
        ("contract { preserve parity (tolerance = 0.1) } verify", "LM", "E512"),
        ("contract { preserve count (tolerance = 0.1) } verify", "Base", "E512"),
    ],
)
def test_role_and_suite_errors(compile_text: Compile, body: str, target: str, code: str) -> None:
    result = compile_text(lm_program(body, target))
    assert code in codes(result), [d.message for d in result.sink.items]


def test_roles_on_weights_from_a_model_directory(compile_text: Compile) -> None:
    # 'weights' cannot run, but a directory with config.json still has a profile.
    result = compile_text(lm_program("inspect LW.layers[1].attn.heads[0]", target="LW"))
    assert result.ok, [d.message for d in result.sink.items]


def test_token_suite_without_tokenizer(compile_text: Compile, workspace: Path) -> None:
    bare = workspace / "bare_llama"
    bare.mkdir()
    for name in ("config.json", "model.safetensors"):
        shutil.copy(workspace / "fixtures" / "tiny_llama" / name, bare / name)
    (bare / "ids.json").write_text(
        json.dumps(
            {
                "kind": "tokens",
                "name": "ids",
                "examples": [{"prompt_ids": [1, 3, 4], "target_id": 999}],
            }
        )
    )
    source = """
    model M = load("bare_llama")
    behavior text = probes("fixtures/tiny_llama/count.json", metric = accuracy)
    behavior ids = probes("bare_llama/ids.json", metric = accuracy)
    program P { target M policy { allow probe }
        contract { preserve text (tolerance = 0.1) preserve ids (tolerance = 0.1) } verify }
    """
    result = compile_text(source)
    assert {"E513", "E514"} <= set(codes(result))


def test_token_suite_validation() -> None:
    with pytest.raises(ValueError, match="exactly one of 'prompt' and 'prompt_ids'"):
        ProbeSuite.model_validate(
            {"kind": "tokens", "name": "x", "examples": [{"prompt": "a", "prompt_ids": [1]}]}
        )
    with pytest.raises(ValueError, match="needs a prompt and a target"):
        ProbeSuite.model_validate(
            {"kind": "tokens", "name": "x", "examples": [{"input": [1.0], "label": 0}]}
        )


# --------------------------------------------------------------------------- execution


def test_runner_accuracy_matches_fixture(fixtures_dir: Path) -> None:
    from weightsmith.synthetic_lm import TASKS

    tensors, arch = BACKEND.load_model(fixtures_dir / "tiny_llama_defective")
    assert isinstance(arch, HFArch)
    assert arch.profile is not None
    truth = json.loads((fixtures_dir / "tiny_llama_defective" / "fixture.json").read_text())
    for task in TASKS:
        suite = load_probe_suite(fixtures_dir / "tiny_llama" / f"{task}.json")
        measured = BACKEND.evaluate(arch, tensors, suite, "accuracy")
        assert measured == pytest.approx(truth["accuracy"][task], abs=1e-4)


def test_candidates_restrict_the_prediction(fixtures_dir: Path) -> None:
    tensors, arch = BACKEND.load_model(fixtures_dir / "tiny_llama")
    assert isinstance(arch, HFArch)
    suite = ProbeSuite.model_validate(
        {
            "kind": "tokens",
            "name": "choice",
            "candidates": ["cold", "hot"],
            "examples": [{"prompt": "opposite : hot", "target": "cold"}],
        }
    )
    assert BACKEND.evaluate(arch, tensors, suite, "accuracy") == 1.0


def test_outlier_neurons_find_exactly_the_planted_units(fixtures_dir: Path) -> None:
    tensors, arch = BACKEND.load_model(fixtures_dir / "tiny_llama_defective")
    assert isinstance(arch, HFArch)
    assert arch.profile is not None
    groups = arch.profile.unit_groups("neurons", [0, 1])
    result = BACKEND.find_units("outlier_neurons", tensors, groups, {}, "M", "id")
    assert [f.locations[0].unit for f in result.findings] == planted(fixtures_dir)["units"]
    assert result.examined == 512
    assert result.noun == "neuron"
    healthy, _ = BACKEND.load_model(fixtures_dir / "tiny_llama")
    assert BACKEND.find_units("outlier_neurons", healthy, groups, {}, "M", "id").findings == ()
    heads = arch.profile.unit_groups("heads", [0, 1])
    assert BACKEND.find_units("outlier_heads", healthy, heads, {}, "M", "id").findings == ()


def test_zeroing_a_head_equals_ablating_its_output(fixtures_dir: Path) -> None:
    tensors, arch = BACKEND.load_model(fixtures_dir / "tiny_llama")
    assert isinstance(arch, HFArch)
    assert arch.profile is not None
    regions = list(arch.profile.resolve(parse_role("layers[1].attn.heads[2]")))
    edited = {**tensors, **BACKEND.edit("zero", tensors, regions, {})}
    reference = dict(tensors)
    o = "model.layers.1.self_attn.o_proj.weight"
    reference[o] = tensors[o].clone()
    reference[o][:, 32:48] = 0
    run = BACKEND.runner(arch, tensors)
    prompts = [[1, 3, 7, 10, 11], [1, 4, 7, 40]]
    assert torch.allclose(
        run.next_token_logits(edited, prompts), run.next_token_logits(reference, prompts)
    )


def test_gpt2_tied_and_sharded(tmp_path: Path, compile_text: Compile, workspace: Path) -> None:
    config = transformers.GPT2Config(
        vocab_size=50,
        n_positions=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
        bos_token_id=1,
        eos_token_id=1,
    )
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config)
    directory = workspace / "gpt2"
    model.save_pretrained(str(directory), max_shard_size="40KB")
    assert (directory / "model.safetensors.index.json").is_file()
    tensors, arch = BACKEND.load_model(directory)
    assert isinstance(arch, HFArch)
    assert arch.profile is not None
    assert arch.profile.tied
    suite = ProbeSuite.model_validate(
        {
            "kind": "tokens",
            "name": "ids",
            "examples": [{"prompt_ids": [1, 2, 3], "target_id": 4}],
        }
    )
    with torch.no_grad():
        expected = model(torch.tensor([[1, 2, 3]])).logits[0, -1].argmax().item()
    suite_ok = suite.model_copy(
        update={"examples": [suite.examples[0].model_copy(update={"target_id": expected})]}
    )
    assert BACKEND.evaluate(arch, tensors, suite_ok, "accuracy") == 1.0
    result = compile_text(
        'model G = load("gpt2")\nprogram P { target G policy { allow mutate } '
        "edit { zero G.unembed[0, :] } }"
    )
    assert result.ok
    assert "W305" in [d.code for d in result.sink.items]


# --------------------------------------------------------------------------- end to end


def test_repair_example_end_to_end(workspace: Path, fixtures_dir: Path) -> None:
    out = workspace / "out"
    result = runner.invoke(
        app, ["run", str(workspace / "repair_tiny_llama.wsm"), "--out", str(out), "--json"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads((out / "report.json").read_text())
    assert {f"{v['clause']} {v['subject']}": v["status"] for v in report["verdicts"]} == {
        "improve count": "verified",
        "preserve opposite": "verified",
        "preserve first": "verified",
        "require finite": "verified",
    }
    truth = planted(fixtures_dir)
    assert report["changes"]["changed_params"] == 2 * len(truth["neurons"]) * 64
    assert sorted(report["changes"]["tensors"]) == sorted(truth["tensors"])
    assert all(f["level"] == "experimentally_supported" for f in report["findings"])
    # The output directory is a working Hugging Face model with the input's tokenizer.
    saved = out / "model"
    assert {"config.json", "tokenizer.json", "model.safetensors"} <= {
        p.name for p in saved.iterdir()
    }
    model = transformers.AutoModelForCausalLM.from_pretrained(str(saved), local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(saved), local_files_only=True)
    ids = tokenizer("count : 7 8 9", return_tensors="pt").input_ids
    with torch.no_grad():
        assert tokenizer.decode(model(ids).logits[0, -1].argmax()) == "10"
    # The patch reverts to the original, side files included.
    reverted = workspace / "reverted"
    applied = runner.invoke(
        app,
        ["apply", str(saved), str(out / "patch"), "--out", str(reverted), "--reverse"],
    )
    assert applied.exit_code == 0, applied.output
    original = BACKEND.load_weights(fixtures_dir / "tiny_llama_defective")
    assert BACKEND.content_hash(BACKEND.load_weights(reverted)) == BACKEND.content_hash(original)
    assert (reverted / "tokenizer.json").is_file()


def test_head_surgery_example(workspace: Path) -> None:
    out = workspace / "out"
    result = runner.invoke(app, ["run", str(workspace / "head_surgery.wsm"), "--out", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads((out / "report.json").read_text())
    assert report["changes"]["changed_params"] == 4 * 16 * 64
    assert report["inspections"][0]["role"] == "layers[1].attn.heads[3]"


def test_inspect_cli_shows_profile_and_units(fixtures_dir: Path) -> None:
    result = runner.invoke(app, ["inspect", str(fixtures_dir / "tiny_llama_defective")])
    assert result.exit_code == 0, result.output
    assert "Architecture profile: family llama" in result.output
    for unit in planted(fixtures_dir)["units"]:
        assert unit in result.output
