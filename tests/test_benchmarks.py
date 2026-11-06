"""Benchmark suites, run as Hugging Face Jobs against a fake Hub (no network, no GPU)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from weightsmith.backends.pytorch import PyTorchBackend
from weightsmith.compiler.driver import CompileResult
from weightsmith.evaluation import benchmarks
from weightsmith.evaluation.probes import Benchmark, ProbeSuite
from weightsmith.runtime.context import RunOptions
from weightsmith.runtime.executor import execute

from .conftest import codes

Compile = Callable[[str], CompileResult]
WSP = Path(__file__).resolve().parent.parent / "examples" / "wsp"
EDITED = "model.layers.0.mlp.down_proj.weight"


class FakeHub:
    """The parts of HfApi a benchmark uses. A job completes on its second inspection and writes
    scores: 10% correct for the unpatched model, 50% for a patched one."""

    def __init__(self, tmp: Path, stage: str = "COMPLETED") -> None:
        self.tmp = tmp
        self.stage = stage
        self.token = "hf_test_token"
        self.files: dict[str, bytes | str] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.polls: dict[str, int] = {}
        self.cancelled: list[str] = []
        self.most_running = 0

    def whoami(self) -> dict[str, str]:
        return {"name": "tester"}

    def create_repo(self, repo: str, private: bool, exist_ok: bool) -> None:
        assert private
        assert exist_ok

    def file_exists(self, repo: str, path: str) -> bool:
        return path in self.files

    def create_commit(self, repo: str, operations: list[Any], commit_message: str) -> None:
        for op in operations:
            data = op.path_or_fileobj  # a temporary file is read now; a model file stays a path
            if isinstance(data, str) and Path(data).stat().st_size < 1 << 26:
                data = Path(data).read_bytes()
            self.files[op.path_in_repo] = data

    def bytes(self, path: str) -> bytes:
        data = self.files[path]
        return data if isinstance(data, bytes) else Path(data).read_bytes()

    def hf_hub_download(self, repo: str, path: str) -> str:
        file = self.tmp / "downloads" / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(self.bytes(path))
        return str(file)

    def run_uv_job(self, script: str, **kw: Any) -> SimpleNamespace:
        running = sum(1 for j in self.jobs if self.polls[j] < 2) + 1
        self.most_running = max(self.most_running, running)
        job = f"job{len(self.jobs)}"
        self.jobs[job] = {"script": script, **kw}
        self.polls[job] = 0
        return SimpleNamespace(id=job)

    def inspect_job(self, job_id: str) -> SimpleNamespace:
        self.polls[job_id] += 1
        done = self.polls[job_id] >= 2
        if done and self.stage == "COMPLETED":
            request = json.loads(self.jobs[job_id]["env"]["WEIGHTSMITH_REQUEST"])
            index, count = request["shard"]
            problems = len(range(index, 500, count))
            total = problems * request["samples"]
            ok = total // 2 if request["patch"] else total // 10
            scores = {"ok": ok, "total": total, "cut": total - ok, "tokens_sum": 7 * total}
            self.files[f"{request['out']}/per_problem.json"] = b"[]"
            self.files[f"{request['out']}/scores.json"] = json.dumps(scores).encode()
        stage = self.stage if done else "RUNNING"
        return SimpleNamespace(
            status=SimpleNamespace(stage=stage), durations=SimpleNamespace(running_secs=900)
        )

    def fetch_job_logs(self, job_id: str, tail: int) -> list[str]:
        return ["Traceback (most recent call last):", "torch.OutOfMemoryError: CUDA out of memory"]

    def cancel_job(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    def list_jobs_hardware(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="a100-large", unit_cost_usd=2.5 / 60, unit_label="minute")]


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    import huggingface_hub

    fake = FakeHub(tmp_path)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)
    monkeypatch.setattr(benchmarks, "POLL_SECONDS", 0.0)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("WEIGHTSMITH_HF_REPO", raising=False)
    return fake


def _suite(workspace: Path, **benchmark: Any) -> None:
    spec = {"name": "math500", "samples": 2, "shards": 100, "only_shards": [0, 1], **benchmark}
    suite = {
        "format": "weightsmith.probes/v0",
        "kind": "benchmark",
        "name": "math",
        "benchmark": spec,
    }
    (workspace / "suites").mkdir(exist_ok=True)
    (workspace / "suites" / "math.json").write_text(json.dumps(suite))


PROGRAM = """
model LM = load("fixtures/tiny_llama")
behavior math = probes("suites/math.json", metric = accuracy)
program P {
    target LM
    policy { allow mutate  allow probe(max_gpu_hours = GPU) }
    edit { scale LM.params["model.layers.0.mlp.down_proj.weight"] (factor = 0.5) }
    contract { improve math (min_delta = 0.3) }
    verify
}
"""


def _run(workspace: Path, compile_text: Compile, gpu: str = "10") -> dict[str, Any]:
    result = compile_text(PROGRAM.replace("GPU", gpu))
    assert result.ok, [d.message for d in result.sink.items]
    out = workspace / f"out{len(list(workspace.glob('out*')))}"
    report = execute(result.program("P"), PyTorchBackend(), RunOptions(out_dir=out))
    return report.to_dict()


def test_benchmark_suites_count_answers_and_gpu_hours() -> None:
    spec = Benchmark(name="math500", samples=4, shards=10, only_shards=[0], timeout_hours=1.5)
    suite = ProbeSuite(name="m", kind="benchmark", benchmark=spec)
    assert (spec.problems, suite.n, suite.gpu_hours) == (50, 200, 1.5)
    assert Benchmark(name="gsm8k", samples=2, shards=10).problems == 1319
    for bad in ({"only_shards": [10]}, {"only_shards": [1, 1]}, {"shards": 501}):
        with pytest.raises(ValidationError):
            Benchmark(name="math500", samples=1, **{"shards": 10, **bad})
    with pytest.raises(ValidationError):
        ProbeSuite(name="m", kind="benchmark")
    with pytest.raises(ValidationError):
        ProbeSuite.model_validate(
            {"name": "m", "benchmark": spec, "examples": [{"input": [1], "label": 0}]}
        )


def test_a_benchmark_runs_jobs_and_sends_only_the_changed_tensors(
    workspace: Path, compile_text: Compile, hub: FakeHub
) -> None:
    _suite(workspace)
    verdict = _run(workspace, compile_text)["verdicts"][0]
    assert verdict["status"] == "verified"
    assert verdict["measurements"]["before"] == 0.1
    assert verdict["measurements"]["after"] == 0.5
    assert len(hub.jobs) == 4  # two shards, before and after
    requests = [json.loads(j["env"]["WEIGHTSMITH_REQUEST"]) for j in hub.jobs.values()]
    assert [r["shard"] for r in requests] == [[0, 100], [1, 100]] * 2
    assert [r["patch"] is None for r in requests] == [True, True, False, False]
    assert {r["repo"] for r in requests} == {"tester/weightsmith-runs"}
    assert requests[0]["sampling"]["max_tokens"] == 32768
    job = next(iter(hub.jobs.values()))
    assert job["secrets"] == {"HF_TOKEN": "hf_test_token"}
    assert job["flavor"] == "a100-large"
    assert job["timeout"] == 7200
    assert Path(job["script"]).name == "hf_bench_job.py"

    from safetensors.torch import load as load_bytes

    patches = [p for p in hub.files if p.startswith("patches/")]
    assert len(patches) == 1
    patch = load_bytes(hub.bytes(patches[0]))
    assert list(patch) == [EDITED]
    bases = {p.split("/")[1] for p in hub.files if p.startswith("bases/")}
    assert len(bases) == 1
    assert "config.json" in json.loads(hub.bytes(f"bases/{bases.pop()}/files.json"))


def test_finished_shards_are_never_run_again(
    workspace: Path, compile_text: Compile, hub: FakeHub, tmp_path: Path
) -> None:
    _suite(workspace)
    _run(workspace, compile_text)
    assert len(hub.jobs) == 4
    verdict = _run(workspace, compile_text)["verdicts"][0]  # from the local cache
    assert len(hub.jobs) == 4
    assert verdict["measurements"]["after"] == 0.5
    for file in (tmp_path / "cache").rglob("scores.json"):
        file.unlink()
    verdict = _run(workspace, compile_text)["verdicts"][0]  # from the repository
    assert len(hub.jobs) == 4
    assert verdict["measurements"]["after"] == 0.5


def test_jobs_run_at_most_parallel_at_once(
    workspace: Path, compile_text: Compile, hub: FakeHub
) -> None:
    _suite(workspace, only_shards=None, shards=4, parallel=2)
    _run(workspace, compile_text, gpu="100")
    assert len(hub.jobs) == 8
    assert hub.most_running == 2


def test_a_failed_job_stops_the_run_with_its_log(
    workspace: Path, compile_text: Compile, hub: FakeHub
) -> None:
    hub.stage = "ERROR"
    _suite(workspace)
    report = _run(workspace, compile_text)
    assert "CUDA out of memory" in report["stop_reason"]


def test_an_interrupt_cancels_the_running_jobs(
    tmp_path: Path, hub: FakeHub, fixtures_dir: Path
) -> None:
    from weightsmith.backends.pytorch.weights import load_weights

    def interrupted(job_id: str) -> None:
        raise KeyboardInterrupt

    hub.inspect_job = interrupted  # type: ignore[method-assign]
    spec = Benchmark(name="math500", samples=1, shards=100, only_shards=[0, 1], parallel=2)
    tensors = load_weights(fixtures_dir / "tiny_llama").tensors
    with pytest.raises(KeyboardInterrupt):
        benchmarks.run_benchmark(spec, fixtures_dir / "tiny_llama", tensors)
    assert hub.cancelled == ["job0", "job1"]


def test_the_gpu_budget_caps_what_jobs_can_cost(
    workspace: Path, compile_text: Compile, hub: FakeHub
) -> None:
    _suite(workspace, timeout_hours=1.5)  # two jobs: at most 3 GPU-hours per evaluation
    report = _run(workspace, compile_text, gpu="4")
    assert report["verdicts"][0]["status"] == "inconclusive"
    assert "max_gpu_hours" in report["verdicts"][0]["reason"]
    assert len(hub.jobs) == 2  # the baseline ran; the second evaluation did not start
    assert report["budgets"]["gpu_hours"] == {"reserved": 3.0, "limit": 4.0}


def test_benchmark_suites_need_a_language_model(workspace: Path, compile_text: Compile) -> None:
    _suite(workspace)
    source = (
        'model Base = load("fixtures/tiny_mlp")\n'
        'behavior math = probes("suites/math.json", metric = accuracy)\n'
        "program P {\n target Base\n policy { allow probe }\n"
        " contract { preserve math (tolerance = 0.0) }\n verify\n}\n"
    )
    assert "E512" in codes(compile_text(source))
    assert "E515" in codes(compile_text(source.replace("accuracy", "success")))


def _job_module() -> Any:
    path = Path(benchmarks.JOB_SCRIPT)
    spec = importlib.util.spec_from_file_location("hf_bench_job", path)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("so \\boxed{\\frac{1}{2}} and then \\boxed{3}", "3"),
        ("<think>maybe \\boxed{1}</think>The answer is \\boxed{ 2 }.", "2"),
        ("<think>\\boxed{1} but I ran out", "1"),
        ("nested \\boxed{\\sqrt{x^{2}}}", "\\sqrt{x^{2}}"),
        ("Qwen style: $\\boxed 5$", "5"),
        ("\\fbox{7}", "7"),
        ("cut inside \\boxed{1 +", None),
        ("empty \\boxed{}", None),
        ("no box at all", None),
    ],
)
def test_the_job_reads_answers_as_the_article_did(text: str, answer: str | None) -> None:
    assert _job_module().extract(text) == answer


# --------------------------------------------------------------------------- the article itself

needs_math = pytest.mark.skipif(
    not (WSP / "fixtures" / "math" / "qwen35" / "model.safetensors").is_file(),
    reason="run examples/wsp/fetch.py to get Qwen3.5-0.8B and the p50c edit",
)


@needs_math
def test_the_math_eval_program_sends_exactly_the_articles_edit(
    hub: FakeHub, tmp_path: Path
) -> None:
    """The pilot's patch is the article's p50c edit, bit for bit; the base is uploaded as is."""
    from weightsmith.compiler.driver import compile_file

    result = compile_file(WSP / "math_pilot.wsm")
    assert result.ok, [d.message for d in result.sink.items]
    report = execute(result.program("MathPilot"), PyTorchBackend(), RunOptions(out_dir=tmp_path))
    assert report.to_dict()["verdicts"][0]["status"] == "verified"
    requests = [json.loads(j["env"]["WEIGHTSMITH_REQUEST"]) for j in hub.jobs.values()]
    assert [r["shard"] for r in requests] == [[0, 10], [0, 10]]
    assert requests[0]["patch"] is None
    assert requests[1]["patch"] is not None

    from safetensors.torch import load as load_bytes

    patch = load_bytes(hub.bytes(requests[1]["patch"]))
    manifest = json.loads(
        (WSP / "fixtures/logs/exp8_math/runs/p50c/edit/manifest.json").read_text()
    )
    assert sorted(patch) == sorted(t["name"] for t in manifest["tensors"])
    for t in manifest["tensors"]:
        import torch

        raw = patch[t["name"]].contiguous().view(torch.int16).numpy().tobytes()
        assert hashlib.sha256(raw).hexdigest() == t["edited_sha256"]
    base = requests[0]["base"]
    uploaded = json.loads(hub.bytes(f"{base}/files.json"))
    assert "model.safetensors" in uploaded
    assert "chat_template.jinja" in uploaded
