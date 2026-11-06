"""Benchmarks: generative evaluations of a language model, run as jobs on Hugging Face Jobs.

A benchmark needs a GPU, so each shard of its problems runs as a job on Hugging Face Jobs: the uv
script ``workers/hf_bench_job.py`` samples the model's answers with vLLM and grades them. The
model goes to a private repository of the logged-in user on the Hub (``<user>/weightsmith-runs``,
or ``WEIGHTSMITH_HF_REPO``) in two parts: the model directory as it was loaded, once, under
``bases/<hash>/``, and the tensors the program changed, under ``patches/<hash>.safetensors``.
Each job writes its scores under ``results/<key>/``, where the key hashes everything the result
depends on (base, patch, benchmark, shard, protocol, pinned packages), so a shard that already ran
is read back, from the local cache or from the repository, and never paid for twice.

Credentials come from the Hugging Face login (``hf auth login`` or ``HF_TOKEN``), never from a
program; the jobs receive the token as a secret.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from weightsmith.errors import UnsupportedOperation
from weightsmith.evaluation.probes import Benchmark
from weightsmith.evaluation.rollouts import WORKERS, cache_dir

JOB_SCRIPT = WORKERS / "hf_bench_job.py"
REPO_NAME = "weightsmith-runs"
# Packages of the job, pinned: they are part of every result's key.
DEPENDENCIES = ("vllm==0.32.0", "math-verify==0.9.0", "pyarrow==26.0.1")
PYTHON = "3.12"

# The test split of each benchmark, at a fixed revision.
DATASETS: dict[str, dict[str, str]] = {
    "math500": {
        "repo": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "file": "test.jsonl",
    },
    "gsm8k": {
        "repo": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "file": "main/test-00000-of-00001.parquet",
    },
}

# Prompt and sampling. think35 is the thinking-mode setting of the Qwen3.5 model card, as used by
# "Sparks of weight-space programming" (Takehi, 2026) to evaluate Qwen3.5-0.8B.
PROTOCOLS: dict[str, dict[str, Any]] = {
    "think35": {
        "thinking": True,
        "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "max_tokens": 32768,
            "seed": 0,
        },
    },
}

TERMINAL = ("COMPLETED", "ERROR", "CANCELED", "DELETED")
POLL_SECONDS = 30.0


def say(text: str) -> None:
    print(f"weightsmith: {text}", file=sys.stderr, flush=True)


@dataclass(slots=True)
class Shard:
    index: int
    key: str
    request: dict[str, Any]
    scores: dict[str, Any] | None = None
    job: str | None = None
    started: float = 0.0
    running_secs: int = 0
    cached: bool = False
    log: list[str] = field(default_factory=list)


def run_benchmark(
    spec: Benchmark, directory: Path, tensors: dict[str, Any], api: Any = None
) -> dict[str, Any]:
    """Evaluate the model loaded from ``directory``, its tensors now ``tensors``, on ``spec``."""
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - the hf extra brings it
            raise UnsupportedOperation(
                "benchmarks need huggingface_hub: pip install 'weightsmith[hf]'"
            ) from exc
        api = HfApi()
    token = _token(api)
    repo = os.environ.get("WEIGHTSMITH_HF_REPO") or f"{api.whoami()['name']}/{REPO_NAME}"
    files = _model_files(directory)
    base = _digest({name: _file_hash(directory / name) for name in files})
    patch, patch_key = _patch(directory, files, tensors)

    shards = []
    for index in spec.selected:
        request = {
            "repo": repo,
            "base": f"bases/{base}",
            "patch": f"patches/{patch_key}.safetensors" if patch_key else None,
            "benchmark": spec.name,
            "dataset": DATASETS[spec.name],
            "shard": [index, spec.shards],
            "samples": spec.samples,
            "protocol": spec.protocol,
            **PROTOCOLS[spec.protocol],
            "dependencies": list(DEPENDENCIES),
        }
        key = _digest(request)
        shards.append(Shard(index, key, {**request, "out": f"results/{key}"}))
    todo = [s for s in shards if not _cached(api, repo, s)]
    say(
        f"{spec.name}: {len(shards)} shard(s), {len(shards) - len(todo)} already evaluated, "
        f"{len(todo)} to run on {spec.flavor}"
    )
    if todo:
        api.create_repo(repo, private=True, exist_ok=True)
        _upload_base(api, repo, base, directory, files)
        if patch is not None:
            _upload_patch(api, repo, patch_key, patch)
        _run_jobs(api, spec, todo, token)
    return _summary(api, spec, shards)


# the model ------------------------------------------------------------------------------------
def _model_files(directory: Path) -> list[str]:
    from weightsmith.models.hf import SIDE_FILES

    names = [n for n in SIDE_FILES if (directory / n).is_file()]
    names += sorted(p.name for p in directory.glob("*.safetensors"))
    if (directory / "model.safetensors.index.json").is_file():
        names.append("model.safetensors.index.json")
    if "config.json" not in names:
        raise UnsupportedOperation(f"{directory} is not a Hugging Face model directory")
    return names


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def _patch(directory: Path, files: list[str], tensors: dict[str, Any]) -> tuple[Any, str | None]:
    """The tensors whose bits differ from the files, in the files' dtypes, and their hash."""
    import torch
    from safetensors import safe_open

    changed: dict[str, torch.Tensor] = {}
    seen: set[str] = set()
    for name in files:
        if not name.endswith(".safetensors"):
            continue
        with safe_open(str(directory / name), framework="pt") as f:
            for tensor in f.keys():  # noqa: SIM118 - safe_open is not a mapping
                seen.add(tensor)
                if tensor not in tensors:
                    continue
                original = f.get_tensor(tensor)
                current = tensors[tensor].detach().cpu().to(original.dtype).contiguous()
                if current.shape != original.shape:
                    raise UnsupportedOperation(f"'{tensor}' changed shape; a patch cannot do that")
                if not torch.equal(_bits(current), _bits(original)):
                    changed[tensor] = current
    extra = sorted(set(tensors) - seen)
    if extra:
        raise UnsupportedOperation(f"'{extra[0]}' is not in the model's files")
    if not changed:
        return None, None
    digest = hashlib.sha256()
    for name in sorted(changed):
        t = changed[name]
        digest.update(f"{name}|{t.dtype}|{list(t.shape)}|".encode())
        digest.update(_bits(t).numpy().tobytes())
    return changed, digest.hexdigest()[:20]


def _bits(t: Any) -> Any:
    import torch

    return t.reshape(-1).view(torch.uint8)


# the repository -------------------------------------------------------------------------------
def _token(api: Any) -> str:
    token = getattr(api, "token", None) or os.environ.get("HF_TOKEN")
    if not token:
        from huggingface_hub import get_token

        token = get_token()
    if not token:
        raise UnsupportedOperation(
            "benchmarks run on Hugging Face Jobs: log in with 'hf auth login' or set HF_TOKEN"
        )
    return token


def _upload_base(api: Any, repo: str, base: str, directory: Path, files: list[str]) -> None:
    if api.file_exists(repo, f"bases/{base}/files.json"):
        return
    from huggingface_hub import CommitOperationAdd

    say(f"uploading the model directory {directory} to {repo}/bases/{base} (once)")
    ops = [CommitOperationAdd(f"bases/{base}/{n}", str(directory / n)) for n in files]
    ops.append(CommitOperationAdd(f"bases/{base}/files.json", json.dumps(files).encode()))
    api.create_commit(repo, ops, commit_message=f"base {base}")


def _upload_patch(api: Any, repo: str, key: str | None, patch: dict[str, Any]) -> None:
    path = f"patches/{key}.safetensors"
    if api.file_exists(repo, path):
        return
    import tempfile

    from huggingface_hub import CommitOperationAdd
    from safetensors.torch import save_file

    say(f"uploading {len(patch)} changed tensor(s) to {repo}/{path}")
    with tempfile.TemporaryDirectory(prefix="weightsmith-patch-") as tmp:
        file = Path(tmp) / "patch.safetensors"
        save_file(patch, str(file), metadata={"format": "pt"})
        api.create_commit(
            repo, [CommitOperationAdd(path, str(file))], commit_message=f"patch {key}"
        )


def _local(shard: Shard) -> Path:
    return cache_dir() / "benchmarks" / shard.key / "scores.json"


def _cached(api: Any, repo: str, shard: Shard) -> bool:
    local = _local(shard)
    if not local.is_file() and api.file_exists(repo, f"results/{shard.key}/scores.json"):
        _download(api, repo, shard)
    if local.is_file():
        shard.scores = json.loads(local.read_text(encoding="utf-8"))
        shard.cached = True
    return shard.cached


def _download(api: Any, repo: str, shard: Shard) -> None:
    local = _local(shard)
    local.parent.mkdir(parents=True, exist_ok=True)
    for name in ("per_problem.json", "scores.json"):  # scores.json last: it marks the shard done
        path = api.hf_hub_download(repo, f"results/{shard.key}/{name}")
        (local.parent / name).write_bytes(Path(path).read_bytes())


# the jobs -------------------------------------------------------------------------------------
def _run_jobs(api: Any, spec: Benchmark, todo: list[Shard], token: str) -> None:
    waiting = list(todo)
    running: list[Shard] = []
    try:
        while waiting or running:
            while waiting and len(running) < spec.parallel:
                shard = waiting.pop(0)
                _launch(api, spec, shard, token)
                running.append(shard)
            time.sleep(POLL_SECONDS)
            for shard in list(running):
                if _finished(api, spec, shard):
                    running.remove(shard)
    except BaseException:
        for shard in running:
            say(f"cancelling job {shard.job}")
            try:
                api.cancel_job(job_id=shard.job)
            except Exception:
                say(f"could not cancel job {shard.job}; cancel it with: hf jobs cancel {shard.job}")
        raise


def _launch(api: Any, spec: Benchmark, shard: Shard, token: str) -> None:
    job = api.run_uv_job(
        str(JOB_SCRIPT),
        dependencies=list(DEPENDENCIES),
        python=PYTHON,
        env={"WEIGHTSMITH_REQUEST": json.dumps(shard.request)},
        secrets={"HF_TOKEN": token},
        flavor=spec.flavor,
        timeout=int(spec.timeout_hours * 3600),
        labels={"weightsmith": "benchmark", "benchmark": spec.name},
    )
    shard.job = job.id
    shard.started = time.monotonic()
    say(f"{spec.name} shard {shard.index}/{spec.shards}: job {job.id} on {spec.flavor}")


def _finished(api: Any, spec: Benchmark, shard: Shard) -> bool:
    info = api.inspect_job(job_id=shard.job)
    stage = str(getattr(info.status, "stage", "")).removeprefix("JobStage.")
    durations = getattr(info, "durations", None)
    shard.running_secs = int(getattr(durations, "running_secs", None) or 0)
    if stage not in TERMINAL:
        return False
    if stage != "COMPLETED":
        tail = list(api.fetch_job_logs(job_id=shard.job, tail=20))
        raise UnsupportedOperation(
            f"{spec.name} shard {shard.index}/{spec.shards} (job {shard.job}) ended {stage}:\n"
            + "\n".join(tail)
        )
    _download(api, shard.request["repo"], shard)
    s = shard.scores = json.loads(_local(shard).read_text(encoding="utf-8"))
    say(
        f"{spec.name} shard {shard.index}/{spec.shards} done: {s['ok']}/{s['total']} correct, "
        f"{s['cut']} cut at the token limit, {shard.running_secs / 60:.1f} min running"
    )
    return True


def _summary(api: Any, spec: Benchmark, shards: list[Shard]) -> dict[str, Any]:
    total = {k: sum(s.scores[k] for s in shards if s.scores) for k in ("ok", "total", "cut")}
    tokens = sum(s.scores["tokens_sum"] for s in shards if s.scores)
    if total["total"] != spec.problems * spec.samples:
        raise UnsupportedOperation(
            f"{spec.name}: {total['total']} graded answers, not {spec.problems * spec.samples}"
        )
    seconds = sum(s.running_secs for s in shards if not s.cached)
    price = _price_per_hour(api, spec.flavor)
    result = {
        "benchmark": spec.name,
        "accuracy": total["ok"] / total["total"],
        "correct": total["ok"],
        "answers": total["total"],
        "cut_share": total["cut"] / total["total"],
        "mean_tokens": tokens / total["total"],
        "shards": [
            {"index": s.index, "key": s.key, "job": s.job, "cached": s.cached} for s in shards
        ],
        "gpu_hours": seconds / 3600,
        "cost_usd": None if price is None else round(seconds / 3600 * price, 2),
    }
    cost = "" if result["cost_usd"] is None else f" (about ${result['cost_usd']:.2f})"
    say(
        f"{spec.name}: {result['accuracy']:.1%} ({total['ok']}/{total['total']}), "
        f"{result['cut_share']:.1%} cut at the token limit; "
        f"{result['gpu_hours']:.2f} GPU-hours on {spec.flavor} in this run{cost}"
    )
    return result


def _price_per_hour(api: Any, flavor: str) -> float | None:
    try:
        for hw in api.list_jobs_hardware():
            if hw.name == flavor:
                per = {"minute": 60, "hour": 1}.get(hw.unit_label)
                return hw.unit_cost_usd * per if per else None
    except Exception:
        return None
    return None
