"""One shard of a benchmark, as a job on Hugging Face Jobs (``weightsmith.evaluation.benchmarks``).

The job reads its request from ``WEIGHTSMITH_REQUEST``: the repository that holds the base model
directory and the patch, the benchmark's dataset and shard, and the protocol (prompt, sampling,
samples per problem). It rebuilds the model (the base directory with the patch's tensors in place
of the originals), samples the answers with vLLM, grades them, and uploads ``scores.json``,
``per_problem.json`` and ``first_samples.jsonl.gz`` to ``results/<key>/`` of the repository.

Prompt, answer extraction and grading are those of "Sparks of weight-space programming" (Rikiya
Takehi, 2026; https://github.com/RikiyaT/lm_understand-logs, ``exp8_math/code/evaluate.py`` and
``mathqa.py``, MIT licence): the user turn is the problem, a newline and the instruction; the
answer is the content of the last ``\\boxed{}`` after any reasoning block; math-verify decides
whether it equals the reference answer.
"""

from __future__ import annotations

import functools
import gzip
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

WORK = Path("/tmp/weightsmith")


def main() -> None:
    request = json.loads(os.environ["WEIGHTSMITH_REQUEST"])
    t0 = time.time()
    model = build_model(request)
    rows = load_rows(request)
    index, count = request["shard"]
    rows = rows[index::count]
    t1 = time.time()

    # vLLM samples top-k / top-p with FlashInfer when it can, but FlashInfer compiles that kernel
    # on first use and the job's image has no CUDA compiler; PyTorch's sampler needs none.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sampling = request["sampling"]
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("EVAL_GPU_MEM", 0.85)),
        max_model_len=sampling["max_tokens"] + 2048,
        seed=0,
    )
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["problem"] + "\n" + request["instruction"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=request["thinking"],
        )
        for r in rows
    ]
    t2 = time.time()
    outs = llm.generate(prompts, SamplingParams(n=request["samples"], **sampling))
    t3 = time.time()
    bad = sum(1 for o in outs for x in o.outputs if x.finish_reason not in ("stop", "length"))
    if bad:
        raise RuntimeError(f"{bad} generations ended without finishing or reaching max_tokens")

    per, ok, total, cut, noans, tokens = [], 0, 0, 0, 0, 0
    for r, o in zip(rows, outs, strict=True):
        marks = []
        for s in o.outputs:
            pred = extract(s.text)
            good = is_correct(pred, r["answer"])
            marks.append(int(good))
            cut += s.finish_reason == "length"
            noans += pred is None and s.finish_reason != "length"
            tokens += len(s.token_ids)
        per.append({"id": r["id"], "correct": marks, "pred_first": extract(o.outputs[0].text)})
        ok += sum(marks)
        total += len(marks)

    import torch
    import vllm

    scores = {
        "benchmark": request["benchmark"],
        "acc": ok / total,
        "n_problems": len(rows),
        "samples": request["samples"],
        "cut_share": cut / total,
        "mean_tokens": tokens / total,
        "ok": ok,
        "total": total,
        "cut": cut,
        "tokens_sum": tokens,
        "no_answer_finished": noans,
        "shard": f"{index}/{count}",
        "seconds": {"setup": round(t1 - t0), "load": round(t2 - t1), "generate": round(t3 - t2)},
        "tokens_per_second": round(tokens / max(t3 - t2, 1e-9)),
        "gpu": torch.cuda.get_device_name(0),
        "vllm": vllm.__version__,
        "request": request,
    }
    print(json.dumps({k: v for k, v in scores.items() if k != "request"}), flush=True)
    upload(request, scores, per, rows, outs)


def build_model(request: dict) -> Path:
    from huggingface_hub import snapshot_download

    base, patch = request["base"], request["patch"]
    patterns = [f"{base}/*"] + ([patch] if patch else [])
    local = Path(
        snapshot_download(request["repo"], allow_patterns=patterns, local_dir=WORK / "repo")
    )
    model = WORK / "model"
    if model.exists():
        shutil.rmtree(model)
    shutil.copytree(local / base, model)
    if not patch:
        return model
    from safetensors.torch import load_file, save_file

    changed = load_file(str(local / patch))
    done = set()
    for file in sorted(model.glob("*.safetensors")):
        tensors = load_file(str(file))
        hit = [n for n in tensors if n in changed]
        for name in hit:
            new = changed[name]
            if new.shape != tensors[name].shape or new.dtype != tensors[name].dtype:
                raise RuntimeError(f"the patch's {name} does not match the model's tensor")
            tensors[name] = new.contiguous()
            done.add(name)
        if hit:
            save_file(tensors, str(file), metadata={"format": "pt"})
    missing = set(changed) - done
    if missing:
        raise RuntimeError(f"the patch has tensors the model lacks: {sorted(missing)[:3]}")
    print(f"patched {len(done)} tensor(s)", flush=True)
    return model


def load_rows(request: dict) -> list[dict]:
    from huggingface_hub import hf_hub_download

    d = request["dataset"]
    path = hf_hub_download(d["repo"], d["file"], repo_type="dataset", revision=d["revision"])
    if request["benchmark"] == "math500":
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        return [
            {"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]} for r in rows
        ]
    if request["benchmark"] == "gsm8k":
        import pyarrow.parquet as pq

        table = pq.read_table(path).to_pylist()
        return [
            {
                "id": f"gsm8k_test_{i}",
                "problem": r["question"],
                "answer": r["answer"].split("####")[-1].strip().replace(",", ""),
            }
            for i, r in enumerate(table)
        ]
    raise ValueError(f"unknown benchmark {request['benchmark']}")


def upload(request: dict, scores: dict, per: list, rows: list, outs: list) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    first = "".join(
        json.dumps({"id": r["id"], "answer": r["answer"], "text": o.outputs[0].text}) + "\n"
        for r, o in zip(rows, outs, strict=True)
    )
    out = request["out"]
    ops = [
        CommitOperationAdd(f"{out}/per_problem.json", json.dumps(per).encode()),
        CommitOperationAdd(f"{out}/first_samples.jsonl.gz", gzip.compress(first.encode())),
        CommitOperationAdd(f"{out}/scores.json", json.dumps(scores, indent=1).encode()),
    ]
    HfApi().create_commit(request["repo"], ops, commit_message=f"results {out}")


# Answer extraction (mathqa.py: answer_text, _box_at, last_boxed, extract) ----------------------
_BOX_FORMS = re.compile(r"\\boxed|\\fbox")


def answer_text(text: str) -> str:
    """The text the extraction reads: any reasoning block cut off (up to the last '</think>')."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def _box_at(text: str, idx: int) -> str | None:
    """The content of the box that starts at text[idx] ('\\boxed' or '\\fbox'), or None."""
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key) :]
    r = rest.lstrip()
    if r.startswith("{"):  # brace matched, as hendrycks' last_boxed_only_string
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = r[1:k].strip()
                    return out or None
        return None  # not closed (e.g. a response cut inside the box)
    if key == "\\boxed" and rest.startswith(" "):  # '\\boxed 5': up to the next '$' or line end
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def last_boxed(text: str) -> str | None:
    """The content of the last '\\boxed' (else the last '\\fbox') of a text, or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract(text: str) -> str | None:
    """The answer of a response, or None."""
    return last_boxed(answer_text(text))


# Grading (mathqa.py: _timeouts, _parsed, is_correct) ------------------------------------------
def _timeouts() -> int | None:
    """math-verify's timeouts use signal.alarm, which works only in the main thread."""
    return 5 if threading.current_thread() is threading.main_thread() else None


@functools.lru_cache(maxsize=1 << 16)
def _parsed(s: str, to: int | None) -> list:
    import math_verify

    return math_verify.parse("\\boxed{" + s + "}", parsing_timeout=to)


def is_correct(pred: str | None, gold: str | None) -> bool:
    """math-verify equivalence of a boxed answer and the gold answer (False for no answer)."""
    if pred is None or gold is None:
        return False
    import math_verify

    to = _timeouts()
    g, p = _parsed(gold, to), _parsed(pred, to)
    if not g or not p:
        return False
    try:
        return bool(math_verify.verify(g, p, timeout_seconds=to))
    except Exception:
        return False


if __name__ == "__main__":
    main()
