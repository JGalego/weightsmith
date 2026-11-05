"""Rollouts: episodes of a policy network in a simulated environment, run by a worker.

An environment such as Kinetix (JAX) has dependencies that cannot share an environment with
Weightsmith's own, so its episodes run in a separate Python environment: the worker script in
``weightsmith/workers`` reads one JSON request on stdin, with the path of a safetensors file of
the policy's weights, and writes one JSON result on stdout. The worker's environment is found
through ``WEIGHTSMITH_<ENV>_PYTHON`` or, by default, in Weightsmith's cache directory; see
``workers/README.md`` for how to build it.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from weightsmith.errors import UnsupportedOperation
from weightsmith.evaluation.probes import Rollouts

WORKERS = Path(__file__).resolve().parent.parent / "workers"


def cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "weightsmith" / "workers"


def worker_python(env: str) -> Path:
    configured = os.environ.get(f"WEIGHTSMITH_{env.upper()}_PYTHON")
    if configured:
        return Path(configured)
    return cache_dir() / env / ".venv" / "bin" / "python"


def run_rollouts(spec: Rollouts, arrays: dict[str, Any]) -> dict[str, Any]:
    """Run the episodes ``spec`` describes with the policy whose weights are ``arrays``."""
    import numpy as np
    from safetensors.numpy import save_file

    python = worker_python(spec.env)
    if not python.is_file():
        raise UnsupportedOperation(
            f"the '{spec.env}' rollout worker is not set up: {python} does not exist "
            f"(see {WORKERS / 'README.md'}, or set WEIGHTSMITH_{spec.env.upper()}_PYTHON)"
        )
    with tempfile.TemporaryDirectory(prefix="weightsmith-rollouts-") as tmp:
        weights = Path(tmp) / "model.safetensors"
        save_file({k: np.ascontiguousarray(v) for k, v in arrays.items()}, str(weights))
        request = {
            "weights": str(weights),
            "level": spec.level,
            "episodes": spec.episodes,
            "seed": spec.seed,
            "mode": spec.mode,
        }
        done = subprocess.run(
            [str(python), "-I", str(WORKERS / f"{spec.env}_worker.py")],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
        )
    if done.returncode != 0:
        tail = "\n".join(done.stderr.strip().splitlines()[-5:])
        raise UnsupportedOperation(f"the '{spec.env}' rollout worker failed:\n{tail}")
    try:
        result = json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise UnsupportedOperation(
            f"the '{spec.env}' rollout worker wrote no JSON result: {exc}"
        ) from exc
    if result.get("episodes") != spec.episodes:
        raise UnsupportedOperation(
            f"the '{spec.env}' rollout worker ran {result.get('episodes')} episodes, "
            f"not {spec.episodes}"
        )
    return result
