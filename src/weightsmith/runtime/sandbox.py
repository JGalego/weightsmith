"""Run Python an editor wrote, in a separate process, on a copy of the weights.

The child process gets the working copy as a dict ``W`` of writable NumPy arrays (``float32`` for
dtypes NumPy lacks, such as ``bfloat16``), plus ``np``. It may import ``torch`` itself. Arrays it
changes in place, or replaces with an array of the same shape, come back as *proposed* changes;
the runtime then applies them through the same budget checks as any other edit.

Each call starts a fresh interpreter in the delegation's working directory, so files written
there persist between calls but variables do not.

This is **not a security boundary**. The child runs with this user's permissions. Its environment
is stripped of credentials (no API keys reach it) and its wall time is limited, but it can read
files and open network connections. Allow ``execute`` only for editors you would let run code on
this machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

Tensors = dict[str, torch.Tensor]
OUTPUT_LIMIT = 20000  # characters of stdout and stderr returned to the editor

_RUNNER = r"""
import hashlib, sys, traceback
import numpy as np
from safetensors.numpy import load_file, save_file

def digest(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()

W = {k: np.array(v) for k, v in load_file("weights.safetensors").items()}
before = {k: (v.shape, v.dtype, digest(v)) for k, v in W.items()}
code = open("cell.py", encoding="utf-8").read()
try:
    exec(compile(code, "cell", "exec"), {"W": W, "np": np, "__name__": "__main__"})
except BaseException:
    traceback.print_exc()
    sys.exit(1)
changed, problems = {}, []
for name, value in W.items():
    if name not in before:
        problems.append(f"'{name}' is not a parameter; new tensors cannot be added")
        continue
    shape, dtype, original = before[name]
    value = np.ascontiguousarray(np.asarray(value).astype(dtype, copy=False))
    if value.shape != shape:
        problems.append(f"'{name}' changed shape from {list(shape)} to {list(value.shape)}")
    elif digest(value) != original:
        changed[name] = value
if problems:
    print("\n".join(problems), file=sys.stderr)
    sys.exit(2)
if changed:
    save_file(changed, "changed.safetensors")
"""


@dataclass(slots=True)
class CellResult:
    stdout: str
    stderr: str
    returncode: int
    seconds: float
    changed: Tensors = field(default_factory=dict)
    timed_out: bool = False


class Sandbox:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "runner.py").write_text(_RUNNER, encoding="utf-8")
        self._written_version: int | None = None

    def _write_weights(self, tensors: Tensors, version: int) -> None:
        if self._written_version == version:
            return
        exported = {
            name: (t.float() if t.dtype in (torch.bfloat16, torch.float16) else t).contiguous()
            for name, t in tensors.items()
        }
        save_file(exported, str(self.directory / "weights.safetensors"))
        self._written_version = version

    def run(self, code: str, tensors: Tensors, version: int, timeout: float) -> CellResult:
        self._write_weights(tensors, version)
        (self.directory / "cell.py").write_text(code, encoding="utf-8")
        changed_file = self.directory / "changed.safetensors"
        changed_file.unlink(missing_ok=True)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.directory),
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
        }
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "runner.py"],
                cwd=self.directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=max(timeout, 0.1),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CellResult(
                _text(exc.stdout), _text(exc.stderr), -1, time.monotonic() - start, timed_out=True
            )
        seconds = time.monotonic() - start
        changed: Tensors = {}
        if proc.returncode == 0 and changed_file.is_file():
            for name, value in load_file(str(changed_file)).items():
                original = tensors[name]
                changed[name] = value.to(original.dtype)
        return CellResult(
            proc.stdout[-OUTPUT_LIMIT:],
            proc.stderr[-OUTPUT_LIMIT:],
            proc.returncode,
            seconds,
            changed,
        )


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return (value.decode(errors="replace") if isinstance(value, bytes) else value)[-OUTPUT_LIMIT:]
