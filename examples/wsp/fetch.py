"""Fetch the logs of "Sparks of weight-space programming" and turn them into Weightsmith inputs.

The article (Rikiya Takehi, 2026, https://rikiyat.github.io/blog/weight-space-programming/)
published every network, prompt, reply and evaluation of its experiments at
https://github.com/RikiyaT/lm_understand-logs (logs, data and networks CC BY 4.0, scripts MIT).
This script downloads that repository at a fixed commit into ``fixtures/logs`` and writes, under
``fixtures/``:

* ``which_game/<id>/``: the four policy networks of Experiment 1 as model directories
  (``model.safetensors``, ``arch.json``, and a ``view.json`` that shows them as the article did:
  ``W1`` .. ``b3``, with W as (inputs, outputs));
* ``which_game/message.txt``: the article's prompt with ``{{weights}}`` where the weights go, and
  ``which_game/instructions.md``, its one-line system instructions;
* ``which_game/article/<id>.json``: Astra's recorded answers as editor scripts, to check the
  programs offline.

It checks that the message Weightsmith will send is byte-for-byte the prompt the article sent.

    uv run python examples/wsp/fetch.py
"""

from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from weightsmith.models.view import View, ViewTensor, parse_format, render

COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
URL = f"https://codeload.github.com/RikiyaT/lm_understand-logs/tar.gz/{COMMIT}"
KEEP = ("README.md", "LICENSE", "harness/", "exp1_which_game/", "exp4_fix_car/", "exp8_math/")

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
LOGS = FIXTURES / "logs"

# Experiment 1: the networks, by the IDs the article's harness used, and their games.
WHICH_GAME = {
    "a9f535cc": "breakout",
    "dfc6cb65": "pong",
    "bdf99279": "pacman",
    "0bbcc3ea": "freeway",
}
MLP_LAYERS = (  # article name -> (Weightsmith layer, .npz prefix, header of W, header of b)
    (
        "1",
        "dense1",
        "actor/dense1",
        "W1 (layer 1 weights), shape {shape}: one line per row (row i = input i), "
        "{cols} values per line",
        "b1 (layer 1 biases), shape {shape}:",
    ),
    (
        "2",
        "dense2",
        "actor/dense2",
        "W2 (layer 2 weights), shape {shape}: one line per row",
        "b2 (layer 2 biases), shape {shape}:",
    ),
    (
        "3",
        "logits",
        "actor/logits",
        "W3 (output layer weights), shape {shape}: one line per row",
        "b3 (output layer biases), shape {shape}:",
    ),
)


def download() -> None:
    stamp = LOGS / ".commit"
    if stamp.is_file() and stamp.read_text().strip() == COMMIT:
        return
    print(f"downloading the article's logs at {COMMIT[:7]} ...")
    with urllib.request.urlopen(URL, timeout=600) as response:
        data = response.read()
    prefix = f"lm_understand-logs-{COMMIT}/"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = []
        for member in archive.getmembers():
            if not member.name.startswith(prefix):
                continue
            member.name = member.name[len(prefix) :]
            if member.name and member.name.startswith(KEEP):
                members.append(member)
        LOGS.mkdir(parents=True, exist_ok=True)
        archive.extractall(LOGS, members=members, filter="data")
    stamp.write_text(COMMIT + "\n")


def which_game() -> None:
    src = LOGS / "exp1_which_game"
    out = FIXTURES / "which_game"
    description = (src / "networks" / "description.txt").read_text()
    for net_id, game in WHICH_GAME.items():
        weights = np.load(src / "networks" / f"{net_id}.npz")
        tensors, layers, view = {}, [], []
        for index, (n, name, key, w_header, b_header) in enumerate(MLP_LAYERS):
            kernel = weights[f"{key}/kernel"]  # (inputs, outputs)
            tensors[f"{name}.weight"] = np.ascontiguousarray(kernel.T)
            tensors[f"{name}.bias"] = weights[f"{key}/bias"]
            layers.append(
                {
                    "type": "linear",
                    "name": name,
                    "in_features": kernel.shape[0],
                    "out_features": kernel.shape[1],
                }
            )
            if index < len(MLP_LAYERS) - 1:
                layers.append({"type": "tanh"})
            view.append(ViewTensor(f"W{n}", f"{name}.weight", True, w_header))
            view.append(ViewTensor(f"b{n}", f"{name}.bias", False, b_header))
        model = out / net_id
        model.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(model / "model.safetensors"))
        arch = {"format": "weightsmith.arch/v0", "kind": "sequential", "inputs": 500}
        (model / "arch.json").write_text(json.dumps({**arch, "layers": layers}, indent=1) + "\n")
        (model / "view.json").write_text(json.dumps(View(tuple(view)).to_dict(), indent=1) + "\n")

        # The message: the article's prompt with the weights replaced by a placeholder.
        run = src / "runs" / game
        prompt = (run / "prompt.txt").read_text()
        actor = (src / "networks" / f"{net_id}_actor.txt").read_text()
        preamble = actor[: actor.index("\n\nW1 (") + 2]
        question = prompt[len(description) : len(prompt) - len(actor)]
        message = description + question + preamble + "{{weights}}\n"
        rendered = render(
            View(tuple(view)),
            parse_format("decimals(3)"),
            {k: v.astype(np.float64) for k, v in tensors.items()},
        )
        if message.replace("{{weights}}", rendered) != prompt:
            raise SystemExit(f"{net_id}: the rebuilt prompt differs from the article's")
        existing = out / "message.txt"
        if existing.is_file() and existing.read_text() != message:
            raise SystemExit("the four prompts were expected to share one message")
        existing.write_text(message)
        (out / "instructions.md").write_text((run / "instructions.md").read_text())

        script = {
            "format": "weightsmith.script/v0",
            "turns": [{"text": (run / "answer.md").read_text()}],
        }
        (out / "article").mkdir(exist_ok=True)
        (out / "article" / f"{net_id}.json").write_text(json.dumps(script, indent=1) + "\n")
    print(f"which_game: 4 networks in {out.relative_to(HERE)}, prompts identical to the article's")


def main() -> None:
    download()
    which_game()


if __name__ == "__main__":
    main()
