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
  programs offline;
* ``fix_car/shown/``: the car policy of Experiment 4 (Kinetix actor, 679 -> 128 -> 5 x 128 -> 16)
  as Astra was shown it, every tensor rounded to 3 significant digits of its largest entry, with
  a view that names and shows its tensors as the article did (``embed.W`` .. ``actor_head.b``);
* ``fix_car/message.txt`` and ``fix_car/instructions.md``: the article's message and system
  instructions; ``fix_car/car_search.json`` (256 episodes) and ``fix_car/car_test.json`` (1,024
  other episodes): rollouts suites in the car-on-ramp level; ``fix_car/article/oneshot.json``:
  Astra's recorded reply as an editor script.

It checks that every message Weightsmith will send is byte-for-byte the prompt the article sent.

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

from weightsmith.models.view import View, ViewTensor, parse_format, render, round_like_shown

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


# Experiment 4: the Kinetix actor, by Weightsmith layer name and Flax parameter path.
_KINETIX = "params/GeneralActorCriticRNN_0/"
CAR_LAYERS = (
    ("embed", "params/Dense_0"),
    *((f"actor{k}", f"{_KINETIX}DenseAndActivation_{k}/Dense_0") for k in range(5)),
    ("actor_head", f"{_KINETIX}Dense_0"),
)
CAR_LEVEL = "l/h2_car_ramp"


def fix_car() -> None:
    src = LOGS / "exp4_fix_car"
    out = FIXTURES / "fix_car"
    fmt = parse_format("int_scale(3)")
    weights = np.load(src / "networks" / "ppo_final_seed0.npz")
    tensors, layers, view = {}, [], []
    for name, key in CAR_LAYERS:
        kernel = round_like_shown(weights[f"{key}/kernel"].astype(np.float64), fmt)  # (in, out)
        bias = round_like_shown(weights[f"{key}/bias"].astype(np.float64), fmt)
        tensors[f"{name}.weight"] = np.ascontiguousarray(kernel.T).astype(np.float32)
        tensors[f"{name}.bias"] = bias.astype(np.float32)
        layers.append(
            {
                "type": "linear",
                "name": name,
                "in_features": kernel.shape[0],
                "out_features": kernel.shape[1],
            }
        )
        if name == "embed":
            layers.append({"type": "relu"})
        elif name != "actor_head":
            layers.append({"type": "tanh"})
        header = "{label} ({shape}), scale {scale}: one line per row (row i)"
        view.append(ViewTensor(f"{name}.W", f"{name}.weight", True, header))
        view.append(ViewTensor(f"{name}.b", f"{name}.bias"))
    model = out / "shown"
    model.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(model / "model.safetensors"))
    arch = {"format": "weightsmith.arch/v0", "kind": "sequential", "inputs": 679}
    (model / "arch.json").write_text(json.dumps({**arch, "layers": layers}, indent=1) + "\n")
    (model / "view.json").write_text(json.dumps(View(tuple(view)).to_dict(), indent=1) + "\n")

    # The message: the article's prompt, whose last part is every weight.
    run = src / "runs" / "oneshot_repair"
    prompt = (run / "reply" / "prompt.txt").read_text()
    start = prompt.index("\nembed.W (679 x 128), scale") + 1
    rendered = render(View(tuple(view)), fmt, {k: v.astype(np.float64) for k, v in tensors.items()})
    message = prompt[:start] + "{{weights}}" + prompt[start + len(rendered) :]
    if message.replace("{{weights}}", rendered) != prompt:
        raise SystemExit("fix_car: the rebuilt prompt differs from the article's")
    (out / "message.txt").write_text(message)
    (out / "instructions.md").write_text((run / "instructions.md").read_text())

    for name, episodes, seed in (("car_search", 256, 0), ("car_test", 1024, 1)):
        suite = {
            "format": "weightsmith.probes/v0",
            "kind": "rollouts",
            "name": name,
            "description": f"{episodes} episodes of the car-on-ramp level, actions sampled",
            "rollouts": {
                "env": "kinetix",
                "level": CAR_LEVEL,
                "episodes": episodes,
                "seed": seed,
                "mode": "sample",
            },
        }
        (out / f"{name}.json").write_text(json.dumps(suite, indent=1) + "\n")

    script = {
        "format": "weightsmith.script/v0",
        "turns": [{"text": (run / "reply" / "answer.md").read_text()}],
    }
    (out / "article").mkdir(exist_ok=True)
    (out / "article" / "oneshot.json").write_text(json.dumps(script, indent=1) + "\n")
    print(
        f"fix_car: the shown network in {out.relative_to(HERE)}, prompt identical to the article's"
    )


def main() -> None:
    download()
    which_game()
    fix_car()


if __name__ == "__main__":
    main()
