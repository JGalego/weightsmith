"""A tiny Llama-style language model with known behaviours and planted defects.

The model is a real Hugging Face ``LlamaForCausalLM`` (2 layers, 4 query heads sharing 2 key/value
heads, 256 MLP neurons per layer) with a word-level tokenizer, saved as an ordinary Hugging Face
model directory. It is trained on three toy tasks, each a next-token prediction:

* ``count``:    ``count : 3 4 5``          -> ``6``
* ``opposite``: ``opposite : hot``         -> ``cold``
* ``first``:    ``first : big up old wet`` -> ``big`` (needs attention back to position 3)

Building it needs the ``hf`` extra (``transformers``). Everything runs on CPU from fixed seeds.
"""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from weightsmith.backends.pytorch.weights import Tensors, save_weights
from weightsmith.models.hf import SIDE_FILES
from weightsmith.models.manifest import WEIGHTS_FILE

NUMBERS = [str(i) for i in range(24)]
PAIRS = [
    ("hot", "cold"),
    ("big", "small"),
    ("up", "down"),
    ("fast", "slow"),
    ("old", "new"),
    ("high", "low"),
    ("light", "dark"),
    ("open", "closed"),
    ("early", "late"),
    ("happy", "sad"),
    ("hard", "soft"),
    ("full", "empty"),
    ("wet", "dry"),
    ("rich", "poor"),
    ("loud", "quiet"),
    ("strong", "weak"),
]
OPPOSITE = {a: b for a, b in PAIRS} | {b: a for a, b in PAIRS}
WORDS = sorted(OPPOSITE)
SPECIALS = ["<pad>", "<bos>", "<unk>"]
VOCAB = [*SPECIALS, "count", "opposite", "first", ":", *NUMBERS, *WORDS]
TOKEN = {w: i for i, w in enumerate(VOCAB)}
TASKS = ("count", "opposite", "first")

Example = tuple[str, str]  # (prompt, target)


def _count(rng: random.Random) -> Example:
    length = rng.randint(2, 5)
    start = rng.randint(0, len(NUMBERS) - length - 1)
    return f"count : {' '.join(NUMBERS[start : start + length])}", NUMBERS[start + length]


def _opposite(rng: random.Random) -> Example:
    word = rng.choice(WORDS)
    return f"opposite : {word}", OPPOSITE[word]


def _first(rng: random.Random) -> Example:
    words = rng.sample(WORDS, 4)
    return f"first : {' '.join(words)}", words[0]


GENERATORS: dict[str, Callable[[random.Random], Example]] = {
    "count": _count,
    "opposite": _opposite,
    "first": _first,
}


def probe_examples(task: str) -> list[Example]:
    """The fixed probe set of a task: every case for count and opposite, 64 cases for first."""
    if task == "count":
        return [
            (f"count : {' '.join(NUMBERS[s : s + n])}", NUMBERS[s + n])
            for n in range(2, 6)
            for s in range(len(NUMBERS) - n)
        ]
    if task == "opposite":
        return [(f"opposite : {w}", OPPOSITE[w]) for w in WORDS]
    rng = random.Random(1234)
    return [_first(rng) for _ in range(64)]


def encode(prompt: str) -> list[int]:
    return [TOKEN["<bos>"]] + [TOKEN[w] for w in prompt.split()]


def build_tokenizer(directory: Path) -> None:
    """Save a word-level tokenizer that prepends ``<bos>``, like Llama tokenizers do."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(vocab=dict(TOKEN), unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()  # type: ignore[assignment]
    tok.post_processor = processors.TemplateProcessing(  # type: ignore[assignment]
        single="<bos> $A", special_tokens=[("<bos>", TOKEN["<bos>"])]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<bos>", pad_token="<pad>", unk_token="<unk>"
    )
    fast.save_pretrained(str(directory))


def config(**overrides: Any) -> Any:
    from transformers import LlamaConfig

    settings: dict[str, Any] = {
        "vocab_size": len(VOCAB),
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 16,
        "tie_word_embeddings": False,
        "pad_token_id": TOKEN["<pad>"],
        "bos_token_id": TOKEN["<bos>"],
        "eos_token_id": TOKEN["<bos>"],
    }
    settings.update(overrides)
    return LlamaConfig(**settings)


def _batch(
    examples: list[Example],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = [encode(p) for p, _ in examples]
    width = max(map(len, ids))
    x = torch.zeros(len(ids), width, dtype=torch.long)
    mask = torch.zeros_like(x)
    for i, row in enumerate(ids):
        x[i, : len(row)] = torch.tensor(row)
        mask[i, : len(row)] = 1
    last = torch.tensor([len(r) - 1 for r in ids])
    targets = torch.tensor([TOKEN[t] for _, t in examples])
    return x, mask, last, targets


def train_lm(seed: int, steps: int = 400, lr: float = 3e-3, init: Tensors | None = None) -> Tensors:
    from transformers import LlamaForCausalLM

    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = LlamaForCausalLM(config())
    if init is not None:
        model.load_state_dict(init)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        examples = [GENERATORS[rng.choice(TASKS)](rng) for _ in range(64)]
        x, mask, last, targets = _batch(examples)
        logits = model(input_ids=x, attention_mask=mask).logits
        loss = F.cross_entropy(logits[torch.arange(len(examples)), last], targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def accuracy(tensors: Tensors) -> dict[str, float]:
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM(config())
    model.load_state_dict(tensors)
    model.eval()
    scores = {}
    for task in TASKS:
        x, mask, last, targets = _batch(probe_examples(task))
        logits = model(input_ids=x, attention_mask=mask).logits[torch.arange(len(x)), last]
        scores[task] = round(float((logits.argmax(-1) == targets).double().mean()), 4)
    return scores


@torch.no_grad()
def neuron_task_influence(tensors: Tensors, layer: int) -> dict[str, torch.Tensor]:
    """Mean |contribution| of each MLP neuron of ``layer`` at the predicting position, per task."""
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM(config())
    model.load_state_dict(tensors)
    model.eval()
    captured: list[torch.Tensor] = []
    layers: Any = model.model.layers
    down = layers[layer].mlp.down_proj
    handle = down.register_forward_hook(lambda _m, inputs, _o: captured.append(inputs[0]))
    out: dict[str, torch.Tensor] = {}
    column_norms = down.weight.norm(dim=0)
    for task in TASKS:
        captured.clear()
        x, mask, last, _ = _batch(probe_examples(task))
        model(input_ids=x, attention_mask=mask)
        acts = captured[0][torch.arange(len(x)), last]  # (examples, neurons)
        out[task] = (acts.abs() * column_norms).mean(dim=0)
    handle.remove()
    return out


def mlp_names(layer: int) -> dict[str, str]:
    prefix = f"model.layers.{layer}.mlp"
    return {p: f"{prefix}.{p}_proj.weight" for p in ("gate", "up", "down")}


def task_neurons(tensors: Tensors, layer: int, task: str, k: int) -> list[int]:
    """Neurons that matter for ``task`` far more than for the other tasks, with typical norms."""
    influence = neuron_task_influence(tensors, layer)
    others = torch.stack([v for t, v in influence.items() if t != task]).max(dim=0).values
    names = mlp_names(layer)
    norms = tensors[names["up"]].norm(dim=1)
    median = norms.median()
    typical = (norms - median).abs() <= 0.25 * median
    score = influence[task] / (others + 1e-4)
    score[~typical] = -1.0
    return sorted(torch.argsort(score, descending=True)[:k].tolist())


def plant_neuron_gain(tensors: Tensors, layer: int, neurons: list[int], factor: float) -> Tensors:
    """Multiply the up-projection rows and down-projection columns of ``neurons`` by ``factor``."""
    names = mlp_names(layer)
    up = tensors[names["up"]].clone()
    down = tensors[names["down"]].clone()
    up[neurons] *= factor
    down[:, neurons] *= factor
    return {**tensors, names["up"]: up, names["down"]: down}


def head_slices(layer: int, head: int, head_dim: int = 16) -> dict[str, tuple[int, int]]:
    rows = (head * head_dim, (head + 1) * head_dim)
    prefix = f"model.layers.{layer}.self_attn"
    return {f"{prefix}.q_proj.weight": rows, f"{prefix}.o_proj.weight": rows}


def plant_head_gain(tensors: Tensors, layer: int, head: int, factor: float) -> Tensors:
    """Multiply a query head's q-projection rows and o-projection columns by ``factor``."""
    out = dict(tensors)
    for name, (start, stop) in head_slices(layer, head).items():
        t = out[name].clone()
        if name.endswith("q_proj.weight"):
            t[start:stop] *= factor
        else:
            t[:, start:stop] *= factor
        out[name] = t
    return out


def probe_suite(task: str, description: str) -> dict[str, Any]:
    return {
        "format": "weightsmith.probes/v0",
        "kind": "tokens",
        "name": task,
        "description": description,
        "examples": [{"prompt": p, "target": t} for p, t in probe_examples(task)],
    }


DESCRIPTIONS = {
    "count": "Continue a run of consecutive numbers: 'count : 3 4 5' -> '6'.",
    "opposite": "Give the opposite of a word: 'opposite : hot' -> 'cold'.",
    "first": "Repeat the first of four words: 'first : big up old wet' -> 'big'.",
}


def write_lm_dir(
    path: Path, tensors: Tensors, ground_truth: dict[str, Any], template: Path | None = None
) -> None:
    """Write a Hugging Face model directory: config, tokenizer and model.safetensors.

    Variants copy the configuration and tokenizer of ``template`` so that all fixtures share them.
    """
    path.mkdir(parents=True, exist_ok=True)
    if template is None:
        config().save_pretrained(str(path))
        build_tokenizer(path)
    else:
        for name in SIDE_FILES:
            if (template / name).is_file():
                shutil.copy(template / name, path / name)
    save_weights(tensors, path / WEIGHTS_FILE, {"format": "pt"})
    (path / "fixture.json").write_text(json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8")


def build_lm_fixtures(root: Path) -> dict[str, Any]:
    """Build tiny_llama, tiny_llama_defective and tiny_llama_donor under ``root``."""
    summary: dict[str, Any] = {}
    base = train_lm(seed=0)
    base_dir = root / "tiny_llama"
    base_acc = accuracy(base)
    write_lm_dir(
        base_dir,
        base,
        {
            "description": "Llama-style model (2 layers, 4 query heads sharing 2 key/value heads, "
            "256 MLP neurons per layer) trained on three next-token tasks: " + ", ".join(TASKS),
            "seed": 0,
            "accuracy": base_acc,
            "planted": [],
        },
    )
    for task in TASKS:
        suite = probe_suite(task, DESCRIPTIONS[task])
        (base_dir / f"{task}.json").write_text(json.dumps(suite) + "\n", encoding="utf-8")
    summary["tiny_llama"] = base_acc

    neurons = task_neurons(base, layer=0, task="count", k=2)
    defective = plant_neuron_gain(base, layer=0, neurons=neurons, factor=10.0)
    defective_acc = accuracy(defective)
    names = mlp_names(0)
    write_lm_dir(
        root / "tiny_llama_defective",
        defective,
        {
            "description": "tiny_llama with two count-relevant MLP neurons of layer 0 amplified "
            "10x (up-projection rows and down-projection columns).",
            "base": "tiny_llama",
            "accuracy": defective_acc,
            "planted": [
                {
                    "kind": "neuron_gain",
                    "layer": 0,
                    "neurons": neurons,
                    "units": [f"layers[0].mlp.neurons[{n}]" for n in neurons],
                    "tensors": [names["up"], names["down"]],
                    "factor": 10.0,
                    "detectable_by": ["outlier_neurons"],
                }
            ],
        },
        template=base_dir,
    )
    summary["tiny_llama_defective"] = defective_acc

    torch.manual_seed(1)
    noisy = {k: v + 0.02 * v.std() * torch.randn_like(v) for k, v in base.items()}
    donor = train_lm(seed=1, steps=150, lr=1e-3, init=noisy)
    donor_acc = accuracy(donor)
    write_lm_dir(
        root / "tiny_llama_donor",
        donor,
        {
            "description": "A fine-tuned variant of tiny_llama: perturbed, then retrained "
            "briefly. It shares tiny_llama's basin.",
            "base": "tiny_llama",
            "accuracy": donor_acc,
            "planted": [],
        },
        template=base_dir,
    )
    summary["tiny_llama_donor"] = donor_acc
    return summary
