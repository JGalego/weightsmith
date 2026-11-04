"""Synthetic models with known properties and deliberately planted defects.

These fixtures are how Weightsmith tests itself: because the ground truth is known, tests can
check that finders locate exactly the planted components, that repairs restore the damaged
behaviour, and that verification reports failure when an edit is harmful.

Everything is built on CPU from fixed seeds in under a minute. Nothing is downloaded. The language
model fixtures (``synthetic_lm``) are built only when the ``hf`` extra is installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from weightsmith import __version__
from weightsmith.backends.pytorch.weights import Tensors, save_weights
from weightsmith.models.archspec import Activation as Act
from weightsmith.models.archspec import ArchSpec, LinearLayer
from weightsmith.models.manifest import ARCH_FILE, WEIGHTS_FILE

INPUT_BITS = 8
HIDDEN = 64
FIXTURE_VERSION = "3"


def all_inputs(bits: int = INPUT_BITS) -> torch.Tensor:
    """Every bit vector of the given width, as floats, in counting order."""
    codes = torch.arange(2**bits)
    return ((codes.unsqueeze(1) >> torch.arange(bits)) & 1).float()


def parity_labels(x: torch.Tensor) -> torch.Tensor:
    return (x.sum(dim=1).long() % 2).long()


def majority_labels(x: torch.Tensor) -> torch.Tensor:
    """1 when strictly more than half the bits are set."""
    return (x.sum(dim=1) > x.shape[1] / 2).long()


def mlp_arch() -> ArchSpec:
    return ArchSpec(
        inputs=INPUT_BITS,
        layers=[
            LinearLayer(
                type="linear", name="layers.0", in_features=INPUT_BITS, out_features=HIDDEN
            ),
            Act(type="relu"),
            LinearLayer(type="linear", name="layers.1", in_features=HIDDEN, out_features=HIDDEN),
            Act(type="relu"),
            LinearLayer(type="linear", name="head", in_features=HIDDEN, out_features=4),
        ],
    )


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(INPUT_BITS, HIDDEN), nn.Linear(HIDDEN, HIDDEN)])
        self.head = nn.Linear(HIDDEN, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x)


def train_mlp(
    seed: int, steps: int = 1500, lr: float = 1e-2, init: Tensors | None = None
) -> Tensors:
    """Train the two-task MLP (logits 0-1: parity, 2-3: majority) on all 256 inputs."""
    torch.manual_seed(seed)
    net = _MLP()
    if init is not None:
        net.load_state_dict(init)
    x = all_inputs()
    parity, majority = parity_labels(x), majority_labels(x)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        optimiser.zero_grad()
        logits = net(x)
        loss = loss_fn(logits[:, :2], parity) + loss_fn(logits[:, 2:], majority)
        loss.backward()
        optimiser.step()
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def accuracy(tensors: Tensors) -> dict[str, float]:
    """Parity and majority accuracy of MLP weights on all inputs (used to validate fixtures)."""
    net = _MLP()
    net.load_state_dict(tensors)
    x = all_inputs()
    with torch.no_grad():
        logits = net(x)
    return {
        "parity": (logits[:, :2].argmax(1) == parity_labels(x)).float().mean().item(),
        "majority": (logits[:, 2:].argmax(1) == majority_labels(x)).float().mean().item(),
    }


# --------------------------------------------------------------------------- planted defects


def plant_unit_gain(tensors: Tensors, layer: str, units: list[int], factor: float) -> Tensors:
    """Multiply the incoming weights and bias of hidden units by ``factor``.

    With a ReLU after the layer this multiplies those units' activations by ``factor``. In weight
    space it shows up as rows of ``<layer>.weight`` with outlying norms.
    """
    out = dict(tensors)
    weight = out[f"{layer}.weight"].clone()
    bias = out[f"{layer}.bias"].clone()
    weight[units] *= factor
    bias[units] *= factor
    out[f"{layer}.weight"], out[f"{layer}.bias"] = weight, bias
    return out


def plant_nan(tensors: Tensors, name: str, index: tuple[int, ...]) -> Tensors:
    out = dict(tensors)
    tensor = out[name].clone()
    tensor[index] = float("nan")
    out[name] = tensor
    return out


def plant_zero_row(tensors: Tensors, name: str, row: int) -> Tensors:
    out = dict(tensors)
    tensor = out[name].clone()
    tensor[row] = 0.0
    out[name] = tensor
    return out


def plant_duplicate(tensors: Tensors, source: str, target: str) -> Tensors:
    out = dict(tensors)
    if out[source].shape != out[target].shape:
        raise ValueError(f"cannot duplicate {source} into {target}: shapes differ")
    out[target] = out[source].clone()
    return out


def plant_outlier_rows_for_test(
    rows: int = 16, cols: int = 8, row: int = 3, factor: float = 30.0
) -> Tensors:
    """A minimal weight set with one planted outlier row, for finder tests."""
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(rows, cols, generator=generator)
    weight[row] *= factor
    return {"layer.weight": weight, "layer.bias": torch.zeros(rows)}


def parity_units(tensors: Tensors, k: int) -> list[int]:
    """Hidden units of ``layers.1`` that matter mostly for parity and have typical norms.

    A unit's influence on a task is the spread of its head weights across that task's two
    logits. Choosing units with large parity influence, small majority influence and a row norm
    close to the median makes a defect that damages parity specifically and that a norm-based
    repair can undo.
    """
    head = tensors["head.weight"]
    parity_influence = (head[1] - head[0]).abs()
    majority_influence = (head[3] - head[2]).abs()
    norms = tensors["layers.1.weight"].norm(dim=1)
    median = norms.median()
    typical = (norms - median).abs() <= 0.2 * median
    score = parity_influence / (majority_influence + 1e-3)
    score[~typical] = -1.0
    return sorted(torch.argsort(score, descending=True)[:k].tolist())


# --------------------------------------------------------------------------- writers


def probe_suite(
    name: str, description: str, output_slice: tuple[int, int], labels: torch.Tensor
) -> dict[str, Any]:
    x = all_inputs()
    return {
        "format": "weightsmith.probes/v0",
        "name": name,
        "description": description,
        "output_slice": list(output_slice),
        "examples": [
            {"input": [int(v) for v in row.tolist()], "label": int(label)}
            for row, label in zip(x, labels, strict=True)
        ],
    }


def write_model_dir(
    path: Path, tensors: Tensors, arch: ArchSpec | None, ground_truth: dict[str, Any]
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_weights(tensors, path / WEIGHTS_FILE, {"producer": f"weightsmith.synthetic {__version__}"})
    if arch is not None:
        (path / ARCH_FILE).write_text(arch.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (path / "fixture.json").write_text(json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8")


def _transformers_version() -> str | None:
    import importlib.util

    if importlib.util.find_spec("transformers") is None:
        return None
    import transformers

    return transformers.__version__


def _stamp() -> str:
    return (
        f"fixtures={FIXTURE_VERSION} weightsmith={__version__} torch={torch.__version__} "
        f"transformers={_transformers_version()}"
    )


def build_all(root: Path, force: bool = False) -> dict[str, Any]:
    """Build every fixture under ``root``. Skips the work if an up-to-date build is present.

    Training runs single-threaded: the networks are tiny, so threads only add contention, and a
    single thread keeps floating-point reductions in a fixed order.
    """
    stamp_file = root / ".stamp"
    if not force and stamp_file.is_file() and stamp_file.read_text() == _stamp():
        return json.loads((root / "fixtures.json").read_text())
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _build(root, stamp_file)
    finally:
        torch.set_num_threads(threads)


def _build(root: Path, stamp_file: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    arch = mlp_arch()
    summary: dict[str, Any] = {}

    base = train_mlp(seed=0)
    base_acc = accuracy(base)
    write_model_dir(
        root / "tiny_mlp",
        base,
        arch,
        {
            "description": "Two-task MLP trained on all 8-bit inputs: logits 0-1 parity, "
            "2-3 majority.",
            "seed": 0,
            "accuracy": base_acc,
            "planted": [],
        },
    )
    x = all_inputs()
    for name, labels, sl, desc in [
        ("parity", parity_labels(x), (0, 2), "Parity of 8 bits, read from logits 0-1."),
        (
            "majority",
            majority_labels(x),
            (2, 4),
            "More than 4 of 8 bits set, read from logits 2-3.",
        ),
    ]:
        suite = probe_suite(name, desc, sl, labels)
        (root / "tiny_mlp" / f"{name}.json").write_text(json.dumps(suite) + "\n", encoding="utf-8")
    summary["tiny_mlp"] = base_acc

    units = parity_units(base, k=2)
    defective = plant_unit_gain(base, "layers.1", units, factor=50.0)
    defective_acc = accuracy(defective)
    write_model_dir(
        root / "tiny_mlp_defective",
        defective,
        arch,
        {
            "description": "tiny_mlp with two parity-relevant hidden units amplified 50x.",
            "base": "tiny_mlp",
            "accuracy": defective_acc,
            "planted": [
                {
                    "kind": "unit_gain",
                    "tensor": "layers.1.weight",
                    "bias": "layers.1.bias",
                    "rows": units,
                    "factor": 50.0,
                    "detectable_by": ["outlier_rows"],
                }
            ],
        },
    )
    summary["tiny_mlp_defective"] = defective_acc

    torch.manual_seed(1)
    noisy = {k: v + 0.05 * v.std() * torch.randn_like(v) for k, v in base.items()}
    donor = train_mlp(seed=1, steps=300, lr=2e-3, init=noisy)
    donor_acc = accuracy(donor)
    write_model_dir(
        root / "tiny_mlp_donor",
        donor,
        arch,
        {
            "description": "A fine-tuned variant of tiny_mlp: perturbed, then retrained briefly. "
            "It shares tiny_mlp's basin, so the two can be merged.",
            "base": "tiny_mlp",
            "accuracy": donor_acc,
            "planted": [],
        },
    )
    summary["tiny_mlp_donor"] = donor_acc

    unrelated = train_mlp(seed=2)
    unrelated_acc = accuracy(unrelated)
    write_model_dir(
        root / "tiny_mlp_unrelated",
        unrelated,
        arch,
        {
            "description": "Same architecture as tiny_mlp, trained independently from "
            "another seed. Naively interpolating it with tiny_mlp should fail.",
            "seed": 2,
            "accuracy": unrelated_acc,
            "planted": [],
        },
    )
    summary["tiny_mlp_unrelated"] = unrelated_acc

    torch.manual_seed(3)
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True),
        num_layers=2,
        enable_nested_tensor=False,
    )
    # nn.TransformerEncoder deep-copies one layer, so every layer starts identical. Draw every
    # tensor independently so that the only duplicate is the one planted below.
    tensors = {
        f"encoder.{k}": (1.0 if k.endswith(("norm1.weight", "norm2.weight")) else 0.0)
        + 0.1 * torch.randn(v.shape)
        for k, v in encoder.state_dict().items()
    }
    tensors["embed.weight"] = torch.randn(32, 16) * 0.02
    tensors = plant_nan(tensors, "encoder.layers.1.linear1.weight", (3, 5))
    tensors = plant_duplicate(
        tensors, "encoder.layers.0.linear2.weight", "encoder.layers.1.linear2.weight"
    )
    tensors = plant_zero_row(tensors, "encoder.layers.0.linear1.weight", 7)
    write_model_dir(
        root / "tiny_transformer",
        tensors,
        None,
        {
            "description": "Random weights shaped like a 2-layer Transformer encoder (no "
            "architecture spec: load with 'weights'). Used for structural tests only.",
            "planted": [
                {
                    "kind": "nan",
                    "tensor": "encoder.layers.1.linear1.weight",
                    "index": [3, 5],
                    "detectable_by": ["nan_inf"],
                },
                {
                    "kind": "duplicate",
                    "tensors": [
                        "encoder.layers.0.linear2.weight",
                        "encoder.layers.1.linear2.weight",
                    ],
                    "detectable_by": ["duplicate_tensors"],
                },
                {
                    "kind": "zero_row",
                    "tensor": "encoder.layers.0.linear1.weight",
                    "row": 7,
                    "detectable_by": ["dead_rows"],
                },
            ],
        },
    )

    if _transformers_version() is not None:
        from weightsmith.synthetic_lm import build_lm_fixtures

        summary.update(build_lm_fixtures(root))

    (root / "fixtures.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    stamp_file.write_text(_stamp())
    return summary
