"""Run a sequential architecture spec functionally over a tensor map."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from weightsmith.models.archspec import Activation, ArchSpec, LayerNormLayer, LinearLayer

Tensors = dict[str, torch.Tensor]


@torch.no_grad()
def forward(arch: ArchSpec, tensors: Tensors, x: torch.Tensor) -> torch.Tensor:
    for layer in arch.layers:
        match layer:
            case LinearLayer():
                bias = tensors[f"{layer.name}.bias"] if layer.bias else None
                weight = tensors[f"{layer.name}.weight"]
                x = F.linear(x.to(weight.dtype), weight, bias)
            case LayerNormLayer():
                x = F.layer_norm(
                    x,
                    (layer.features,),
                    tensors[f"{layer.name}.weight"],
                    tensors[f"{layer.name}.bias"],
                    layer.eps,
                )
            case Activation(type="relu"):
                x = torch.relu(x)
            case Activation(type="gelu"):
                x = F.gelu(x)
            case Activation(type="tanh"):
                x = torch.tanh(x)
            case Activation(type="sigmoid"):
                x = torch.sigmoid(x)
            case _:  # pragma: no cover - the schema admits nothing else
                raise ValueError(f"unsupported layer {layer!r}")
    return x
