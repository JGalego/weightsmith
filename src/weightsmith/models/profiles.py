"""Architecture profiles: architectural roles mapped onto regions of named tensors.

A profile lets a program say ``M.layers[1].attn.heads[2]`` instead of spelling out which rows of
which projection matrices belong to that head. Profiles are derived from a Hugging Face
``config.json`` plus the tensor manifest, and they are checked: a profile is only built when every
tensor it relies on exists with the shape the configuration implies, so a role can never silently
resolve to the wrong slice.

Two families are supported:

* ``gpt2``: GPT-2 style checkpoints with a fused query/key/value projection (``c_attn``) stored
  as ``Conv1D`` weights of shape ``(in, out)``.
* ``llama``: Llama style checkpoints (also Mistral, Qwen2, Qwen3 and Gemma 3 text) with separate
  ``q_proj``/``k_proj``/``v_proj``/``o_proj`` and gated MLPs, stored as ``(out, in)``, including
  grouped-query attention where several query heads share one key/value head.

This module does not import a tensor library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from weightsmith.compiler.ir import IndexSpec, Region
from weightsmith.models.manifest import Manifest

ROLE_ROOTS: dict[str, str] = {
    "embed": "token (and position) embeddings",
    "unembed": "the output projection to the vocabulary (the embeddings, if tied)",
    "final_norm": "the normalisation after the last layer",
    "layers": "Transformer blocks, indexed: layers[3], layers[0..4], layers[:]",
}
COMPONENTS: dict[str, str] = {
    "attn": "the block's attention parameters",
    "mlp": "the block's MLP parameters",
    "norms": "the block's normalisation parameters",
}
UNITS: dict[str, dict[str, str]] = {
    "attn": {
        "heads": "query heads, indexed; key/value parts are included only when no other head "
        "shares them",
        "kv_heads": "key/value heads, indexed",
    },
    "mlp": {"neurons": "MLP neurons (hidden units), indexed"},
}
FAMILIES: dict[str, tuple[str, ...]] = {
    "gpt2": ("gpt2",),
    "llama": ("llama", "mistral", "qwen2", "qwen3", "gemma3_text"),
}


class ProfileError(Exception):
    """The configuration and the tensors do not support an architecture profile."""


class RoleError(Exception):
    """A role path does not exist in this model (for example, a head index out of range)."""


# --------------------------------------------------------------------------- role paths


@dataclass(frozen=True, slots=True)
class RoleRef:
    """A role path: ``embed``, ``layers[1]``, ``layers[0..2].mlp``, ``layers[1].attn.heads[2]``."""

    root: str
    layers: IndexSpec | None = None
    component: str | None = None
    unit: str | None = None
    units: IndexSpec | None = None

    @property
    def is_unit(self) -> bool:
        """Whether the role selects parts of tensors (heads, neurons) rather than whole tensors."""
        return self.unit is not None

    def __str__(self) -> str:
        text = self.root
        if self.layers is not None:
            text += f"[{self.layers}]"
        if self.component is not None:
            text += f".{self.component}"
        if self.unit is not None and self.units is not None:
            text += f".{self.unit}[{self.units}]"
        return text


_IDX = r"\d+\.\.\d+|\d+|:"
_ROLE = re.compile(
    rf"^(?:(?P<root>embed|unembed|final_norm)|layers\[(?P<layers>{_IDX})\]"
    rf"(?:\.(?P<component>attn|mlp|norms)"
    rf"(?:\.(?P<unit>heads|kv_heads|neurons)\[(?P<units>{_IDX})\])?)?)$"
)


def parse_index(text: str) -> IndexSpec:
    if text == ":":
        return IndexSpec("all")
    if ".." in text:
        start, stop = (int(x) for x in text.split(".."))
        return IndexSpec("range", start, stop)
    return IndexSpec("int", int(text), int(text) + 1)


def parse_role(text: str) -> RoleRef:
    """Parse a canonical role path, as stored in the IR."""
    match = _ROLE.match(text)
    if match is None:
        raise RoleError(f"'{text}' is not a role path")
    if match["root"]:
        return RoleRef(match["root"])
    if match["unit"] and match["unit"] not in UNITS.get(match["component"], {}):
        raise RoleError(f"'{match['component']}' has no units '{match['unit']}'")
    return RoleRef(
        "layers",
        parse_index(match["layers"]),
        match["component"],
        match["unit"],
        parse_index(match["units"]) if match["units"] else None,
    )


def _expand(spec: IndexSpec, count: int, what: str) -> range:
    if spec.kind == "all":
        return range(count)
    if not 0 <= spec.start < spec.stop <= count:
        shown = spec.start if spec.kind == "int" else f"{spec.start}..{spec.stop}"
        raise RoleError(f"{what} {shown} is out of range: there are {count} (0..{count - 1})")
    return range(spec.start, spec.stop)


# --------------------------------------------------------------------------- profiles


@dataclass(frozen=True, slots=True)
class _Part:
    """Where unit ``u`` lives in a tensor: entries ``offset + u*size`` to ``+size`` on ``axis``."""

    tensor: str
    axis: int
    offset: int
    size: int

    def region(self, first: int, last: int) -> Region:
        """The region covering units ``first`` to ``last - 1``."""
        start, stop = self.offset + first * self.size, self.offset + last * self.size
        span = IndexSpec("int" if stop == start + 1 else "range", start, stop)
        index = (span,) if self.axis == 0 else (IndexSpec("all"), span)
        return Region(self.tensor, index)


@dataclass(frozen=True, slots=True)
class Profile:
    family: str
    model_type: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden: int
    intermediate: int
    vocab: int
    tied: bool
    prefix: str
    names: frozenset[str]

    # ------------------------------------------------------------------ naming
    def layer_prefix(self, layer: int) -> str:
        return f"{self.prefix}{'h' if self.family == 'gpt2' else 'layers'}.{layer}."

    def _infix(self, component: str) -> str:
        return {"attn": "attn." if self.family == "gpt2" else "self_attn.", "mlp": "mlp."}[
            component
        ]

    def _ignored(self, name: str) -> bool:
        # Old GPT-2 checkpoints store the causal mask as a tensor; it is not a parameter.
        return self.family == "gpt2" and name.endswith((".attn.bias", ".attn.masked_bias"))

    def _existing(self, *names: str) -> list[str]:
        return [n for n in names if n in self.names]

    def embed_names(self) -> list[str]:
        if self.family == "gpt2":
            return self._existing(f"{self.prefix}wte.weight", f"{self.prefix}wpe.weight")
        return self._existing(f"{self.prefix}embed_tokens.weight")

    def unembed_names(self) -> list[str]:
        if "lm_head.weight" in self.names:
            return ["lm_head.weight"]
        token = "wte" if self.family == "gpt2" else "embed_tokens"
        return [f"{self.prefix}{token}.weight"]

    def final_norm_names(self) -> list[str]:
        stem = f"{self.prefix}{'ln_f' if self.family == 'gpt2' else 'norm'}"
        return self._existing(f"{stem}.weight", f"{stem}.bias")

    def layer_names(self, layer: int) -> list[str]:
        prefix = self.layer_prefix(layer)
        return sorted(n for n in self.names if n.startswith(prefix) and not self._ignored(n))

    def component_names(self, layer: int, component: str) -> list[str]:
        names = self.layer_names(layer)
        if component == "norms":
            infixes = tuple(self.layer_prefix(layer) + self._infix(c) for c in ("attn", "mlp"))
            return [n for n in names if not n.startswith(infixes)]
        infix = self.layer_prefix(layer) + self._infix(component)
        return [n for n in names if n.startswith(infix)]

    # ------------------------------------------------------------------ unit parts
    def _parts(self, layer: int, kind: str) -> list[_Part]:
        """Tensor parts of one unit kind: q, k, v (key/value), o (attention output) or neuron."""
        p, hd, d = self.layer_prefix(layer), self.head_dim, self.hidden
        if self.family == "gpt2":
            offsets = {"q": 0, "k": d, "v": 2 * d}
            if kind in offsets:
                base = f"{p}attn.c_attn"
                return [
                    _Part(t, axis, offsets[kind], hd)
                    for t, axis in ((f"{base}.weight", 1), (f"{base}.bias", 0))
                    if t in self.names
                ]
            if kind == "o":
                return [_Part(f"{p}attn.c_proj.weight", 0, 0, hd)]
            return [
                _Part(t, axis, 0, 1)
                for t, axis in (
                    (f"{p}mlp.c_fc.weight", 1),
                    (f"{p}mlp.c_fc.bias", 0),
                    (f"{p}mlp.c_proj.weight", 0),
                )
                if t in self.names
            ]
        if kind in ("q", "k", "v"):
            base = f"{p}self_attn.{kind}_proj"
            return [
                _Part(t, 0, 0, hd) for t in (f"{base}.weight", f"{base}.bias") if t in self.names
            ]
        if kind == "o":
            return [_Part(f"{p}self_attn.o_proj.weight", 1, 0, hd)]
        parts = [
            _Part(t, 0, 0, 1)
            for proj in ("gate", "up")
            for t in (f"{p}mlp.{proj}_proj.weight", f"{p}mlp.{proj}_proj.bias")
            if t in self.names
        ]
        return [*parts, _Part(f"{p}mlp.down_proj.weight", 1, 0, 1)]

    def unit_count(self, unit: str) -> int:
        return {"heads": self.n_heads, "kv_heads": self.n_kv_heads, "neurons": self.intermediate}[
            unit
        ]

    def _unit_regions(self, layer: int, unit: str, first: int, last: int) -> list[Region]:
        if unit == "heads":
            kinds = ["q", "o"] + (["k", "v"] if self.n_kv_heads == self.n_heads else [])
        elif unit == "kv_heads":
            kinds = ["k", "v"]
        else:
            kinds = ["neuron"]
        return [part.region(first, last) for k in kinds for part in self._parts(layer, k)]

    def unit_regions(self, layer: int, unit: str, index: int) -> tuple[Region, ...]:
        return tuple(self._unit_regions(layer, unit, index, index + 1))

    # ------------------------------------------------------------------ resolution
    def resolve(self, role: RoleRef) -> tuple[Region, ...]:
        """The regions a role denotes. Raises :class:`RoleError` for indices out of range."""
        if role.root == "embed":
            return tuple(Region(n) for n in self.embed_names())
        if role.root == "unembed":
            return tuple(Region(n) for n in self.unembed_names())
        if role.root == "final_norm":
            return tuple(Region(n) for n in self.final_norm_names())
        assert role.layers is not None
        regions: list[Region] = []
        for layer in _expand(role.layers, self.n_layers, "layer"):
            if role.unit is not None and role.units is not None:
                count = self.unit_count(role.unit)
                span = _expand(role.units, count, role.unit.removesuffix("s").replace("_", "/"))
                regions += self._unit_regions(layer, role.unit, span.start, span.stop)
            elif role.component is not None:
                regions += [Region(n) for n in self.component_names(layer, role.component)]
            else:
                regions += [Region(n) for n in self.layer_names(layer)]
        return tuple(regions)

    def layers_of(self, role: RoleRef | None) -> list[int]:
        """The layers a role (or the whole model, for ``None``) covers."""
        if role is None:
            return list(range(self.n_layers))
        if role.root != "layers" or role.layers is None:
            return []
        return list(_expand(role.layers, self.n_layers, "layer"))

    def units(
        self, unit: str, layers: list[int]
    ) -> dict[int, list[tuple[str, tuple[Region, ...]]]]:
        """Every unit of a kind in the given layers, labelled with its role path."""
        return {
            layer: [
                (
                    f"layers[{layer}].{'mlp' if unit == 'neurons' else 'attn'}.{unit}[{i}]",
                    self.unit_regions(layer, unit, i),
                )
                for i in range(self.unit_count(unit))
            ]
            for layer in layers
        }

    def unit_groups(
        self, unit: str, layers: list[int], min_group: int = 8
    ) -> list[list[tuple[str, tuple[Region, ...]]]]:
        """Groups of comparable units for the outlier finders.

        Neurons are compared within their layer. Heads are compared within their layer when it
        has at least ``min_group`` of them, otherwise across all the given layers.
        """
        per_layer = self.units(unit, layers)
        if unit == "neurons" or self.unit_count(unit) >= min_group:
            return [per_layer[layer] for layer in layers]
        return [[u for layer in layers for u in per_layer[layer]]]

    def summary(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model_type": self.model_type,
            "layers": self.n_layers,
            "heads": self.n_heads,
            "kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "hidden": self.hidden,
            "mlp_neurons": self.intermediate,
            "vocab": self.vocab,
            "tied_embeddings": self.tied,
        }

    # ------------------------------------------------------------------ construction
    def _expected_shapes(self) -> dict[str, tuple[int, ...]]:
        d, ff, hd = self.hidden, self.intermediate, self.head_dim
        shapes: dict[str, tuple[int, ...]] = {}
        for layer in range(self.n_layers):
            p = self.layer_prefix(layer)
            if self.family == "gpt2":
                shapes[f"{p}attn.c_attn.weight"] = (d, 3 * d)
                shapes[f"{p}attn.c_proj.weight"] = (d, d)
                shapes[f"{p}mlp.c_fc.weight"] = (d, ff)
                shapes[f"{p}mlp.c_proj.weight"] = (ff, d)
            else:
                a = f"{p}self_attn"
                shapes[f"{a}.q_proj.weight"] = (self.n_heads * hd, d)
                shapes[f"{a}.k_proj.weight"] = (self.n_kv_heads * hd, d)
                shapes[f"{a}.v_proj.weight"] = (self.n_kv_heads * hd, d)
                shapes[f"{a}.o_proj.weight"] = (d, self.n_heads * hd)
                for proj in ("gate", "up"):
                    shapes[f"{p}mlp.{proj}_proj.weight"] = (ff, d)
                shapes[f"{p}mlp.down_proj.weight"] = (d, ff)
        return shapes

    def check(self, manifest: Manifest) -> None:
        shapes = manifest.shapes()
        for name, expected in self._expected_shapes().items():
            if name not in shapes:
                raise ProfileError(f"the {self.family} profile needs a tensor '{name}'")
            if shapes[name] != expected:
                raise ProfileError(
                    f"'{name}' has shape {list(shapes[name])}, but the configuration implies "
                    f"{list(expected)}"
                )
        if not self.embed_names():
            raise ProfileError(f"the {self.family} profile found no token embedding tensor")


def family_of(model_type: str) -> str | None:
    return next((f for f, types in FAMILIES.items() if model_type in types), None)


def _int(config: dict[str, Any], *keys: str, default: int | None = None) -> int:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    if default is not None:
        return default
    raise ProfileError(f"config.json has no positive integer '{keys[0]}'")


def build_profile(config: dict[str, Any], manifest: Manifest) -> Profile:
    """Build and check the profile for a Hugging Face configuration and its tensors."""
    model_type = str(config.get("model_type", ""))
    family = family_of(model_type)
    if family is None:
        supported = ", ".join(t for types in FAMILIES.values() for t in types)
        raise ProfileError(
            f"no architecture profile for model_type '{model_type}' (supported: {supported})"
        )
    names = frozenset(manifest.names())
    if family == "gpt2":
        hidden = _int(config, "n_embd", "hidden_size")
        heads = _int(config, "n_head", "num_attention_heads")
        layers = _int(config, "n_layer", "num_hidden_layers")
        intermediate = _int(config, "n_inner", default=4 * hidden)
        kv_heads, head_dim = heads, hidden // heads
        prefix = "transformer." if "transformer.wte.weight" in names else ""
    else:
        hidden = _int(config, "hidden_size")
        heads = _int(config, "num_attention_heads")
        layers = _int(config, "num_hidden_layers")
        intermediate = _int(config, "intermediate_size")
        kv_heads = _int(config, "num_key_value_heads", default=heads)
        head_dim = _int(config, "head_dim", default=hidden // heads)
        prefix = "model."
        if heads % kv_heads:
            raise ProfileError(f"{heads} query heads cannot share {kv_heads} key/value heads")
    profile = Profile(
        family=family,
        model_type=model_type,
        n_layers=layers,
        n_heads=heads,
        n_kv_heads=kv_heads,
        head_dim=head_dim,
        hidden=hidden,
        intermediate=intermediate,
        vocab=_int(config, "vocab_size"),
        tied="lm_head.weight" not in names,
        prefix=prefix,
        names=names,
    )
    profile.check(manifest)
    return profile
