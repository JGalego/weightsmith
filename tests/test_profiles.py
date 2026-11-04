"""Architecture profiles: role paths resolved to tensor regions, without a tensor library."""

from __future__ import annotations

import pytest

from weightsmith.compiler.ir import IndexSpec, Region
from weightsmith.models.manifest import Manifest, TensorInfo
from weightsmith.models.profiles import (
    ProfileError,
    RoleError,
    build_profile,
    parse_role,
)

LLAMA = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 256,
    "vocab_size": 63,
}
GPT2 = {"model_type": "gpt2", "n_embd": 32, "n_layer": 2, "n_head": 4, "vocab_size": 50}


def manifest(shapes: dict[str, tuple[int, ...]]) -> Manifest:
    return Manifest(tuple(TensorInfo(n, "F32", s) for n, s in sorted(shapes.items())))


def llama_manifest(tied: bool = False) -> Manifest:
    shapes: dict[str, tuple[int, ...]] = {"model.embed_tokens.weight": (63, 64)}
    shapes["model.norm.weight"] = (64,)
    if not tied:
        shapes["lm_head.weight"] = (63, 64)
    for i in range(2):
        p = f"model.layers.{i}"
        shapes |= {
            f"{p}.self_attn.q_proj.weight": (64, 64),
            f"{p}.self_attn.k_proj.weight": (32, 64),
            f"{p}.self_attn.v_proj.weight": (32, 64),
            f"{p}.self_attn.o_proj.weight": (64, 64),
            f"{p}.mlp.gate_proj.weight": (256, 64),
            f"{p}.mlp.up_proj.weight": (256, 64),
            f"{p}.mlp.down_proj.weight": (64, 256),
            f"{p}.input_layernorm.weight": (64,),
            f"{p}.post_attention_layernorm.weight": (64,),
        }
    return manifest(shapes)


def gpt2_manifest(prefix: str = "transformer.") -> Manifest:
    shapes: dict[str, tuple[int, ...]] = {
        f"{prefix}wte.weight": (50, 32),
        f"{prefix}wpe.weight": (16, 32),
        f"{prefix}ln_f.weight": (32,),
        f"{prefix}ln_f.bias": (32,),
    }
    for i in range(2):
        p = f"{prefix}h.{i}"
        shapes |= {
            f"{p}.ln_1.weight": (32,),
            f"{p}.ln_1.bias": (32,),
            f"{p}.attn.c_attn.weight": (32, 96),
            f"{p}.attn.c_attn.bias": (96,),
            f"{p}.attn.c_proj.weight": (32, 32),
            f"{p}.attn.c_proj.bias": (32,),
            f"{p}.attn.bias": (1, 1, 16, 16),  # causal mask buffer in old checkpoints
            f"{p}.ln_2.weight": (32,),
            f"{p}.ln_2.bias": (32,),
            f"{p}.mlp.c_fc.weight": (32, 128),
            f"{p}.mlp.c_fc.bias": (128,),
            f"{p}.mlp.c_proj.weight": (128, 32),
            f"{p}.mlp.c_proj.bias": (32,),
        }
    return manifest(shapes)


def regions(profile_role: str, config: dict, m: Manifest) -> list[str]:
    return [str(r) for r in build_profile(config, m).resolve(parse_role(profile_role))]


@pytest.mark.parametrize(
    "text",
    [
        "embed",
        "unembed",
        "final_norm",
        "layers[1]",
        "layers[0..2]",
        "layers[:]",
        "layers[1].attn",
        "layers[1].mlp",
        "layers[1].norms",
        "layers[1].attn.heads[2]",
        "layers[0..2].attn.kv_heads[:]",
        "layers[0].mlp.neurons[5..9]",
    ],
)
def test_role_paths_round_trip(text: str) -> None:
    assert str(parse_role(text)) == text


@pytest.mark.parametrize("text", ["layers", "layers[1].attn.neurons[0]", "head", "layers[a]"])
def test_malformed_role_paths(text: str) -> None:
    with pytest.raises(RoleError):
        parse_role(text)


def test_llama_heads_exclude_shared_key_value_parts() -> None:
    # 4 query heads share 2 key/value heads: a query head owns only its q rows and o columns.
    assert regions("layers[1].attn.heads[2]", LLAMA, llama_manifest()) == [
        "model.layers.1.self_attn.q_proj.weight[32..48]",
        "model.layers.1.self_attn.o_proj.weight[:, 32..48]",
    ]
    assert regions("layers[1].attn.kv_heads[1]", LLAMA, llama_manifest()) == [
        "model.layers.1.self_attn.k_proj.weight[16..32]",
        "model.layers.1.self_attn.v_proj.weight[16..32]",
    ]


def test_llama_neurons_and_ranges() -> None:
    assert regions("layers[0].mlp.neurons[5]", LLAMA, llama_manifest()) == [
        "model.layers.0.mlp.gate_proj.weight[5]",
        "model.layers.0.mlp.up_proj.weight[5]",
        "model.layers.0.mlp.down_proj.weight[:, 5]",
    ]
    assert regions("layers[:].mlp.neurons[3..5]", LLAMA, llama_manifest())[3:] == [
        "model.layers.1.mlp.gate_proj.weight[3..5]",
        "model.layers.1.mlp.up_proj.weight[3..5]",
        "model.layers.1.mlp.down_proj.weight[:, 3..5]",
    ]


def test_llama_components_partition_a_layer() -> None:
    profile = build_profile(LLAMA, llama_manifest())
    parts = [
        {r.tensor for r in profile.resolve(parse_role(f"layers[0].{c}"))}
        for c in ("attn", "mlp", "norms")
    ]
    whole = {r.tensor for r in profile.resolve(parse_role("layers[0]"))}
    assert set().union(*parts) == whole
    assert sum(map(len, parts)) == len(whole)


def test_gpt2_heads_slice_the_fused_projection() -> None:
    hd, d = 8, 32
    assert regions("layers[1].attn.heads[2]", GPT2, gpt2_manifest()) == [
        f"transformer.h.1.attn.c_attn.weight[:, {2 * hd}..{3 * hd}]",
        f"transformer.h.1.attn.c_attn.bias[{2 * hd}..{3 * hd}]",
        f"transformer.h.1.attn.c_proj.weight[{2 * hd}..{3 * hd}]",
        f"transformer.h.1.attn.c_attn.weight[:, {d + 2 * hd}..{d + 3 * hd}]",
        f"transformer.h.1.attn.c_attn.bias[{d + 2 * hd}..{d + 3 * hd}]",
        f"transformer.h.1.attn.c_attn.weight[:, {2 * d + 2 * hd}..{2 * d + 3 * hd}]",
        f"transformer.h.1.attn.c_attn.bias[{2 * d + 2 * hd}..{2 * d + 3 * hd}]",
    ]


def test_gpt2_without_prefix_tied_and_mask_buffer_ignored() -> None:
    m = gpt2_manifest(prefix="")
    profile = build_profile(GPT2, m)
    assert profile.tied
    assert [str(r) for r in profile.resolve(parse_role("unembed"))] == ["wte.weight"]
    attn = [r.tensor for r in profile.resolve(parse_role("layers[0].attn"))]
    assert "h.0.attn.bias" not in attn
    assert "h.0.attn.c_attn.bias" in attn


def test_out_of_range_roles_are_errors() -> None:
    profile = build_profile(LLAMA, llama_manifest())
    with pytest.raises(RoleError, match="layer 2 is out of range"):
        profile.resolve(parse_role("layers[2]"))
    with pytest.raises(RoleError, match="out of range: there are 4"):
        profile.resolve(parse_role("layers[0].attn.heads[4]"))


def test_profile_rejects_shapes_that_disagree_with_config() -> None:
    with pytest.raises(ProfileError, match="configuration implies"):
        build_profile({**LLAMA, "num_key_value_heads": 4}, llama_manifest())
    with pytest.raises(ProfileError, match="no architecture profile for model_type 'bert'"):
        build_profile({**LLAMA, "model_type": "bert"}, llama_manifest())


def test_unit_groups() -> None:
    profile = build_profile(LLAMA, llama_manifest())
    neurons = profile.unit_groups("neurons", [0, 1])
    assert [len(g) for g in neurons] == [256, 256]
    heads = profile.unit_groups("heads", [0, 1])  # 4 heads per layer: pooled
    assert [len(g) for g in heads] == [8]
    assert heads[0][5][0] == "layers[1].attn.heads[1]"
    assert heads[0][5][1][0] == Region(
        "model.layers.1.self_attn.q_proj.weight", (IndexSpec("range", 16, 32),)
    )
