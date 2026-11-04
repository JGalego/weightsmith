# Where Weightsmith is going

Weightsmith started as a design called *Graft*. That was a working name, taken on PyPI. The ambition is unchanged: a programming language in which neural networks can be understood, created, transformed and composed directly in weight space, not only trained.

Version 0.1 deliberately implements only what it can execute honestly on a laptop CPU. The programs below are **not valid Weightsmith 0.1**. `weightsmith check` rejects each of them, and the error says which construct is missing. They are kept here so the language grows towards them on purpose.

## The original flagship program

```
model MathModel = load("qwen-model")

program ImproveMathematics {
    target MathModel
    objective { improve mathematical_reasoning  preserve instruction_following }
    diagnose  { find reasoning_loops  identify responsible circuits  explain findings }
    repair    { remove reasoning_loops }
    constraints { no_gradient_descent  max_changed_weights = 0.02%  max_forward_passes = 50 }
    verify    { benchmark MATH500  benchmark GSM8K }
}
```

What it needs, and what 0.1 already does about each part:

| Need | Status in 0.1 | Planned |
|---|---|---|
| Behaviours as declared measurements | `behavior b = probes(...)` | Generative benchmarks with sampling, scoring functions, held-out splits |
| One place for requirements | `contract { improve / preserve / require }` replaces `objective` and the `preserve` clauses scattered through `repair` | |
| One place for effects and budgets | `policy` replaces `constraints`; `forbid train` replaces `no_gradient_descent` | |
| Honest budgets | The checker warns when verification needs more forward passes than allowed. The example's 50 passes cannot cover MATH500's 500 problems and GSM8K's 1,319. | Per-token budgets for generation |
| Running Hugging Face models | Not supported: only `arch.json` sequential specs | Phase 4: architecture profiles, the `hf` extra |
| `find reasoning_loops`, `identify responsible circuits` | Not supported: only structural finders | Phase 5: probe-based discovery producing `Circuit` and `Feature` values with evidence levels |
| `remove reasoning_loops` | `repair` from findings with a finite strategy registry | Strategies grounded in causal evidence (ablation, activation patching, rank-one edits) |
| `explain findings` | Findings carry evidence, measurements and limitations | Optional reasoning providers that propose hypotheses, recorded and replayable, entering at level `hypothesised` |

## Composition beyond merging

```
compose MathModel {
    add circuit AdvancedReasoning
    preserve base_capabilities
}
```

This needs circuits as first-class, transferable values with a compatibility check between models: a `transfer` operation, and permutation alignment before interpolating unrelated networks. The `merge_unrelated.wsm` example shows why. Version 0.1 already composes weight sets that share a basin (`interpolate`, `transplant`) and verifies the result.

## Creating components

```
write circuit Reasoning
```

Synthesis of parameters from a specification is a research problem. The language will reserve `create` for it, as a separate verb from `save`/`output`, once a strategy exists that can be verified.

## Roadmap

1. **Foundations** (done in 0.1): grammar, parser, AST, diagnostics, types, effects, IR, `check`, `compile`.
2. **Execution** (done in 0.1): loading, architecture specs, content hashing, PyTorch backend, inspection, finders, dry runs.
3. **Transformations and composition** (done in 0.1): edits, compose, repair planning, patches, rollback, diff, provenance, verification.
4. **Architecture profiles**: Transformer roles (`layers[i].attn.heads[j]`, `mlp`, `embed`) mapped onto selections; Hugging Face loading; generative probes; head-level finders.
5. **Epistemic layer and reasoning**: circuit and feature discovery with interventional evidence; `transfer` of findings between compatible models; optional reasoning providers with record/replay.
6. **Ecosystem**: language server, editor support, package format, more backends.
