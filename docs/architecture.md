# Architecture

```
source (.wsm)
  -> parser (Lark, LALR)            language/grammar.lark, parser.py, nodes.py
  -> name resolution + type check   compiler/typecheck.py   (reads safetensors headers, no torch)
  -> lowering to IR                 compiler/lower.py, ir.py  (JSON-serialisable)
  -> effect check against policy    compiler/effects.py
  -> planner                        runtime/planner.py    (backend capabilities, repair planning)
  -> executor                       runtime/executor.py   (working copy, budgets, verification)
  -> backend                        backends/base.py (protocol), backends/pytorch/
  -> artifacts + report             model/, patch/, report.json
```

| Package | Responsibility |
|---|---|
| `language` | Grammar, parser, AST nodes with source spans, highlighter, hover docs for the language server |
| `compiler` | Types, name resolution and checking, effects and policies, IR, lowering |
| `registry` | Names and parameter schemas of finders, operations, strategies and metrics; torch-free |
| `models` | Tensor manifests from safetensors headers; architecture specs; Hugging Face configs and architecture profiles (roles to tensor regions); weight views |
| `runtime` | Planner, executor, working copy, budgets, editor sessions |
| `backends` | Backend protocol with capability declaration; the PyTorch backend |
| `evaluation` | Probe suites, rollouts and benchmarks, verdict semantics, run reports |
| `artifacts` | Patch manifest schema, provenance |
| `evidence` | Findings, evidence levels |
| `reasoning` | Editors: Anthropic, OpenAI and scripted providers; the edit language; recording and exact replay |
| `workers` | Code that runs outside this environment: the Kinetix rollout worker and the Hugging Face Jobs benchmark script |
| `synthetic`, `synthetic_lm` | Fixture models with planted defects and known ground truth, including a tiny Llama-style language model |

The language itself is defined in [semantics.md](semantics.md).
