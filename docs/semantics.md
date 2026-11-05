# Weightsmith language semantics

This document defines the language as currently implemented; it has not been released. The Python package is one implementation of it; where the implementation and this document disagree, the implementation has a bug.

## 1. Model of computation

A Weightsmith program operates on one **target**: a weight set loaded from disk. Execution never changes the loaded original. Every mutating statement acts on a copy-on-write **working copy** of the target. That gives every program three things for free:

- a **baseline** for before/after measurement,
- a **patch** from the original to the working copy,
- a guarantee that **inputs are never overwritten**.

Statements run in source order. Declarations (`model`, `weights`, `behavior`, `selection`, `policy`) are order-independent within a file. The `policy` and `contract` blocks of a program are declarative and apply to the whole program wherever they appear.

## 2. Lexical structure

| Element | Form |
|---|---|
| Whitespace | Insignificant, including newlines. Every statement starts with a keyword. |
| Comments | `//` to end of line. |
| Names | `[A-Za-z_][A-Za-z0-9_]*`. Keywords are reserved. |
| Integers | `42` |
| Floats | `0.5`, `1e-3`, `2.5e+2`. A digit is required after the decimal point. |
| Fractions | `5%` means 0.05. `12.5%` is allowed. |
| Strings | `"..."` with JSON escapes. |
| Booleans | `true`, `false` |
| Ranges | `a..b` inside a subscript: the half-open interval [a, b). |

Keywords: `model weights behavior selection policy program target let inspect find in edit compose repair contract verify output load probes allow forbid preserve improve require true false`.

## 3. Grammar

The normative grammar is `src/weightsmith/language/grammar.lark`. In EBNF:

```ebnf
file          = { decl } ;
decl          = load_decl | behavior_decl | selection_decl | editor_decl | policy_block | program ;
load_decl     = ( "model" | "weights" ) NAME "=" "load" "(" STRING ")" ;
behavior_decl = "behavior" NAME "=" "probes" "(" STRING { "," kwarg } ")" ;
selection_decl= "selection" NAME "=" expr ;
editor_decl   = "editor" NAME "=" "provider" "(" STRING { "," kwarg } ")" ;
policy_block  = "policy" "{" { ( "allow" | "forbid" ) NAME [ kwargs ] } "}" ;
program       = "program" NAME "{" "target" NAME { stmt } "}" ;

stmt          = "let" NAME "=" expr
              | "inspect" [ expr ]
              | "find" NAME [ "in" expr ] [ kwargs ]
              | "edit" "{" { op_stmt } "}"
              | "compose" "{" { op_stmt } "}"
              | "repair" [ kwargs ]
              | "delegate" NAME [ kwargs ]
              | "contract" "{" { ( "preserve" | "improve" | "require" ) NAME [ kwargs ] } "}"
              | "verify"
              | "output" "{" [ NAME { "," NAME } ] "}"
              | policy_block ;
op_stmt       = NAME expr [ kwargs ] ;

kwargs        = "(" [ kwarg { "," kwarg } ] ")" ;
kwarg         = NAME "=" expr ;
expr          = atom { "." NAME | "[" index { "," index } "]" } ;
atom          = NAME | INT | FLOAT | PERCENT | STRING | BOOL | "[" [ expr { "," expr } ] "]" ;
index         = INT | STRING | INT ".." INT | ":" ;
```

All arguments are named. There are no positional arguments and no general function calls.

## 4. Declarations

Paths in `load` and `probes` are resolved relative to the directory of the source file.

| Declaration | Meaning |
|---|---|
| `weights W = load(p)` | `p` is a `.safetensors` file or a directory containing `model.safetensors`. `W` has type `WeightSet`. It can be inspected, searched, edited and composed, but never run. |
| `model M = load(p)` | `p` is a directory with weights and an architecture: either `arch.json` (a Weightsmith sequential network) or a Hugging Face `config.json` (a causal language model). `M` has type `Model`, a `WeightSet` with an architecture, so it can also be probed. Running a Hugging Face model needs the `hf` extra; checking it does not. |
| `behavior B = probes(p, metric = m)` | A probe suite (`weightsmith.probes/v0` JSON) plus a metric. `B` has type `Behavior`. |
| `selection S = expr` | A named selection, usable in every program of the file. |
| `editor E = provider(p, ...)` | An editor (section 7a): a language model, or a script, that can act on the target through tools. `E` has type `Editor`. Credentials never appear in a program. |
| `policy { ... }` | The file-level policy (section 8). At most one per file. |
| `program P { target T ... }` | A program whose target is the `model` or `weights` binding `T`. |

A model directory's weights are `model.safetensors`, or the shards named by `model.safetensors.index.json`. A weight set loaded with `weights` from a directory that has a `config.json` gets an architecture profile (section 6) but still cannot run.

A behaviour is operational. `behavior parity = probes(...)` fixes how parity is *measured*. It says nothing about where or how a model computes parity.

**Probe suites** come in two kinds, both in `weightsmith.probes/v0` JSON:

| `kind` | Examples | Model | Prediction |
|---|---|---|---|
| `vectors` (default) | `{"input": [...], "label": k}` | an `arch.json` model | argmax over `output_slice` of the outputs |
| `tokens` | `{"prompt": "...", "target": "..."}`, or `prompt_ids` and `target_id` | a Hugging Face model | the most likely next token after the prompt |

A text prompt is tokenized with the model's tokenizer, special tokens included, as the model would see it. A text target must be exactly one token when tokenized without special tokens; include a leading space if the tokenizer needs it. A `tokens` suite may list `candidates` (text) or `candidate_ids`: the prediction is then the most likely candidate, which makes the suite a multiple-choice test. Text needs tokenizer files in the model directory; token ids do not.

## 5. Types

| Kind | Types | Notes |
|---|---|---|
| Structural | `WeightSet(b)`, `Model(b)`, `Params(b)`, `Selection(b)`, `Selection(b, sliced)` | `b` is the binding the value is rooted in. Derivable from files. |
| Epistemic | `Behavior`, and run-time `Finding`, `Evidence` | Findings carry the content hash of the weights they describe. |
| Transformational | run-time `Patch` | Bound to a base content hash. |
| Specification | `Contract`, `Policy` | Blocks, not first-class values. |
| Scalar | `Int`, `Float`, `Fraction`, `String`, `Bool`, `List[T]`, `Symbol[registry]` | |

Assignability: `Model(b) <: WeightSet(b)`; `Int <: Float`; `Fraction <: Float`; a `Float` or `Int` literal between 0 and 1 is accepted where a `Fraction` is expected. A `Symbol[r]` is a bare name that must name an entry of registry `r` (section 7).

Version 0.1 syntax cannot name `Finding`, `Evidence` or `Patch` values directly. They flow through `find`, `repair`, `verify` and `output`. `Circuit` and `Feature` are reserved for a later version and do not exist yet.

## 6. Expressions and statements

**Selections.** `M.params["glob"]` selects every parameter of `M` whose name matches the glob (`*`, `?`, `[...]`, as in `fnmatch`, case-sensitive). A glob matching nothing is an error. `S[i, a..b, :]` restricts an unsliced selection to an index region, one index per leading dimension. Sliced selections cannot be sliced again. `M` alone, where a selection is expected, means every parameter of `M`.

**Roles.** A weight set with an architecture profile can also be selected by role:

| Role | Selects |
|---|---|
| `M.embed` | token embeddings, and position embeddings if the model has them |
| `M.unembed` | the output projection; the token embeddings if they are tied (warning W305) |
| `M.final_norm` | the normalisation after the last block |
| `M.layers[i]` | every parameter of block `i`; `i` is an index, a range `a..b` or `:` |
| `M.layers[i].attn`, `.mlp`, `.norms` | the attention, MLP or normalisation parameters of those blocks |
| `M.layers[i].attn.heads[j]` | query head `j`: its rows of the query projection and its slice of the output projection, plus its key and value parts when no other head shares them |
| `M.layers[i].attn.kv_heads[g]` | key/value head `g`: its rows of the key and value projections |
| `M.layers[i].mlp.neurons[j]` | MLP neuron `j`: its row of each input projection (gate and up, or `c_fc`) and its slice of the output projection |

A head or neuron selection is a set of regions in several tensors; it counts as sliced, so it cannot be sliced again, searched, or passed to an operation that needs whole tensors. A whole-tensor role such as `M.layers[1].mlp` or `M.embed` can be sliced like any other selection. Indices are checked against the profile.

Profiles exist for two families, chosen by `model_type` in `config.json`: `gpt2` (fused query/key/value projection stored as `(in, out)`) and `llama` (`llama`, `mistral`, `qwen2`, `qwen3`, `gemma3_text`; separate projections stored as `(out, in)`, with grouped-query attention). A profile is built only if every tensor it relies on exists with the shape the configuration implies. Other models load and run, but only `params` selections are available.

**`let x = e`** binds a selection inside a program. Names cannot be rebound or shadow declarations.

**`inspect [e]`** records per-tensor statistics (shape, dtype, count, mean, std, min, max, L2 norm, zeros, non-finite) for `e`, or for the whole target. On the target it reads the working copy as it stands at that point.

**`find f [in e] [(args)]`** runs finder `f` over `e`, default the whole target, and records findings at evidence level `observed`. `e` must be unsliced. `anomalies (kinds = [...])` runs several finders with their default parameters. Unit finders (`outlier_heads`, `outlier_neurons`) need an architecture profile, and their scope must be the weight set or a layer role (`M.layers[...]`, optionally `.attn` for heads or `.mlp` for neurons).

**`edit { op e (args) ... }`** applies edit operations to selections of the target, in order. An operand rooted in another binding is an error: use `compose`.

**`compose { op e (args) ... }`** combines the working copy with another weight set: `e` is a weight set or a selection of one, never of the target, and may be a slice or a role such as `Donor.layers[0].attn.heads[1]`. Only the selected entries change. Names and shapes must match.

**`repair (strategies = [...])`** is the one declarative mutation. The planner takes the findings about the target that describe the current working copy, and pairs each with the first listed strategy that handles its kind. Findings no strategy handles are reported as unrepaired. If there are findings and none can be handled, the run fails with "no applicable strategy". Findings made before a later edit are stale and are not used. The program must contain a `find` on the target before `repair`, and at least one listed strategy must handle a finder the program runs.

**`delegate E (task = ..., tools = [...], practice = [...])`** hands the working copy to editor `E` (section 7a) until it stops calling tools or a budget runs out. Every change it makes is a mutation of the working copy like any other: budgeted, patched and verified by the contract. Give exactly one of `task` (a string), `task_file` (a text file, read at compile time so the program hash covers it) or `message_file` (a text file sent as the whole first message, see section 7a). `practice` lists the behaviours the editor may measure. A behaviour the contract verifies should not also be a practice behaviour (W306): the verdict would not be independent of the editor's search.

**`contract { ... }`** declares what `verify` checks:

- `improve B (min_delta = d [, min_n = k])`
- `preserve B (tolerance = t [, min_n = k])`
- `require finite`
- `require changed_params (max = f)`
- `require answer (editor = E, field = "F", matches = "regex")`

`improve` and `preserve` need the target to be a `Model`.

**`verify`** evaluates the contract (section 9) against the working copy at that point.

**`output { ... }`** must be the last statement. `model` and `patch` write artifacts and need the `save` effect. `findings`, `evidence`, `evaluation` and `summary` select what is shown on the console. The full JSON report is written to the output directory on every non-dry run.

## 7. Registries

Unknown names are compile-time errors that list what is available.

**Finders**, all at evidence level `observed`:

| Name | Parameters | Finds |
|---|---|---|
| `nan_inf` | | tensors containing NaN or infinities |
| `zero_tensors` | | tensors that are entirely zero |
| `dead_rows` | | all-zero rows of 2-D tensors |
| `outlier_rows` | `threshold = 8.0`, `min_ratio = 2.0` | rows whose L2 norm has a median/MAD robust z-score above the threshold and is at least `min_ratio` times the median row norm. Tensors with fewer than 8 rows are skipped and reported as skipped. |
| `outlier_tensors` | `threshold = 8.0`, `min_ratio = 2.0` | tensors whose norm is an outlier, in the same sense, among at least 3 same-shape tensors |
| `outlier_heads` | `threshold = 8.0`, `min_ratio = 2.0` | attention heads whose parameter norm is an outlier among comparable heads: those of the same layer when it has at least 8, otherwise all heads in scope. Needs a profile. |
| `outlier_neurons` | `threshold = 8.0`, `min_ratio = 2.0` | MLP neurons whose parameter norm is an outlier among the neurons of their layer. Needs a profile. |
| `duplicate_tensors` | | bit-identical non-constant tensors |
| `low_effective_rank` | `threshold = 10%` | 2-D tensors whose entropy effective rank is below `threshold * min(shape)` |
| `anomalies` | `kinds = [...]` | runs the listed finders; by default every finder that needs no profile |

The `min_ratio` condition matters when few items are compared or they are nearly equal: the median absolute deviation is then tiny and z-scores alone flag ordinary variation. Groups of fewer than 8 heads or neurons are skipped and reported as skipped.

**Edit operations** (operand: a selection of the target):

| Name | Parameters | Effect on the selected region |
|---|---|---|
| `zero` | | set to 0 |
| `scale` | `factor` | multiply |
| `fill` | `value` | set to a constant |
| `prune_magnitude` | `fraction` | zero the smallest-magnitude `fraction` of entries, per tensor |
| `low_rank` | `rank` | best rank-k approximation (truncated SVD); whole 2-D tensors only |
| `replace_nonfinite` | `value = 0.0` | replace NaN and infinities |

**Compose operations** (operand: another weight set or a selection of one):

| Name | Parameters | Effect |
|---|---|---|
| `interpolate` | `alpha` in [0, 1] | `w = (1 - alpha) * w + alpha * donor` on the selected entries, computed in float64 |
| `transplant` | | copy the donor's selected entries |

**Repair strategies**:

| Name | Handles | Effect |
|---|---|---|
| `rescale_outlier_rows` | `outlier_rows` | scale each flagged row to the tensor's median row norm. If `<prefix>.bias` has one entry per row, scale that entry by the same factor. |
| `rescale_outlier_units` | `outlier_heads`, `outlier_neurons` | for each part of the unit (for a neuron: its gate row, up row and down column), if the part's norm is at least the finder's `min_ratio` times the median norm of the same part across the compared units, scale the part to that median. Other parts are left alone. |
| `replace_nonfinite` | `nan_inf` | replace non-finite entries with 0 |

**Requirements** (written `require <name>` in a contract):

| Name | Parameters | Holds when |
|---|---|---|
| `finite` | | no parameter of the working copy is NaN or infinite |
| `changed_params` | `max` | the fraction of changed parameters is at most `max` |
| `answer` | `editor`, `field`, `matches` | the last reply of editor `editor` in this run has a line `field: value` (markdown around the field name is ignored, and a heading `## field` takes the next line as its value) whose value matches the regular expression `matches`. The pattern is part of the program, never of the editor's messages. |

**Metrics**: `accuracy`, the fraction of probe examples predicted correctly (section 4). Higher is better.

## 7a. Editors

An editor is declared with a provider and settings. Providers:

| Provider | Settings | Needs |
|---|---|---|
| `anthropic` | `model = "claude-opus-5-5"`, `effort = "high"` (`low` to `max`), `max_tokens = 64000` | the `anthropic` extra; credentials from the environment |
| `openai` | `model` (required), `api` (`responses` by default; `chat` for Chat Completions, the default when `base_url` is set), `base_url` for any server that speaks Chat Completions, including local ones, `effort` (`none` to `max`, as the model allows), `max_tokens = 64000` | the `openai` extra; `OPENAI_API_KEY` unless `base_url` is local |
| `scripted` | `script`: a `weightsmith.script/v0` JSON file of turns, each with optional `text` and `calls` | nothing; deterministic |

`delegate` arguments:

- one of `task`, `task_file` or `message_file`;
- `tools` (default `[describe, read_weights, stats]`; `[]` for none);
- `protocol = "tools"` or `"blocks"` (below);
- `system` or `system_file`: system instructions that replace Weightsmith's own;
- `practice` (default none);
- `max_steps = 20` (editor turns);
- `max_read = 20000` (numbers one read may return);
- `include_weights = false` (put every weight, as text, in the briefing, for small networks);
- `weights_format`: how weights are written as text, `decimals(n)` or `int_scale(n)` (below).

With the OpenAI Responses API, requests are sent with `store: false` and the encrypted reasoning items the model returns are sent back with the next request, so reasoning carries over between turns and every request is complete in itself.

**Messages and weights as text.** By default the first message is a briefing written by Weightsmith. With `message_file`, the file is sent as it is instead, with `{{weights}}` replaced by every weight as text (no other placeholder exists, E367); this is how a program sends a prompt exactly as published elsewhere. Weights are written through the target's *view*: an optional `view.json` (`weightsmith.view/v0`) in the model directory lists the tensors to show, in order, each with the `label` the editor sees, the `param` it stands for, whether it is shown `transpose`d, and a `header` template (`{label}`, `{shape}` such as `500 x 32`, `{rows}`, `{cols}`, and for `int_scale` `{scale}` such as `1e-3`). Without a view every parameter is shown under its own name. Tensors are separated by a blank line and matrices are written one line per row. `decimals(n)` writes every value with `n` decimals (`-0.000` as `0.000`); `int_scale(n)` writes each tensor as integers times one power of ten chosen so that its largest entry keeps `n` significant digits, and starts each matrix row with its index. Both are lossy: an editor that edits what it read edits the rounded network.

**Protocols.** With `protocol = "tools"` the editor calls function tools. With `protocol = "blocks"` it is given no function tools: it acts by writing a fenced block whose opening line names a tool, and only the last such block of a reply is read. A reply without one ends the delegation. Only tools that have a block form can be granted under this protocol (E368); with `tools = []` the editor answers once in text, which `require answer` can then check.

**Tools.** A tool's effects must be allowed by the policy (E410).

| Tool | Effect | Does |
|---|---|---|
| `describe` | inspect | lists tensors, the profile, budgets left, practice behaviours, what the contract verifies, edit operations and finders |
| `read_weights` | inspect | values of a selection as text |
| `stats` | inspect | per-tensor statistics of a selection |
| `find` | inspect | runs a finder on the working copy |
| `run_probes` | probe | accuracy on a practice behaviour, with failing examples; one forward pass per example |
| `forward` | probe | next-token candidates for prompts, or outputs for input vectors; one forward pass per input |
| `edit` | mutate | an edit operation (section 7) on a selection |
| `set_weights` | mutate | writes explicit values into a selection, region by region in row-major order |
| `undo` | mutate | reverts the editor's last change |
| `python` | execute | runs Python in a separate process with the weights as NumPy arrays `W`; arrays it changes are applied as edits, which also need `mutate` (W403) |

Tools name selections with strings: `all`, `params["glob"]` with an optional slice, or a role path such as `layers[1].attn.heads[2]` (section 6).

**The loop.** The runtime sends a briefing: the task, the target, the profile, budgets, practice behaviours and the contract's items, but never the contract's probes. Each turn, every tool call is checked and run; a failing call, or one that would exceed a budget, returns an error to the editor and the run continues. The editor stops by replying without a tool call; its account is recorded as a hypothesis. The delegation also stops at `max_steps`, at the `reason` budget, or on a refusal. Then the program continues with its next statement.

**Recording and replay.** Every request and response is written to `<out>/transcripts/NN-E.json` (`weightsmith.transcript/v0`) as the run proceeds. `weightsmith run --replay DIR` answers each editor call from the recording instead of the provider. Requests are compared by SHA-256: if the weights, tools, budgets, task or any tool result differ from the recording, the replay stops with an error rather than continuing on a different path. Nothing derived from wall-clock time enters a request.

**The `python` tool is not a security boundary.** Code runs as the current user, in a fresh process per call, with a minimal environment (no credentials) and a time budget. Files written in its working directory persist between calls. Allow `execute` only for editors you would let run code on your machine.

## 8. Effects and policies

| Effect | Performed by | Budgets |
|---|---|---|
| `load` | loading weights and probe suites | `max_bytes` |
| `inspect` | `inspect`, `find`, `verify` with only `require` items, and the reading tools of `delegate` | |
| `probe` | `verify` with `preserve`/`improve` items, and `run_probes`/`forward` | `max_forward_passes`: one per probe example or input per evaluation |
| `mutate` | `edit`, `compose`, `repair`, and the editing tools of `delegate` | `max_changed_params`: a fraction of the target's parameters |
| `train` | nothing yet | `max_steps` |
| `reason` | `delegate`: every call to an editor; may send model data off the machine | `max_calls` (editor turns), `max_tokens` (input plus output, as the provider reports them) |
| `execute` | the `python` tool | `max_seconds`: wall time over all calls (default 600) |
| `save` | `output { model }`, `output { patch }` | |

**Defaults.** With no policy, `load` and `inspect` are allowed and everything else is forbidden. A policy block starts from the defaults and applies its rules. Each effect may appear once.

**Scoping.** A file-level policy applies to every program without its own. A program policy is checked against the file policy and may only tighten it. Allowing an effect the file forbids, or raising a budget, is an error. Budgets the program omits are inherited.

**Enforcement.** The compiler rejects any operation whose effect is not allowed (E410). It warns when verification visibly needs more forward passes than the probe budget allows (W402). At run time the policy is re-checked, because compiled IR can be edited, and budgets are enforced before work is done:

- A mutation that would exceed `max_changed_params` is not applied, and the run stops.
- A behaviour check that would exceed `max_forward_passes` is not run and becomes `inconclusive`. The run stops after that `verify`.
- Inside `delegate`, a tool call that would exceed a budget is refused and the editor is told; an exhausted `reason` budget ends the delegation, not the run.

Budgets are never silently exceeded.

## 9. Verification

Each contract item gets one verdict:

| Verdict | Meaning |
|---|---|
| `verified` | Measured within budget on at least `min_n` examples (default 20), and the comparator held. |
| `failed` | Measured, and the comparator did not hold. |
| `inconclusive` | No trustworthy measurement: budget exhausted, suite smaller than `min_n`, threshold finer than the suite's resolution 1/n, or no architecture. |
| `skipped` | Deliberately not evaluated (dry run). |

Comparators, for a higher-is-better metric with `before` measured on the original target and `after` on the working copy:

- `improve`: `after - before >= min_delta`.
- `preserve`: `before - after <= tolerance`. Improvement satisfies `preserve`.
- `require finite`: no parameter of the working copy is NaN or infinite.
- `require changed_params`: changed parameters / total parameters `<= max`.
- `require answer`: `verified` when the editor's `field` line matches `matches`, `failed` when it does not, `inconclusive` when the editor gave no such line or was not asked.

A parameter counts as **changed** when its stored bits differ from the original. Baselines are measured once per behaviour and reused across `verify` statements.

A verdict is a statement about the declared probe suite. It is not a statistical guarantee about the behaviour in general. Nothing is ever promoted from `inconclusive` to `verified`.

## 10. Findings and evidence

A finding records its kind, the binding and content hash of the weights it describes, the locations it points to, a list of evidence, and its known limitations. Its level is the strongest level among its evidence:

| Level | Operational definition |
|---|---|
| `hypothesised` | Asserted without runtime evidence, for example by a reasoning provider. |
| `observed` | A deterministic statistic of the weights. No forward pass was run. |
| `experimentally_supported` | An intervention in this run changed a measured behaviour as predicted. |
| `independently_verified` | Reproduced by a second method or on held-out probes. Never assigned in 0.1. |

In 0.1, `verify` raises a repaired finding to `experimentally_supported` only under three conditions. Every mutation in the run must have come from `repair`. An `improve` item must be verified. The evidence must name the other findings repaired in the same step, because the attribution is joint.

## 11. Static checks

`weightsmith check` runs four stages, each reporting diagnostics with source spans:

1. **syntax**: E1xx.
2. **names**: unknown or duplicate names, unknown registry entries (E2xx).
3. **types**: argument types and ranges, statement rules, contract rules (E3xx), including roles (E316 no profile, E317 unknown part, E318 missing or bad index, E319 unit-finder scope) and editors (E329 a setting outside its choices, E360 delegate to a non-editor, E361 task or task_file, E362 running tools on weights that cannot run, E363 unreadable task file, E364 credentials in a program, E365 both `system` and `system_file`, E366 an unknown weight format, E367 an unknown placeholder in a message file, E368 a tool without a block form under `protocol = "blocks"`, E369 an invalid `answer` pattern; W306 practising on a verified behaviour, W307 practice without run_probes). Structural checks against tensor manifests read from safetensors headers and against `config.json` (E5xx): glob matches, slice bounds, role indices (E510, E511), 2-D requirements, compose compatibility, probe-suite widths against `arch.json`, suite kind against the model (E512), tokenizer presence (E513) and token ids against the vocabulary (E514).
4. **effects**: policy rules (E40x), effect permission (E410), and W403 when `python` is granted without `mutate`.

When an input file does not exist yet, structural checks are deferred to run time and a warning (W201, W202) says so. `check` never imports a tensor library.

## 12. Runtime behaviour

The runtime checks, before executing anything, that the backend declares every capability and registry entry the program needs. It then checks the policy and prepares the output directory. It refuses to write into a non-empty directory unless told to overwrite, and refuses any directory inside an input's directory.

`weightsmith run` exit codes:

| Code | Meaning |
|---|---|
| 0 | completed; every contract item verified |
| 1 | the program is invalid |
| 2 | runtime error, or stopped by a budget |
| 3 | completed; at least one contract item failed |
| 4 | completed; no failures, at least one item inconclusive |

A dry run (`--dry-run`) loads, inspects, finds and plans. It computes mutations in memory to report their size. It measures nothing, marks every contract item `skipped`, and writes no file.

## 13. Intermediate representation

A program lowers to a linear list of operations over numbered values (`weightsmith.ir/v0`):

| Op | Meaning | Effects |
|---|---|---|
| `Load` | load a model, weight set or behaviour | load |
| `Select` | select parameters by glob or by role (a canonical role path such as `layers[1].attn.heads[2]`, resolved against the profile at run time), or slice a selection | none |
| `Inspect` | statistics | inspect |
| `Find` | run a finder | inspect |
| `Transform` | edit operation on the working copy | mutate |
| `Compose` | compose operation on the working copy | mutate |
| `Repair` | planner-expanded repair | mutate |
| `Delegate` | an editor acting through tools | reason, plus the effects of the granted tools |
| `Verify` | evaluate contract items | probe or inspect |
| `Emit` | write artifacts | save, if model or patch |

Loaded values and selections are immutable. Mutating ops act on the target's implicit working copy. The IR names operations and parameters, never tensors or framework types, and serialises to JSON (`weightsmith compile`). `weightsmith run` accepts that JSON directly. A program's content hash is computed over its ops and policy, without source locations, and is recorded in reports and patches.

## 14. Versioning

The language, IR (`weightsmith.ir/v0`), architecture specs (`weightsmith.arch/v0`), probe suites (`weightsmith.probes/v0`), patches (`weightsmith.patch/v0`) and reports (`weightsmith.report/v0`) are versioned independently. Readers reject versions they do not know.
