# Weightsmith language semantics

Language version 0.1. This document defines the language. The Python package is one implementation of it; where the implementation and this document disagree, the implementation has a bug.

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
decl          = load_decl | behavior_decl | selection_decl | policy_block | program ;
load_decl     = ( "model" | "weights" ) NAME "=" "load" "(" STRING ")" ;
behavior_decl = "behavior" NAME "=" "probes" "(" STRING { "," kwarg } ")" ;
selection_decl= "selection" NAME "=" expr ;
policy_block  = "policy" "{" { ( "allow" | "forbid" ) NAME [ kwargs ] } "}" ;
program       = "program" NAME "{" "target" NAME { stmt } "}" ;

stmt          = "let" NAME "=" expr
              | "inspect" [ expr ]
              | "find" NAME [ "in" expr ] [ kwargs ]
              | "edit" "{" { op_stmt } "}"
              | "compose" "{" { op_stmt } "}"
              | "repair" [ kwargs ]
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
| `model M = load(p)` | `p` is a directory with `model.safetensors` and `arch.json`. `M` has type `Model`, a `WeightSet` with an architecture, so it can also be probed. |
| `behavior B = probes(p, metric = m)` | A probe suite (`weightsmith.probes/v0` JSON) plus a metric. `B` has type `Behavior`. |
| `selection S = expr` | A named selection, usable in every program of the file. |
| `policy { ... }` | The file-level policy (section 8). At most one per file. |
| `program P { target T ... }` | A program whose target is the `model` or `weights` binding `T`. |

A behaviour is operational. `behavior parity = probes(...)` fixes how parity is *measured*. It says nothing about where or how a model computes parity.

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

**`let x = e`** binds a selection inside a program. Names cannot be rebound or shadow declarations.

**`inspect [e]`** records per-tensor statistics (shape, dtype, count, mean, std, min, max, L2 norm, zeros, non-finite) for `e`, or for the whole target. On the target it reads the working copy as it stands at that point.

**`find f [in e] [(args)]`** runs finder `f` over `e`, default the whole target, and records findings at evidence level `observed`. `e` must be unsliced. `anomalies (kinds = [...])` runs several finders with their default parameters.

**`edit { op e (args) ... }`** applies edit operations to selections of the target, in order. An operand rooted in another binding is an error: use `compose`.

**`compose { op e (args) ... }`** combines the working copy with another weight set: `e` is a weight set or an unsliced selection of one, never of the target. Names and shapes must match.

**`repair (strategies = [...])`** is the one declarative mutation. The planner takes the findings about the target that describe the current working copy, and pairs each with the first listed strategy that handles its kind. Findings no strategy handles are reported as unrepaired. If there are findings and none can be handled, the run fails with "no applicable strategy". Findings made before a later edit are stale and are not used. The program must contain a `find` on the target before `repair`, and at least one listed strategy must handle a finder the program runs.

**`contract { ... }`** declares what `verify` checks:

- `improve B (min_delta = d [, min_n = k])`
- `preserve B (tolerance = t [, min_n = k])`
- `require finite`
- `require changed_params (max = f)`

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
| `outlier_rows` | `threshold = 8.0` | rows whose L2 norm has a median/MAD robust z-score above the threshold. Tensors with fewer than 8 rows are skipped and reported as skipped. |
| `outlier_tensors` | `threshold = 8.0` | tensors whose norm is a robust outlier among at least 3 same-shape tensors |
| `duplicate_tensors` | | bit-identical non-constant tensors |
| `low_effective_rank` | `threshold = 10%` | 2-D tensors whose entropy effective rank is below `threshold * min(shape)` |
| `anomalies` | `kinds = [...]` | runs the listed finders |

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
| `interpolate` | `alpha` in [0, 1] | `w = (1 - alpha) * w + alpha * donor`, computed in float64 |
| `transplant` | | copy the donor's tensors |

**Repair strategies**:

| Name | Handles | Effect |
|---|---|---|
| `rescale_outlier_rows` | `outlier_rows` | scale each flagged row to the tensor's median row norm. If `<prefix>.bias` has one entry per row, scale that entry by the same factor. |
| `replace_nonfinite` | `nan_inf` | replace non-finite entries with 0 |

**Requirements** (written `require <name>` in a contract):

| Name | Parameters | Holds when |
|---|---|---|
| `finite` | | no parameter of the working copy is NaN or infinite |
| `changed_params` | `max` | the fraction of changed parameters is at most `max` |

**Metrics**: `accuracy`, the fraction of probe examples whose argmax over the suite's `output_slice` equals the label. Higher is better.

## 8. Effects and policies

| Effect | Performed by | Budgets |
|---|---|---|
| `load` | loading weights and probe suites | `max_bytes` |
| `inspect` | `inspect`, `find`, `verify` with only `require` items | |
| `probe` | `verify` with `preserve`/`improve` items | `max_forward_passes`: one per probe example per evaluation |
| `mutate` | `edit`, `compose`, `repair` | `max_changed_params`: a fraction of the target's parameters |
| `train` | nothing in 0.1 | `max_steps` |
| `reason` | nothing in 0.1; reserved for external reasoning providers | `max_calls` |
| `save` | `output { model }`, `output { patch }` | |

**Defaults.** With no policy, `load` and `inspect` are allowed and everything else is forbidden. A policy block starts from the defaults and applies its rules. Each effect may appear once.

**Scoping.** A file-level policy applies to every program without its own. A program policy is checked against the file policy and may only tighten it. Allowing an effect the file forbids, or raising a budget, is an error. Budgets the program omits are inherited.

**Enforcement.** The compiler rejects any operation whose effect is not allowed (E410). It warns when verification visibly needs more forward passes than the probe budget allows (W402). At run time the policy is re-checked, because compiled IR can be edited, and budgets are enforced before work is done:

- A mutation that would exceed `max_changed_params` is not applied, and the run stops.
- A behaviour check that would exceed `max_forward_passes` is not run and becomes `inconclusive`. The run stops after that `verify`.

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
3. **types**: argument types and ranges, statement rules, contract rules (E3xx). Structural checks against tensor manifests read from safetensors headers (E5xx): glob matches, slice bounds, 2-D requirements, compose compatibility, and probe-suite widths against `arch.json`.
4. **effects**: policy rules (E40x) and effect permission (E410).

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
| `Select` | glob-select parameters, or slice a selection | none |
| `Inspect` | statistics | inspect |
| `Find` | run a finder | inspect |
| `Transform` | edit operation on the working copy | mutate |
| `Compose` | compose operation on the working copy | mutate |
| `Repair` | planner-expanded repair | mutate |
| `Verify` | evaluate contract items | probe or inspect |
| `Emit` | write artifacts | save, if model or patch |

Loaded values and selections are immutable. Mutating ops act on the target's implicit working copy. The IR names operations and parameters, never tensors or framework types, and serialises to JSON (`weightsmith compile`). `weightsmith run` accepts that JSON directly. A program's content hash is computed over its ops and policy, without source locations, and is recorded in reports and patches.

## 14. Versioning

The language, IR (`weightsmith.ir/v0`), architecture specs (`weightsmith.arch/v0`), probe suites (`weightsmith.probes/v0`), patches (`weightsmith.patch/v0`) and reports (`weightsmith.report/v0`) are versioned independently. Readers reject versions they do not know.
