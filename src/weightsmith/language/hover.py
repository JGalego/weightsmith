"""Hover documentation for Weightsmith source, as Markdown.

Used by the language server (``weightsmith lsp``). Everything shown comes from the registries, so
the documentation of a finder, an operation or a delegate parameter is the one the compiler checks
against. The context of a word is read from the Pygments lexer's tokens rather than a parse tree,
so hovering works in programs that do not parse yet.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pygments.token import Comment, Keyword, Name, Punctuation, String, _TokenType

from weightsmith import registry
from weightsmith.compiler.effects import BUDGETS, Effect
from weightsmith.language.highlight import DECLARATIONS, WeightsmithLexer
from weightsmith.registry import Entry, ParamSpec

KEYWORD_DOCS: dict[str, str] = {
    "model": "`model M = load(path)`: a directory with weights and an architecture "
    "(`arch.json`, or a Hugging Face `config.json`). A model can be probed. In `output { }`, "
    "saves the working model.",
    "weights": "`weights W = load(path)`: a weight set (a `.safetensors` file or a directory "
    "holding one). It can be inspected, searched, edited and composed, but never run.",
    "behavior": "`behavior B = probes(path, metric = m)`: a probe suite plus a metric. It fixes "
    "how a behaviour is measured, not where the model computes it.",
    "selection": "`selection S = expr`: a named selection, usable in every program of the file.",
    "editor": "`editor E = provider(name, ...)`: a language model, or a script, that can act on "
    "the target. Credentials never appear in a program.",
    "program": "`program P { target T ... }`: statements run against a working copy of the "
    "target `T`.",
    "let": "`let x = expr`: binds a selection inside a program. Names cannot be rebound.",
    "policy": "`policy { allow e (budgets) forbid e }`: which effects may happen, and their "
    "budgets. A program policy may only tighten the file policy.",
    "target": "`target T`: the `model` or `weights` binding the program works on. The program "
    "edits a working copy; the original is never changed.",
    "inspect": "`inspect [expr]`: records per-tensor statistics of a selection, or of the whole "
    "working copy.",
    "find": "`find f [in expr] [(args)]`: runs finder `f` and records its findings at evidence "
    "level `observed`.",
    "in": "`find f in expr`: the scope a finder searches; the whole target by default.",
    "edit": "`edit { op expr (args) ... }`: applies edit operations to selections of the target, "
    "in order.",
    "compose": "`compose { op expr (args) ... }`: combines the working copy with another weight "
    "set. Only the selected entries change.",
    "repair": "`repair (strategies = [...])`: pairs each current finding with the first listed "
    "strategy that handles its kind, and applies it.",
    "delegate": "`delegate E (task = ..., tools = [...])`: hands the working copy to editor `E` "
    "until it stops or a budget runs out. Its changes are budgeted, patched and verified like "
    "any other mutation.",
    "contract": "`contract { improve ... preserve ... require ... }`: what `verify` checks.",
    "verify": "`verify`: evaluates the contract against the working copy at this point. Each item "
    "gets a verdict: `verified`, `failed`, `inconclusive` or `skipped`.",
    "output": "`output { model, patch, findings, evidence, evaluation, summary }`: the last "
    "statement; what to save and what to show.",
    "allow": "`allow effect (budgets)`: permits an effect, optionally within budgets.",
    "forbid": "`forbid effect`: forbids an effect.",
    "preserve": "",  # from registry.CLAUSES
    "improve": "",  # from registry.CLAUSES
    "require": "`require item (args)`: a structural contract item (`finite`, `changed_params`, "
    "`answer`).",
    "true": "Boolean literal.",
    "false": "Boolean literal.",
}

FUNCTION_DOCS: dict[str, str] = {
    "load": "`load(path)`: loads weights from a path relative to this file.",
    "probes": "`probes(path, metric = m)`: a probe suite (`weightsmith.probes/v0` JSON) measured "
    "with metric `m`.",
    "provider": "`provider(name, ...)`: an editor provider; its settings depend on the provider.",
}

EFFECT_DOCS: dict[Effect, str] = {
    Effect.LOAD: "Read model files and probe suites from disk.",
    Effect.INSPECT: "Read parameter values: `inspect`, `find`, and the reading tools of "
    "`delegate`.",
    Effect.PROBE: "Run forward passes: `verify` with `preserve`/`improve`, and probing tools.",
    Effect.MUTATE: "Change parameters of the working copy: `edit`, `compose`, `repair`, and the "
    "editing tools of `delegate`.",
    Effect.TRAIN: "Run an optimiser. No operation does this yet.",
    Effect.REASON: "Call a language model acting as an editor. May send model data off the "
    "machine.",
    Effect.EXECUTE: "Run code an editor wrote, on this machine (the `python` tool).",
    Effect.SAVE: "Write model or patch artifacts (`output { model }`, `output { patch }`).",
}

BUDGET_DOCS: dict[str, str] = {
    "max_bytes": "Most bytes loaded from disk.",
    "max_forward_passes": "Most forward passes: one per probe example or input per evaluation.",
    "max_gpu_hours": (
        "Most GPU hours of benchmark jobs, each evaluation charged as if every job ran to its "
        "timeout: a cap on what the jobs can cost."
    ),
    "max_changed_params": "Largest fraction of the target's parameters that may change.",
    "max_steps": "Most optimiser steps.",
    "max_calls": "Most editor turns.",
    "max_tokens": "Most tokens, input plus output, as the provider reports them.",
    "max_seconds": "Most wall-clock seconds over all code runs.",
}

# Parameters whose value is a list of registry names: parameter -> (kind, registry).
_KWARG_LISTS: dict[str, tuple[str, dict[str, Entry]]] = {
    "tools": ("tool", registry.TOOLS),  # type: ignore[dict-item]
    "strategies": ("strategy", registry.STRATEGIES),  # type: ignore[dict-item]
}


@dataclass(frozen=True, slots=True)
class _Tok:
    start: int
    line: int
    kind: _TokenType
    text: str

    @property
    def end(self) -> int:
        return self.start + len(self.text)


def hover(source: str, offset: int) -> str | None:
    """Markdown for the word at ``offset`` in ``source``, or None if there is nothing to say."""
    tokens = _tokens(source)
    index = next((i for i, t in enumerate(tokens) if t.start <= offset < t.end), None)
    if index is None:
        return None
    tok = tokens[index]
    if tok.kind in Comment:
        return None
    if tok.kind in String:
        return _string(tokens, index)
    if tok.kind in Name.Attribute:
        return _kwarg(tokens, index)
    if tok.kind in Keyword:
        if _list_owner(tokens, index) == "tools" and tok.text in registry.TOOLS:
            return _entry("tool", registry.TOOLS[tok.text])
        return _keyword(tok.text)
    if tok.kind in Name:
        return _name(source, tokens, index)
    return None


# --------------------------------------------------------------------------- what a word means


def _keyword(word: str) -> str | None:
    if word in registry.CLAUSES:
        return _entry("clause", registry.CLAUSES[word])
    doc = KEYWORD_DOCS.get(word)
    return f"**{word}** (keyword)\n\n{doc}" if doc else None


def _name(source: str, tokens: list[_Tok], index: int) -> str | None:
    word = tokens[index].text
    if word in FUNCTION_DOCS and _next(tokens, index, "("):
        return f"**{word}** (built-in)\n\n{FUNCTION_DOCS[word]}"
    before = _previous(tokens, index)
    if before is not None:
        lead = before.text
        if lead == "find" and word in registry.FINDERS:
            return _entry("finder", registry.FINDERS[word])
        if lead == "require" and word in registry.REQUIREMENTS:
            return _entry("contract item", registry.REQUIREMENTS[word])
        if lead in ("allow", "forbid") and word in set(Effect):
            return _effect(Effect(word))
    owner = _list_owner(tokens, index)
    if owner in _KWARG_LISTS and word in _KWARG_LISTS[owner][1]:
        kind, table = _KWARG_LISTS[owner]
        return _entry(kind, table[word])
    if "metric" in (owner, _assigned_to(tokens, index)) and word in registry.METRICS:
        return _entry("metric", registry.METRICS[word])
    if _block_of(tokens, index) == "output" and word in registry.OUTPUTS:
        return f"**{word}** (output)\n\n{registry.OUTPUTS[word]}"
    block = _block_of(tokens, index)
    if block in ("edit", "compose") and _starts_statement(tokens, index):
        ops = registry.EDIT_OPS if block == "edit" else registry.COMPOSE_OPS
        if word in ops:
            return _entry(f"{block} operation", ops[word])
    declared = _declaration(source, tokens, word)
    if declared is not None:
        return declared
    return _any_registry(word)


def _kwarg(tokens: list[_Tok], index: int) -> str | None:
    name = tokens[index].text
    params = _params_of_call(tokens, index)
    if params is None:
        return None
    owner, specs = params
    if isinstance(specs, dict):  # budgets
        if name in specs:
            return f"**{name}**: {specs[name]} (budget of `{owner}`)\n\n{BUDGET_DOCS.get(name, '')}"
        return None
    spec = next((p for p in specs if p.name == name), None)
    return f"{_param(spec)}\n\n*Argument of `{owner}`.*" if spec else None


def _string(tokens: list[_Tok], index: int) -> str | None:
    # The provider name: `provider("openai", ...)`.
    opening = _previous(tokens, index)
    head = _previous(tokens, index - 1) if opening and opening.text == "(" else None
    if head is not None and head.text == "provider":
        value = tokens[index].text.strip('"')
        if value in registry.PROVIDERS:
            return _entry("provider", registry.PROVIDERS[value])
    return None


def _params_of_call(
    tokens: list[_Tok], index: int
) -> tuple[str, tuple[ParamSpec, ...] | dict[str, object]] | None:
    """The parameters accepted by the call whose argument list contains token ``index``."""
    opening = _unmatched(tokens, index, "(", ")")
    if opening is None:
        return None
    head = _statement_head(tokens, opening)
    if not head:
        return None
    words = [t.text for t in head]
    first = words[0]
    if words[-1] == "probes":
        return "probes", registry.BEHAVIOR_PARAMS
    if words[-1] == "provider":
        name = _first_string(tokens, opening)
        spec = registry.PROVIDERS.get(name or "")
        return (f'provider("{name}")', spec.params) if spec else None
    if first == "delegate":
        return "delegate", registry.DELEGATE_PARAMS
    if first == "repair":
        return "repair", registry.REPAIR_PARAMS
    if first in ("allow", "forbid") and len(words) > 1 and words[1] in set(Effect):
        budgets = BUDGETS[Effect(words[1])]
        return words[1], {k: str(v) for k, v in budgets.items()}
    tables: dict[str, dict[str, Entry]] = {
        "find": registry.FINDERS,  # type: ignore[dict-item]
        "require": registry.REQUIREMENTS,  # type: ignore[dict-item]
    }
    if first in tables and len(words) > 1 and words[1] in tables[first]:
        return words[1], tables[first][words[1]].params
    if first in registry.CLAUSES:
        return first, registry.CLAUSES[first].params
    block = _block_of(tokens, opening)
    ops = {"edit": registry.EDIT_OPS, "compose": registry.COMPOSE_OPS}.get(block or "")
    if ops is not None and first in ops:
        return first, ops[first].params
    return None


# --------------------------------------------------------------------------- formatting


def _entry(kind: str, entry: Entry) -> str:
    lines = [f"**{entry.name}** ({kind})", "", entry.doc]
    if entry.params:
        lines += ["", *(f"- {_param(p)}" for p in entry.params)]
    return "\n".join(lines)


def _param(p: ParamSpec) -> str:
    facts = [str(p.type)]
    if p.required:
        facts.append("required")
    elif p.default is not None:
        facts.append(f"default `{_literal(p.default)}`")
    if p.choices:
        facts.append("one of " + ", ".join(f"`{c}`" for c in p.choices))
    if p.minimum is not None or p.maximum is not None:
        low = "" if p.minimum is None else f"{p.minimum:g} ≤ "
        high = "" if p.maximum is None else f" ≤ {p.maximum:g}"
        facts.append(f"{low}value{high}")
    return f"`{p.name}` ({', '.join(facts)}): {p.doc}"


def _literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    return str(value)


def _effect(effect: Effect) -> str:
    budgets = BUDGETS[effect]
    lines = [f"**{effect.value}** (effect)", "", EFFECT_DOCS[effect]]
    if budgets:
        lines += ["", *(f"- `{k}` ({v}): {BUDGET_DOCS[k]}" for k, v in budgets.items())]
    return "\n".join(lines)


def _any_registry(word: str) -> str | None:
    tables: Iterable[tuple[str, dict[str, Entry]]] = (
        ("finder", registry.FINDERS),  # type: ignore[arg-type]
        ("edit operation", registry.EDIT_OPS),  # type: ignore[arg-type]
        ("compose operation", registry.COMPOSE_OPS),  # type: ignore[arg-type]
        ("strategy", registry.STRATEGIES),  # type: ignore[arg-type]
        ("metric", registry.METRICS),  # type: ignore[arg-type]
        ("contract item", registry.REQUIREMENTS),  # type: ignore[arg-type]
        ("tool", registry.TOOLS),  # type: ignore[arg-type]
    )
    found = [_entry(kind, table[word]) for kind, table in tables if word in table]
    if word in set(Effect):
        found.append(_effect(Effect(word)))
    if word in registry.OUTPUTS:
        found.append(f"**{word}** (output)\n\n{registry.OUTPUTS[word]}")
    return "\n\n---\n\n".join(found) or None


def _declaration(source: str, tokens: list[_Tok], word: str) -> str | None:
    """The declaration of a name the program defines, shown as written."""
    for i, tok in enumerate(tokens):
        if tok.kind in Keyword.Declaration and tok.text in DECLARATIONS:
            name = _next_token(tokens, i)
            if name is not None and name.text == word:
                end = _declaration_end(tokens, i)
                text = " ".join(source[tok.start : tokens[end].end].split())
                text = text.replace("( ", "(").replace(" )", ")")
                return f"```weightsmith\n{text}\n```"
    return None


def _declaration_end(tokens: list[_Tok], start: int) -> int:
    """The last token of the declaration at ``start``: its closing bracket, or its line end."""
    depth = 0
    for i in range(start, len(tokens)):
        text = tokens[i].text
        if tokens[start].text == "program" and text == "{":
            return i
        if text in ("(", "{", "["):
            depth += 1
        elif text in (")", "}", "]"):
            depth -= 1
            if depth <= 0:
                return i
        elif depth == 0 and i > start and _ends_line(tokens, i):
            return i
    return len(tokens) - 1


# --------------------------------------------------------------------------- token context


def _tokens(source: str) -> list[_Tok]:
    out, pos, line = [], 0, 0
    for kind, text in WeightsmithLexer().get_tokens(source):
        if out and kind in String and out[-1].kind in String and out[-1].end == pos:
            out[-1] = _Tok(out[-1].start, out[-1].line, String, out[-1].text + text)
        elif text.strip():
            out.append(_Tok(pos, line, kind, text))
        pos += len(text)
        line += text.count("\n")
    return out


def _previous(tokens: list[_Tok], index: int) -> _Tok | None:
    return tokens[index - 1] if index > 0 else None


def _next_token(tokens: list[_Tok], index: int) -> _Tok | None:
    return tokens[index + 1] if index + 1 < len(tokens) else None


def _next(tokens: list[_Tok], index: int, text: str) -> bool:
    following = _next_token(tokens, index)
    return following is not None and following.text == text


def _unmatched(tokens: list[_Tok], index: int, opening: str, closing: str) -> int | None:
    depth = 0
    for i in range(index - 1, -1, -1):
        text = tokens[i].text
        if text == closing:
            depth += 1
        elif text == opening:
            if depth == 0:
                return i
            depth -= 1
    return None


def _statement_head(tokens: list[_Tok], opening: int) -> list[_Tok]:
    """The tokens between the start of a statement and its argument list at ``opening``."""
    head: list[_Tok] = []
    depth = 0
    for i in range(opening - 1, -1, -1):
        tok = tokens[i]
        if tok.text in ("]", ")"):
            depth += 1
        elif tok.text in ("[", "("):
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and tok.text in ("{", "}", "=", ","):
            break
        head.insert(0, tok)
        if depth == 0 and ((tok.kind in Keyword and tok.text != "in") or tok.text in FUNCTION_DOCS):
            break
    return [t for t in head if t.kind not in Punctuation or t.text == "."]


def _first_string(tokens: list[_Tok], opening: int) -> str | None:
    for tok in tokens[opening + 1 :]:
        if tok.kind in String:
            return tok.text.strip('"')
        if tok.text in (",", ")"):
            return None
    return None


def _list_owner(tokens: list[_Tok], index: int) -> str | None:
    """For a word inside ``name = [ ... ]``, the name."""
    opening = _unmatched(tokens, index, "[", "]")
    if opening is None:
        return None
    return _assigned_to(tokens, opening)


def _assigned_to(tokens: list[_Tok], index: int) -> str | None:
    """For the value after ``name =``, the name."""
    equals = _previous(tokens, index)
    name = _previous(tokens, index - 1) if equals is not None and equals.text == "=" else None
    return name.text if name is not None and name.kind in Name.Attribute else None


def _block_of(tokens: list[_Tok], index: int) -> str | None:
    """The keyword of the innermost ``{ }`` block around token ``index``."""
    opening = _unmatched(tokens, index, "{", "}")
    keyword = _previous(tokens, opening) if opening is not None else None
    return keyword.text if keyword is not None else None


def _starts_statement(tokens: list[_Tok], index: int) -> bool:
    before = _previous(tokens, index)
    return before is None or before.text in ("{", "}", ")") or _ends_line(tokens, index - 1)


def _ends_line(tokens: list[_Tok], index: int) -> bool:
    following = _next_token(tokens, index)
    return following is None or following.line != tokens[index].line


__all__ = ["BUDGET_DOCS", "EFFECT_DOCS", "FUNCTION_DOCS", "KEYWORD_DOCS", "hover"]
