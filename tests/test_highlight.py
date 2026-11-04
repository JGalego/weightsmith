"""The highlighters must cover the language: the Pygments lexer against the grammar and the
registries, the VS Code grammar against the lexer, and the snippet script's output."""

from __future__ import annotations

import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest
from pygments.token import Error, Keyword, Name, Number, String, _TokenType

from weightsmith import registry
from weightsmith.compiler.effects import Effect
from weightsmith.language import highlight
from weightsmith.language.highlight import WeightsmithLexer

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "examples").glob("*.wsm"))
TM_GRAMMAR = json.loads(
    (ROOT / "editors" / "vscode" / "syntaxes" / "weightsmith.tmLanguage.json").read_text()
)


def tokens(source: str) -> list[tuple[_TokenType, str]]:
    return [(t, v) for t, v in WeightsmithLexer().get_tokens(source) if v.strip()]


def grammar_words() -> set[str]:
    grammar = (ROOT / "src" / "weightsmith" / "language" / "grammar.lark").read_text()
    grammar = re.sub(r"//[^\n]*", "", grammar)
    words = set(re.findall(r'"([a-z]+)"', grammar))
    words |= set(re.search(r"BOOL\.2: /\((\w+)\|(\w+)\)", grammar).groups())  # type: ignore[union-attr]
    return words


def test_every_grammar_word_is_highlighted() -> None:
    assert grammar_words() <= set(highlight.KEYWORDS) | set(highlight.FUNCTIONS)


def test_every_registry_name_is_highlighted() -> None:
    names = {
        *registry.FINDERS,
        *registry.EDIT_OPS,
        *registry.COMPOSE_OPS,
        *registry.STRATEGIES,
        *registry.METRICS,
        *registry.REQUIREMENTS,
        *registry.OUTPUTS,
        *(e.value for e in Effect),
    }
    assert names <= set(highlight.BUILTINS) | set(highlight.KEYWORDS)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_examples_lex_without_errors(path: Path) -> None:
    source = path.read_text()
    all_tokens = list(WeightsmithLexer().get_tokens(source))
    assert not [v for t, v in all_tokens if t in Error]
    assert "".join(v for _, v in all_tokens) == source


def test_token_kinds() -> None:
    source = (
        'model Base = load("m")\n'
        "program P {\n"
        "    target Base\n"
        "    policy { allow mutate(max_changed_params = 10%) }\n"
        '    edit { scale Base.params["a.\\"b"][0..2, :] (factor = 1.5e-3) }\n'
        "    contract { require finite }\n"
        "}\n"
    )
    kinds: dict[str, _TokenType] = {}
    for t, v in tokens(source):
        kinds.setdefault(v, t)
    assert kinds["model"] in Keyword.Declaration
    assert kinds["Base"] in Name.Variable
    assert kinds["load"] in Name.Builtin
    assert kinds["P"] in Name.Class
    assert kinds["allow"] in Keyword.Pseudo
    assert kinds["max_changed_params"] in Name.Attribute
    assert kinds["10%"] in Number
    assert kinds["params"] in Name.Property
    assert kinds['\\"'] in String.Escape
    assert kinds["1.5e-3"] in Number.Float
    assert kinds["finite"] in Name.Builtin


def tm_words(rule: dict) -> set[str]:
    match = re.fullmatch(r"\\b\(\?:(.*)\)\\b", rule["match"])
    assert match, rule
    return set(match.group(1).split("|"))


def test_vscode_grammar_mirrors_the_lexer() -> None:
    repo = TM_GRAMMAR["repository"]
    by_scope = {r["name"]: tm_words(r) for r in repo["keyword"]["patterns"]}
    assert by_scope["storage.type.weightsmith"] == set(highlight.DECLARATIONS)
    assert by_scope["keyword.control.weightsmith"] == set(highlight.STATEMENTS)
    assert by_scope["keyword.other.rule.weightsmith"] == set(highlight.RULES)
    assert by_scope["constant.language.boolean.weightsmith"] == set(highlight.CONSTANTS)
    assert tm_words(repo["builtin"]) == set(highlight.BUILTINS)


def load_snippet_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("snippet", ROOT / "scripts" / "snippet.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("theme", ["auto", "light", "dark"])
def test_snippet_svg_is_well_formed(theme: str) -> None:
    snippet = load_snippet_script()
    source = (ROOT / "examples" / "repair_mlp.wsm").read_text()
    svg = ET.fromstring(snippet.to_svg(source, "repair_mlp.wsm & <friends>", theme))
    lines = [t for t in svg.iter("{http://www.w3.org/2000/svg}text") if t.get("class") != "title"]
    assert len(lines) == len(source.rstrip("\n").split("\n"))
    assert "".join(lines[0].itertext()) == source.split("\n")[0]


def test_snippet_html_keeps_the_text() -> None:
    snippet = load_snippet_script()
    source = (ROOT / "examples" / "merge_mlps.wsm").read_text()
    html = snippet.to_html(source, None, "auto")
    body = re.sub(r"<[^>]+>", "", html.split("<pre>")[1].split("</pre>")[0])
    assert body.replace("&quot;", '"').replace("&amp;", "&") == source.rstrip("\n")
