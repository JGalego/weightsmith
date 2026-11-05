"""Pygments lexer for Weightsmith source files (``.wsm``).

The word lists here are the source of truth for every highlighter: the VS Code grammar in
``editors/vscode`` mirrors them, and tests check both against the Lark grammar and the registries.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Whitespace,
)

from weightsmith import registry
from weightsmith.compiler.effects import Effect

# Keywords that introduce a name.
DECLARATIONS = ("model", "weights", "behavior", "selection", "editor", "program", "let")
# Keywords that start a block or a statement.
STATEMENTS = (
    "policy",
    "target",
    "inspect",
    "find",
    "in",
    "edit",
    "compose",
    "repair",
    "delegate",
    "contract",
    "verify",
    "output",
)
# Policy rules and contract clauses.
RULES = ("allow", "forbid", "preserve", "improve", "require")
CONSTANTS = ("true", "false")
KEYWORDS = DECLARATIONS + STATEMENTS + RULES + CONSTANTS

# Built-in functions of the declaration syntax, then every name a registry or the effect system
# defines. Keywords win where a name is both (`inspect`, `model`).
FUNCTIONS = ("load", "probes", "provider")
BUILTINS = tuple(
    sorted(
        (
            set(FUNCTIONS)
            | set(registry.FINDERS)
            | set(registry.EDIT_OPS)
            | set(registry.COMPOSE_OPS)
            | set(registry.STRATEGIES)
            | set(registry.METRICS)
            | set(registry.REQUIREMENTS)
            | set(registry.OUTPUTS)
            | set(registry.PROVIDERS)
            | set(registry.TOOLS)
            | {e.value for e in Effect}
        )
        - set(KEYWORDS)
    )
)

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"


class WeightsmithLexer(RegexLexer):
    name = "Weightsmith"
    aliases: ClassVar[Sequence[str]] = ("weightsmith", "wsm")
    filenames: ClassVar[Sequence[str]] = ("*.wsm",)
    mimetypes: ClassVar[Sequence[str]] = ("text/x-weightsmith",)
    url = "https://github.com/JGalego/weightsmith"

    tokens: ClassVar[dict] = {
        "root": [
            (r"\s+", Whitespace),
            (r"//.*?$", Comment.Single),
            (r'"', String.Double, "string"),
            (rf"(program)(\s+)({_NAME})", bygroups(Keyword.Declaration, Whitespace, Name.Class)),
            (
                rf"({'|'.join(d for d in DECLARATIONS if d != 'program')})(\s+)({_NAME})",
                bygroups(Keyword.Declaration, Whitespace, Name.Variable),
            ),
            (rf"(target)(\s+)({_NAME})", bygroups(Keyword, Whitespace, Name.Variable)),
            (words(DECLARATIONS, suffix=r"\b"), Keyword.Declaration),
            (words(STATEMENTS, suffix=r"\b"), Keyword),
            (words(RULES, suffix=r"\b"), Keyword.Pseudo),
            (words(CONSTANTS, suffix=r"\b"), Keyword.Constant),
            (r"\d+(\.\d+)?%", Number),
            (r"\d+\.\d+([eE][+-]?\d+)?|\d+[eE][+-]?\d+", Number.Float),
            (r"\d+", Number.Integer),
            (r"\.\.", Operator),
            (rf"(\.)({_NAME})", bygroups(Punctuation, Name.Property)),
            (rf"({_NAME})(\s*)(=)", bygroups(Name.Attribute, Whitespace, Operator)),
            (words(BUILTINS, suffix=r"\b"), Name.Builtin),
            (_NAME, Name),
            (r"=", Operator),
            (r"[{}()\[\],:]", Punctuation),
        ],
        "string": [
            (r"\\.", String.Escape),
            (r'"', String.Double, "#pop"),
            (r'[^"\\\n]+', String.Double),
        ],
    }
