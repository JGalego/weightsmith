# Weightsmith for VS Code

Syntax highlighting, comment toggling, bracket matching and hover documentation for Weightsmith (`.wsm`) programs.

Build and install it from the repository root with `just vscode`.

## Hover

Hovering a word shows what it does: keywords, built-ins (finders, operations, strategies, metrics, contract items, tools, effects, outputs, providers), the arguments of a call such as `delegate E (message_file = ...)` or `allow reason(max_calls = ...)`, and the declaration of a name the file defines. The text comes from the same registries the compiler checks programs against.

Hover needs the language server, `weightsmith lsp`, from the `lsp` extra (`pip install 'weightsmith[lsp]'`; `uv sync` in this repository includes it). The extension runs, in order of preference:

1. the command in the `weightsmith.server.command` setting, for example `["uv", "run", "weightsmith"]` (`lsp` is appended);
2. `.venv/bin/weightsmith` in the workspace;
3. `weightsmith` on the PATH.

Highlighting works without the server.

## Keeping it in sync

The keyword and built-in lists in `syntaxes/weightsmith.tmLanguage.json` mirror the Pygments lexer in `src/weightsmith/language/highlight.py`; `tests/test_highlight.py` fails when they drift apart. `tests/test_hover.py` fails when a keyword, effect or budget has no hover text.
