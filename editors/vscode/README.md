# Weightsmith for VS Code

Syntax highlighting, comment toggling and bracket matching for Weightsmith (`.wsm`) programs.

Build and install it from the repository root with `just vscode`.

The keyword and built-in lists in `syntaxes/weightsmith.tmLanguage.json` mirror the Pygments lexer
in `src/weightsmith/language/highlight.py`; `tests/test_highlight.py` fails when they drift apart.
