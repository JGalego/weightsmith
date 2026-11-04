"""Render Weightsmith source as a syntax-highlighted SVG or HTML snippet.

    uv run python scripts/snippet.py examples/merge_mlps.wsm -o merge.svg
    uv run python scripts/snippet.py examples/repair_mlp.wsm --title -o repair.svg
    sed -n '21,46p' README.md | uv run python scripts/snippet.py - --theme dark

SVG output is a self-contained image (it renders in a GitHub README); HTML output is a <pre> with
its own stylesheet. With --theme auto (the default) the colours follow the viewer's light or dark
preference.
"""

from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path

from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    _TokenType,
)

from weightsmith.language.highlight import WeightsmithLexer

# CSS class, then (light, dark) colours, then extra CSS. Looked up from the most specific token
# type up, so a token without its own entry takes its parent's.
PALETTE: dict[_TokenType, tuple[str, tuple[str, str], str]] = {
    Comment: ("c", ("#64748B", "#94A3B8"), "font-style:italic"),
    Keyword: ("k", ("#B45309", "#FBBF24"), "font-weight:600"),
    Keyword.Declaration: ("kd", ("#B45309", "#FBBF24"), "font-weight:600"),
    Keyword.Pseudo: ("kp", ("#C2410C", "#FB923C"), ""),
    Keyword.Constant: ("kc", ("#7C3AED", "#C4B5FD"), ""),
    Name.Class: ("nc", ("#1D4ED8", "#93C5FD"), "font-weight:600"),
    Name.Variable: ("nv", ("#1D4ED8", "#93C5FD"), ""),
    Name.Builtin: ("nb", ("#0F766E", "#5EEAD4"), ""),
    Name.Attribute: ("na", ("#475569", "#CBD5E1"), ""),
    Name.Property: ("np", ("#0369A1", "#7DD3FC"), ""),
    String: ("s", ("#15803D", "#86EFAC"), ""),
    Number: ("m", ("#7C3AED", "#C4B5FD"), ""),
    Operator: ("o", ("#64748B", "#94A3B8"), ""),
    Punctuation: ("p", ("#64748B", "#94A3B8"), ""),
}
BACKGROUND = ("#F8FAFC", "#0F172A")
BORDER = ("#E2E8F0", "#1E293B")
FOREGROUND = ("#1E293B", "#E2E8F0")
TITLE = ("#64748B", "#94A3B8")

FONT = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"
FONT_SIZE = 14
CHAR_WIDTH = 8.45  # advance of one character at FONT_SIZE, generous enough for common fonts
LINE_HEIGHT = 21
PAD = 20

FILE_NAME = object()  # --title given without a value


def css_class(token: _TokenType) -> str | None:
    while token not in PALETTE and token.parent is not None:
        token = token.parent
    return PALETTE[token][0] if token in PALETTE else None


def lines_of(source: str) -> list[list[tuple[str | None, str]]]:
    """Split the token stream into lines of (css class, text) runs."""
    lines: list[list[tuple[str | None, str]]] = [[]]
    for token, value in WeightsmithLexer().get_tokens(source):
        for i, part in enumerate(value.split("\n")):
            if i:
                lines.append([])
            if part:
                lines[-1].append((css_class(token), part))
    while lines and not lines[-1]:
        lines.pop()
    return lines


def stylesheet(theme: str, text_prop: str, selector: str) -> str:
    def rules(dark: int) -> str:
        out = [f"{selector}{{{text_prop}:{FOREGROUND[dark]}}}"]
        out += [
            f"{selector} .{cls}{{{text_prop}:{colours[dark]}{';' + extra if extra else ''}}}"
            for cls, colours, extra in PALETTE.values()
        ]
        out += [
            f"{selector} .bg{{fill:{BACKGROUND[dark]};background:{BACKGROUND[dark]};"
            f"stroke:{BORDER[dark]};border-color:{BORDER[dark]}}}",
            f"{selector} .title{{{text_prop}:{TITLE[dark]}}}",
        ]
        return "".join(out)

    if theme == "light":
        return rules(0)
    if theme == "dark":
        return rules(1)
    return rules(0) + f"@media (prefers-color-scheme:dark){{{rules(1)}}}"


def to_svg(source: str, title: str | None, theme: str) -> str:
    lines = lines_of(source)
    columns = max((sum(len(text) for _, text in line) for line in lines), default=0)
    top = PAD + (LINE_HEIGHT + 6 if title else 0)
    width = round(2 * PAD + columns * CHAR_WIDTH)
    height = round(top + len(lines) * LINE_HEIGHT + PAD - (LINE_HEIGHT - FONT_SIZE))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title or "Weightsmith code")}">',
        f"<style>{stylesheet(theme, 'fill', 'svg')}"
        f"text{{font-family:{FONT};font-size:{FONT_SIZE}px;white-space:pre}}</style>",
        f'<rect class="bg" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8"/>',
    ]
    if title:
        y = PAD + FONT_SIZE - 2
        out.append(f'<text class="title" x="{PAD}" y="{y}">{escape(title)}</text>')
    for i, line in enumerate(lines):
        y = top + FONT_SIZE + i * LINE_HEIGHT
        spans = "".join(
            f'<tspan class="{cls}">{escape(text)}</tspan>' if cls else escape(text)
            for cls, text in line
        )
        out.append(f'<text x="{PAD}" y="{y}" xml:space="preserve">{spans}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def to_html(source: str, title: str | None, theme: str) -> str:
    body = "\n".join(
        "".join(
            f'<span class="{cls}">{escape(text)}</span>' if cls else escape(text)
            for cls, text in line
        )
        for line in lines_of(source)
    )
    caption = f'<div class="title">{escape(title)}</div>' if title else ""
    style = (
        stylesheet(theme, "color", ".wsm")
        + f".wsm .bg{{font-family:{FONT};font-size:{FONT_SIZE}px;line-height:{LINE_HEIGHT}px;"
        f"padding:{PAD}px;border:1px solid;border-radius:8px;overflow-x:auto}}"
        ".wsm pre{margin:0;font:inherit}.wsm .title{margin-bottom:6px}"
    )
    return (
        f"<style>{style}</style>\n"
        f'<div class="wsm"><div class="bg">{caption}<pre>{body}</pre></div></div>\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("source", help="a .wsm file, or - to read standard input")
    parser.add_argument("-o", "--output", type=Path, help="output file (default: standard output)")
    parser.add_argument(
        "-f", "--format", choices=["svg", "html"], help="default: from --output, else svg"
    )
    parser.add_argument(
        "-t",
        "--title",
        nargs="?",
        const=FILE_NAME,
        help="caption above the code; on its own, the file name (default: no caption)",
    )
    parser.add_argument("--theme", choices=["auto", "light", "dark"], default="auto")
    args = parser.parse_args()

    if args.source == "-":
        source, file_name = sys.stdin.read(), None
    else:
        path = Path(args.source)
        source, file_name = path.read_text(encoding="utf-8"), path.name
    title = file_name if args.title is FILE_NAME else args.title
    fmt = args.format or (
        "html" if args.output and args.output.suffix in {".html", ".htm"} else "svg"
    )
    rendered = (to_svg if fmt == "svg" else to_html)(source, title, args.theme)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
