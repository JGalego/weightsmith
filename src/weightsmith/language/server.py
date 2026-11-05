"""The Weightsmith language server (``weightsmith lsp``), for editors that speak LSP.

It answers hover requests with the documentation of the word under the cursor (see ``hover``).
It needs the ``lsp`` extra (pygls).
"""

from __future__ import annotations

from typing import Any

from weightsmith import __version__
from weightsmith.language.hover import hover


def build() -> Any:
    from lsprotocol import types
    from pygls.lsp.server import LanguageServer

    server = LanguageServer("weightsmith", __version__)

    @server.feature(types.TEXT_DOCUMENT_HOVER)
    def on_hover(ls: LanguageServer, params: types.HoverParams) -> types.Hover | None:
        document = ls.workspace.get_text_document(params.text_document.uri)
        text = hover(document.source, document.offset_at_position(params.position))
        if text is None:
            return None
        return types.Hover(types.MarkupContent(types.MarkupKind.Markdown, text))

    return server


def serve() -> None:
    build().start_io()
