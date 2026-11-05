"""Fenced blocks in an editor's reply, for delegations with ``protocol = "blocks"``.

Under this protocol the editor is given no function tools. It acts by writing a fenced block whose
opening line names the tool, for example::

    ```edits
    actor0.W[6, 122] = 80
    ```

Only the last block of a granted kind in a reply is read. A reply with no such block is a plain
answer.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

_FENCE = re.compile(
    r"^```([A-Za-z_][A-Za-z0-9_]*)[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL
)


@dataclass(frozen=True, slots=True)
class Block:
    kind: str
    body: str


def last_block(text: str, kinds: Collection[str]) -> Block | None:
    found = [Block(m.group(1), m.group(2)) for m in _FENCE.finditer(text) if m.group(1) in kinds]
    return found[-1] if found else None
