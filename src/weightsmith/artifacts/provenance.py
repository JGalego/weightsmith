"""Provenance records attached to reports and patches."""

from __future__ import annotations

import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

from weightsmith import __version__


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def collect(**extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "weightsmith": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _version("torch"),
        "safetensors": _version("safetensors"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    record.update(extra)
    return record
