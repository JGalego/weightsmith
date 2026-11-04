"""Hugging Face model directories: ``config.json`` plus safetensors weights (one file or shards).

Reading a directory's configuration, tokenizer presence and architecture profile needs no tensor
library, so ``weightsmith check`` can validate roles and probe suites statically. Running such a
model needs the optional ``hf`` extra (``transformers``); see ``backends/pytorch/hf.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from weightsmith.errors import LoadError
from weightsmith.models.archspec import ArchSpec
from weightsmith.models.manifest import ARCH_FILE, Manifest
from weightsmith.models.profiles import Profile, ProfileError, build_profile

CONFIG_FILE = "config.json"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)
# Files copied next to saved weights so that the output directory loads like the input.
SIDE_FILES = (CONFIG_FILE, "generation_config.json", *TOKENIZER_FILES)


@dataclass(frozen=True, slots=True)
class HFArch:
    """A Hugging Face architecture: the configuration of a model directory, plus its profile."""

    directory: Path
    model_type: str
    vocab_size: int
    has_tokenizer: bool
    profile: Profile | None
    profile_error: str | None = None
    config: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


Architecture = ArchSpec | HFArch


def model_format(path: Path) -> str | None:
    """``"weightsmith"`` for an ``arch.json`` directory, ``"hf"`` for ``config.json``, else None."""
    if not path.is_dir():
        return None
    if (path / ARCH_FILE).is_file():
        return "weightsmith"
    if (path / CONFIG_FILE).is_file():
        return "hf"
    return None


def read_config(directory: Path) -> dict[str, Any]:
    file = directory / CONFIG_FILE
    try:
        config = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoadError(f"cannot read {file}: {exc}") from exc
    if not isinstance(config, dict):
        raise LoadError(f"{file} is not a JSON object")
    return config


def read_hf_arch(directory: Path, manifest: Manifest) -> HFArch:
    """Describe a Hugging Face model directory. A missing profile is recorded, not an error."""
    config = read_config(directory)
    vocab = config.get("vocab_size")
    if not isinstance(vocab, int) or vocab <= 0:
        raise LoadError(f"{directory / CONFIG_FILE} has no positive integer 'vocab_size'")
    profile: Profile | None = None
    error: str | None = None
    try:
        profile = build_profile(config, manifest)
    except ProfileError as exc:
        error = str(exc)
    return HFArch(
        directory=directory,
        model_type=str(config.get("model_type", "")),
        vocab_size=vocab,
        has_tokenizer=any((directory / f).is_file() for f in ("tokenizer.json", "tokenizer.model")),
        profile=profile,
        profile_error=error,
        config=config,
    )


def read_profile(path: Path, manifest: Manifest) -> tuple[Profile | None, str | None]:
    """The profile of the weights at ``path``, if it is a directory with a ``config.json``."""
    if not (path.is_dir() and (path / CONFIG_FILE).is_file()):
        return None, "no config.json next to the weights, so no architecture profile"
    arch = read_hf_arch(path, manifest)
    return arch.profile, arch.profile_error
