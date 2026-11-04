"""Run Hugging Face causal language models over a Weightsmith tensor map.

Needs the optional ``hf`` extra (``transformers``). The module is built once per model directory
from its ``config.json``; every evaluation then calls it functionally with the tensors of the
working copy (``torch.func.functional_call``), so the module is never mutated and tied weights
stay tied.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
from torch.func import functional_call

from weightsmith.errors import IncompatibleError, UnsupportedOperation
from weightsmith.evaluation.probes import ProbeSuite
from weightsmith.models.hf import HFArch

Tensors = dict[str, torch.Tensor]
BATCH = 32


def available() -> bool:
    return importlib.util.find_spec("transformers") is not None


def _require() -> None:
    if not available():
        raise UnsupportedOperation(
            "running a Hugging Face model needs the 'hf' extra: pip install 'weightsmith[hf]'"
        )


class HFRunner:
    def __init__(self, arch: HFArch, tensors: Tensors) -> None:
        _require()
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(str(arch.directory), local_files_only=True)
        floats = [t.dtype for t in tensors.values() if t.is_floating_point()]
        dtype = max(set(floats), key=floats.count) if floats else torch.float32
        self.model: Any = AutoModelForCausalLM.from_config(config, dtype=dtype)
        self.model.eval()
        self.arch = arch
        self.keymap, self.unused = self._map_names(tensors)
        self._tokenizer: Any = None

    def _map_names(self, tensors: Tensors) -> tuple[dict[str, str], list[str]]:
        """Map file tensor names to module state names (some checkpoints omit the prefix)."""
        state = self.model.state_dict()
        prefix = getattr(self.model, "base_model_prefix", "")
        keymap: dict[str, str] = {}
        unused: list[str] = []
        for name, tensor in tensors.items():
            key = name if name in state else f"{prefix}.{name}" if prefix else name
            if key not in state:
                unused.append(name)
                continue
            if tuple(state[key].shape) != tuple(tensor.shape):
                raise IncompatibleError(
                    f"'{name}' has shape {list(tensor.shape)}, but the model built from "
                    f"config.json expects {list(state[key].shape)}"
                )
            keymap[name] = key
        params = dict(self.model.named_parameters())  # tied parameters appear once
        covered = set(keymap.values())
        missing = [k for k in params if k not in covered]
        if missing:
            raise IncompatibleError(
                f"{self.arch.directory}: the weights lack {len(missing)} parameter(s) the model "
                f"needs, for example '{missing[0]}'"
            )
        return keymap, unused

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            if not self.arch.has_tokenizer:
                raise IncompatibleError(f"{self.arch.directory} has no tokenizer files")
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.arch.directory), local_files_only=True
            )
        return self._tokenizer

    def _single_token(self, text: str, what: str) -> int:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise IncompatibleError(
                f"{what} {text!r} is {len(ids)} tokens for this tokenizer; it must be exactly one"
            )
        return int(ids[0])

    def encode(self, suite: ProbeSuite) -> tuple[list[list[int]], list[int], list[int] | None]:
        prompts: list[list[int]] = []
        targets: list[int] = []
        for example in suite.token_examples:
            if example.prompt_ids is not None:
                prompts.append(list(example.prompt_ids))
            else:
                assert example.prompt is not None
                prompts.append(list(self.tokenizer(example.prompt)["input_ids"]))
            if example.target_id is not None:
                targets.append(example.target_id)
            else:
                assert example.target is not None
                targets.append(self._single_token(example.target, "target"))
        candidates = suite.candidate_ids
        if suite.candidates is not None:
            candidates = [self._single_token(c, "candidate") for c in suite.candidates]
        return prompts, targets, candidates

    @torch.no_grad()
    def next_token_logits(self, tensors: Tensors, prompts: list[list[int]]) -> torch.Tensor:
        """Logits for the token after each prompt, shape (prompts, vocabulary)."""
        state = {self.keymap[n]: t for n, t in tensors.items() if n in self.keymap}
        rows = []
        for start in range(0, len(prompts), BATCH):
            chunk = prompts[start : start + BATCH]
            width = max(map(len, chunk))
            ids = torch.zeros(len(chunk), width, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, prompt in enumerate(chunk):
                ids[i, : len(prompt)] = torch.tensor(prompt)
                mask[i, : len(prompt)] = 1
            out = functional_call(
                self.model, state, args=(), kwargs={"input_ids": ids, "attention_mask": mask}
            )
            last = torch.tensor([len(p) - 1 for p in chunk])
            rows.append(out.logits[torch.arange(len(chunk)), last].float())
        return torch.cat(rows)

    def accuracy(self, tensors: Tensors, suite: ProbeSuite) -> float:
        prompts, targets, candidates = self.encode(suite)
        logits = self.next_token_logits(tensors, prompts)
        target = torch.tensor(targets)
        if candidates is not None:
            choices = torch.tensor(candidates)
            if not bool(torch.isin(target, choices).all()):
                raise IncompatibleError(
                    f"probe suite '{suite.name}' has a target outside its candidates"
                )
            predicted = choices[logits[:, choices].argmax(dim=1)]
        else:
            predicted = logits.argmax(dim=1)
        return float((predicted == target).double().mean())
