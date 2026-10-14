"""Entailment: does a cited passage support the segment that cites it.

The site's backend runs this check on every answer before it goes out. An
evaluation run has to score with the same model and the same rule, or the
figure is about some other check:

    model      MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, fp32 ONNX
    rule       a segment is dropped when P(entailment) < 0.5

`Nli` is the seam. Tests pass a stand-in. `OnnxNli` is the real one. It reads a
directory holding the same `model.onnx` and `tokenizer.json` the backend pins,
and it reports the model file's sha256 so the two can be told to be the same
file. It needs the `nli` extra (`uv sync --extra nli` in packages/bench), which
nothing else here installs.

Rough edge, said plainly: OnnxNli has never been run. No test loads the model
and CI does not download it. The label order is read from the directory's
config.json where there is one.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# Under this, the backend drops the segment.
THRESHOLD = 0.5
MODEL_DIR_ENV = "SPATIAL_NLI_MODEL_DIR"
MAX_TOKENS = 512
# The model card's order. config.json wins when it is there.
ENTAILMENT_INDEX = 0


class NliError(Exception):
    pass


class Nli(Protocol):
    @property
    def name(self) -> str:
        """What scored: the model, and which file of it where that is known."""

    def entailment(self, premise: str, hypothesis: str) -> float:
        """P(entailment) of ``hypothesis`` given ``premise``, from 0 to 1."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class OnnxNli:
    """The model itself, through the ONNX runtime. Loaded once, on first use."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        model = directory / "model.onnx"
        if not model.is_file() or not (directory / "tokenizer.json").is_file():
            raise NliError(f"{directory} needs model.onnx and tokenizer.json")
        self.sha256 = _sha256(model)
        self._session: Any = None
        self._tokenizer: Any = None
        self._entailment = ENTAILMENT_INDEX
        config = directory / "config.json"
        if config.is_file():
            labels = json.loads(config.read_text(encoding="utf-8")).get("id2label", {})
            found = [int(i) for i, label in labels.items() if str(label).lower() == "entailment"]
            if found:
                self._entailment = found[0]

    @property
    def name(self) -> str:
        return f"{MODEL}@sha256:{self.sha256[:16]}"

    def _load(self) -> None:
        try:
            runtime = importlib.import_module("onnxruntime")
            tokenizers = importlib.import_module("tokenizers")
        except ImportError as error:
            raise NliError(f"the NLI model needs the nli extra installed: {error}") from error
        self._session = runtime.InferenceSession(str(self.directory / "model.onnx"))
        self._tokenizer = tokenizers.Tokenizer.from_file(str(self.directory / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)

    def entailment(self, premise: str, hypothesis: str) -> float:
        if self._session is None:
            self._load()
        numpy = importlib.import_module("numpy")
        encoded = self._tokenizer.encode(premise, hypothesis)
        # int64 [1, n], the way the backend's translator builds them.
        have = {
            "input_ids": encoded.ids,
            "attention_mask": encoded.attention_mask,
            "token_type_ids": encoded.type_ids,
        }
        wanted = {wants.name for wants in self._session.get_inputs()}
        feeds = {
            name: numpy.asarray([values], dtype=numpy.int64)
            for name, values in have.items()
            if name in wanted
        }
        logits = [float(value) for value in self._session.run(None, feeds)[0][0]]
        peak = max(logits)
        exps = [math.exp(value - peak) for value in logits]
        return exps[self._entailment] / sum(exps)
