"""Strict record/replay transcripts (spec §10).

CallRecorder wraps a live backend, writing one JSONL row per call keyed
on (model, prompt, seed) — NUL-separated SHA-256, same as robigo, so no
shorter/longer field combination can collide onto one digest.
CallReplayer is STRICT: a miss raises TranscriptMiss, never falls
through to live or to a canned value. Same key recorded N times replays
in order, N times. Recorded infrastructure errors replay by re-raising
the same type.
"""

import hashlib
import json
from collections import deque
from pathlib import Path

from assay.backends.base import Backend, BackendCaps, ModelInfo, Reply
from assay.errors import AssayError, ContractViolation, InfrastructureError


def key_for(model: str, prompt: str, seed: int) -> str:
    """sha256 over model NUL seed NUL prompt — collision-proof framing."""
    material = f"{model}\x00{seed}\x00{prompt}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class TranscriptMiss(AssayError):
    """The transcript has no (remaining) row for this call."""


_ERROR_TYPES = {
    "InfrastructureError": InfrastructureError,
    "ContractViolation": ContractViolation,
}


class CallRecorder:
    """Backend wrapper that records every call to a JSONL transcript."""

    def __init__(self, inner: Backend, path: Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self.caps = inner.caps
        self.model = inner.model
        self._path.write_text("", encoding="utf-8")  # fresh transcript

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        max_tokens: int,
        num_ctx: int | None = None,
    ) -> Reply:
        key = key_for(self.model, prompt, seed)
        try:
            reply = self._inner.generate(
                prompt, seed=seed, max_tokens=max_tokens, num_ctx=num_ctx
            )
        except InfrastructureError as error:
            self._write_row(
                key=key,
                seed=seed,
                outcome="error",
                text=None,
                tokens_in=None,
                tokens_out=None,
                stop_reason=None,
                error_type=type(error).__name__,
            )
            raise
        self._write_row(
            key=key,
            seed=seed,
            outcome="reply",
            text=reply.text,
            tokens_in=reply.tokens_in,
            tokens_out=reply.tokens_out,
            stop_reason=reply.stop_reason,
            error_type=None,
        )
        return reply

    def model_info(self) -> ModelInfo:
        return self._inner.model_info()

    def _write_row(self, **fields) -> None:
        row = {"model": self.model, **fields}
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


class CallReplayer:
    """Backend that replays a recorded transcript, strictly."""

    def __init__(self, path: Path, *, model: str, caps: BackendCaps) -> None:
        self._path = Path(path)
        self.model = model
        self.caps = caps
        self._rows: dict[str, deque] = {}
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._rows.setdefault(row["key"], deque()).append(row)

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        max_tokens: int,
        num_ctx: int | None = None,
    ) -> Reply:
        key = key_for(self.model, prompt, seed)
        queue = self._rows.get(key)
        if not queue:
            raise TranscriptMiss(
                f"transcript {self._path} has no remaining row for "
                f"model={self.model!r} seed={seed} key={key[:12]}"
            )
        row = queue.popleft()
        if row["outcome"] == "error":
            error_cls = _ERROR_TYPES.get(row["error_type"], InfrastructureError)
            raise error_cls(f"replayed {row['error_type']} for key {key[:12]}")
        return Reply(
            text=row["text"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            stop_reason=row["stop_reason"],
            raw={"replayed": True, "key": row["key"]},
        )

    def model_info(self) -> ModelInfo:
        raise TranscriptMiss("model_info is not recorded in transcripts")
