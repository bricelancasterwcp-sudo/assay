"""Strict record/replay transcripts (spec §10).

CallRecorder wraps a live backend, writing one JSONL row per call keyed
on (model, prompt, seed) — NUL-separated SHA-256, same as robigo, so no
shorter/longer field combination can collide onto one digest.
CallReplayer is STRICT: a miss raises TranscriptMiss, never falls
through to live or to a canned value. Same key recorded N times replays
in order, N times. Recorded infrastructure errors replay by re-raising
the same type — as does a recorded ToolsUnsupported, which is a
capability fact the evidence trail must not lose.

Rows carry ``kind`` ("generate" | "chat_tools") because the two call
shapes share one keyspace. A row written before the tool surface
existed has no ``kind`` field at all, and means "generate": the v1.5
transcripts under docs/superpowers/evidence/ are evidence, and evidence
does not get rewritten to suit a later schema.
"""

import hashlib
import json
from collections import deque
from pathlib import Path

from assay.backends.base import (
    Backend,
    BackendCaps,
    ModelInfo,
    Reply,
    ToolCall,
    ToolReply,
    ToolsUnsupported,
)
from assay.errors import AssayError, ContractViolation, InfrastructureError


def key_for(model: str, prompt: str, seed: int) -> str:
    """sha256 over model NUL seed NUL prompt — collision-proof framing."""
    material = f"{model}\x00{seed}\x00{prompt}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def tools_key_material(messages: list[dict], tools: list[dict]) -> str:
    """The prompt-slot string identifying one tool call.

    A tool call has no prompt: its identity is the WHOLE payload, both
    the conversation and the schemas offered, since the same messages
    with a different toolset are a different question. ``sort_keys``
    makes that identity content-addressed rather than an artefact of the
    order a caller happened to build its dicts in — without it, a record
    and a replay of the same call can hash differently. Callers that
    need the payload's size (budget charging) measure this same string,
    so the instrument charges for exactly what it keys on.
    """
    return json.dumps(
        {"messages": messages, "tools": tools},
        sort_keys=True,
        separators=(",", ":"),
    )


class TranscriptMiss(AssayError):
    """The transcript has no (remaining) row for this call."""


_ERROR_TYPES = {
    "InfrastructureError": InfrastructureError,
    "ContractViolation": ContractViolation,
    # By NAME: ToolsUnsupported is deliberately not an AssayError, so it
    # would never arrive through a subclass lookup.
    "ToolsUnsupported": ToolsUnsupported,
}

_RECORDED_ERRORS = (InfrastructureError, ToolsUnsupported)
"""The outcomes a transcript stores as an error row and re-raises.

Anything else escapes unrecorded: a bug in assay is not a measurement,
and writing it down would make the transcript claim the endpoint did
something it never did.
"""

KIND_GENERATE = "generate"
KIND_CHAT_TOOLS = "chat_tools"
"""The row vocabulary, one value per call shape on the Backend protocol."""


def _recorded_error(row: dict, key: str) -> Exception:
    """The exception a recorded error row replays as.

    An unknown name replays as InfrastructureError: a transcript written
    by a future assay may carry an error type this build has never heard
    of, and "the endpoint failed us" is the safe reading — never a
    silent success, and never a capability fact this build cannot check.
    """
    error_cls = _ERROR_TYPES.get(row["error_type"], InfrastructureError)
    return error_cls(f"replayed {row['error_type']} for key {key[:12]}")


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
                kind=KIND_GENERATE,
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
            kind=KIND_GENERATE,
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

    def chat_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        seed: int,
        max_tokens: int,
    ) -> ToolReply:
        key = key_for(self.model, tools_key_material(messages, tools), seed)
        try:
            reply = self._inner.chat_tools(
                messages, tools, seed=seed, max_tokens=max_tokens
            )
        except _RECORDED_ERRORS as error:
            self._write_row(
                kind=KIND_CHAT_TOOLS,
                key=key,
                seed=seed,
                outcome="error",
                text=None,
                tool_calls=None,
                tokens_in=None,
                tokens_out=None,
                stop_reason=None,
                error_type=type(error).__name__,
            )
            raise
        self._write_row(
            kind=KIND_CHAT_TOOLS,
            key=key,
            seed=seed,
            outcome="reply",
            text=reply.text,
            tool_calls=[
                {"name": call.name, "arguments": call.arguments}
                for call in reply.tool_calls
            ],
            tokens_in=reply.tokens_in,
            tokens_out=reply.tokens_out,
            stop_reason=reply.stop_reason,
            error_type=None,
        )
        return reply

    def model_info(self) -> ModelInfo:
        return self._inner.model_info()

    def _write_row(self, *, kind: str, **fields) -> None:
        row = {"model": self.model, "kind": kind, **fields}
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
        row = self._take(key, KIND_GENERATE, seed)
        if row["outcome"] == "error":
            raise _recorded_error(row, key)
        return Reply(
            text=row["text"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            stop_reason=row["stop_reason"],
            raw={"replayed": True, "key": row["key"]},
        )

    def chat_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        seed: int,
        max_tokens: int,
    ) -> ToolReply:
        key = key_for(self.model, tools_key_material(messages, tools), seed)
        row = self._take(key, KIND_CHAT_TOOLS, seed)
        if row["outcome"] == "error":
            raise _recorded_error(row, key)
        return ToolReply(
            text=row["text"],
            tool_calls=tuple(
                # `arguments` is replayed as recorded: None stays None
                # (unreadable on the wire), {} stays {} (called with no
                # arguments). Conflating them would invent a measurement.
                ToolCall(name=call["name"], arguments=call["arguments"])
                for call in row["tool_calls"]
            ),
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            stop_reason=row["stop_reason"],
            raw={"replayed": True, "key": row["key"]},
        )

    def model_info(self) -> ModelInfo:
        raise TranscriptMiss("model_info is not recorded in transcripts")

    def _take(self, key: str, kind: str, seed: int) -> dict:
        """Pop the next unplayed row for this key, or raise a strict miss.

        A row of the OTHER kind is a miss, not a match. The two call
        shapes share one keyspace — a generate prompt that happens to be
        the serialized tool payload keys identically — and a tool row
        must never answer a generate call. The declined row is left in
        the queue: a call it was not recorded for does not consume it.
        """
        queue = self._rows.get(key)
        if not queue:
            raise TranscriptMiss(
                f"transcript {self._path} has no remaining {kind} row for "
                f"model={self.model!r} seed={seed} key={key[:12]}"
            )
        found = queue[0].get("kind", KIND_GENERATE)
        if found != kind:
            raise TranscriptMiss(
                f"transcript {self._path} has a {found} row where this "
                f"{kind} call keys — model={self.model!r} seed={seed} "
                f"key={key[:12]}"
            )
        return queue.popleft()
