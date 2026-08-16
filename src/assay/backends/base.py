"""Backend protocol and response contract (spec §3).

One protocol, two implementations (Ollama-native, OpenAI-compat).
`validate_reply` is the 11.5k-bug detector: a 200 that promised token
counts but delivered none becomes ContractViolation, never a model
result. No field defaults to a value that looks like a measurement —
unreported is None, always.

The tool surface (v1.6) adds the second classification the instrument
depends on: an endpoint that refuses the `tools` parameter is reporting
a CAPABILITY, not failing — `raise_for_tools_status` separates the two
so the probe can record `tools_supported=False` as data.
"""

import json
from dataclasses import dataclass
from typing import Callable, Protocol

from assay.errors import ContractViolation, InfrastructureError

_CONTRACT_FIELDS = ("tokens_in", "tokens_out", "stop_reason")

PROBE_TEMPERATURE = 0.2
"""Every probe call pins the sampler; an unpinned temperature leaves the
endpoint's default (Ollama: 0.8) as an uncontrolled variable of the
instrument. Measured live (2026-08-12, evidence/live run 1): at the
daemon default, qwen2.5-coder:7b emitted semantically correct
search_replace fixes with the indentation stripped in 15/15 probes —
0% landing — where robigo's stage 2, same model, same daemon, pinned at
0.2, measured 100%. Landing is a property of model x codec x directive
x SAMPLER; this constant pins the sampler and travels in
provenance. 0.2, not 0.0, for lineage: every robigo number this
project's profiles are compared against was measured at 0.2."""


@dataclass(frozen=True)
class Reply:
    text: str
    tokens_in: int | None  # None = backend did not report; NEVER estimated
    tokens_out: int | None
    stop_reason: str | None  # "stop" | "length" | None (unreported)
    raw: dict  # verbatim response body, for evidence trails


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation a model emitted, as delivered.

    ``arguments`` is None when arguments were present but could not be
    read as an object (the OpenAI wire carries them as a JSON string, and
    a malformed one is DATA the probe scores as an invalid call). None is
    "unreadable"; ``{}`` is "called with no arguments" — never conflated.
    """

    name: str
    arguments: dict | None


@dataclass(frozen=True)
class ToolReply:
    text: str
    tool_calls: tuple[ToolCall, ...]
    tokens_in: int | None  # None = backend did not report; NEVER estimated
    tokens_out: int | None
    stop_reason: str | None
    raw: dict  # verbatim response body, for evidence trails


class ToolsUnsupported(Exception):
    """The endpoint refused the tools parameter.

    A CAPABILITY FACT the probe records as ``tools_supported=False``,
    never an infrastructure failure — deliberately NOT an AssayError so
    no ``except InfrastructureError`` handler can swallow it. Carries
    ``.raw``, the verbatim error body, for the evidence trail.
    """

    raw: dict | None = None


def _error_text(body) -> str:
    """The error string a server's body carries, however it nests it.

    Ollama returns ``{"error": "<model> does not support tools"}``;
    OpenAI-compat servers nest an object (``{"error": {"message": ...,
    "param": "tools"}}``) where only ``param`` may name the culprit. The
    whole error value is serialized so the signal is not lost in either
    shape.
    """
    payload = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, default=str)
    except (TypeError, ValueError):
        # Belt and braces: a body off the wire is always serializable and
        # default=str absorbs the rest, but a classifier must not be the
        # thing that crashes while explaining a failure.
        return str(payload)


ArgumentsReader = Callable[[object], dict | None]
"""How one wire's tool arguments become a dict (or None if unreadable)."""


def _dict_arguments(raw) -> dict | None:
    """The Ollama rule: arguments arrive parsed, anything else is unreadable."""
    return raw if isinstance(raw, dict) else None


def parse_tool_calls(
    message: dict, read_arguments: ArgumentsReader = _dict_arguments
) -> tuple[ToolCall, ...]:
    """Read ``message.tool_calls[*].function`` on either wire.

    The traversal is identical for both backends; only how ``arguments``
    is read differs (Ollama: already a dict; OpenAI: a JSON string), so
    that is the one injected part. A junk entry is kept as a nameless
    call rather than dropped: a malformed call the model emitted is DATA
    the probe scores, and a silently vanished entry would instead read as
    "no call was made".
    """
    entries = message.get("tool_calls")
    if not isinstance(entries, list):
        return ()
    calls = []
    for entry in entries:
        function = entry.get("function") if isinstance(entry, dict) else None
        function = function if isinstance(function, dict) else {}
        name = function.get("name")
        calls.append(
            ToolCall(
                name=name if isinstance(name, str) else "",
                arguments=read_arguments(function.get("arguments")),
            )
        )
    return tuple(calls)


def raise_for_tools_status(status: int, body, source: str) -> None:
    """Classify a tool-call response status; return None if it is fine.

    A 4xx whose error text mentions "tool" (case-insensitive) is the
    endpoint declining to speak the tool protocol → ``ToolsUnsupported``.
    Every other non-2xx is the endpoint failing us → InfrastructureError.
    The rule keys on behavior CLASS, not exact wording: error text is not
    a stable target (Ollama says "does not support tools", others vary),
    while "4xx and the complaint is about tools" is.
    """
    if 200 <= status < 300:
        return None
    if 400 <= status < 500 and "tool" in _error_text(body).lower():
        unsupported = ToolsUnsupported(
            f"HTTP {status} from {source}: endpoint refused the tools parameter"
        )
        unsupported.raw = body
        raise unsupported
    error = InfrastructureError(f"HTTP {status} from {source}")
    error.raw = body
    raise error


@dataclass(frozen=True)
class BackendCaps:
    reports_counts: bool | None  # None = unknown until calibration (openai_compat)
    per_request_ctx: bool  # can set context per request (options.num_ctx)
    truncate_control: bool  # can request hard-fail instead of silent truncation
    metadata_access: bool  # can read model architecture metadata (spec §4)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    quant: str | None
    weights_bytes: int | None
    training_ctx: int | None
    block_count: int | None
    kv_head_count: int | None
    head_dim: int | None
    loaded: bool | None  # True if the daemon reports the model resident
    source: str  # "api_show" | "gguf_blob" | "openai_models"


class Backend(Protocol):
    caps: BackendCaps
    model: str

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        max_tokens: int,
        num_ctx: int | None = None,
    ) -> Reply: ...

    def chat_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        seed: int,
        max_tokens: int,
    ) -> ToolReply: ...

    def model_info(self) -> ModelInfo: ...


def validate_reply(
    reply: Reply | ToolReply, caps: BackendCaps
) -> Reply | ToolReply:
    """Enforce the response contract a backend's caps promise.

    If ``caps.reports_counts is True`` and any contract field is None,
    raise ContractViolation carrying the raw body (the exact signature
    of the Ollama ~11.5k bug: valid-looking content, no stats). When
    ``reports_counts`` is False or None, missing counts pass through
    untouched — calibration decides later (spec §5). Tool replies carry
    the same fields and are held to the same contract.
    """
    if caps.reports_counts is not True:
        return reply
    missing = [name for name in _CONTRACT_FIELDS if getattr(reply, name) is None]
    if missing:
        error = ContractViolation(
            f"backend promised token counts but reply lacks {', '.join(missing)}"
        )
        error.raw = reply.raw
        raise error
    return reply
