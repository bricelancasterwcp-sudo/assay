"""Backend protocol and response contract (spec §3).

One protocol, two implementations (Ollama-native, OpenAI-compat).
`validate_reply` is the 11.5k-bug detector: a 200 that promised token
counts but delivered none becomes ContractViolation, never a model
result. No field defaults to a value that looks like a measurement —
unreported is None, always.
"""

from dataclasses import dataclass
from typing import Protocol

from assay.errors import ContractViolation

_CONTRACT_FIELDS = ("tokens_in", "tokens_out", "stop_reason")


@dataclass(frozen=True)
class Reply:
    text: str
    tokens_in: int | None  # None = backend did not report; NEVER estimated
    tokens_out: int | None
    stop_reason: str | None  # "stop" | "length" | None (unreported)
    raw: dict  # verbatim response body, for evidence trails


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

    def model_info(self) -> ModelInfo: ...


def validate_reply(reply: Reply, caps: BackendCaps) -> Reply:
    """Enforce the response contract a backend's caps promise.

    If ``caps.reports_counts is True`` and any contract field is None,
    raise ContractViolation carrying the raw body (the exact signature
    of the Ollama ~11.5k bug: valid-looking content, no stats). When
    ``reports_counts`` is False or None, missing counts pass through
    untouched — calibration decides later (spec §5).
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
