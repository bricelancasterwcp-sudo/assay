"""Ollama-native backend (spec §3, plan Task 4).

Talks to the native API at the daemon root: POST /api/generate for
replies, POST /api/show + GET /api/tags + GET /api/ps for metadata.
Wire facts (previously measured, do not re-derive):

- ``truncate`` is TOP-LEVEL in the generate payload, not inside
  ``options`` (robigo gotcha).
- Weights size comes from /api/tags — /api/show has no size (robigo
  gotcha).
- Every reply goes through ``validate_reply``: the ~11.5k stats-free
  200 becomes ContractViolation here, never a model result.
"""

import json
import urllib.error
import urllib.request
from functools import partial
from typing import Callable

from assay.backends.base import (
    PROBE_TEMPERATURE,
    BackendCaps,
    ModelInfo,
    Reply,
    ToolReply,
    parse_tool_calls,
    raise_for_tools_status,
    validate_reply,
)
from assay.errors import ContractViolation, InfrastructureError
from assay.geometry import (
    attention_layer_count,
    recurrent_state_bytes,
    serving_block_count,
)

HttpPost = Callable[[str, dict], tuple[int, dict]]  # (url, payload) -> (status, body)
HttpGet = Callable[[str], tuple[int, dict]]

DEFAULT_TIMEOUT = 120.0


def _send(request: urllib.request.Request, timeout: float) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read().decode("utf-8"))
        except (ValueError, OSError):
            body = {}
        return error.code, body if isinstance(body, dict) else {}
    except OSError as error:  # URLError, timeout, refused connection
        raise InfrastructureError(
            f"transport failure for {request.full_url}: {error}"
        ) from error
    try:
        body = json.loads(text)
    except ValueError as error:
        raise ContractViolation(
            f"non-JSON body from {request.full_url}"
        ) from error
    return status, body


def _default_post(
    url: str, payload: dict, *, timeout: float = DEFAULT_TIMEOUT
) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _send(request, timeout)


def _default_get(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> tuple[int, dict]:
    request = urllib.request.Request(url, method="GET")
    return _send(request, timeout)


def _arch_value(model_info: dict, suffix: str):
    """Read an architecture-prefixed /api/show key (e.g. qwen2.block_count).

    Prefers the prefix named by general.architecture; falls back to a
    suffix scan so unknown architectures still resolve. Missing → None,
    never a guess.
    """
    arch = model_info.get("general.architecture")
    if isinstance(arch, str):
        value = model_info.get(f"{arch}.{suffix}")
        if value is not None:
            return value
    dotted = f".{suffix}"
    for key, value in model_info.items():
        if key.endswith(dotted) and not key.startswith("general."):
            return value
    return None


def _head_dim(arch_info: dict) -> int | None:
    """Attention head dimension: the STATED value first, the derivation after.

    ``attention.key_length`` is what the model file says the head width
    is. ``embedding_length // head_count`` only DERIVES it, and the
    derivation assumes attention width == embedding width / heads —
    false for architectures that size attention independently (qwen3-moe
    q4: embedding 2048, 32 heads, stated key_length 128, so the
    derivation says 64). A head_dim off by 2x silently halves every
    kv-cache number the window law is built on, so the explicit reading
    wins whenever the metadata carries it. Neither present → None.
    """
    key_length = _arch_value(arch_info, "attention.key_length")
    if key_length is not None:
        return key_length
    embedding_length = _arch_value(arch_info, "embedding_length")
    head_count = _arch_value(arch_info, "attention.head_count")
    if embedding_length is None or not head_count:
        return None
    return embedding_length // head_count


def _layer_geometry(
    arch_info: dict,
) -> tuple[int | None, int | None, int | None, int | None]:
    """(block_count, attention_layer_count, mtp_layer_count, recurrent_bytes).

    Reads the keys; ``geometry.py`` owns the rules they feed, in the
    order they compose — R6 takes the MTP layers off the raw block
    count, R3 divides what serves by the stated attention interval, and
    R4 sizes the layers that are left, which are the recurrent ones.

    When the file states an interval this implementation cannot apply,
    the raw block count is WITHHELD along with the derived one.
    ``kv_bytes_per_token`` falls back to ``block_count`` when no
    attention count was derived — correct for a dense file, which states
    no interval — so reporting the raw count for a file that states one
    would route it straight into the all-blocks charge R3 exists to
    forbid. Unmeasurable, and reported as such (R8): ``plan_window``
    then returns None and the run records the geometry as unavailable.
    """
    block_count = _arch_value(arch_info, "block_count")
    mtp_layer_count = _arch_value(arch_info, "nextn_predict_layers")
    serving = serving_block_count(block_count, mtp_layer_count)
    attention = attention_layer_count(
        serving, _arch_value(arch_info, "full_attention_interval")
    )
    if attention is None:
        return None, None, mtp_layer_count, None
    return (
        block_count,
        attention,
        mtp_layer_count,
        recurrent_state_bytes(
            serving - attention,
            conv_kernel=_arch_value(arch_info, "ssm.conv_kernel"),
            state_size=_arch_value(arch_info, "ssm.state_size"),
            group_count=_arch_value(arch_info, "ssm.group_count"),
            inner_size=_arch_value(arch_info, "ssm.inner_size"),
        ),
    )


class OllamaNative:
    """Backend for Ollama's native API."""

    caps = BackendCaps(
        reports_counts=True,
        per_request_ctx=True,
        truncate_control=True,
        metadata_access=True,
    )

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        http_post: HttpPost = _default_post,
        http_get: HttpGet = _default_get,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        if http_post is _default_post:
            http_post = partial(_default_post, timeout=timeout)
        if http_get is _default_get:
            http_get = partial(_default_get, timeout=timeout)
        self._http_post = http_post
        self._http_get = http_get

    def _checked(self, status: int, body, path: str) -> dict:
        if not 200 <= status < 300:
            error = InfrastructureError(f"HTTP {status} from {path}")
            error.raw = body
            raise error
        if not isinstance(body, dict):
            raise ContractViolation(
                f"non-dict JSON body from {path}: {type(body).__name__}"
            )
        return body

    def _post(self, path: str, payload: dict) -> dict:
        status, body = self._http_post(f"{self.base_url}{path}", payload)
        return self._checked(status, body, path)

    def _get(self, path: str) -> dict:
        status, body = self._http_get(f"{self.base_url}{path}")
        return self._checked(status, body, path)

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        max_tokens: int,
        num_ctx: int | None = None,
    ) -> Reply:
        options: dict = {
            "seed": seed,
            "num_predict": max_tokens,
            "temperature": PROBE_TEMPERATURE,
        }
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Top-level, NOT inside options (robigo gotcha): ask the
            # daemon to hard-fail instead of silently truncating.
            "truncate": False,
            # Thinking-default models (qwen3.8, r1-class) spend the whole
            # generation budget on reasoning and return an EMPTY visible
            # response under capped probes (measured live 2026-08-14 on
            # qwen3.8:27b: 32 thinking tokens, empty text). Probes measure
            # the VISIBLE channel; reasoning-off is part of the
            # instrument and is recorded in provenance/lens.
            "think": False,
            "options": options,
        }
        body = self._post("/api/generate", payload)
        reply = Reply(
            text=body.get("response", ""),
            tokens_in=body.get("prompt_eval_count"),
            tokens_out=body.get("eval_count"),
            stop_reason=body.get("done_reason"),
            raw=body,
        )
        return validate_reply(reply, self.caps)

    def chat_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        seed: int,
        max_tokens: int,
    ) -> ToolReply:
        """Native tool call: POST /api/chat with the same pinned sampler.

        A 4xx refusing tools is a capability fact, not a failure — the
        classifier in base.py decides, and the raw body travels with it.
        """
        path = "/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            # Same reason as generate: probes measure the VISIBLE channel.
            "think": False,
            "options": {
                "seed": seed,
                "num_predict": max_tokens,
                "temperature": PROBE_TEMPERATURE,
            },
        }
        status, body = self._http_post(f"{self.base_url}{path}", payload)
        raise_for_tools_status(status, body, path)
        if not isinstance(body, dict):
            raise ContractViolation(
                f"non-dict JSON body from {path}: {type(body).__name__}"
            )
        message = body.get("message")
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        reply = ToolReply(
            text=content if isinstance(content, str) else "",
            # Ollama's arguments are already a dict — the default rule.
            tool_calls=parse_tool_calls(message),
            tokens_in=body.get("prompt_eval_count"),
            tokens_out=body.get("eval_count"),
            stop_reason=body.get("done_reason"),
            raw=body,
        )
        return validate_reply(reply, self.caps)

    def model_info(self) -> ModelInfo:
        show = self._post("/api/show", {"model": self.model})
        arch_info = show.get("model_info")
        arch_info = arch_info if isinstance(arch_info, dict) else {}
        details = show.get("details")
        details = details if isinstance(details, dict) else {}
        block_count, attention_layers, mtp_layers, recurrent_bytes = (
            _layer_geometry(arch_info)
        )

        return ModelInfo(
            name=self.model,
            quant=details.get("quantization_level"),
            weights_bytes=self._weights_bytes(),
            training_ctx=_arch_value(arch_info, "context_length"),
            block_count=block_count,
            kv_head_count=_arch_value(arch_info, "attention.head_count_kv"),
            head_dim=_head_dim(arch_info),
            loaded=self._loaded(),
            source="api_show",
            # Present only on MoE files; a dense model reports neither
            # and keeps both None (never 0 — see ModelInfo).
            expert_count=_arch_value(arch_info, "expert_count"),
            expert_used_count=_arch_value(arch_info, "expert_used_count"),
            # Hybrid layer geometry (R3/R4/R6). A dense file states none
            # of these keys and lands on the dense identity: every block
            # an attention layer, no MTP layer, no recurrent state.
            attention_layer_count=attention_layers,
            recurrent_state_bytes=recurrent_bytes,
            mtp_layer_count=mtp_layers,
            # MLA value width (R9). Verbatim reading, no derivation: an
            # MLA file states this separately from key_length.
            value_length=_arch_value(arch_info, "attention.value_length"),
        )

    def _is_this_model(self, entry) -> bool:
        return isinstance(entry, dict) and self.model in (
            entry.get("name"),
            entry.get("model"),
        )

    def _weights_bytes(self) -> int | None:
        # /api/show has no size; /api/tags does (robigo gotcha).
        tags = self._get("/api/tags")
        for entry in tags.get("models") or []:
            if self._is_this_model(entry):
                size = entry.get("size")
                return size if isinstance(size, int) else None
        return None

    def _loaded(self) -> bool:
        ps = self._get("/api/ps")
        return any(self._is_this_model(entry) for entry in ps.get("models") or [])
