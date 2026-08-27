"""OpenAI-compatible backend (spec §3).

Bare ``/chat/completions`` calls: ``{model, messages, max_tokens, seed}``.
Token counts come from ``usage`` when the server includes it, else None —
whether counts are reliably reported is determined empirically during
calibration (spec §5), so ``caps.reports_counts`` is None here, never
assumed. ``model_info`` is honestly absent: no metadata access means
every architecture field is None, not a guess.
"""

import json
import urllib.error
import urllib.request
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

HttpPost = Callable[[str, dict], tuple[int, dict]]
HttpGet = Callable[[str], tuple[int, dict]]

DEFAULT_TIMEOUT = 120.0


def _read_json(url: str, data: bytes | None, timeout: float) -> tuple[int, dict]:
    """Return ``(status, body)`` — an error status is data, not a raise.

    A non-2xx body is the only evidence that distinguishes "this endpoint
    does not do tools" from "this endpoint is broken", so it is read and
    handed back (``{}`` when unreadable) exactly as the Ollama transport
    does. Deciding what a status MEANS belongs to the caller: generate
    raises on any non-200, chat_tools classifies first.
    """
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except (ValueError, OSError):
            body = {}
        return exc.code, body if isinstance(body, dict) else {}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise InfrastructureError(f"transport failure for {url}: {exc}") from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"non-JSON body from {url}") from exc
    if not isinstance(body, dict):
        raise ContractViolation(f"JSON body from {url} is not an object")
    return status, body


def _default_post(url: str, payload: dict, *,
                  timeout: float = DEFAULT_TIMEOUT) -> tuple[int, dict]:
    return _read_json(url, json.dumps(payload).encode("utf-8"), timeout)


def _default_get(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> tuple[int, dict]:
    return _read_json(url, None, timeout)


def _decode_arguments(raw) -> dict | None:
    """Tool arguments come over this wire as a JSON **string**.

    A malformed string is DATA — the probe scores it as an invalid call —
    so it becomes ``arguments=None``, never an exception. Anything that
    parses to something other than an object is equally unreadable.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


class OpenAICompat:
    """Backend for any /chat/completions-speaking server."""

    caps = BackendCaps(
        reports_counts=None,  # unknown until calibration (spec §5)
        per_request_ctx=False,
        truncate_control=False,
        metadata_access=False,
    )

    def __init__(self, base_url: str, model: str, *,
                 http_post: HttpPost | None = None,
                 http_get: HttpGet | None = None,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        if http_post is None:
            http_post = lambda url, payload: _default_post(  # noqa: E731
                url, payload, timeout=timeout
            )
        if http_get is None:
            http_get = lambda url: _default_get(url, timeout=timeout)  # noqa: E731
        self._http_post = http_post
        self._http_get = http_get

    def generate(self, prompt: str, *, seed: int, max_tokens: int,
                 num_ctx: int | None = None) -> Reply:
        # num_ctx is deliberately not sent: caps.per_request_ctx is False.
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "seed": seed,
            "temperature": PROBE_TEMPERATURE,
        }
        status, body = self._http_post(url, payload)
        if status != 200:
            raise InfrastructureError(f"HTTP {status} from {url}")
        if not isinstance(body, dict):
            error = ContractViolation(f"body from {url} is not a JSON object")
            error.raw = {"body": body}
            raise error
        try:
            choice = body["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            error = ContractViolation(
                f"200 from {url} lacks choices[0].message.content"
            )
            error.raw = body
            raise error from exc
        if not isinstance(text, str):
            error = ContractViolation(f"message content from {url} is not text")
            error.raw = body
            raise error
        usage = body.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        reply = Reply(
            text=text,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            stop_reason=choice.get("finish_reason"),
            raw=body,
        )
        return validate_reply(reply, self.caps)

    def chat_tools(self, messages: list[dict], tools: list[dict], *,
                   seed: int, max_tokens: int) -> ToolReply:
        """Tool call over /chat/completions; a refusal is classified, not raised.

        The status is handed to the shared classifier BEFORE the body is
        read as a reply: a 4xx complaining about tools is the endpoint
        declaring a capability, and only the raw error body says so.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "seed": seed,
            "temperature": PROBE_TEMPERATURE,
        }
        status, body = self._http_post(url, payload)
        raise_for_tools_status(status, body, url)
        if not isinstance(body, dict):
            error = ContractViolation(f"body from {url} is not a JSON object")
            error.raw = {"body": body}
            raise error
        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            error = ContractViolation(f"200 from {url} lacks choices[0].message")
            error.raw = body
            raise error from exc
        if not isinstance(message, dict):
            error = ContractViolation(f"message from {url} is not an object")
            error.raw = body
            raise error
        # content is null on this wire when the model calls a tool: an
        # empty visible answer, not a broken response.
        content = message.get("content")
        usage = body.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        reply = ToolReply(
            text=content if isinstance(content, str) else "",
            tool_calls=parse_tool_calls(message, _decode_arguments),
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            stop_reason=choice.get("finish_reason"),
            raw=body,
        )
        return validate_reply(reply, self.caps)

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            quant=None,
            weights_bytes=None,
            training_ctx=None,
            block_count=None,
            kv_head_count=None,
            head_dim=None,
            loaded=None,
            source="openai_models",
            # No metadata access: MoE routing is unreported, not absent.
            expert_count=None,
            expert_used_count=None,
            # ...and so is the hybrid layer geometry (R3/R4/R6). None
            # here is "this wire cannot say", which is why 0 recurrent
            # bytes — the answer for an architecture that states no ssm
            # keys — must not be written by a backend that read no keys.
            attention_layer_count=None,
            recurrent_state_bytes=None,
            mtp_layer_count=None,
        )
