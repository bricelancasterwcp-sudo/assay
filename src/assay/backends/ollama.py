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
    validate_reply,
)
from assay.errors import ContractViolation, InfrastructureError

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

    def model_info(self) -> ModelInfo:
        show = self._post("/api/show", {"model": self.model})
        arch_info = show.get("model_info")
        arch_info = arch_info if isinstance(arch_info, dict) else {}
        details = show.get("details")
        details = details if isinstance(details, dict) else {}

        embedding_length = _arch_value(arch_info, "embedding_length")
        head_count = _arch_value(arch_info, "attention.head_count")
        head_dim = None
        if embedding_length is not None and head_count:
            head_dim = embedding_length // head_count

        return ModelInfo(
            name=self.model,
            quant=details.get("quantization_level"),
            weights_bytes=self._weights_bytes(),
            training_ctx=_arch_value(arch_info, "context_length"),
            block_count=_arch_value(arch_info, "block_count"),
            kv_head_count=_arch_value(arch_info, "attention.head_count_kv"),
            head_dim=head_dim,
            loaded=self._loaded(),
            source="api_show",
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
