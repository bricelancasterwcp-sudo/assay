"""assay backends: one protocol, two implementations (spec §3)."""

from assay.backends.base import (
    Backend,
    BackendCaps,
    ModelInfo,
    Reply,
    validate_reply,
)
from assay.backends.openai_compat import OpenAICompat, _default_get
from assay.errors import InfrastructureError

__all__ = [
    "Backend",
    "BackendCaps",
    "ModelInfo",
    "OpenAICompat",
    "Reply",
    "detect_backend",
    "validate_reply",
]

_V1_SUFFIX = "/v1"


def _native_root(base_url: str) -> str:
    """The Ollama-native API lives at the host root, never under /v1."""
    root = base_url.rstrip("/")
    if root.endswith(_V1_SUFFIX):
        root = root[: -len(_V1_SUFFIX)]
    return root


def _backend_kwargs(http_post, http_get) -> dict:
    kwargs = {}
    if http_post is not None:
        kwargs["http_post"] = http_post
    if http_get is not None:
        kwargs["http_get"] = http_get
    return kwargs


def _make_ollama(base_url: str, model: str, http_post, http_get) -> Backend:
    # Imported lazily: keeps this package importable without the native
    # backend module and avoids an import cycle.
    from assay.backends.ollama import OllamaNative

    return OllamaNative(
        _native_root(base_url), model, **_backend_kwargs(http_post, http_get)
    )


def detect_backend(base_url: str, model: str, *, forced: str | None = None,
                   http_post=None, http_get=None) -> Backend:
    """Pick the backend for ``base_url``.

    ``forced`` ("ollama" | "openai") wins outright. Otherwise a GET on
    ``{root}/api/tags`` answering 200 with a ``models`` key means Ollama
    (the body shape decides — any other 200 is NOT Ollama). OllamaNative
    gets the base_url with a trailing ``/v1`` stripped (the native API
    lives at the root); OpenAICompat keeps whatever the user gave.
    """
    if forced is not None:
        if forced == "ollama":
            return _make_ollama(base_url, model, http_post, http_get)
        if forced == "openai":
            return OpenAICompat(
                base_url, model, **_backend_kwargs(http_post, http_get)
            )
        raise ValueError(f"unknown backend kind: {forced!r}")

    get = http_get if http_get is not None else _default_get
    try:
        status, body = get(f"{_native_root(base_url)}/api/tags")
    except InfrastructureError:
        status, body = None, None
    if status == 200 and isinstance(body, dict) and "models" in body:
        return _make_ollama(base_url, model, http_post, http_get)
    return OpenAICompat(base_url, model, **_backend_kwargs(http_post, http_get))
