"""Ollama-native backend (plan Task 4, spec §3).

Wire facts pinned here (previously measured, never re-derived):
top-level ``truncate`` (not inside options), counts from
prompt_eval_count/eval_count, arch-prefixed /api/show keys, weights
size from /api/tags, loaded flag from /api/ps, and the stats-free-200
~11.5k signature becoming ContractViolation.
"""

import pytest

from assay.backends.base import ModelInfo
from assay.backends.ollama import OllamaNative
from assay.errors import ContractViolation, InfrastructureError

BASE_URL = "http://fake-host:11434"
MODEL = "qwen2.5-coder:7b-instruct-q8_0"

GOOD_GENERATE_BODY = {
    "response": "hi there",
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 10,
    "eval_count": 2,
}

QWEN_SHOW_BODY = {
    "details": {"quantization_level": "Q8_0"},
    "model_info": {
        "general.architecture": "qwen2",
        "qwen2.block_count": 28,
        "qwen2.attention.head_count": 28,
        "qwen2.attention.head_count_kv": 4,
        "qwen2.embedding_length": 3584,
        "qwen2.context_length": 32768,
    },
}

QWEN_TAGS_BODY = {"models": [{"name": MODEL, "size": 8098524160}]}


class FakeTransport:
    """Captures requests; routes canned (status, body) responses by URL suffix."""

    def __init__(self, post_routes=None, get_routes=None):
        self.post_calls: list[tuple[str, dict]] = []
        self.get_calls: list[str] = []
        self._post_routes = dict(post_routes or {})
        self._get_routes = dict(get_routes or {})

    def post(self, url: str, payload: dict) -> tuple[int, dict]:
        self.post_calls.append((url, payload))
        return self._route(self._post_routes, url)

    def get(self, url: str) -> tuple[int, dict]:
        self.get_calls.append(url)
        return self._route(self._get_routes, url)

    @staticmethod
    def _route(routes, url):
        for suffix, response in routes.items():
            if url.endswith(suffix):
                return response
        raise AssertionError(f"unexpected request: {url}")


def make_backend(transport: FakeTransport) -> OllamaNative:
    return OllamaNative(
        BASE_URL, MODEL, http_post=transport.post, http_get=transport.get
    )


def test_generate_sends_topLevel_truncate_and_seeded_options():
    transport = FakeTransport(post_routes={"/api/generate": (200, GOOD_GENERATE_BODY)})
    backend = make_backend(transport)

    reply = backend.generate("count to three", seed=7, max_tokens=64)

    url, payload = transport.post_calls[0]
    assert url == f"{BASE_URL}/api/generate"
    # truncate is TOP-LEVEL, not inside options (robigo gotcha).
    assert payload["truncate"] is False
    # thinking disabled at the wire: probes measure the VISIBLE channel
    assert payload["think"] is False
    assert "truncate" not in payload["options"]
    assert payload["options"]["seed"] == 7
    assert payload["options"]["num_predict"] == 64
    # num_ctx omitted entirely when the caller did not pass it.
    assert "num_ctx" not in payload["options"]
    assert "num_ctx" not in payload
    assert payload["stream"] is False
    assert payload["model"] == MODEL
    assert payload["prompt"] == "count to three"
    assert reply.text == "hi there"
    assert reply.tokens_in == 10
    assert reply.tokens_out == 2
    assert reply.stop_reason == "stop"


def test_num_ctx_passes_through_when_given():
    transport = FakeTransport(post_routes={"/api/generate": (200, GOOD_GENERATE_BODY)})
    backend = make_backend(transport)

    backend.generate("hello", seed=0, max_tokens=32, num_ctx=8192)

    _, payload = transport.post_calls[0]
    assert payload["options"]["num_ctx"] == 8192


def test_stats_free_200_raises_contract_violation():
    # The measured ~11.5k signature verbatim: valid-looking content,
    # done false, no counts, no done_reason.
    body = {"response": "valid-looking text", "done": False}
    transport = FakeTransport(post_routes={"/api/generate": (200, body)})
    backend = make_backend(transport)

    with pytest.raises(ContractViolation) as excinfo:
        backend.generate("hello", seed=0, max_tokens=32)
    assert excinfo.value.raw == body


def test_http_5xx_raises_infrastructure_error():
    transport = FakeTransport(post_routes={"/api/generate": (500, {"error": "boom"})})
    backend = make_backend(transport)

    with pytest.raises(InfrastructureError) as excinfo:
        backend.generate("hello", seed=0, max_tokens=32)
    # A 5xx is transport-level failure, never a contract violation.
    assert not isinstance(excinfo.value, ContractViolation)


def test_non_dict_json_body_raises_contract_violation():
    transport = FakeTransport(
        post_routes={"/api/generate": (200, ["not", "a", "dict"])}
    )
    backend = make_backend(transport)

    with pytest.raises(ContractViolation):
        backend.generate("hello", seed=0, max_tokens=32)


def test_model_info_reads_arch_prefixed_keys_and_tags_size():
    transport = FakeTransport(
        post_routes={"/api/show": (200, QWEN_SHOW_BODY)},
        get_routes={
            "/api/tags": (200, QWEN_TAGS_BODY),
            "/api/ps": (200, {"models": []}),
        },
    )
    backend = make_backend(transport)

    info = backend.model_info()

    assert isinstance(info, ModelInfo)
    assert info.name == MODEL
    assert info.block_count == 28
    assert info.kv_head_count == 4  # head_count_kv, NOT head_count
    assert info.head_dim == 128  # embedding_length 3584 // head_count 28
    assert info.training_ctx == 32768
    # Weights size comes from /api/tags (/api/show has no size — robigo gotcha).
    assert info.weights_bytes == 8098524160
    assert info.quant == "Q8_0"
    assert info.source == "api_show"
    # /api/show was asked about this model.
    show_url, show_payload = transport.post_calls[0]
    assert show_url == f"{BASE_URL}/api/show"
    assert show_payload == {"model": MODEL}


def test_model_info_missing_metadata_is_none_not_guessed():
    # None-vs-zero: absent architecture keys and an absent tags entry
    # yield None fields, never defaults that look like measurements.
    transport = FakeTransport(
        post_routes={"/api/show": (200, {"model_info": {}, "details": {}})},
        get_routes={
            "/api/tags": (200, {"models": [{"name": "other:latest", "size": 1}]}),
            "/api/ps": (200, {"models": []}),
        },
    )
    backend = make_backend(transport)

    info = backend.model_info()

    assert info.block_count is None
    assert info.kv_head_count is None
    assert info.head_dim is None
    assert info.training_ctx is None
    assert info.weights_bytes is None
    assert info.quant is None


@pytest.mark.parametrize(
    ("ps_models", "expected"),
    [
        ([{"name": MODEL}], True),
        ([{"name": "other:latest"}], False),
        ([], False),
    ],
)
def test_ps_sets_loaded_flag(ps_models, expected):
    transport = FakeTransport(
        post_routes={"/api/show": (200, QWEN_SHOW_BODY)},
        get_routes={
            "/api/tags": (200, QWEN_TAGS_BODY),
            "/api/ps": (200, {"models": ps_models}),
        },
    )
    backend = make_backend(transport)

    assert backend.model_info().loaded is expected


def test_generate_pins_the_probe_temperature():
    # An unpinned temperature leaves the daemon default (0.8) as an
    # uncontrolled variable of the instrument. Measured live 2026-08-12:
    # qwen at the default stripped indentation in 15/15 search_replace
    # probes (0% landing) where robigo's 0.2-pinned probe measured 100%.
    transport = FakeTransport(post_routes={"/api/generate": (200, GOOD_GENERATE_BODY)})
    backend = make_backend(transport)

    backend.generate("anything", seed=0, max_tokens=8)

    _, payload = transport.post_calls[0]
    assert payload["options"]["temperature"] == 0.2
