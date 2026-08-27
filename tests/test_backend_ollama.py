"""Ollama-native backend (plan Task 4, spec §3).

Wire facts pinned here (previously measured, never re-derived):
top-level ``truncate`` (not inside options), counts from
prompt_eval_count/eval_count, arch-prefixed /api/show keys, weights
size from /api/tags, loaded flag from /api/ps, and the stats-free-200
~11.5k signature becoming ContractViolation.
"""

import pytest

from assay.backends.base import ModelInfo, ToolCall, ToolsUnsupported
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

MOE_MODEL = "qwen3:30b-a3b-q4_K_M"
#: A real MoE shape (qwen3-moe): the attention width is NOT
#: embedding_length // head_count (2048 // 32 = 64), the file states
#: key_length 128, and the routing metadata is present.
MOE_SHOW_BODY = {
    "details": {"quantization_level": "Q4_K_M"},
    "model_info": {
        "general.architecture": "qwen3moe",
        "qwen3moe.block_count": 48,
        "qwen3moe.attention.head_count": 32,
        "qwen3moe.attention.head_count_kv": 4,
        "qwen3moe.attention.key_length": 128,
        "qwen3moe.embedding_length": 2048,
        "qwen3moe.context_length": 40960,
        "qwen3moe.expert_count": 128,
        "qwen3moe.expert_used_count": 8,
    },
}

MOE_TAGS_BODY = {"models": [{"name": MOE_MODEL, "size": 18_600_000_000}]}


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
    # No attention.key_length in this file: the derivation is the
    # FALLBACK reading, and it is the right one here.
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


# --- MoE metadata and the head_dim reading (v1.6) --------------------------


def moe_backend() -> OllamaNative:
    transport = FakeTransport(
        post_routes={"/api/show": (200, MOE_SHOW_BODY)},
        get_routes={
            "/api/tags": (200, MOE_TAGS_BODY),
            "/api/ps": (200, {"models": []}),
        },
    )
    return OllamaNative(
        BASE_URL, MOE_MODEL, http_post=transport.post, http_get=transport.get
    )


def test_model_info_prefers_the_stated_key_length_over_the_derivation():
    # embedding_length // head_count is a DERIVATION, and it is wrong for
    # architectures whose attention width is set independently of the
    # embedding width (this qwen3-moe shape: 2048 // 32 = 64, stated 128).
    # A head_dim off by 2x silently halves every kv-cache number the
    # window law is built on, so the explicit reading wins when present.
    info = moe_backend().model_info()

    assert info.head_dim == 128
    assert info.head_dim != 2048 // 32


def test_model_info_reads_the_expert_counts_of_a_moe_model():
    info = moe_backend().model_info()

    assert info.expert_count == 128
    assert info.expert_used_count == 8


def test_dense_model_reports_no_experts_rather_than_zero():
    # None-vs-zero at the metadata layer: a dense model is not a
    # 0-expert MoE, and 0 would render downstream as a measured routing
    # fact about a model that does no routing.
    transport = FakeTransport(
        post_routes={"/api/show": (200, QWEN_SHOW_BODY)},
        get_routes={
            "/api/tags": (200, QWEN_TAGS_BODY),
            "/api/ps": (200, {"models": []}),
        },
    )

    info = make_backend(transport).model_info()

    assert info.expert_count is None
    assert info.expert_used_count is None


# --- hybrid architecture metadata: R3 / R4 / R6 ----------------------------

HYBRID_MODEL = "qwen3.6:35b-a3b-reap48"
#: The REAP-48 Qwen3.6-35B-A3B shape AS CONVERTED: one attention layer in
#: four, an MTP layer the converter counted into ``block_count`` (40 + 1),
#: and a full ``ssm.*`` set for the Gated-DeltaNet layers. Keys and values
#: as the artifact states them (gguf-geometry vector
#: ``qwen3.6-35b-a3b-reap48-mtp-trap``); one fixture exercises all three
#: rules because on this file they compose.
HYBRID_SHOW_BODY = {
    "details": {"quantization_level": "Q4_K_M"},
    "model_info": {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 41,
        "qwen35moe.context_length": 262144,
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.full_attention_interval": 4,
        "qwen35moe.nextn_predict_layers": 1,
        "qwen35moe.expert_count": 133,
        "qwen35moe.expert_used_count": 8,
        "qwen35moe.ssm.conv_kernel": 4,
        "qwen35moe.ssm.state_size": 128,
        "qwen35moe.ssm.group_count": 16,
        "qwen35moe.ssm.time_step_rank": 32,
        "qwen35moe.ssm.inner_size": 4096,
    },
}


def hybrid_backend(model_info: dict | None = None) -> OllamaNative:
    body = (
        HYBRID_SHOW_BODY
        if model_info is None
        else {"details": {}, "model_info": model_info}
    )
    transport = FakeTransport(
        post_routes={"/api/show": (200, body)},
        get_routes={
            "/api/tags": (200, {"models": []}),
            "/api/ps": (200, {"models": []}),
        },
    )
    return OllamaNative(
        BASE_URL, HYBRID_MODEL, http_post=transport.post, http_get=transport.get
    )


def test_model_info_derives_the_hybrid_layer_geometry():
    info = hybrid_backend().model_info()

    # The raw count is REPORTED as the file states it...
    assert info.block_count == 41
    assert info.mtp_layer_count == 1
    # ...and the count the kv cache is charged on is neither that nor the
    # serving count: R6 takes the MTP layer off (41 - 1 = 40), R3 divides
    # by the stated interval (40 // 4 = 10).
    assert info.attention_layer_count == 10
    # R4: the 30 layers that are not attention layers hold recurrent
    # state, sized from the ssm dimensions.
    assert info.recurrent_state_bytes == 65_863_680


def test_dense_metadata_reports_every_block_as_an_attention_layer():
    transport = FakeTransport(
        post_routes={"/api/show": (200, QWEN_SHOW_BODY)},
        get_routes={
            "/api/tags": (200, QWEN_TAGS_BODY),
            "/api/ps": (200, {"models": []}),
        },
    )

    info = make_backend(transport).model_info()

    # No interval key: the dense identity, so the kv arithmetic is
    # unchanged for every model that was already correct.
    assert info.attention_layer_count == 28
    assert info.mtp_layer_count is None
    # No ssm keys: this architecture HAS no recurrent layers, which is
    # the one case where 0 is a measurement rather than a default.
    assert info.recurrent_state_bytes == 0


def test_metadata_free_show_body_reports_no_layer_geometry_at_all():
    # None-vs-zero: nothing was read, so nothing is claimed — including
    # the recurrent term, which must not read as "measured: none".
    info = hybrid_backend(model_info={}).model_info()

    assert info.block_count is None
    assert info.attention_layer_count is None
    assert info.mtp_layer_count is None
    assert info.recurrent_state_bytes is None


def test_an_interval_that_cannot_be_applied_withholds_the_layer_count():
    # R8. `full_attention_interval` 4 over 2 serving blocks means no
    # layer satisfies llama.cpp's `(i+1) % 4 == 0` rule. The raw block
    # count is withheld along with the derived one: `kv_bytes_per_token`
    # falls back to `block_count` when no attention count was derived,
    # so reporting 2 here would route this file straight into the
    # all-blocks charge R3 exists to forbid.
    stated = dict(HYBRID_SHOW_BODY["model_info"])
    stated["qwen35moe.block_count"] = 2
    stated["qwen35moe.nextn_predict_layers"] = 0

    info = hybrid_backend(model_info=stated).model_info()

    assert info.attention_layer_count is None
    assert info.block_count is None
    assert info.recurrent_state_bytes is None
    # The key itself is still reported: the refusal is about the derived
    # counts, not about hiding what the file said.
    assert info.mtp_layer_count == 0


# --- chat_tools (v1.6) -----------------------------------------------------

MESSAGES = [{"role": "user", "content": "read tiny.py, then tell me the bug"}]
TOOLSET = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]

GOOD_CHAT_BODY = {
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "tiny.py"}}}
        ],
    },
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 41,
    "eval_count": 9,
}


def chat_transport(response) -> FakeTransport:
    return FakeTransport(post_routes={"/api/chat": response})


def test_chat_tools_posts_api_chat_with_pinned_sampler():
    transport = chat_transport((200, GOOD_CHAT_BODY))
    backend = make_backend(transport)

    backend.chat_tools(MESSAGES, TOOLSET, seed=7, max_tokens=256)

    url, payload = transport.post_calls[0]
    assert url == f"{BASE_URL}/api/chat"
    assert payload["model"] == MODEL
    assert payload["messages"] == MESSAGES
    assert payload["tools"] == TOOLSET
    assert payload["stream"] is False
    # Thinking off here too: the probe measures the VISIBLE channel.
    assert payload["think"] is False
    assert payload["options"] == {
        "seed": 7,
        "num_predict": 256,
        "temperature": 0.2,
    }
    # /api/chat takes no truncate flag; nothing extra goes on the wire.
    assert set(payload) == {"model", "messages", "tools", "stream", "think", "options"}


def test_chat_tools_parses_native_tool_call():
    transport = chat_transport((200, GOOD_CHAT_BODY))
    backend = make_backend(transport)

    reply = backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)

    assert reply.text == ""
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert call.name == "read_file"
    # Ollama delivers arguments already parsed as a dict.
    assert call.arguments == {"path": "tiny.py"}
    assert reply.tokens_in == 41
    assert reply.tokens_out == 9
    assert reply.stop_reason == "stop"
    assert reply.raw == GOOD_CHAT_BODY


def test_chat_tools_prose_answer_is_data_not_an_error():
    # A model answering in prose instead of calling is a measurement the
    # probe scores, never an exception.
    body = {
        "message": {"role": "assistant", "content": "You should read tiny.py."},
        "done_reason": "stop",
        "prompt_eval_count": 30,
        "eval_count": 7,
    }
    backend = make_backend(chat_transport((200, body)))

    reply = backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)

    assert reply.tool_calls == ()
    assert reply.text == "You should read tiny.py."


def test_chat_tools_missing_content_is_empty_text():
    body = dict(GOOD_CHAT_BODY)
    body["message"] = {"role": "assistant", "tool_calls": [
        {"function": {"name": "read_file", "arguments": {}}}
    ]}
    backend = make_backend(chat_transport((200, body)))

    reply = backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)

    assert reply.text == ""
    # An empty arguments dict is "called with no arguments" — NOT None.
    assert reply.tool_calls[0].arguments == {}


def test_chat_tools_non_dict_arguments_become_none():
    body = dict(GOOD_CHAT_BODY)
    body["message"] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "read_file", "arguments": "tiny.py"}}],
    }
    backend = make_backend(chat_transport((200, body)))

    reply = backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)

    assert reply.tool_calls[0].name == "read_file"
    assert reply.tool_calls[0].arguments is None


def test_chat_tools_keeps_a_malformed_entry_instead_of_dropping_it():
    # A junk entry the model emitted is DATA the probe scores as an
    # invalid call; dropping it would read as "no call was made" and
    # quietly turn a bad call into a missing one.
    body = dict(GOOD_CHAT_BODY)
    body["message"] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "no-function-here"},
            {"function": {"name": "read_file", "arguments": {"path": "t.py"}}},
        ],
    }
    backend = make_backend(chat_transport((200, body)))

    reply = backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)

    assert len(reply.tool_calls) == 2
    assert reply.tool_calls[0] == ToolCall(name="", arguments=None)
    assert reply.tool_calls[1].name == "read_file"


def test_chat_tools_stats_free_200_raises_contract_violation():
    # The ~11.5k signature again: a valid-looking tool call, no counts.
    body = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "t.py"}}}
            ],
        },
        "done": False,
    }
    backend = make_backend(chat_transport((200, body)))

    with pytest.raises(ContractViolation) as excinfo:
        backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)
    assert excinfo.value.raw == body


def test_chat_tools_400_without_tool_support_is_a_capability_fact():
    body = {"error": "registry.ollama.ai/library/gemma2:9b does not support tools"}
    backend = make_backend(chat_transport((400, body)))

    with pytest.raises(ToolsUnsupported) as excinfo:
        backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)
    assert excinfo.value.raw == body
    # Never an infrastructure failure: the probe records it as data.
    assert not isinstance(excinfo.value, InfrastructureError)


def test_chat_tools_5xx_is_infrastructure_error():
    backend = make_backend(chat_transport((500, {"error": "boom"})))

    with pytest.raises(InfrastructureError) as excinfo:
        backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)
    assert not isinstance(excinfo.value, ContractViolation)


def test_chat_tools_non_dict_body_raises_contract_violation():
    backend = make_backend(chat_transport((200, ["not", "a", "dict"])))

    with pytest.raises(ContractViolation):
        backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)


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
