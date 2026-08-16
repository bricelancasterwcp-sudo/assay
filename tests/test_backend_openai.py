"""Tests for the OpenAI-compat backend and backend auto-detect (spec §3).

The OllamaNative class lives in a PARALLEL task's module; detection is
tested against a local fake injected at ``assay.backends.ollama`` so this
file never imports another task's code.
"""

import io
import sys
import types
import urllib.error
import urllib.request

import pytest

from assay.backends import detect_backend
from assay.backends.base import ToolsUnsupported
from assay.backends.openai_compat import OpenAICompat, _default_get, _default_post
from assay.errors import ContractViolation, InfrastructureError


class RecordingPost:
    """Fake HttpPost capturing (url, payload) and returning a canned body."""

    def __init__(self, status=200, body=None):
        self.status = status
        self.body = body if body is not None else {}
        self.calls = []

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        return self.status, self.body


class RecordingGet:
    """Fake HttpGet capturing urls and returning a canned body (or raising)."""

    def __init__(self, status=200, body=None, raises=None):
        self.status = status
        self.body = body if body is not None else {}
        self.raises = raises
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if self.raises is not None:
            raise self.raises
        return self.status, self.body


def forbidden_get(url):
    raise AssertionError(f"http_get must not be called (got {url})")


def forbidden_post(url, payload):
    raise AssertionError(f"http_post must not be called (got {url})")


def chat_body(text="hello", usage=None, finish_reason="stop"):
    choice = {"index": 0, "message": {"role": "assistant", "content": text}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    body = {"object": "chat.completion", "choices": [choice]}
    if usage is not None:
        body["usage"] = usage
    return body


class FakeOllamaNative:
    """Local stand-in for the parallel task's OllamaNative (constructor only)."""

    def __init__(self, base_url, model, *, http_post=None, http_get=None,
                 timeout=120.0):
        self.base_url = base_url
        self.model = model
        self.http_post = http_post
        self.http_get = http_get


@pytest.fixture
def fake_ollama_module(monkeypatch):
    module = types.ModuleType("assay.backends.ollama")
    module.OllamaNative = FakeOllamaNative
    monkeypatch.setitem(sys.modules, "assay.backends.ollama", module)
    return module


# --- OpenAICompat.generate -------------------------------------------------


def test_usage_counts_extracted_when_present():
    post = RecordingPost(
        body=chat_body(
            "hi there",
            usage={"prompt_tokens": 12, "completion_tokens": 3},
        )
    )
    backend = OpenAICompat(
        "http://x:8080/v1", "m", http_post=post, http_get=forbidden_get
    )

    reply = backend.generate("prompt text", seed=7, max_tokens=64)

    assert reply.text == "hi there"
    assert reply.tokens_in == 12
    assert reply.tokens_out == 3
    assert reply.stop_reason == "stop"
    url, payload = post.calls[0]
    assert url == "http://x:8080/v1/chat/completions"
    assert payload == {
        "model": "m",
        "messages": [{"role": "user", "content": "prompt text"}],
        "max_tokens": 64,
        "seed": 7,
        "temperature": 0.2,
    }


def test_missing_usage_yields_none_counts_without_error():
    post = RecordingPost(body=chat_body("bare", usage=None, finish_reason=None))
    backend = OpenAICompat(
        "http://x:8080/v1", "m", http_post=post, http_get=forbidden_get
    )

    reply = backend.generate("p", seed=0, max_tokens=8)

    # reports_counts is None (unknown until calibration) so validate_reply
    # must let missing counts through — unreported is None, never estimated.
    assert reply.tokens_in is None
    assert reply.tokens_out is None
    assert reply.stop_reason is None
    assert reply.text == "bare"


def test_num_ctx_is_not_sent_on_the_wire():
    post = RecordingPost(body=chat_body())
    backend = OpenAICompat(
        "http://x:8080/v1", "m", http_post=post, http_get=forbidden_get
    )

    backend.generate("p", seed=1, max_tokens=8, num_ctx=4096)

    _, payload = post.calls[0]
    assert payload == {
        "model": "m",
        "messages": [{"role": "user", "content": "p"}],
        "max_tokens": 8,
        "seed": 1,
        "temperature": 0.2,
    }


def test_http_error_status_raises_infrastructure_error():
    post = RecordingPost(status=500, body={"error": "boom"})
    backend = OpenAICompat(
        "http://x:8080/v1", "m", http_post=post, http_get=forbidden_get
    )

    with pytest.raises(InfrastructureError):
        backend.generate("p", seed=0, max_tokens=8)


def test_body_without_choices_is_contract_violation():
    post = RecordingPost(body={"object": "chat.completion"})
    backend = OpenAICompat(
        "http://x:8080/v1", "m", http_post=post, http_get=forbidden_get
    )

    with pytest.raises(ContractViolation):
        backend.generate("p", seed=0, max_tokens=8)


def test_model_info_is_honestly_absent():
    backend = OpenAICompat(
        "http://x:8080/v1", "the-model",
        http_post=forbidden_post, http_get=forbidden_get,
    )

    info = backend.model_info()

    assert info.name == "the-model"
    assert info.source == "openai_models"
    assert info.quant is None
    assert info.weights_bytes is None
    assert info.training_ctx is None
    assert info.block_count is None
    assert info.kv_head_count is None
    assert info.head_dim is None
    assert info.loaded is None


def test_caps_declare_counts_unknown_until_calibration():
    assert OpenAICompat.caps.reports_counts is None
    assert OpenAICompat.caps.per_request_ctx is False
    assert OpenAICompat.caps.truncate_control is False
    assert OpenAICompat.caps.metadata_access is False


# --- detect_backend --------------------------------------------------------


def test_detect_prefers_native_when_api_tags_answers(fake_ollama_module):
    get = RecordingGet(status=200, body={"models": [{"name": "m"}]})

    backend = detect_backend(
        "http://h:11434/v1", "m", http_post=forbidden_post, http_get=get
    )

    assert isinstance(backend, FakeOllamaNative)
    # /v1 stripped: the native API lives at the root.
    assert backend.base_url == "http://h:11434"
    # The detection probe itself must hit the stripped root.
    assert get.calls[0] == "http://h:11434/api/tags"


def test_detect_without_v1_suffix_keeps_base_for_native(fake_ollama_module):
    get = RecordingGet(status=200, body={"models": []})

    backend = detect_backend(
        "http://h:11434", "m", http_post=forbidden_post, http_get=get
    )

    assert isinstance(backend, FakeOllamaNative)
    assert backend.base_url == "http://h:11434"


def test_detect_200_without_models_key_is_not_ollama(fake_ollama_module):
    # A non-Ollama server that answers 200 on /api/tags with some other
    # shape must NOT be mistaken for Ollama: the body shape decides.
    get = RecordingGet(status=200, body={"object": "list", "data": []})

    backend = detect_backend(
        "http://h:8080/v1", "m", http_post=forbidden_post, http_get=get
    )

    assert isinstance(backend, OpenAICompat)


def test_detect_non_200_falls_back_to_openai_keeping_v1(fake_ollama_module):
    get = RecordingGet(status=404, body={})

    backend = detect_backend(
        "http://h:8080/v1", "m", http_post=forbidden_post, http_get=get
    )

    assert isinstance(backend, OpenAICompat)
    # The user's /v1 suffix is preserved for the compat backend.
    assert backend.base_url == "http://h:8080/v1"


def test_detect_transport_failure_falls_back_to_openai(fake_ollama_module):
    get = RecordingGet(raises=InfrastructureError("connection refused"))

    backend = detect_backend(
        "http://h:8080/v1", "m", http_post=forbidden_post, http_get=get
    )

    assert isinstance(backend, OpenAICompat)


def test_forced_backend_overrides_detection(fake_ollama_module):
    # forced="openai" even though /api/tags WOULD answer like Ollama —
    # and no detection GET may happen at all.
    backend = detect_backend(
        "http://h:11434/v1", "m", forced="openai",
        http_post=forbidden_post, http_get=forbidden_get,
    )
    assert isinstance(backend, OpenAICompat)
    assert backend.base_url == "http://h:11434/v1"

    forced_native = detect_backend(
        "http://h:11434/v1", "m", forced="ollama",
        http_post=forbidden_post, http_get=forbidden_get,
    )
    assert isinstance(forced_native, FakeOllamaNative)
    assert forced_native.base_url == "http://h:11434"


def test_forced_unknown_kind_is_rejected(fake_ollama_module):
    with pytest.raises(ValueError):
        detect_backend(
            "http://h:1/v1", "m", forced="vllm",
            http_post=forbidden_post, http_get=forbidden_get,
        )


def test_generate_pins_the_probe_temperature():
    # Same invariant as the ollama backend: the sampler is part of the
    # instrument and must be pinned, not left to the server default.
    post = RecordingPost(body=chat_body("ok"))
    backend = OpenAICompat(
        "http://x:8080/v1", "m", http_post=post, http_get=forbidden_get
    )

    backend.generate("anything", seed=0, max_tokens=8)

    assert post.calls[0][1]["temperature"] == 0.2


# --- the transport refactor (chat_tools needs the error body) --------------


class FakeHTTPError(urllib.error.HTTPError):
    """A real HTTPError with a readable body, as urlopen would raise."""

    def __init__(self, url, code, payload: bytes):
        super().__init__(url, code, "Error", {}, io.BytesIO(payload))


def install_urlopen(monkeypatch, error):
    """Make urlopen raise ``error`` — no socket is ever opened."""

    def fake_urlopen(request, timeout=None):
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_http_error_returns_status_and_body_instead_of_raising(monkeypatch):
    # The classifier needs the body a 4xx carries; raising here would
    # discard the only evidence that tools were refused.
    url = "http://x:8080/v1/chat/completions"
    install_urlopen(
        monkeypatch, FakeHTTPError(url, 400, b'{"error": "no tools here"}')
    )

    assert _default_post(url, {"model": "m"}) == (400, {"error": "no tools here"})


def test_unreadable_error_body_becomes_an_empty_dict(monkeypatch):
    url = "http://x:8080/v1/chat/completions"
    install_urlopen(monkeypatch, FakeHTTPError(url, 503, b"<html>gateway</html>"))

    assert _default_post(url, {"model": "m"}) == (503, {})


def test_non_dict_error_body_becomes_an_empty_dict(monkeypatch):
    url = "http://x:8080/v1/models"
    install_urlopen(monkeypatch, FakeHTTPError(url, 404, b"[1, 2, 3]"))

    assert _default_get(url) == (404, {})


def test_transport_failure_still_raises_infrastructure_error(monkeypatch):
    url = "http://x:8080/v1/chat/completions"
    install_urlopen(monkeypatch, urllib.error.URLError("connection refused"))

    with pytest.raises(InfrastructureError):
        _default_post(url, {"model": "m"})


# --- OpenAICompat.chat_tools ----------------------------------------------

MESSAGES = [{"role": "user", "content": "read tiny.py"}]
TOOLSET = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
]


def tool_body(content=None, arguments='{"path": "tiny.py"}', name="read_file",
              usage=None, finish_reason="tool_calls"):
    message = {"role": "assistant", "content": content}
    if arguments is not None:
        message["tool_calls"] = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ]
    choice = {"index": 0, "message": message, "finish_reason": finish_reason}
    body = {"object": "chat.completion", "choices": [choice]}
    if usage is not None:
        body["usage"] = usage
    return body


def tools_backend(status=200, body=None):
    post = RecordingPost(status=status, body=body if body is not None else tool_body())
    backend = OpenAICompat(
        "http://x:8080/v1", "m", http_post=post, http_get=forbidden_get
    )
    return backend, post


def test_chat_tools_sends_the_tools_payload():
    backend, post = tools_backend()

    backend.chat_tools(MESSAGES, TOOLSET, seed=7, max_tokens=256)

    url, payload = post.calls[0]
    assert url == "http://x:8080/v1/chat/completions"
    assert payload == {
        "model": "m",
        "messages": MESSAGES,
        "tools": TOOLSET,
        "max_tokens": 256,
        "seed": 7,
        "temperature": 0.2,
    }


def test_chat_tools_parses_json_string_arguments():
    backend, _ = tools_backend(
        body=tool_body(usage={"prompt_tokens": 41, "completion_tokens": 9})
    )

    reply = backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)

    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "read_file"
    # arguments arrives as a JSON STRING on this wire and must be parsed.
    assert reply.tool_calls[0].arguments == {"path": "tiny.py"}
    # content: null (the OpenAI shape when the model calls a tool) is "".
    assert reply.text == ""
    assert reply.tokens_in == 41
    assert reply.tokens_out == 9
    assert reply.stop_reason == "tool_calls"


@pytest.mark.parametrize(
    "arguments",
    [
        '{"path": "tiny.py"',  # truncated — the classic malformed call
        "not json at all",
        "[1, 2]",  # valid JSON, but not an arguments object
        "null",
    ],
)
def test_malformed_arguments_are_data_not_an_exception(arguments):
    backend, _ = tools_backend(body=tool_body(arguments=arguments))

    reply = backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)

    assert reply.tool_calls[0].name == "read_file"
    assert reply.tool_calls[0].arguments is None


def test_already_parsed_arguments_pass_through():
    # Some compat servers deliver an object where the spec says string.
    # That is a readable call, not a malformed one — scoring it invalid
    # would be an instrument error, not a model result.
    backend, _ = tools_backend(body=tool_body(arguments={"path": "tiny.py"}))

    reply = backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)

    assert reply.tool_calls[0].arguments == {"path": "tiny.py"}


def test_chat_tools_prose_answer_yields_an_empty_tuple():
    backend, _ = tools_backend(
        body=tool_body(content="I would just open tiny.py.", arguments=None,
                       finish_reason="stop")
    )

    reply = backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)

    assert reply.tool_calls == ()
    assert reply.text == "I would just open tiny.py."
    assert reply.stop_reason == "stop"


def test_chat_tools_missing_usage_leaves_counts_none():
    # reports_counts is None here (unknown until calibration), so a
    # count-free tool reply passes — unreported is None, never estimated.
    backend, _ = tools_backend(body=tool_body(usage=None))

    reply = backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)

    assert reply.tokens_in is None
    assert reply.tokens_out is None


def test_chat_tools_4xx_naming_tools_is_a_capability_fact():
    body = {
        "error": {
            "message": "'tools' is not supported by this model",
            "type": "invalid_request_error",
            "param": "tools",
        }
    }
    backend, _ = tools_backend(status=400, body=body)

    with pytest.raises(ToolsUnsupported) as excinfo:
        backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)
    assert excinfo.value.raw == body
    assert not isinstance(excinfo.value, InfrastructureError)


def test_chat_tools_5xx_is_infrastructure_error():
    backend, _ = tools_backend(status=500, body={"error": "boom"})

    with pytest.raises(InfrastructureError):
        backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)


def test_chat_tools_body_without_choices_is_contract_violation():
    backend, _ = tools_backend(body={"object": "chat.completion"})

    with pytest.raises(ContractViolation):
        backend.chat_tools(MESSAGES, TOOLSET, seed=0, max_tokens=256)
