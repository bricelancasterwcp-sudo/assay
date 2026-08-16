"""Backend protocol and reply contract validation (plan Task 2, spec §3).

`validate_reply` is the 11.5k-bug detector: a 200 that promised token
counts but delivered none is a ContractViolation, never a model result.
The tool surface (v1.6) adds the second classification the instrument
depends on: a refused `tools` parameter is a CAPABILITY FACT, not an
infrastructure failure.
"""

from dataclasses import FrozenInstanceError

import pytest

from assay.backends.base import (
    BackendCaps,
    Reply,
    ToolCall,
    ToolReply,
    ToolsUnsupported,
    raise_for_tools_status,
    validate_reply,
)
from assay.errors import ContractViolation, InfrastructureError

PROMISING_CAPS = BackendCaps(
    reports_counts=True,
    per_request_ctx=True,
    truncate_control=True,
    metadata_access=True,
)


def make_reply(**overrides) -> Reply:
    fields = {
        "text": "hello",
        "tokens_in": 12,
        "tokens_out": 3,
        "stop_reason": "stop",
        "raw": {"done": True},
    }
    fields.update(overrides)
    return Reply(**fields)


@pytest.mark.parametrize("missing_field", ["tokens_in", "tokens_out", "stop_reason"])
def test_missing_counts_with_promised_counts_is_contract_violation(missing_field):
    # Each contract field individually missing violates — including
    # stop_reason alone while both counts are present.
    reply = make_reply(**{missing_field: None})
    with pytest.raises(ContractViolation):
        validate_reply(reply, PROMISING_CAPS)


def test_complete_reply_passes_validation_when_counts_promised():
    reply = make_reply()
    assert validate_reply(reply, PROMISING_CAPS) is reply


@pytest.mark.parametrize("reports_counts", [False, None])
def test_missing_counts_without_promise_pass_through(reports_counts):
    # reports_counts=False (known absent) and None (unknown until
    # calibration, openai_compat) both let count-free replies through.
    caps = BackendCaps(
        reports_counts=reports_counts,
        per_request_ctx=False,
        truncate_control=False,
        metadata_access=False,
    )
    reply = make_reply(tokens_in=None, tokens_out=None, stop_reason=None)
    assert validate_reply(reply, caps) is reply


def test_contract_violation_carries_raw_body():
    # The evidence trail: the exception exposes the verbatim response body.
    raw = {"response": "valid-looking text", "done": False}
    reply = make_reply(tokens_in=None, tokens_out=None, stop_reason=None, raw=raw)
    with pytest.raises(ContractViolation) as excinfo:
        validate_reply(reply, PROMISING_CAPS)
    assert excinfo.value.raw == raw


# --- tool reply types ------------------------------------------------------


def make_tool_reply(**overrides) -> ToolReply:
    fields = {
        "text": "",
        "tool_calls": (ToolCall(name="read_file", arguments={"path": "tiny.py"}),),
        "tokens_in": 41,
        "tokens_out": 9,
        "stop_reason": "stop",
        "raw": {"done": True},
    }
    fields.update(overrides)
    return ToolReply(**fields)


def test_tool_types_are_frozen():
    call = ToolCall(name="read_file", arguments={"path": "tiny.py"})
    with pytest.raises(FrozenInstanceError):
        call.name = "write_file"
    with pytest.raises(FrozenInstanceError):
        make_tool_reply().text = "rewritten"


def test_unparseable_arguments_are_none_not_an_empty_dict():
    # None-vs-zero for tool arguments: {} is "the model called with no
    # arguments", None is "arguments came back and could not be read".
    assert ToolCall(name="read_file", arguments=None).arguments is None
    assert ToolCall(name="read_file", arguments={}).arguments == {}


@pytest.mark.parametrize("missing_field", ["tokens_in", "tokens_out", "stop_reason"])
def test_tool_reply_obeys_the_same_count_contract(missing_field):
    # A stats-free tool reply from a counts-promising backend is the same
    # ~11.5k bug wearing a different response shape.
    reply = make_tool_reply(**{missing_field: None})
    with pytest.raises(ContractViolation) as excinfo:
        validate_reply(reply, PROMISING_CAPS)
    assert excinfo.value.raw == {"done": True}


def test_complete_tool_reply_passes_validation():
    reply = make_tool_reply()
    assert validate_reply(reply, PROMISING_CAPS) is reply


# --- the unsupported classifier --------------------------------------------


def test_tools_unsupported_is_never_an_infrastructure_error():
    # The probe records it as tools_supported=False; an
    # `except InfrastructureError` handler must not swallow it.
    assert not issubclass(ToolsUnsupported, InfrastructureError)


@pytest.mark.parametrize(
    "body",
    [
        # Ollama's wording, measured.
        {"error": "registry.ollama.ai/library/gemma2:9b does not support tools"},
        # Case-insensitive: the wording is not a stable target.
        {"error": "This model does not support TOOLS"},
        # OpenAI-compat servers nest the error object.
        {"error": {"message": "Unsupported parameter", "param": "tools"}},
    ],
)
def test_4xx_naming_tools_raises_tools_unsupported_carrying_raw(body):
    with pytest.raises(ToolsUnsupported) as excinfo:
        raise_for_tools_status(400, body, "/api/chat")
    assert excinfo.value.raw == body


def test_4xx_about_something_else_is_infrastructure():
    with pytest.raises(InfrastructureError) as excinfo:
        raise_for_tools_status(404, {"error": "model 'nope' not found"}, "/api/chat")
    assert not isinstance(excinfo.value, ToolsUnsupported)


def test_5xx_mentioning_tools_is_still_infrastructure():
    # The rule keys on behavior CLASS, not on the word: a 5xx is the
    # endpoint failing us however its body is worded.
    with pytest.raises(InfrastructureError) as excinfo:
        raise_for_tools_status(503, {"error": "tool runner crashed"}, "/api/chat")
    assert not isinstance(excinfo.value, ToolsUnsupported)


def test_infrastructure_error_from_the_classifier_carries_raw():
    body = {"error": "boom"}
    with pytest.raises(InfrastructureError) as excinfo:
        raise_for_tools_status(500, body, "/api/chat")
    assert excinfo.value.raw == body


@pytest.mark.parametrize("status", [200, 201, 204])
def test_success_status_raises_nothing(status):
    assert raise_for_tools_status(status, {"message": {}}, "/api/chat") is None


@pytest.mark.parametrize("body", [["not", "a", "dict"], None, "plain text"])
def test_unreadable_error_body_still_classifies_as_infrastructure(body):
    # The classifier never crashes on a body shape it did not expect.
    with pytest.raises(InfrastructureError):
        raise_for_tools_status(400, body, "/api/chat")


def test_bare_body_naming_tools_counts_when_there_is_no_error_key():
    body = {"message": "the model does not support tools"}
    with pytest.raises(ToolsUnsupported):
        raise_for_tools_status(400, body, "/api/chat")


def test_an_echoed_request_body_does_not_fake_an_unsupported_verdict():
    # A server that echoes the failing request echoes the `tools` array
    # we sent. Scanning the whole body would read our own payload back as
    # the endpoint's refusal — a FABRICATED capability fact, the exact
    # bug class this instrument exists to catch. Only error-bearing
    # fields are scanned.
    body = {
        "message": "unknown field 'temprature'",
        "request": {
            "model": "m",
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        },
    }
    with pytest.raises(InfrastructureError) as excinfo:
        raise_for_tools_status(400, body, "/api/chat")
    assert not isinstance(excinfo.value, ToolsUnsupported)


def test_an_echo_nested_inside_the_error_object_is_not_scanned_either():
    # Same fabrication one level down: serializing the whole error object
    # would read an echoed `tools` array back as a refusal. The descent
    # into a dict-valued key reads ITS error fields, not all of it.
    body = {
        "error": {
            "message": "unknown field 'temprature'",
            "type": "invalid_request_error",
            "request": {"tools": [{"type": "function"}]},
        }
    }
    with pytest.raises(InfrastructureError) as excinfo:
        raise_for_tools_status(400, body, "/api/chat")
    assert not isinstance(excinfo.value, ToolsUnsupported)


def test_a_null_error_field_does_not_hide_the_real_refusal():
    # The mirror gap: `error` is PRESENT but null, so a default-on-missing
    # lookup stops there and a genuine refusal misfiles as infrastructure.
    body = {"error": None, "message": "'tools' not supported by this model"}
    with pytest.raises(ToolsUnsupported) as excinfo:
        raise_for_tools_status(400, body, "/api/chat")
    assert excinfo.value.raw == body
