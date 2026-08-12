"""Backend protocol and reply contract validation (plan Task 2, spec §3).

`validate_reply` is the 11.5k-bug detector: a 200 that promised token
counts but delivered none is a ContractViolation, never a model result.
"""

import pytest

from assay.backends.base import BackendCaps, Reply, validate_reply
from assay.errors import ContractViolation

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
