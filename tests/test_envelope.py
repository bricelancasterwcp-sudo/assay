"""Envelope fidelity probe (plan Task 8, spec §6).

N seeded probes ask for an exact one-line ``VERB ARG`` reply; fidelity
is the fraction that land exactly. Invalid replies are classified
(prose / shape / refusal) because a 60%-from-chattiness model and a
60%-from-refusals model need different application responses.
Zero completed probes is fidelity ``None``, never ``0.0``.
"""

import pytest

from assay.backends.base import BackendCaps, Reply
from assay.budget import Budget, BudgetMeter
from assay.envelope import Envelope, probe_envelope

_VERBS = ("ALPHA", "BRAVO", "CHARLIE")


def expected_line(i: int) -> str:
    return f"{_VERBS[i % 3]} {i}"


def ample_meter() -> BudgetMeter:
    return BudgetMeter(Budget(max_calls=1000, max_prompt_tokens=1_000_000))


class ScriptedBackend:
    """Fake backend replying with scripted texts in call order."""

    caps = BackendCaps(
        reports_counts=True,
        per_request_ctx=True,
        truncate_control=True,
        metadata_access=True,
    )
    model = "scripted"

    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = []  # (prompt, seed, max_tokens)

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
        self.calls.append((prompt, seed, max_tokens))
        text = self._texts[len(self.calls) - 1]
        return Reply(
            text=text,
            tokens_in=50,
            tokens_out=5,
            stop_reason="stop",
            raw={},
        )

    def model_info(self):
        raise NotImplementedError("envelope probe never asks for model_info")


NO_FAILURES = {"prose": 0, "shape": 0, "refusal": 0}


@pytest.mark.parametrize("text", ["ALPHA 0", "ALPHA 0\n", "  ALPHA 0  "])
def test_exact_line_counts_valid(text):
    # Valid = reply, stripped, is exactly "{verb_i} {i}".
    backend = ScriptedBackend([text])
    result = probe_envelope(backend, ample_meter(), n=1)
    assert result == Envelope(fidelity=1.0, n=1, failures=NO_FAILURES)


def test_correct_line_wrapped_in_prose_counts_prose():
    backend = ScriptedBackend(
        ["Sure! Here is my reply:\nALPHA 0\nLet me know if you need more."]
    )
    result = probe_envelope(backend, ample_meter(), n=1)
    assert result.fidelity == 0.0
    assert result.failures == {"prose": 1, "shape": 0, "refusal": 0}


@pytest.mark.parametrize(
    "text",
    [
        "I'm Sorry, but I will not comply with that request.",
        "I can't help with that.",
        "As an assistant I cannot produce that output.",
    ],
)
def test_refusal_classified(text):
    # Refusal markers (case-insensitive) with no valid line anywhere.
    backend = ScriptedBackend([text])
    result = probe_envelope(backend, ample_meter(), n=1)
    assert result.fidelity == 0.0
    assert result.failures == {"prose": 0, "shape": 0, "refusal": 1}


def test_valid_line_beside_refusal_marker_counts_prose_not_refusal():
    # Plan Task 8: refusal requires NO valid line. A reply carrying both
    # a refusal marker and the exact expected line is extractable
    # chattiness — the prose-before-refusal precedence.
    backend = ScriptedBackend(["I'm sorry about earlier.\nALPHA 0"])
    result = probe_envelope(backend, ample_meter(), n=1)
    assert result.failures == {"prose": 1, "shape": 0, "refusal": 0}


def test_wrong_shape_without_markers_is_shape():
    backend = ScriptedBackend(["BANANA 99"])
    result = probe_envelope(backend, ample_meter(), n=1)
    assert result.failures == {"prose": 0, "shape": 1, "refusal": 0}


def test_zero_probes_is_none_not_zero():
    # None-vs-zero invariant: budget dies before the first call, so
    # fidelity was never measured — None, never 0.0.
    backend = ScriptedBackend([])
    meter = BudgetMeter(Budget(max_calls=0, max_prompt_tokens=1_000_000))
    result = probe_envelope(backend, meter, n=5)
    assert result.n == 0
    assert result.fidelity is None
    assert backend.calls == []


def test_fidelity_over_scripted_mix():
    # 7 valid / 3 mixed (one prose, one refusal, one shape) => 0.7 with
    # the right failure histogram.
    texts = [expected_line(i) for i in range(10)]
    texts[2] = f"Sure thing:\n{expected_line(2)}\nHope that helps!"
    texts[5] = "I'm sorry, I cannot do that."
    texts[8] = "FOO BAR"
    backend = ScriptedBackend(texts)
    result = probe_envelope(backend, ample_meter(), n=10)
    assert result.n == 10
    assert result.fidelity == pytest.approx(0.7)
    assert result.failures == {"prose": 1, "shape": 1, "refusal": 1}


def test_budget_death_mid_run_reports_partial():
    # Budget exhaustion mid-run: report what was measured, n = completed.
    backend = ScriptedBackend([expected_line(i) for i in range(10)])
    meter = BudgetMeter(Budget(max_calls=4, max_prompt_tokens=1_000_000))
    result = probe_envelope(backend, meter, n=10)
    assert result.n == 4
    assert result.fidelity == 1.0
    assert len(backend.calls) == 4


def test_prompt_is_exact_and_seeds_offset_from_base():
    # Pins the wire behavior: the exact prompt text, the verb cycle,
    # and seed = seed_base + i.
    backend = ScriptedBackend([expected_line(i) for i in range(4)])
    probe_envelope(backend, ample_meter(), n=4, seed_base=100)
    prompts = [call[0] for call in backend.calls]
    seeds = [call[1] for call in backend.calls]
    assert prompts[0] == (
        "Reply with exactly one line of the form: VERB ARG\n"
        "VERB must be one of: ALPHA, BRAVO, CHARLIE. ARG must be the number 0.\n"
        "Example of a valid reply: ALPHA 7\n"
        "Reply with the single line only. Verb to use: ALPHA"
    )
    assert seeds == [100, 101, 102, 103]
    assert prompts[1].endswith("Verb to use: BRAVO")
    assert "ARG must be the number 1." in prompts[1]
    assert prompts[2].endswith("Verb to use: CHARLIE")
    assert prompts[3].endswith("Verb to use: ALPHA")
