"""Tests for the empirical ceiling probe (spec §5, plan Task 7)."""

import random

import pytest

from assay.backends.base import BackendCaps, Reply
from assay.budget import Budget, BudgetMeter
from assay.ceiling import (
    Calibration,
    build_filler,
    calibrate,
    classify_call,
    probe_ceiling,
)
from assay.errors import BudgetExhausted, ContractViolation, InfrastructureError

CANARY = "ASSAY-0"


def _reply(
    text: str,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    stop_reason: str | None = None,
) -> Reply:
    return Reply(
        text=text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        stop_reason=stop_reason,
        raw={},
    )


class TestClassifyCall:
    """One test per row of the spec §5 decision table, in table order."""

    def test_contract_violation_is_missing_stats(self):
        signal = classify_call(
            ContractViolation("200 with no prompt_eval_count"),
            sent_est=1000,
            canary=CANARY,
            counts_available=True,
        )
        assert signal == "missing_stats"

    def test_other_infrastructure_error_is_hard_error(self):
        signal = classify_call(
            InfrastructureError("HTTP 500"),
            sent_est=1000,
            canary=CANARY,
            counts_available=True,
        )
        assert signal == "hard_error"

    def test_low_tokens_in_is_silent_truncation_even_with_canary(self):
        # The counts rule outranks the canary: a truncated count is a
        # daemon result even when the canary happens to survive.
        reply = _reply(f"{CANARY} fine", tokens_in=100, tokens_out=4, stop_reason="stop")
        signal = classify_call(
            reply, sent_est=1000, canary=CANARY, counts_available=True
        )
        assert signal == "silent_truncation"

    def test_absent_canary_with_full_counts_is_attention_loss(self):
        reply = _reply("I summarized instead.", tokens_in=1000, tokens_out=5, stop_reason="stop")
        signal = classify_call(
            reply, sent_est=1000, canary=CANARY, counts_available=True
        )
        assert signal == "attention_loss"

    def test_absent_canary_without_counts_is_canary_loss(self):
        reply = _reply("I summarized instead.")
        signal = classify_call(
            reply, sent_est=1000, canary=CANARY, counts_available=False
        )
        assert signal == "canary_loss"

    def test_canary_present_and_counts_fine_is_ok(self):
        reply = _reply(f"{CANARY} understood.", tokens_in=1000, tokens_out=3, stop_reason="stop")
        signal = classify_call(
            reply, sent_est=1000, canary=CANARY, counts_available=True
        )
        assert signal == "ok"

    def test_canary_present_without_counts_is_ok(self):
        reply = _reply(f"{CANARY} understood.")
        signal = classify_call(
            reply, sent_est=1000, canary=CANARY, counts_available=False
        )
        assert signal == "ok"

    def test_attention_loss_requires_counts_that_contradict_truncation(self):
        # Canary absent + counts ~ sent: the prompt demonstrably arrived
        # whole, so this is the MODEL dropping the instruction — never
        # the ambiguous canary_loss.
        reply = _reply("no canary here", tokens_in=990, tokens_out=4, stop_reason="stop")
        signal = classify_call(
            reply, sent_est=1000, canary=CANARY, counts_available=True
        )
        assert signal == "attention_loss"
        assert signal != "canary_loss"


def test_filler_is_seed_deterministic_and_sized():
    target = int(500 * 3.0)
    one = build_filler(random.Random(7), 500, 3.0)
    two = build_filler(random.Random(7), 500, 3.0)
    other = build_filler(random.Random(8), 500, 3.0)
    assert one == two
    assert one != other
    # Sized to est_tokens * chars_per_token, within one word of target.
    assert target <= len(one) <= target + 14
    # Naturalistic mixed words, not one repeated token.
    assert len(set(one.split())) > 25


# --- scripted fake backends for the ladder -------------------------------

CPT = 3.0  # the chars-per-token every fake and calibration below agree on
CAL = Calibration(chars_per_token=CPT, counts_available=True, deterministic=True)
CANARY_INSTRUCTION_PREFIX = "Begin your reply with exactly the word ASSAY-"


def _meter(calls: int = 10_000, tokens: int = 10**9) -> BudgetMeter:
    return BudgetMeter(Budget(max_calls=calls, max_prompt_tokens=tokens))


class _FakeBackend:
    caps = BackendCaps(
        reports_counts=True,
        per_request_ctx=True,
        truncate_control=True,
        metadata_access=True,
    )
    model = "fake-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def model_info(self):  # pragma: no cover - never used by the probe
        raise NotImplementedError


class FrontTruncatingBackend(_FakeBackend):
    """Silently keeps only the LAST `limit` prompt tokens, like Ollama.

    Front truncation eats the canary instruction; the reported
    tokens_in caps at the limit.
    """

    def __init__(self, limit: int = 4096) -> None:
        super().__init__()
        self.limit = limit

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None) -> Reply:
        self.prompts.append(prompt)
        sent = round(len(prompt) / CPT)
        kept = prompt[-int(self.limit * CPT):]
        word = f"ASSAY-{seed}"
        text = f"{word} understood." if word in kept else "understood."
        return Reply(
            text=text,
            tokens_in=min(sent, self.limit),
            tokens_out=4,
            stop_reason="stop",
            raw={},
        )


class MissingStatsBackend(_FakeBackend):
    """Reproduces the Ollama ~11.5k bug: stats-free 200 past the ceiling."""

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None) -> Reply:
        self.prompts.append(prompt)
        sent = round(len(prompt) / CPT)
        if sent > 11500:
            raise ContractViolation("200 with no prompt_eval_count")
        return Reply(
            text=f"ASSAY-{seed} ok",
            tokens_in=sent,
            tokens_out=3,
            stop_reason="stop",
            raw={},
        )


class HonestBackend(_FakeBackend):
    """Full counts, canary always honored."""

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None) -> Reply:
        self.prompts.append(prompt)
        sent = round(len(prompt) / CPT)
        return Reply(
            text=f"ASSAY-{seed} ok",
            tokens_in=sent,
            tokens_out=3,
            stop_reason="stop",
            raw={},
        )


class CanaryIgnoringBackend(_FakeBackend):
    """Honest counts, but the model never follows the canary instruction."""

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None) -> Reply:
        self.prompts.append(prompt)
        sent = round(len(prompt) / CPT)
        return Reply(
            text="I would prefer to summarize the passage instead.",
            tokens_in=sent,
            tokens_out=9,
            stop_reason="stop",
            raw={},
        )


class DeadBackend(_FakeBackend):
    """Every call is an infrastructure failure."""

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None) -> Reply:
        self.prompts.append(prompt)
        raise InfrastructureError("connection reset")


# --- ladder + bisection ---------------------------------------------------


def test_silent_truncation_fake_is_found_by_bisection():
    backend = FrontTruncatingBackend(limit=4096)
    ceiling = probe_ceiling(
        backend, _meter(), cap_tokens=16384, seeds=(0, 1), calibration=CAL
    )
    assert ceiling.failure_mode == "silent_truncation"
    assert ceiling.max_verified is not None
    assert ceiling.first_failure is not None
    # The 0.8 truncation rule makes a 4096-token truncator detectable
    # from 4096 / 0.8 = 5120 tokens up; bisection must land within 10%
    # of that boundary.
    boundary = 4096 / 0.8
    assert 4096 <= ceiling.max_verified <= boundary
    assert boundary < ceiling.first_failure <= boundary * 1.1
    # Bisection resolution invariant.
    assert ceiling.first_failure - ceiling.max_verified <= max(
        ceiling.max_verified // 10, 256
    )
    # The canary instruction rides at the FRONT so front-truncation
    # eats it — this pins the placement.
    assert backend.prompts
    assert all(p.startswith(CANARY_INSTRUCTION_PREFIX) for p in backend.prompts)


def test_missing_stats_fake_reproduces_the_ollama_bug():
    ceiling = probe_ceiling(
        MissingStatsBackend(), _meter(), cap_tokens=16384, seeds=(0, 1), calibration=CAL
    )
    assert ceiling.failure_mode == "missing_stats"
    assert ceiling.max_verified is not None
    assert ceiling.first_failure is not None
    assert ceiling.max_verified <= 11500 < ceiling.first_failure
    assert ceiling.counts_available is True


def test_honest_server_reports_none_up_to_cap():
    ceiling = probe_ceiling(
        HonestBackend(), _meter(), cap_tokens=16384, seeds=(0, 1), calibration=CAL
    )
    assert ceiling.failure_mode == "none_up_to_cap"
    assert ceiling.first_failure is None
    assert ceiling.max_verified == 16384


def test_budget_exhaustion_mid_ladder_reports_partial_not_raise():
    # 3 calls admitted: 1024 x both seeds, then 2048 seed 0; the 4th
    # charge dies. The probe must report what it verified, not raise.
    meter = BudgetMeter(Budget(max_calls=3, max_prompt_tokens=10**9))
    ceiling = probe_ceiling(
        HonestBackend(), meter, cap_tokens=16384, seeds=(0, 1), calibration=CAL
    )
    assert ceiling.max_verified == 1024
    assert ceiling.first_failure is None
    assert ceiling.failure_mode == "budget"
    assert sum(1 for e in ceiling.evidence if e.signal == "ok") == 3
    assert any(e.signal == "budget" for e in ceiling.evidence)


def test_budget_dead_before_any_measurement_raises():
    meter = BudgetMeter(Budget(max_calls=0, max_prompt_tokens=10**9))
    with pytest.raises(BudgetExhausted):
        probe_ceiling(
            HonestBackend(), meter, cap_tokens=16384, seeds=(0, 1), calibration=CAL
        )


def test_attention_loss_does_not_stop_the_ladder():
    ceiling = probe_ceiling(
        CanaryIgnoringBackend(), _meter(), cap_tokens=16384, seeds=(0, 1), calibration=CAL
    )
    assert ceiling.failure_mode == "none_up_to_cap"
    assert ceiling.first_failure is None
    assert ceiling.max_verified == 16384
    # Every rung was climbed despite the dropped canary...
    sizes = sorted({e.est_tokens for e in ceiling.evidence})
    assert sizes == [1024, 2048, 4096, 8192, 16384]
    # ...and the model result is on the record as evidence.
    assert any(e.signal == "attention_loss" for e in ceiling.evidence)


def test_first_rung_failure_yields_none_max_verified_not_zero():
    # None-vs-zero: nothing verified is None, never 0.
    ceiling = probe_ceiling(
        DeadBackend(), _meter(), cap_tokens=16384, seeds=(0, 1), calibration=CAL
    )
    assert ceiling.max_verified is None
    assert ceiling.first_failure is not None
    assert ceiling.first_failure <= 1024
    assert ceiling.failure_mode == "hard_error"


# --- calibration ----------------------------------------------------------


class DeterministicCountingBackend(_FakeBackend):
    """tokens_in = len(prompt) / 4; identical seeded calls, same text."""

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None) -> Reply:
        self.prompts.append(prompt)
        return Reply(
            text=f"ASSAY-{seed} ready",
            tokens_in=round(len(prompt) / 4.0),
            tokens_out=3,
            stop_reason="stop",
            raw={},
        )


class CountlessBackend(_FakeBackend):
    caps = BackendCaps(
        reports_counts=None,
        per_request_ctx=False,
        truncate_control=False,
        metadata_access=False,
    )

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None) -> Reply:
        self.prompts.append(prompt)
        return Reply(
            text="a steady reply",
            tokens_in=None,
            tokens_out=None,
            stop_reason=None,
            raw={},
        )


def test_calibrate_measures_chars_per_token_and_determinism():
    meter = _meter()
    cal = calibrate(DeterministicCountingBackend(), meter, seed=0)
    assert cal.counts_available is True
    assert cal.chars_per_token == pytest.approx(4.0, rel=0.01)
    assert cal.deterministic is True
    assert meter.spent.calls == 2  # one probe + one repeat, both charged


def test_calibrate_without_counts_reports_none_not_a_guess():
    cal = calibrate(CountlessBackend(), _meter(), seed=0)
    # None-vs-zero: unmeasured chars_per_token is None, never the 3.0
    # sizing fallback dressed up as a measurement.
    assert cal.chars_per_token is None
    assert cal.counts_available is False
    assert cal.deterministic is True
