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

    def test_absent_canary_and_count_free_reply_is_canary_loss_despite_calibration(self):
        # Calibration measured counts, but THIS reply carries none (an
        # OpenAI-compat server that reports usage on small prompts and
        # omits it while silently front-truncating large ones): nothing
        # verifies the rung — never "ok".
        reply = _reply("I summarized the passage instead.")
        signal = classify_call(
            reply, sent_est=8192, canary=CANARY, counts_available=True
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


class DefaultCtxDaemonBackend(_FakeBackend):
    """Real-Ollama shape: the flagship ~11.5k scenario (spec §12.1).

    The serving window is options.num_ctx when the request carries it,
    else the daemon DEFAULT (4096); the prompt is front-truncated into
    that window; and the genuine serving-path ceiling is ~11.5k
    (stats-free 200s past it). A ladder that never sends num_ctx
    measures the default-window knob (~5k silent_truncation) instead
    of the daemon ceiling.
    """

    DEFAULT_NUM_CTX = 4096
    CEILING = 11500

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[int, int | None]] = []  # (sent_est, num_ctx)

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None) -> Reply:
        self.prompts.append(prompt)
        sent = round(len(prompt) / CPT)
        self.requests.append((sent, num_ctx))
        if sent > self.CEILING:
            raise ContractViolation("200 with no prompt_eval_count")
        window = num_ctx if num_ctx is not None else self.DEFAULT_NUM_CTX
        kept = prompt[-int(window * CPT):]
        word = f"ASSAY-{seed}"
        text = f"{word} ok" if word in kept else "ok"
        return Reply(
            text=text,
            tokens_in=min(sent, window),
            tokens_out=3,
            stop_reason="stop",
            raw={},
        )


class MixedSignalBackend(_FakeBackend):
    """Past the threshold, the failing signal depends on the seed:
    listed seeds raise plain InfrastructureError (hard_error), all
    others raise ContractViolation (missing_stats)."""

    def __init__(self, hard_error_seeds=(0,), threshold=3000) -> None:
        super().__init__()
        self.hard_error_seeds = set(hard_error_seeds)
        self.threshold = threshold

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None) -> Reply:
        self.prompts.append(prompt)
        sent = round(len(prompt) / CPT)
        if sent > self.threshold:
            if seed in self.hard_error_seeds:
                raise InfrastructureError("HTTP 500")
            raise ContractViolation("200 with no prompt_eval_count")
        return Reply(
            text=f"ASSAY-{seed} ok",
            tokens_in=sent,
            tokens_out=3,
            stop_reason="stop",
            raw={},
        )


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
    # BOTH seeds really probed every rung — the 2-seed dimension of the
    # measurement (spec §5), not one seed asked twice.
    for size in (1024, 2048, 4096, 8192, 16384):
        assert {e.seed for e in ceiling.evidence if e.est_tokens == size} == {0, 1}


def test_ladder_widens_num_ctx_past_the_daemon_default():
    # Without per-request num_ctx the ladder measures the daemon's
    # DEFAULT window (4096 → silent_truncation near ~5k), not the
    # serving-path ceiling — the flagship ~11.5k missing_stats
    # measurement would be unreproducible (spec §12.1).
    backend = DefaultCtxDaemonBackend()
    ceiling = probe_ceiling(
        backend, _meter(), cap_tokens=16384, seeds=(0, 1), calibration=CAL
    )
    assert ceiling.failure_mode == "missing_stats"
    assert ceiling.max_verified is not None
    assert ceiling.max_verified <= 11500
    assert ceiling.first_failure is not None
    assert ceiling.first_failure > 11500
    # Every call asked for a serving window covering prompt + reply.
    assert backend.requests
    for sent, num_ctx in backend.requests:
        assert num_ctx is not None
        assert num_ctx >= sent + 32


def test_no_per_request_ctx_backend_is_never_sent_num_ctx():
    class NoCtxHonestBackend(HonestBackend):
        caps = BackendCaps(
            reports_counts=True,
            per_request_ctx=False,
            truncate_control=False,
            metadata_access=False,
        )

        def __init__(self) -> None:
            super().__init__()
            self.num_ctxs: list[int | None] = []

        def generate(self, prompt, *, seed, max_tokens, num_ctx=None) -> Reply:
            self.num_ctxs.append(num_ctx)
            return super().generate(prompt, seed=seed, max_tokens=max_tokens)

    backend = NoCtxHonestBackend()
    probe_ceiling(backend, _meter(), cap_tokens=4096, seeds=(0,), calibration=CAL)
    assert backend.num_ctxs
    assert all(num_ctx is None for num_ctx in backend.num_ctxs)


def test_failure_mode_is_the_majority_across_seeds():
    # seed 0 hard_error, seeds 1-2 missing_stats: the 2-1 majority wins
    # (plan Task 7 step 7) — not the first failing signal.
    ceiling = probe_ceiling(
        MixedSignalBackend(hard_error_seeds=(0,)),
        _meter(),
        cap_tokens=16384,
        seeds=(0, 1, 2),
        calibration=CAL,
    )
    assert ceiling.first_failure is not None
    assert ceiling.failure_mode == "missing_stats"


def test_failure_mode_tie_breaks_toward_table_order():
    # 1-1 tie between hard_error (seed 0) and missing_stats (seed 1):
    # the decision-table order puts missing_stats first.
    ceiling = probe_ceiling(
        MixedSignalBackend(hard_error_seeds=(0,)),
        _meter(),
        cap_tokens=16384,
        seeds=(0, 1),
        calibration=CAL,
    )
    assert ceiling.first_failure is not None
    assert ceiling.failure_mode == "missing_stats"


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


def test_calibrate_requests_a_covering_num_ctx_when_available():
    class RecordingBackend(DeterministicCountingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.num_ctxs: list[int | None] = []

        def generate(self, prompt, *, seed, max_tokens, num_ctx=None) -> Reply:
            self.num_ctxs.append(num_ctx)
            return super().generate(prompt, seed=seed, max_tokens=max_tokens)

    backend = RecordingBackend()
    calibrate(backend, _meter(), seed=0)
    assert len(backend.num_ctxs) == 2
    # ~500-token probe + reply headroom, on a per_request_ctx backend.
    assert all(num_ctx is not None and num_ctx >= 500 + 32 for num_ctx in backend.num_ctxs)


def test_calibrate_without_counts_reports_none_not_a_guess():
    cal = calibrate(CountlessBackend(), _meter(), seed=0)
    # None-vs-zero: unmeasured chars_per_token is None, never the 3.0
    # sizing fallback dressed up as a measurement.
    assert cal.chars_per_token is None
    assert cal.counts_available is False
    assert cal.deterministic is True
