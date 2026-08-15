"""v1.2 speed probes: server timings preferred, fallbacks named."""

from assay.backends.base import Reply
from assay.budget import Budget, BudgetMeter
from assay.speed import Speed, probe_speed, server_timings


def reply(raw=None, tokens_in=None, tokens_out=None):
    return Reply(text="x", tokens_in=tokens_in, tokens_out=tokens_out,
                 stop_reason="stop", raw=raw or {})


class SpeedFake:
    def __init__(self, raw=None, tokens_in=None, tokens_out=None):
        self._reply = reply(raw, tokens_in, tokens_out)
        self.calls = 0

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
        self.calls += 1
        return self._reply


def ticking_clock(step=2.0):
    state = {"t": 0.0}

    def clock():
        state["t"] += step
        return state["t"]

    return clock


OLLAMA_RAW = {
    "eval_count": 64, "eval_duration": 4_000_000_000,          # 16 tok/s
    "prompt_eval_count": 2048, "prompt_eval_duration": 2_000_000_000,  # 1024
}
LLAMA_SERVER_RAW = {
    "timings": {"predicted_per_second": 12.5, "prompt_per_second": 800.0}
}


class VaryingSpeedFake:
    """A backend whose per-call decode rate actually moves.

    A fake that answers every call identically cannot tell a probe that
    records each call apart from one that records the mean N times —
    the samples must carry the spread that made averaging necessary."""

    def __init__(self, decode_rates, prefill_rate=800.0):
        self._decode_rates = list(decode_rates)
        self._prefill_rate = prefill_rate
        self.decode_calls = 0
        self.calls = 0

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
        self.calls += 1
        if max_tokens == 1:  # the prefill probe: one token out
            return reply({"timings": {"prompt_per_second": self._prefill_rate}})
        rate = self._decode_rates[self.decode_calls % len(self._decode_rates)]
        self.decode_calls += 1
        return reply({"timings": {"predicted_per_second": rate}})


def meter():
    return BudgetMeter(Budget(max_calls=99, max_prompt_tokens=10**9))


# --- server_timings ---------------------------------------------------------

def test_ollama_nanosecond_timings_extracted():
    prefill, decode = server_timings(reply(OLLAMA_RAW))
    assert decode == 16.0
    assert prefill == 1024.0


def test_llama_server_precomputed_rates_extracted():
    prefill, decode = server_timings(reply(LLAMA_SERVER_RAW))
    assert decode == 12.5
    assert prefill == 800.0


def test_no_timings_yield_nones_never_guesses():
    assert server_timings(reply({})) == (None, None)


# --- probe_speed ------------------------------------------------------------

def test_server_timings_evidence_class():
    speed = probe_speed(SpeedFake(OLLAMA_RAW), meter(), calibration=None,
                        clock=ticking_clock())
    assert speed.decode_tps == 16.0
    assert speed.prefill_tps == 1024.0
    assert speed.evidence == "server_timings"
    assert speed.n_decode == 1 and speed.n_prefill == 1


def test_wall_clock_counts_fallback_uses_real_counts():
    # The ticking clock advances 1.0 per call, so each generate spans
    # exactly 1.0s: 64 tokens out -> 64 tok/s decode; 2048 in ->
    # 2048 tok/s prefill. Counts present, no server timings.
    fake = SpeedFake(raw={}, tokens_in=2048, tokens_out=64)
    speed = probe_speed(fake, meter(), calibration=None,
                        clock=ticking_clock(step=1.0))
    assert speed.decode_tps == 64.0
    assert speed.prefill_tps == 2048.0
    assert speed.evidence == "wall_clock_counts"


def test_count_free_backend_is_estimated_and_says_so():
    fake = SpeedFake(raw={}, tokens_in=None, tokens_out=None)
    speed = probe_speed(fake, meter(), calibration=None,
                        clock=ticking_clock(step=1.0))
    assert speed.decode_tps is not None
    assert speed.evidence == "wall_clock_estimated"


def test_budget_death_before_any_call_is_unmeasured_not_zero():
    dead = BudgetMeter(Budget(max_calls=0, max_prompt_tokens=1))
    speed = probe_speed(SpeedFake(OLLAMA_RAW), dead, calibration=None,
                        clock=ticking_clock())
    assert speed.decode_tps is None
    assert speed.prefill_tps is None
    assert speed.evidence == "unmeasured"
    assert isinstance(speed, Speed)


# --- per-call samples (v1.5) ------------------------------------------------

def test_decode_samples_record_every_call_not_only_their_mean():
    # A single mean cannot say whether the endpoint was steady or
    # lurching, and a diff of two means cannot say whether the gap is
    # bigger than the run-to-run noise. The samples are the evidence.
    fake = VaryingSpeedFake([10.0, 20.0, 60.0])
    speed = probe_speed(fake, meter(), calibration=None, decode_calls=3,
                        clock=ticking_clock())
    assert speed.decode_samples == (10.0, 20.0, 60.0)
    assert speed.n_decode == 3
    assert len(speed.decode_samples) == speed.n_decode
    assert abs(sum(speed.decode_samples) / len(speed.decode_samples)
               - speed.decode_tps) < 1e-9


def test_prefill_sample_is_a_one_tuple_never_a_bare_number():
    # One prefill call is still a sample: a 1-tuple says "measured
    # once", which is a different claim from "not recorded".
    fake = VaryingSpeedFake([10.0], prefill_rate=777.0)
    speed = probe_speed(fake, meter(), calibration=None, decode_calls=1,
                        clock=ticking_clock())
    assert speed.prefill_samples == (777.0,)
    assert len(speed.prefill_samples) == speed.n_prefill == 1


def test_samples_are_tuples_not_lists():
    # The profile is a frozen dataclass compared by equality after a
    # JSON round-trip; a list here would break that comparison.
    speed = probe_speed(VaryingSpeedFake([10.0, 30.0]), meter(),
                        calibration=None, decode_calls=2,
                        clock=ticking_clock())
    assert isinstance(speed.decode_samples, tuple)
    assert isinstance(speed.prefill_samples, tuple)


def test_wall_clock_samples_record_the_same_rates_that_were_averaged():
    # The fallback classes sample too — whatever rate was accepted into
    # the mean is the rate recorded.
    fake = SpeedFake(raw={}, tokens_in=2048, tokens_out=64)
    speed = probe_speed(fake, meter(), calibration=None, decode_calls=2,
                        clock=ticking_clock(step=1.0))
    assert speed.decode_samples == (64.0, 64.0)
    assert speed.prefill_samples == (2048.0,)


def test_unmeasured_probe_records_no_samples_but_not_none():
    # Zero samples is a MEASUREMENT ("we sampled, nothing landed"), and
    # None is its absence ("this probe predates sampling"). The dead
    # budget produced the former, so the tuple is empty, not None.
    dead = BudgetMeter(Budget(max_calls=0, max_prompt_tokens=1))
    speed = probe_speed(SpeedFake(OLLAMA_RAW), dead, calibration=None,
                        clock=ticking_clock())
    assert speed.decode_samples == ()
    assert speed.prefill_samples == ()


def test_old_speed_payload_still_parses_with_none_samples():
    # A v1.4 profile on disk has no samples at all. It must still
    # construct, and it must say "not recorded" — never an empty tuple
    # (which would claim a sampling run that never happened).
    payload = {"decode_tps": 66.0, "prefill_tps": 3765.0,
               "evidence": "server_timings", "n_decode": 1, "n_prefill": 1}
    speed = Speed(**payload)
    assert speed.decode_samples is None
    assert speed.prefill_samples is None
