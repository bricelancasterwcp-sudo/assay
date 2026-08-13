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
