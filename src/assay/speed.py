"""Speed probes: decode and prefill throughput (v1.2).

Two numbers, two audiences. **decode tok/s** is chat usability — the
rate a person watches tokens appear. **prefill tok/s** is agent
usability — an agent loop re-reads long context constantly, so its
wall-clock is dominated by prompt processing, not generation. A model
can be a fine agent engine and an annoying chatbot, or the reverse;
one number would hide that.

Measurement prefers the SERVER'S OWN TIMINGS over wall clock, and the
evidence class is always named (the ceiling probe's honesty pattern):

- ``server_timings`` — extracted from the reply body. Ollama native
  reports ``eval_count``/``eval_duration`` and ``prompt_eval_count``/
  ``prompt_eval_duration`` (nanoseconds); llama-server's OpenAI-compat
  replies carry ``timings.predicted_per_second`` /
  ``timings.prompt_per_second``.
- ``wall_clock_counts`` — no server timings, but the reply carries real
  token counts: rate = counts / measured elapsed. Decode uses a short
  prompt (prefill negligible); prefill uses ``max_tokens=1`` (decode
  negligible). Cruder, stated.
- ``wall_clock_estimated`` — no timings AND no counts: token totals
  fall back to the calibration's chars-per-token estimate. The weakest
  class, named so a consumer can discount it.

Every accepted per-call rate is kept beside its mean
(``decode_samples`` / ``prefill_samples``, v1.5). Two runs of the same
model differ by some amount of run-to-run noise; a reader comparing
their means cannot tell a real regression from that noise unless the
samples the means came from are on the page.

The clock is injectable; the suite never depends on real time.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

from assay.backends.base import Backend, Reply
from assay.budget import BudgetMeter
from assay.ceiling import Calibration, build_filler
from assay.errors import BudgetExhausted

DECODE_MAX_TOKENS = 64
DECODE_PROMPT = (
    "Count upward from one, one number per line, and keep counting "
    "until you are stopped."
)
PREFILL_EST_TOKENS = 2048
_SPEED_SEED = 900


@dataclass(frozen=True)
class Speed:
    decode_tps: float | None    # None = unmeasured, never fake zero
    prefill_tps: float | None
    evidence: str               # server_timings | wall_clock_counts |
                                # wall_clock_estimated | mixed
    n_decode: int
    n_prefill: int
    # Every per-call rate that went into the mean above (v1.5). A diff
    # of two runs cannot say whether a gap is real without knowing the
    # run-to-run spread, and a mean of three calls hides it entirely.
    # None-vs-empty is load-bearing: None = this probe predates
    # sampling (a pre-v1.5 profile reloaded from disk), () = sampling
    # ran and accepted nothing, and a 1-tuple = measured exactly once.
    decode_samples: tuple[float, ...] | None = None
    prefill_samples: tuple[float, ...] | None = None


def server_timings(reply: Reply) -> tuple[float | None, float | None]:
    """(prefill_tps, decode_tps) from the server's own report, or Nones.

    Only rates the body actually supports are returned — a reply with
    generation timings but no prompt timings yields (None, decode)."""
    raw = reply.raw or {}
    prefill = decode = None
    # Ollama native: counts + nanosecond durations.
    if raw.get("eval_count") and raw.get("eval_duration"):
        decode = raw["eval_count"] / (raw["eval_duration"] / 1e9)
    if raw.get("prompt_eval_count") and raw.get("prompt_eval_duration"):
        prefill = raw["prompt_eval_count"] / (raw["prompt_eval_duration"] / 1e9)
    # llama-server OpenAI-compat: precomputed rates.
    timings = raw.get("timings") or {}
    if decode is None and timings.get("predicted_per_second"):
        decode = float(timings["predicted_per_second"])
    if prefill is None and timings.get("prompt_per_second"):
        prefill = float(timings["prompt_per_second"])
    return prefill, decode


def _rate_from_wall_clock(
    tokens: int | None, elapsed: float, estimated_tokens: float,
) -> tuple[float | None, str]:
    if elapsed <= 0:
        return None, "wall_clock_counts"
    if tokens is not None:
        return tokens / elapsed, "wall_clock_counts"
    if estimated_tokens > 0:
        return estimated_tokens / elapsed, "wall_clock_estimated"
    return None, "wall_clock_estimated"


def probe_speed(
    backend: Backend,
    meter: BudgetMeter,
    *,
    calibration: Calibration | None,
    decode_calls: int = 1,
    prefill_est_tokens: int = PREFILL_EST_TOKENS,
    clock: Callable[[], float] = time.monotonic,
) -> Speed:
    """Measure decode and prefill throughput within the budget.

    Budget exhaustion mid-probe reports what was measured; a family
    with zero completed calls yields None rates (unmeasured, named by
    the caller in ``dropped``). Infrastructure errors propagate."""
    chars_per_token = 3.0
    if calibration is not None and calibration.chars_per_token is not None:
        chars_per_token = calibration.chars_per_token

    decode_rates: list[float] = []
    prefill_rates: list[float] = []
    evidence: set[str] = set()

    for i in range(decode_calls):
        try:
            meter.charge(max(1, len(DECODE_PROMPT) // 3))
        except BudgetExhausted:
            break
        start = clock()
        reply = backend.generate(
            DECODE_PROMPT, seed=_SPEED_SEED + i, max_tokens=DECODE_MAX_TOKENS
        )
        elapsed = clock() - start
        _, decode = server_timings(reply)
        if decode is not None:
            decode_rates.append(decode)
            evidence.add("server_timings")
        else:
            rate, kind = _rate_from_wall_clock(
                reply.tokens_out, elapsed, DECODE_MAX_TOKENS
            )
            if rate is not None:
                decode_rates.append(rate)
                evidence.add(kind)

    prefill_prompt = build_filler(
        random.Random(_SPEED_SEED), prefill_est_tokens, chars_per_token
    )
    try:
        meter.charge(prefill_est_tokens)
        start = clock()
        reply = backend.generate(
            prefill_prompt, seed=_SPEED_SEED, max_tokens=1
        )
        elapsed = clock() - start
        prefill, _ = server_timings(reply)
        if prefill is not None:
            prefill_rates.append(prefill)
            evidence.add("server_timings")
        else:
            rate, kind = _rate_from_wall_clock(
                reply.tokens_in, elapsed, prefill_est_tokens
            )
            if rate is not None:
                prefill_rates.append(rate)
                evidence.add(kind)
    except BudgetExhausted:
        pass

    return Speed(
        decode_tps=(sum(decode_rates) / len(decode_rates)
                    if decode_rates else None),
        prefill_tps=(sum(prefill_rates) / len(prefill_rates)
                     if prefill_rates else None),
        evidence=("mixed" if len(evidence) > 1
                  else next(iter(evidence)) if evidence else "unmeasured"),
        n_decode=len(decode_rates),
        n_prefill=len(prefill_rates),
        # The samples ARE the rates that were averaged — same list, not
        # a re-derivation that could drift from the reported mean.
        decode_samples=tuple(decode_rates),
        prefill_samples=tuple(prefill_rates),
    )
