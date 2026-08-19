"""v1.7 parallel degradation: k lanes, serialized detection, server-timed rates.

The instrument's two clocks are tested apart, because the module keeps
them apart: the BACKEND decides what came back (a timed reply, an
untimed one, or an exception) and the RUNNER decides the wall-clock
spans a client would have seen. Rates may only come from the first;
the serialized/parallel fact may only come from the second.
"""

import threading
from dataclasses import FrozenInstanceError

import pytest

from assay.backends.base import Reply
from assay.budget import Budget, BudgetMeter
from assay.errors import BudgetExhausted, InfrastructureError
from assay.parallel import (
    OVERLAP_FRACTION,
    OVERLAP_PROVENANCE,
    PARALLEL_SEED_BASE,
    Parallel,
    ParallelRow,
    classify_mode,
    probe_parallel,
)
from assay.speed import DECODE_MAX_TOKENS, DECODE_PROMPT

# --- doubles ----------------------------------------------------------------


def timed_reply(decode_tps: float) -> Reply:
    """A reply carrying the server's own decode rate."""
    return Reply(
        text="1\n2\n3\n4",
        tokens_in=20,
        tokens_out=4,
        stop_reason="stop",
        raw={"timings": {"predicted_per_second": decode_tps}},
    )


UNTIMED_REPLY = Reply(
    text="1\n2\n3\n4", tokens_in=20, tokens_out=4, stop_reason="stop", raw={}
)


class LaneFake:
    """Per-lane outcomes keyed by SEED, never by call order.

    Threads finish in whatever order the OS picks, so a fake scripted by
    arrival order would be deterministic under the synthetic runner and
    a coin flip under the real threaded one. The seed is the probe's own
    lane identity, so keying on it works under both.
    """

    def __init__(self, by_seed: dict, before_reply=None) -> None:
        self._by_seed = dict(by_seed)
        self._before_reply = before_reply
        self.requests: list[tuple[str, int, int]] = []
        self._lock = threading.Lock()

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
        with self._lock:
            self.requests.append((prompt, seed, max_tokens))
        if self._before_reply is not None:
            self._before_reply()
        if seed not in self._by_seed:
            raise AssertionError(f"lane fake got an unscripted seed: {seed}")
        outcome = self._by_seed[seed]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def seeds(self) -> list[int]:
        return sorted(seed for _, seed, _ in self.requests)


class PinnedSpans:
    """A runner that really calls each lane and pins its wall-clock span.

    One batch per k, in order. The length assertion is load-bearing: it
    pins that the probe builds exactly k lane callables for k.
    """

    def __init__(self, *batches: tuple[tuple[float, float], ...]) -> None:
        self._batches = list(batches)
        self.calls = 0

    def __call__(self, callables):
        assert self.calls < len(self._batches), (
            "runner called more times than the test scripted"
        )
        spans = self._batches[self.calls]
        self.calls += 1
        assert len(spans) == len(callables), (
            f"scripted {len(spans)} spans but the probe built "
            f"{len(callables)} lanes"
        )
        results = []
        for (start, end), call in zip(spans, callables):
            try:
                value = call()
            except Exception as exc:  # the runner's contract: reply_or_exc
                value = exc
            results.append((start, end, value))
        return results


def counting_clock(step: float = 1.0):
    """A thread-safe monotonic counter — this file's tests never read
    real time.

    Narrower than "the suite never reads real time" (M1, final fix
    wave, 2026-08-18): `tests/test_run.py`'s full-`probe()` pin has no
    seam to script the parallel family's lane spans through and paces
    its fake in real time instead, deliberately — see `parallel.py`'s
    module docstring and CARRIED-DEBT.md for why. Every test in THIS
    file stays clock-free by construction: synthetic spans in, a
    classified mode out, this clock counting ticks rather than
    sampling the wall.
    """
    state = {"t": 0.0}
    lock = threading.Lock()

    def clock() -> float:
        with lock:
            state["t"] += step
            return state["t"]

    return clock


def meter(calls: int = 99, tokens: int = 10**9) -> BudgetMeter:
    return BudgetMeter(Budget(max_calls=calls, max_prompt_tokens=tokens))


def two_lanes(*, spans, by_seed, baseline=60.0, meter_=None):
    """Run one k=2 probe over pinned spans and scripted lane replies."""
    return probe_parallel(
        LaneFake(by_seed),
        meter_ or meter(),
        baseline_decode_tps=baseline,
        ks=(2,),
        runner=PinnedSpans(spans),
    )


# --- (a) serialized vs parallel, decided by client clocks only --------------


def test_stacked_spans_classify_serialized():
    # Lane 2 begins exactly when lane 1 ends: the endpoint queued them.
    result = two_lanes(
        spans=((0.0, 1.0), (1.0, 2.0)),
        by_seed={1720: timed_reply(30.0), 1721: timed_reply(30.0)},
    )
    assert result.rows[0].mode == "serialized"


def test_overlapping_spans_classify_parallel():
    result = two_lanes(
        spans=((0.0, 1.0), (0.1, 1.1)),
        by_seed={1720: timed_reply(30.0), 1721: timed_reply(30.0)},
    )
    assert result.rows[0].mode == "parallel"


def test_overlap_exactly_at_the_fraction_is_still_serialized():
    # Pinned to the second decimal against OVERLAP_FRACTION = 0.25:
    # both spans are 1.0 s long, so the overlap (1.0 - 0.75 = 0.25 s)
    # is exactly 0.25 of the shorter span. The rule is strict `>`, so
    # exactly-at-the-fraction still reads serialized.
    result = two_lanes(
        spans=((0.0, 1.0), (0.75, 1.75)),
        by_seed={1720: timed_reply(30.0), 1721: timed_reply(30.0)},
    )
    # The behavior first, the constant second: a moved fraction must
    # break the CLASSIFICATION here, not merely a bookkeeping equality.
    assert result.rows[0].mode == "serialized"
    assert OVERLAP_FRACTION == 0.25  # the spans above are pinned to it


def test_one_hundredth_past_the_fraction_is_parallel():
    # The other side of the same boundary: the overlap is now
    # 1.0 - 0.74 = 0.26 of the 1.0 s span, just past the 0.25 fraction.
    result = two_lanes(
        spans=((0.0, 1.0), (0.74, 1.74)),
        by_seed={1720: timed_reply(30.0), 1721: timed_reply(30.0)},
    )
    assert result.rows[0].mode == "parallel"


def test_spans_are_sorted_by_start_before_classifying():
    # Lane order is arrival order, not time order; the later-starting
    # lane came back first here and the classification must not care.
    result = two_lanes(
        spans=((1.0, 2.0), (0.0, 1.0)),
        by_seed={1720: timed_reply(30.0), 1721: timed_reply(30.0)},
    )
    assert result.rows[0].mode == "serialized"


def test_concurrent_lanes_read_parallel_at_every_time_scale():
    """The defect this rule replaces was scale-dependence, so the test
    is scale-swept. Under the old absolute tolerance, lanes shorter
    than 0.25 s read `serialized` no matter how completely they
    overlapped — the rule was really "each lane must last longer than
    0.25 s", which is a statement about the endpoint's SPEED, not its
    scheduling. 0.222 s is not hypothetical: it is the pure-decode span
    of the fastest model on the committed matrix.
    """
    for duration in (0.05, 0.1, 0.222, 0.25, 0.3, 1.0):
        spans = [(0.0, duration), (0.0, duration)]
        assert classify_mode(spans) == "parallel", f"{duration}s lanes"


def test_serialized_lanes_read_serialized_at_every_time_scale():
    """The other half: the new rule must not buy concurrency detection
    by losing serialization detection."""
    for duration in (0.05, 0.222, 1.0):
        spans = [(0.0, duration), (duration, 2 * duration)]
        assert classify_mode(spans) == "serialized", f"{duration}s lanes"


def test_dispatch_skew_cannot_manufacture_a_parallel_reading():
    """The guard the old absolute tolerance existed to provide, kept.

    Two lanes that very nearly serialize — 2 ms of overlap on a 200 ms
    span — must still read `serialized`. That is 1% overlap against a
    25% floor. This test is load-bearing: it is the only thing standing
    between client-side skew and a false `parallel`.
    """
    assert classify_mode([(0.0, 0.200), (0.198, 0.398)]) == "serialized"


def test_a_zero_length_span_cannot_divide_by_itself():
    """A degenerate span makes any overlap infinite in ratio terms, and
    a pair that measured nothing must not be allowed to classify
    anything (CARRIED-DEBT I2, final fix wave, 2026-08-18).

    Before this fix the zero-length guard stopped the ratio computation
    but still fell through to the function's default return, so a pair
    of lanes with no measurable duration at all read `serialized` —
    and `_parallel_verdict`'s rule 1 has no way to tell that answer
    apart from a genuinely queued endpoint, so it published `risky` for
    a k that measured nothing. `classify_mode` already returns `None`
    for the below-two-lanes case on exactly this principle ("None,
    never the reassuring answer"); a degenerate span is the same
    nothing-was-measured fact one level in, and now gets the same
    honest answer. `_parallel_verdict` already routes `mode is None` to
    `unmeasured` (its rule 1), so this is a route-through fix, not a
    new consumer-side branch.
    """
    assert classify_mode([(0.0, 0.0), (0.0, 0.0)]) is None


def test_a_single_returned_lane_has_no_mode():
    # "Serialized" is a claim about two lanes' relationship; with one
    # lane there is nothing to relate, and None says so rather than
    # defaulting to the reassuring answer.
    result = probe_parallel(
        LaneFake({1710: timed_reply(30.0)}),
        meter(),
        baseline_decode_tps=60.0,
        ks=(1,),
        runner=PinnedSpans(((0.0, 1.0),)),
    )
    assert result.rows[0].mode is None
    assert result.rows[0].per_lane_decode_tps == 30.0


# --- (b) degradation arithmetic ---------------------------------------------


def test_degradation_ratio_is_per_lane_over_the_single_lane_baseline():
    result = two_lanes(
        spans=((0.0, 1.0), (0.1, 1.1)),
        by_seed={1720: timed_reply(30.0), 1721: timed_reply(30.0)},
        baseline=60.0,
    )
    row = result.rows[0]
    assert row.per_lane_decode_tps == 30.0
    # Every returned lane reported, so the total is a real total: the
    # sum of the lane rates, not the mean scaled by k.
    assert row.total_throughput_tps == 30.0 + 30.0
    assert row.degradation_ratio == 0.5
    assert row.n_lanes_ok == 2
    assert row.evidence == "server_timings"


def test_rates_come_from_server_timings_not_from_the_span_widths():
    # The spans say each lane took 1 second; the server says 30 tok/s.
    # A client-clock rate would be 64/1.0 = 64.0 (DECODE_MAX_TOKENS over
    # the span). The instrument must report what the server reported.
    result = two_lanes(
        spans=((0.0, 1.0), (0.1, 1.1)),
        by_seed={1720: timed_reply(30.0), 1721: timed_reply(30.0)},
    )
    assert result.rows[0].per_lane_decode_tps == 30.0


# --- (c) an errored lane is named, never a zero -----------------------------


def test_errored_lane_is_named_and_excluded_from_the_mean():
    # Three lanes at 30/30/60 mean 40.0. A fourth lane counted as 0.0
    # would mean 30.0 — the exact lie this test exists to catch.
    result = probe_parallel(
        LaneFake(
            {
                1740: timed_reply(30.0),
                1741: InfrastructureError("transport failure: connection reset"),
                1742: timed_reply(30.0),
                1743: timed_reply(60.0),
            }
        ),
        meter(),
        baseline_decode_tps=60.0,
        ks=(4,),
        runner=PinnedSpans(((0.0, 1.0), (0.1, 0.2), (0.1, 1.1), (0.1, 1.1))),
    )
    row = result.rows[0]
    assert row.per_lane_decode_tps == 40.0
    # An errored lane never "returned", so it does not block the total
    # the way a timing-free reply does: all three lanes that came back
    # reported, and their sum is a total of what came back. (It is also
    # not per_lane * k, which would be 160.0.)
    assert row.total_throughput_tps == 30.0 + 30.0 + 60.0
    assert row.degradation_ratio == pytest.approx(40.0 / 60.0)
    assert row.n_lanes_ok == 3
    assert len(row.lane_errors) == 1
    assert "InfrastructureError" in row.lane_errors[0]
    assert "connection reset" in row.lane_errors[0]
    assert "lane 1" in row.lane_errors[0]


# --- (d) every lane errored: unmeasured, never zero -------------------------


def test_all_lanes_errored_reports_nothing_measured_never_zero():
    result = two_lanes(
        spans=((0.0, 1.0), (0.1, 1.1)),
        by_seed={
            1720: InfrastructureError("transport failure: connection refused"),
            1721: InfrastructureError("transport failure: connection refused"),
        },
    )
    row = result.rows[0]
    assert row.per_lane_decode_tps is None
    assert row.total_throughput_tps is None
    assert row.degradation_ratio is None
    assert row.mode is None
    assert row.n_lanes_ok == 0
    assert len(row.lane_errors) == 2
    assert row.evidence == "unmeasured"


# --- (e) missing server timings -> named, and no client-clock guess ---------


def test_missing_server_timings_names_the_class_and_reports_no_rate():
    result = two_lanes(
        spans=((0.0, 1.0), (0.1, 1.1)),
        by_seed={1720: UNTIMED_REPLY, 1721: UNTIMED_REPLY},
    )
    row = result.rows[0]
    assert row.evidence == "unmeasured"
    assert row.per_lane_decode_tps is None
    assert row.total_throughput_tps is None
    assert row.degradation_ratio is None
    # The lanes came back — they just came back without timings. That is
    # a different fact from an errored lane, and n_lanes_ok keeps them
    # apart: two healthy replies, no rate.
    assert row.n_lanes_ok == 2
    assert row.lane_errors == ()
    # The scheduling fact survives: client clocks are allowed to decide
    # this one thing even when no rate can be read.
    assert row.mode == "parallel"


def test_row_evidence_names_the_weakest_class_among_lanes():
    result = two_lanes(
        spans=((0.0, 1.0), (0.1, 1.1)),
        by_seed={1720: timed_reply(30.0), 1721: UNTIMED_REPLY},
    )
    row = result.rows[0]
    assert row.evidence == "unmeasured"
    # One lane reported; the mean is over that lane alone, not over two.
    # A mean over a subset is still a per-lane rate, so it survives with
    # its evidence class beside it.
    assert row.per_lane_decode_tps == 30.0
    assert row.n_lanes_ok == 2
    # The TOTAL does not survive. A sum over one of two returned lanes
    # is not a total of anything: 30.0 here would be a field named
    # "total throughput" reporting half the endpoint's aggregate, and no
    # reader could tell that from a genuinely halved endpoint. Nor may
    # it be extrapolated to per_lane * k (60.0) — that is a guess, not a
    # measurement. Unmeasurable totals are None.
    assert row.total_throughput_tps is None


# --- (f) the meter: all-or-nothing per k ------------------------------------


def test_meter_refusal_at_k4_keeps_the_k2_row_and_skips_k4():
    tight = meter(calls=2)
    runner = PinnedSpans(((0.0, 1.0), (0.1, 1.1)))
    result = probe_parallel(
        LaneFake({1720: timed_reply(30.0), 1721: timed_reply(30.0)}),
        tight,
        baseline_decode_tps=60.0,
        ks=(2, 4),
        runner=runner,
    )
    assert [row.k for row in result.rows] == [2]
    # ...and the k that did NOT run is NAMED. An absent row cannot say
    # whether k=4 was never asked for, was refused by the meter, or was
    # simply not in this run's ks — three different findings that read
    # identically as "one row". The measured row survives beside it.
    assert result.skipped == ("k=4: budget refused 4 concurrent lanes",)
    assert result.rows[0].per_lane_decode_tps == 30.0
    # Nothing was charged for the k the probe never ran.
    assert tight.spent.calls == 2
    assert runner.calls == 1


def test_prompt_token_limit_refuses_a_k_all_or_nothing():
    per_lane_tokens = max(1, len(DECODE_PROMPT) // 3)
    tight = meter(calls=99, tokens=per_lane_tokens * 3)  # room for 3 lanes
    result = probe_parallel(
        LaneFake({1720: timed_reply(30.0), 1721: timed_reply(30.0)}),
        tight,
        baseline_decode_tps=60.0,
        ks=(2, 4),
        runner=PinnedSpans(((0.0, 1.0), (0.1, 1.1))),
    )
    assert [row.k for row in result.rows] == [2]
    # k=4 needed four lanes and only one lane's worth was left: the
    # partial charge is never made, so the third lane's tokens stay
    # unspent rather than paying for lanes that never launched.
    assert tight.spent.prompt_tokens == per_lane_tokens * 2
    # The token meter refuses by the same rule and names the k the same
    # way: a skipped k is a budget finding whichever meter stopped it.
    assert result.skipped == ("k=4: budget refused 4 concurrent lanes",)


def test_a_clock_that_trips_between_ks_keeps_the_k2_row_and_names_k4():
    # The wall-clock ceiling (v1.7) is a THIRD limit, and this family's
    # affordability question has to consult it: a k=4 admitted on a
    # closed clock would charge four lanes, launch them, and then die
    # mid-k — recording calls for lanes that never came back and
    # discarding the k=2 row that was already measured. Refused BEFORE
    # any lane launches, named like any other refused k.
    fake = LaneFake({1720: timed_reply(30.0), 1721: timed_reply(30.0)})
    # Time passes when the endpoint is CALLED: five seconds per lane, so
    # k=2's two lanes put the run past a five-second ceiling.
    tight = BudgetMeter(
        Budget(max_calls=99, max_prompt_tokens=10**9, max_seconds=5.0),
        clock=lambda: len(fake.requests) * 5.0,
    )
    runner = PinnedSpans(((0.0, 1.0), (0.1, 1.1)))

    result = probe_parallel(
        fake, tight, baseline_decode_tps=60.0, ks=(2, 4), runner=runner,
    )

    assert [row.k for row in result.rows] == [2]
    assert result.rows[0].per_lane_decode_tps == 30.0
    assert result.skipped == ("k=4: budget refused 4 concurrent lanes",)
    # Two lanes launched and two lanes were charged: the clock closed the
    # meter between the ks, never in the middle of one.
    assert tight.spent.calls == 2
    assert len(fake.requests) == 2
    assert runner.calls == 1


def test_budget_exhausted_before_any_k_propagates():
    dead = meter(calls=1)
    with pytest.raises(BudgetExhausted):
        probe_parallel(
            LaneFake({1720: timed_reply(30.0), 1721: timed_reply(30.0)}),
            dead,
            baseline_decode_tps=60.0,
            ks=(2, 4),
            runner=PinnedSpans(((0.0, 1.0), (0.1, 1.1))),
        )
    assert dead.spent.calls == 0


def test_probe_parallel_refuses_a_missing_baseline_before_spending():
    """CARRIED-DEBT item 18. `degradation_ratio` divides by the
    baseline, so a None reached the division as a TypeError AFTER the
    lanes were charged and run — the caller paid for a measurement it
    could not express. The family's own contract says the baseline is
    the caller's obligation; a contract worth stating is worth
    enforcing before the spend.

    The lanes here REPORT timings deliberately: with timing-free lanes
    `per_lane` is None, which short-circuits the comparison and hides
    the defect behind a silent `degradation_ratio=None`.
    """
    seeds = {PARALLEL_SEED_BASE + 2 * 10 + lane: timed_reply(50.0)
             for lane in range(2)}
    m = meter()
    with pytest.raises(ValueError, match="baseline"):
        probe_parallel(LaneFake(seeds), m, baseline_decode_tps=None, ks=(2,),
                       runner=PinnedSpans([(0.0, 1.0), (0.0, 1.0)]))
    assert m.spent.calls == 0, "the guard must fire before any lane is charged"


# --- the request shape mirrors the serial speed probe -----------------------


def test_lanes_send_the_speed_probe_request_with_pinned_per_lane_seeds():
    fake = LaneFake({1720: timed_reply(30.0), 1721: timed_reply(30.0)})
    probe_parallel(
        fake,
        meter(),
        baseline_decode_tps=60.0,
        ks=(2,),
        runner=PinnedSpans(((0.0, 1.0), (0.1, 1.1))),
    )
    assert fake.seeds == [1720, 1721]  # seed_base + k*10 + lane
    for prompt, _, max_tokens in fake.requests:
        assert prompt == DECODE_PROMPT
        assert max_tokens == DECODE_MAX_TOKENS


def test_seed_base_is_injectable_and_lanes_stay_distinguishable():
    fake = LaneFake({2540: timed_reply(30.0), 2541: timed_reply(30.0),
                     2542: timed_reply(30.0), 2543: timed_reply(30.0)})
    probe_parallel(
        fake,
        meter(),
        baseline_decode_tps=60.0,
        ks=(4,),
        seed_base=2500,
        runner=PinnedSpans(((0.0, 1.0),) * 4),
    )
    assert fake.seeds == [2540, 2541, 2542, 2543]


# --- the envelope carries its provisional tolerance -------------------------


def test_result_carries_the_fraction_and_its_chosen_provenance():
    result = two_lanes(
        spans=((0.0, 1.0), (0.1, 1.1)),
        by_seed={1720: timed_reply(30.0), 1721: timed_reply(30.0)},
        baseline=60.0,
    )
    assert isinstance(result, Parallel)
    assert isinstance(result.rows[0], ParallelRow)
    assert isinstance(result.rows, tuple)
    assert result.baseline_decode_tps == 60.0
    # Every k the run asked for ran, so nothing is named skipped. The
    # empty tuple is the measured claim "no k was refused" — the field
    # is never absent, because a reader must be able to tell that from a
    # run whose budget quietly dropped a k.
    assert result.skipped == ()
    # v10: a new run records the FRACTION it classified under, never a
    # seconds tolerance — that field pair is v9-and-earlier only, and a
    # live run leaves it None rather than populating a field whose name
    # would misdescribe what was recorded.
    assert result.overlap_fraction == OVERLAP_FRACTION
    # A CHOSEN threshold travels with the fact that it was chosen, so a
    # reader never mistakes it for a derived one.
    assert result.overlap_provenance == OVERLAP_PROVENANCE == "chosen-2026-08-18"
    assert result.tolerance_s is None
    assert result.tolerance_provenance is None


def test_a_new_run_records_the_fraction_and_not_a_seconds_tolerance():
    """The rename is the point: a fraction stored in a field named
    seconds would be a false claim, and — worse — `assay diff` would
    compare `tolerance_s: 0.25` against `tolerance_s: 0.25`, find them
    byte-equal, and report no change across a break where the whole
    classification rule changed.
    """
    result = two_lanes(
        spans=((0.0, 1.0), (0.0, 1.0)),
        by_seed={1720: timed_reply(30.0), 1721: timed_reply(30.0)},
    )
    assert result.overlap_fraction == OVERLAP_FRACTION == 0.25
    assert result.overlap_provenance == OVERLAP_PROVENANCE == "chosen-2026-08-18"
    assert result.tolerance_s is None
    assert result.tolerance_provenance is None


def test_rows_are_frozen():
    row = two_lanes(
        spans=((0.0, 1.0), (0.1, 1.1)),
        by_seed={1720: timed_reply(30.0), 1721: timed_reply(30.0)},
    ).rows[0]
    with pytest.raises(FrozenInstanceError):
        row.mode = "serialized"


# --- the default runner really runs threads ---------------------------------


def test_default_threaded_runner_launches_lanes_concurrently():
    # No injected runner: this exercises the module's own threading
    # default against an in-process fake (no sockets). The barrier is
    # the proof — a lane cannot pass it until BOTH lanes are inside
    # generate(), so a runner that ran lanes one after another would
    # deadlock until the timeout and surface as a lane error.
    barrier = threading.Barrier(2, timeout=10)
    fake = LaneFake(
        {1720: timed_reply(30.0), 1721: timed_reply(10.0)},
        before_reply=barrier.wait,
    )
    result = probe_parallel(
        fake,
        meter(),
        baseline_decode_tps=60.0,
        ks=(2,),
        clock=counting_clock(),
    )
    row = result.rows[0]
    assert row.lane_errors == ()
    assert row.n_lanes_ok == 2
    assert row.per_lane_decode_tps == 20.0
    assert row.total_throughput_tps == 40.0
    # Both starts are clocked before either lane clears the barrier, so
    # the counting clock hands out {1,2} to the starts and {3,4} to the
    # ends: overlapping spans, deterministically.
    assert row.mode == "parallel"
    assert fake.seeds == [1720, 1721]


def test_no_lane_vanishes_from_the_threaded_runner():
    # A BaseException escaping a worker thread would leave that lane
    # with no result at all, and the row would report "one lane came
    # back" — a failure that shrank the look without ever being named.
    fake = LaneFake({1720: SystemExit("lane killed"), 1721: timed_reply(30.0)})
    result = probe_parallel(
        fake, meter(), baseline_decode_tps=60.0, ks=(2,), clock=counting_clock()
    )
    row = result.rows[0]
    assert row.n_lanes_ok == 1
    assert len(row.lane_errors) == 1
    assert "SystemExit" in row.lane_errors[0]
    assert row.per_lane_decode_tps == 30.0
