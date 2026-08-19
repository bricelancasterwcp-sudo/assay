"""Parallel degradation: what k concurrent requests do to one endpoint (v1.7).

The agent-fleet question. A profile measured one request at a time says
nothing about a box running four agents, and the two failure shapes
look nothing alike: an endpoint that BATCHES shares its throughput
(each lane slower, total roughly held), while one that QUEUES simply
serializes (each lane's latency multiplied by k). For a consumer that
is the headline, so the family reports it as `mode`.

Verdict as of v1.8 (`profile._parallel_verdict`): the worst measured k
decides, a `serialized` k caps at risky, and a refused k leaves the
cell unmeasured. Three honesty rules hold the MEASUREMENT up.

**Rates come from the server, never from the client's clock.** Lane
throughput is read out of the reply body by ``assay.speed``'s
extractor, so the number means the same thing it means in the serial
speed family — same source, same fallback vocabulary. Under
concurrency a client-side rate would be a measurement of the
scheduler, not of the model, so when a lane reports no timings its
rate is None and the row's ``evidence`` names the weakest class any
lane produced. (``wall_clock_counts`` / ``wall_clock_estimated`` are
in the vocabulary and unreachable here BY DESIGN: this family never
derives a rate from a span.)

**Client clocks decide exactly one thing: the scheduling fact.**
Whether the lanes' wall-clock spans stacked or overlapped is an
observation about arrival and completion, not a rate, so it is the one
place the caller's clock is trusted. The overlap tolerance is a CHOSEN
constant, and it travels with ``TOLERANCE_PROVENANCE`` saying so — a
threshold should be derived from measurement, and one that is not must
be flagged until the campaign's live rows can sanity-check it.

**An errored lane is infrastructure evidence, never a zero.** It is
named in ``lane_errors`` and dropped from the means; a zero would
silently drag a mean toward "degraded" and report an unreachable
endpoint as a slow one. When every lane errors the row carries None
rates and no mode — nothing was measured, and 0.0 is a measurement.

The clock and the concurrency runner are both injectable; the suite
never depends on real time, and the default threaded runner is
exercised against an in-process fake rather than a socket.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from assay.backends.base import Backend, Reply
from assay.budget import BudgetMeter
from assay.errors import BudgetExhausted
from assay.speed import DECODE_MAX_TOKENS, DECODE_PROMPT, server_timings

OVERLAP_TOLERANCE_S = 0.25  # CHOSEN, not derived — sanity-checked by
                            # the campaign's live rows (spec §3)
TOLERANCE_PROVENANCE = "chosen-2026-08-17"

DEFAULT_KS = (2, 4)
PARALLEL_SEED_BASE = 1700
_SEED_STRIDE = 10  # seed = seed_base + k*10 + lane: k-block, lane-offset

# Weakest first. The row reports the WEAKEST class any lane produced,
# not a "mixed" summary: a consumer discounting the number needs to
# know how bad the worst lane's evidence was, and one timing-free lane
# is enough to make the row's mean partial.
_EVIDENCE_WEAKEST_FIRST = (
    "unmeasured",
    "wall_clock_estimated",
    "wall_clock_counts",
    "server_timings",
)

LaneResult = tuple[float, float, "Reply | BaseException"]
Runner = Callable[[list[Callable[[], Reply]]], list[LaneResult]]


@dataclass(frozen=True)
class ParallelRow:
    k: int
    per_lane_decode_tps: float | None   # None = no lane reported timings
    # None unless EVERY returned lane reported timings — a partial total
    # is not a total. When they all reported it is the sum over them;
    # errored lanes never "returned" and are named in lane_errors.
    total_throughput_tps: float | None
    degradation_ratio: float | None     # per-lane / single-lane baseline
    mode: str | None                    # "parallel" | "serialized" | None
    n_lanes_ok: int                     # lanes that returned a reply at all
    lane_errors: tuple[str, ...]        # named infra failures, never zeros
    evidence: str                       # weakest timing class among lanes


@dataclass(frozen=True)
class Parallel:
    rows: tuple[ParallelRow, ...]
    baseline_decode_tps: float
    tolerance_s: float                  # == OVERLAP_TOLERANCE_S as run
    tolerance_provenance: str           # == TOLERANCE_PROVENANCE
    # The k values the meter refused, NAMED — ``LongOutput.skipped``'s
    # rule applied here. An absent row is silent about why it is absent:
    # "only k=2 was asked for", "k=4 was refused by the budget" and "this
    # run used different ks" all read as one row and mean three different
    # things. Defaulted, so a document written before the list existed
    # parses as the honest "no k was named skipped".
    skipped: tuple[str, ...] = ()


def _threaded_runner(clock: Callable[[], float]) -> Runner:
    """The default seam: one stdlib thread per lane, clocked at the edges.

    ``clock()`` is read immediately before and after the lane's call, so
    the span covers the request and nothing else. A raising lane is
    RETURNED, not propagated — the runner's contract is
    ``(start, end, reply_or_exc)`` and the probe, not the runner, is
    where a failure becomes named evidence. The end clock is read on
    that path too: a lane that failed still occupied the endpoint for
    some span.

    ``BaseException`` and not ``Exception``: an escaping BaseException
    would kill the thread with its slot never filled, and a lane that
    VANISHES reads downstream as "k-1 lanes came back" — an unnamed
    failure quietly shrinking the look. Every lane leaves a result.
    """

    def run(callables: list[Callable[[], Reply]]) -> list[LaneResult]:
        results: list[LaneResult | None] = [None] * len(callables)

        def lane(index: int, call: Callable[[], Reply]) -> None:
            start = clock()
            try:
                outcome: Reply | BaseException = call()
            except BaseException as exc:
                outcome = exc
            results[index] = (start, clock(), outcome)

        threads = [
            threading.Thread(target=lane, args=(index, call), daemon=True)
            for index, call in enumerate(callables)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # Narrowing, not a data path: every lane above fills its slot.
        return [result for result in results if result is not None]

    return run


def _lane_call(backend: Backend, seed: int) -> Callable[[], Reply]:
    """One lane's request: the serial decode probe, with this lane's seed.

    Deliberately identical to what ``assay.speed`` sends serially —
    same prompt, same ``max_tokens`` — so the parallel numbers are
    comparable with the baseline they are divided by. Only the seed
    differs, so lanes stay distinguishable in a transcript.
    """

    def call() -> Reply:
        return backend.generate(
            DECODE_PROMPT, seed=seed, max_tokens=DECODE_MAX_TOKENS
        )

    return call


def _affordable(meter: BudgetMeter, lanes: int, prompt_tokens: int) -> bool:
    """True when ALL `lanes` calls can be charged — asked before any is.

    All-or-nothing on purpose: charging lane by lane until one is
    refused would record calls for lanes that never launched, and
    ``spent`` is the run's evidence of what the box was actually asked
    to do. Half a k is not a smaller k, either — three lanes of a
    four-lane look would answer a question nobody asked.

    The question goes to the METER rather than being re-derived here
    (v1.7). A hand-rolled comparison against two budget fields knows
    nothing about the third: once ``max_seconds`` existed, this function
    admitted a k the clock had already closed, charged its lanes, and
    let ``charge`` raise in the middle of the k — which recorded calls
    for lanes that never launched and threw away the k the run HAD
    measured. One batch question, all three limits.
    """
    return not meter.would_exceed_n(lanes, lanes * prompt_tokens)


def classify_mode(
    spans: list[tuple[float, float]], fraction: float = OVERLAP_TOLERANCE_S
) -> str | None:
    """"serialized" if the lanes' spans stacked, "parallel" if they overlapped.

    Sorted by start; parallel iff some consecutive pair overlaps by
    more than `fraction` of the SHORTER of the two spans.

    The fraction is relative on purpose. An absolute seconds tolerance
    subtracted from a duration-dependent comparison can only be correct
    at one time scale: for lanes launched together, overlap is
    approximately the lane duration, so "overlap > 0.25 s" reduces to
    "each lane lasted longer than 0.25 s" — a statement about the
    endpoint's SPEED, not its scheduling. A fast endpoint would read
    `serialized` while serving every lane at once.

    What the fraction still absorbs is the client-side skew between "the
    previous lane's reply finished arriving" and "the next lane's
    request went out". That skew is milliseconds, and 25% of a span is
    far above it at any scale — two lanes that nearly serialize overlap
    by ~1% and correctly stay `serialized`.

    None below two lanes: "serialized" is a statement about how two
    lanes relate, and with one lane there is nothing to relate. None,
    never the reassuring answer.
    """
    if len(spans) < 2:
        return None
    ordered = sorted(spans)
    for (prev_start, prev_end), (next_start, next_end) in zip(ordered, ordered[1:]):
        overlap = min(prev_end, next_end) - max(prev_start, next_start)
        shortest = min(prev_end - prev_start, next_end - next_start)
        # A zero-length span makes any overlap infinite in ratio terms;
        # it measured nothing, so it classifies nothing.
        if shortest > 0 and overlap > fraction * shortest:
            return "parallel"
    return "serialized"


def _weakest(classes: list[str]) -> str:
    if not classes:
        return "unmeasured"
    return min(classes, key=_EVIDENCE_WEAKEST_FIRST.index)


def _row(
    k: int,
    results: list[LaneResult],
    baseline_decode_tps: float,
    fraction: float,
) -> ParallelRow:
    """One k's lanes, folded into a row without conflating the failures.

    Three states stay separate all the way through: a lane that raised
    (named, excluded, not counted ok), a lane that replied without
    timings (counted ok, excluded from the mean, named in `evidence`),
    and a lane that replied with timings (counted everywhere).

    The mean and the total take that middle state differently, on
    purpose. A MEAN over the reporting lanes is still a per-lane rate —
    speed.py's honest partial, with its evidence class beside it. A SUM
    over a subset is not a total of anything: a k=4 look with two
    timing-free lanes would put roughly half the endpoint's aggregate
    under a field named "total throughput", and no consumer reading a
    number could tell that from a genuinely halved endpoint. So the
    total is None unless every returned lane reported.
    """
    rates: list[float] = []
    classes: list[str] = []
    errors: list[str] = []
    spans: list[tuple[float, float]] = []

    for lane, (start, end, outcome) in enumerate(results):
        if isinstance(outcome, BaseException):
            errors.append(f"lane {lane}: {type(outcome).__name__}: {outcome}")
            continue
        spans.append((start, end))
        _, decode = server_timings(outcome)
        if decode is None:
            classes.append("unmeasured")
        else:
            rates.append(decode)
            classes.append("server_timings")

    per_lane = sum(rates) / len(rates) if rates else None
    every_returned_lane_reported = bool(rates) and len(rates) == len(spans)
    return ParallelRow(
        k=k,
        per_lane_decode_tps=per_lane,
        total_throughput_tps=(
            sum(rates) if every_returned_lane_reported else None
        ),
        degradation_ratio=(
            per_lane / baseline_decode_tps
            if per_lane is not None and baseline_decode_tps > 0
            else None
        ),
        mode=classify_mode(spans, fraction),
        n_lanes_ok=len(spans),
        lane_errors=tuple(errors),
        evidence=_weakest(classes),
    )


def probe_parallel(
    backend: Backend,
    meter: BudgetMeter,
    *,
    baseline_decode_tps: float,
    ks: tuple[int, ...] = DEFAULT_KS,
    seed_base: int = PARALLEL_SEED_BASE,
    clock: Callable[[], float] = time.monotonic,
    runner: Runner | None = None,
) -> Parallel:
    """Send k concurrent decode requests per k, and report what happened.

    The baseline is the caller's obligation: `degradation_ratio` is
    against the SAME RUN's single-lane ``speed.decode_tps``, and the
    family never runs without it (the orchestrator drops it by name
    instead of dividing by a number from somewhere else).

    Budget: each k is charged in full before any of its lanes launches.
    A k the meter refuses is skipped — its row is absent rather than
    empty, because a k that never ran measured nothing — and NAMED in
    ``skipped``, because an absent row cannot say whether the k was
    refused or never asked for. Refusal before ANY k has run propagates
    ``BudgetExhausted`` so the caller drops the whole family by name.
    """
    if baseline_decode_tps is None:
        # Before the spend, not after: the ratio's divisor is the
        # caller's obligation and a lane charged against a baseline
        # that does not exist measures something nobody can express.
        raise ValueError(
            "probe_parallel needs a single-lane baseline_decode_tps from "
            "the same run; the orchestrator drops the family by name "
            "when speed is unmeasured")
    run = runner if runner is not None else _threaded_runner(clock)
    lane_tokens = max(1, len(DECODE_PROMPT) // 3)
    rows: list[ParallelRow] = []
    skipped: list[str] = []

    for k in ks:
        if not _affordable(meter, k, lane_tokens):
            if rows:
                # This k is skipped and SAID SO; what was measured stands.
                skipped.append(f"k={k}: budget refused {k} concurrent lanes")
                continue
            raise BudgetExhausted(
                f"parallel: k={k} needs {k} calls / {k * lane_tokens} prompt "
                f"tokens and the budget refused before any lane ran; spent so "
                f"far: {meter.spent.calls} calls / "
                f"{meter.spent.prompt_tokens} prompt tokens"
            )
        for _ in range(k):
            meter.charge(lane_tokens)
        results = run(
            [
                _lane_call(backend, seed_base + k * _SEED_STRIDE + lane)
                for lane in range(k)
            ]
        )
        rows.append(_row(k, results, baseline_decode_tps, OVERLAP_TOLERANCE_S))

    return Parallel(
        rows=tuple(rows),
        baseline_decode_tps=baseline_decode_tps,
        tolerance_s=OVERLAP_TOLERANCE_S,
        tolerance_provenance=TOLERANCE_PROVENANCE,
        skipped=tuple(skipped),
    )
