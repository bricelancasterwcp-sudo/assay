"""Probe orchestrator (spec §9, plan Task 11).

Order: detect/build backend (recorded when asked) -> model_info ->
geometry (pure, no model calls) -> calibrate -> ceiling -> envelope ->
codecs -> loop -> speed -> parallel -> long_output -> tools -> verdicts
-> Profile.

Budget discipline: ``budget`` has NO default — a library consumer
burning a user's GPU time must say how much (spec §9). Any probe
family hitting BudgetExhausted stops the remaining families; every
unfinished family is None and named in ``provenance.dropped`` — never
partial-pretending-complete. If the budget dies before ANY family
completed, BudgetExhausted propagates (the CLI maps it to exit 2).
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from assay import __version__
from assay.backends import Backend, detect_backend
from assay.backends.base import PROBE_TEMPERATURE
from assay.backends.ollama import OllamaNative
from assay.backends.openai_compat import OpenAICompat
from assay.budget import Budget, BudgetMeter
from assay.ceiling import (CALIBRATION_CALLS, Calibration, Ceiling,
                           ShapeCeiling, calibrate, ladder_steps,
                           probe_ceiling, probe_fixed_shapes,
                           shape_probe_sizes)
from assay.codecs import (CodecDirectives, DEFAULT_PRESENTATION, GRADES_FOR,
                          JSON_CODECS, Landing, PATCH_CODECS, cell_attempts,
                          probe_codecs, stopped_on_rule)
from assay.fixtures import FIXTURE_SET
from assay.long_output import RUNGS, LongOutput, probe_long_output
from assay.loop import Loop, error_turn_prompts, probe_loop, turn_prompts
from assay.envelope import probe_envelope
from assay.errors import BudgetExhausted
from assay.geometry import free_vram_mib, plan_window
from assay.parallel import DEFAULT_KS, Parallel, probe_parallel
from assay.profile import (PROFILE_VERSION, Profile, best_patch_cell,
                           compute_verdicts, verdict_cell)
from assay.replay import CallRecorder
from assay.speed import Speed, probe_speed
from assay.stats import LOOK_SCHEDULE, stopping_rule_name
from assay.tools import (TOOLS_LOOK_SCHEDULE, TURNS_PER_TASK, Tools,
                         probe_tools, task_cap)


_QUICK_CEILING_CAP = 16384
_FULL_CEILING_CAP = 32768


@dataclass(frozen=True)
class ModeParams:
    seeds: tuple[int, ...]
    envelope_n: int
    codecs_n_per_cell: int
    loop_runs: int
    shape_probes: tuple[int, ...]
    # The tallest rung the ladder may reach in this mode, before the
    # model's training_ctx and the user's --window-cap narrow it further
    # (``ceiling_cap_for``). Lives on the params so the cost table can
    # price the family from the params alone.
    ceiling_cap: int
    # None = fixed-n codec sampling; a schedule = sequential looks, and
    # then codecs_n_per_cell is the cap the schedule already carries.
    codec_look_schedule: tuple[int, ...] | None
    # None = the v1 five-task prefix at fixed n; a schedule = looks over
    # the twenty-task pool (v1.7). Quick stays fixed so its numbers keep
    # comparing across the version boundary.
    tools_look_schedule: tuple[int, ...] | None
    speed_decode_calls: int


MODE_PARAMS = {
    "quick": ModeParams(seeds=(0,), envelope_n=10, codecs_n_per_cell=5,
                        loop_runs=3, shape_probes=(2048, 4096, 8192),
                        ceiling_cap=_QUICK_CEILING_CAP,
                        codec_look_schedule=None, tools_look_schedule=None,
                        speed_decode_calls=1),
    # v1.5: the default mode samples codec cells SEQUENTIALLY — every
    # cell is examined at 5/10/20/35 and stops at the first look whose
    # Wilson-95 interval decides a rung. A decided cell costs what it
    # costs (an unusable one settles at 5); an undecided one runs to 35,
    # the smallest n at which a perfect cell clears `ready`
    # non-provisionally (Wilson lower on 35/35 is 0.9011 against the 0.9
    # floor). That is why the honest mode is now affordable enough to be
    # the default: the old fixed n=10 bought verdicts that were almost
    # always provisional, and the old thorough spent 315 calls to buy
    # certainty this rule usually reaches for far less.
    "full": ModeParams(seeds=(0, 1), envelope_n=30, codecs_n_per_cell=35,
                       loop_runs=5, shape_probes=(2048, 4096, 8192),
                       ceiling_cap=_FULL_CEILING_CAP,
                       codec_look_schedule=LOOK_SCHEDULE,
                       tools_look_schedule=TOOLS_LOOK_SCHEDULE,
                       speed_decode_calls=3),
    # thorough is an ALIAS of full (v1.5): its whole point was buying a
    # decidable `ready` at n=35, which is exactly the sequential cap.
    # Kept as its own key so the documented --thorough flag still
    # parses and old invocations keep working.
    "thorough": ModeParams(seeds=(0, 1), envelope_n=30, codecs_n_per_cell=35,
                           loop_runs=5, shape_probes=(2048, 4096, 8192),
                           ceiling_cap=_FULL_CEILING_CAP,
                           codec_look_schedule=LOOK_SCHEDULE,
                           tools_look_schedule=TOOLS_LOOK_SCHEDULE,
                           speed_decode_calls=3),
}


def ceiling_cap_for(
    mode: str, training_ctx: int | None, window_cap: int | None
) -> int:
    """The ladder cap: mode table, then the user's cap if tighter."""
    # thorough shares full's cap: its extra budget buys codec samples,
    # not a taller ladder. An unrecognised mode is priced as full, which
    # is what this function has always done.
    params = MODE_PARAMS.get(mode, MODE_PARAMS["full"])
    cap = params.ceiling_cap
    if mode != "quick":
        cap = min(training_ctx or cap, cap)
    if window_cap is not None:
        cap = min(cap, window_cap)
    return cap


def _codec_half_calls(codecs: tuple[str, ...], params: ModeParams) -> int:
    """What one half of the codec matrix costs, cell by cell."""
    return sum(
        cell_attempts(codec, grade, n_per_cell=params.codecs_n_per_cell,
                      look_schedule=params.codec_look_schedule)
        for codec in codecs
        for grade in GRADES_FOR[codec]
    )


#: family -> what it costs when NOTHING decides early: every probe runs,
#: every cell reaches its cap, every task is scored. Each entry DERIVES
#: from the constants the probe itself consumes — the schedules, the task
#: pools, the fixture counts, the script lengths, the ladder's own step
#: enumeration — so a family that grows re-prices itself here instead of
#: waiting for a hand count to be corrected after a mid-family death.
#:
#: The codec matrix is split into its two halves because that is how a
#: consumer under a call ceiling buys it: `structured_extraction` and
#: `patch_editing` are separate verdicts and separate preflights.
#:
#: The ceiling entry is the LADDER: every rung at every seed, which is
#: what a run that finds no failure spends. A ladder that DOES fail stops
#: early and bisects instead, and the bisection's own worst case
#: (``ceiling.bisection_worst_case_steps`` x seeds — 8 calls in full
#: mode, 4 in quick) is the run-level headroom the default budgets carry
#: rather than a per-family reservation, because no run pays for both
#: tails. Those two numbers are the derivation, not an estimate of it:
#: the budget block above ``cli.DEFAULT_BUDGETS`` quotes THEM (552 + 8 of
#: 610 full, 117 + 4 of 130 quick), and tests/test_run.py sums this table
#: against the metered runs.
_WORST_CASE = {
    # Geometry is arithmetic over model_info plus a VRAM reading: it asks
    # the model nothing, and 0 is a measurement, not a placeholder.
    "geometry": lambda p: 0,
    "calibration": lambda p: CALIBRATION_CALLS,
    "ceiling": lambda p: len(ladder_steps(p.ceiling_cap)) * len(p.seeds),
    "ceiling_shapes": lambda p: sum(len(shape_probe_sizes(shape))
                                    for shape in p.shape_probes),
    "envelope": lambda p: p.envelope_n,
    "codecs-json": lambda p: _codec_half_calls(JSON_CODECS, p),
    "codecs-patch": lambda p: _codec_half_calls(PATCH_CODECS, p),
    # The decode calls, plus probe_speed's single prefill probe.
    "speed": lambda p: p.speed_decode_calls + 1,
    # Every run plays both scripts: the golden turns and the error turns.
    "loop": lambda p: p.loop_runs * (len(turn_prompts())
                                     + len(error_turn_prompts())),
    "long_output": lambda p: len(RUNGS),
    "tools": lambda p: task_cap(p.tools_look_schedule) * TURNS_PER_TASK,
    # Charged in full before either k launches; quick names the family
    # instead of running it, which is the caller's decision, not a cost.
    "parallel": lambda p: sum(DEFAULT_KS),
}


def worst_case_calls(family: str, params: ModeParams) -> int:
    """The calls ``family`` can cost in this mode — declared, never guessed.

    For a consumer deciding whether a family FITS before starting it: what
    starts, finishes, so the number has to bound the whole family rather
    than describe a typical run. Families that stop early (a decided codec
    cell, a tools pool that settles at the first look, a rung above the
    measured ceiling) cost less; none costs more.

    An unknown family raises ``KeyError``. There is no default cost: a
    guessed number is how a run starts something it cannot pay for.
    """
    return _WORST_CASE[family](params)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _endpoint_kind(live: Backend, forced: str | None) -> str:
    if forced is not None:
        return forced
    if isinstance(live, OllamaNative):
        return "ollama"
    if isinstance(live, OpenAICompat):
        return "openai"
    return "injected"  # a test override: named honestly, never guessed


def _calibration_payload(calibration: Calibration | None) -> dict | None:
    if calibration is None:
        return None
    return {
        "chars_per_token": calibration.chars_per_token,
        "counts_available": calibration.counts_available,
        "deterministic": calibration.deterministic,
    }


#: The name of the rule that ended each codec cell's sampling. The
#: renderer itself moved to ``assay.stats`` in v1.7, when the tools
#: family began naming its own schedule: one function, two families, so
#: the string in a codec lens and the string in a tools lens cannot
#: drift apart. Kept under the old name because that is what the codec
#: call site reads.
_stopping_rule = stopping_rule_name


def _codec_n_used(
    codecs: dict[str, dict[str, Landing]] | None
) -> dict[str, int]:
    """The n each codec verdict was actually computed from.

    Read from the SAME cells compute_verdicts grades, so the lens can
    never quote an n belonging to a cell no verdict used. An unmeasured
    cell gets NO entry (spec §8 None-vs-zero): ``n_used: 0`` would read
    as a verdict graded on zero samples.
    """
    used: dict[str, int] = {}
    json_cell = verdict_cell(codecs, "json_object")
    if json_cell is not None and json_cell.n > 0:
        used["structured_extraction"] = json_cell.n
    patch_cell = best_patch_cell(codecs)
    if patch_cell is not None and patch_cell.n > 0:
        used["patch_editing"] = patch_cell.n
    return used


def _codecs_were_cut_off(
    codecs: dict[str, dict[str, Landing]], params: ModeParams
) -> bool:
    """Did the meter end the codec matrix, or did the matrix end itself?

    Fixed n: any cell short of n_per_cell was cut off. Sequential: a
    short cell is the stopping rule working, so each cell is asked
    whether IT ended on the rule (``codecs.stopped_on_rule``).
    """
    cells = ((codec, cell)
             for codec, grades in codecs.items()
             for cell in grades.values())
    if params.codec_look_schedule is None:
        return any(cell.n < params.codecs_n_per_cell for _, cell in cells)
    return any(not stopped_on_rule(codec, cell, params.codec_look_schedule)
               for codec, cell in cells)


def _read_geometry(
    active: Backend,
    info,
    calibration: Calibration | None,
    window_cap: int | None,
    dropped: list[str],
):
    """Plan the window from the SERVING state, and name it if it cannot be.

    Geometry reads AFTER calibration's first live call (v1.1, live
    validation finding 3): a cold model's weights are not yet resident,
    so a pre-load VRAM reading double-counts them and reports
    usable_window 0. Post-calibration, model_info's `loaded` flag and
    the VRAM reading describe the serving state the probes actually ran
    against. If calibration never ran, the pre-load info is used and the
    weaker evidence is what it honestly is.

    Shared by both orchestrators: the rule and its reason belong to the
    measurement, not to one mode's call order.
    """
    geometry_info = active.model_info() if calibration is not None else info
    geometry = plan_window(
        geometry_info, vram_free_mib=free_vram_mib(), user_cap=window_cap
    )
    if geometry is None:
        dropped.append(
            "geometry: kv arithmetic or training_ctx unavailable "
            f"(source={geometry_info.source})"
        )
    return geometry


def _spend_payload(budget: Budget, meter: BudgetMeter) -> dict:
    """What was granted and what was spent — seconds only when metered.

    ``spent.seconds`` is 0.0 on a run with no ``max_seconds``, and that
    zero is a FLOOR rather than a measurement: the meter never asked the
    clock what time it is (assay.budget), so publishing it would report
    a run that took no time. The key exists exactly when a ceiling was
    granted — which is also exactly when the meter watched the clock, so
    a run that was watched but never charged still reports its (zero)
    reading rather than going silent about a ceiling it was held to.
    """
    granted = {
        "max_calls": budget.max_calls,
        "max_prompt_tokens": budget.max_prompt_tokens,
    }
    spent = {
        "calls": meter.spent.calls,
        "prompt_tokens": meter.spent.prompt_tokens,
    }
    if budget.max_seconds is not None:
        granted["max_seconds"] = budget.max_seconds
        spent["seconds"] = meter.spent.seconds
    return {"budget": granted, "spent": spent}


def _finish_profile(
    *,
    base_url: str,
    kind: str,
    autodetected: bool,
    info,
    mode: str,
    params: ModeParams,
    budget: Budget,
    meter: BudgetMeter,
    started: str,
    directives: CodecDirectives | None,
    tier: str | None,
    emulated: bool | None,
    calibration: Calibration | None,
    geometry,
    ceiling: Ceiling | None,
    ceiling_shapes: tuple[ShapeCeiling, ...] | None,
    envelope,
    codecs: dict[str, dict[str, Landing]] | None,
    speed: Speed | None,
    loop: Loop | None,
    long_output: LongOutput | None,
    tools: Tools | None,
    parallel: Parallel | None,
    dropped: list[str],
    budget_death: BudgetExhausted | None,
) -> Profile:
    """Assemble the Profile both orchestrators return — or refuse to.

    The exit-2 guard first: a run whose budget died before ANY family
    completed measured nothing, and a document full of Nones is not a
    result the caller should be handed as one. The condition asks about
    every family rather than the four v1 ones, and the two readings
    agree on ``probe``'s order (a family after the ceiling cannot be
    measured while the ceiling is None — a dead meter skips all of
    them). Under budget mode's priority they do NOT agree: speed can be
    the only family a tiny budget bought, and discarding it because the
    ceiling never fit would throw away the run's only measurement.
    """
    measured = (geometry, ceiling, ceiling_shapes, envelope, codecs, speed,
                loop, long_output, tools, parallel)
    if budget_death is not None and all(value is None for value in measured):
        raise budget_death  # nothing completed: the caller must know (exit 2)

    presentation = "custom" if directives is not None else DEFAULT_PRESENTATION
    return Profile(
        assay_profile_version=PROFILE_VERSION,
        probe_version=__version__,
        endpoint={
            "kind": kind,
            "base_url": base_url,
            "autodetected": autodetected,
        },
        model={
            "name": info.name,
            "quant": info.quant,
            "weights_bytes": info.weights_bytes,
            "training_ctx": info.training_ctx,
        },
        geometry=geometry,
        ceiling=ceiling,
        ceiling_shapes=ceiling_shapes,
        envelope=envelope,
        codecs=codecs,
        speed=speed,
        loop=loop,
        long_output=long_output,
        tools=tools,
        # No verdict argument: parallel is measurement-only in v1.7.
        parallel=parallel,
        verdicts=compute_verdicts(
            geometry, ceiling, envelope, codecs, speed, loop, long_output,
            tools,
            presentation=presentation,
            stopping_rule=_stopping_rule(params.codec_look_schedule),
            n_used=_codec_n_used(codecs),
        ),
        provenance={
            "started": started,
            "finished": _utc_now(),
            "mode": mode,
            "tier": tier,
            "emulated": emulated,
            "presentation": presentation,
            "fixture_set": FIXTURE_SET,
            "temperature": PROBE_TEMPERATURE,
            "thinking": "disabled",
            "seeds": list(params.seeds),
            **_spend_payload(budget, meter),
            "calibration": _calibration_payload(calibration),
            "dropped": dropped,
        },
    )


def probe(
    base_url: str,
    model: str,
    *,
    budget: Budget,
    mode: str = "quick",
    backend: str | None = None,
    record: Path | None = None,
    window_cap: int | None = None,
    directives: CodecDirectives | None = None,
    tier: str | None = None,
    emulated: bool | None = None,
    _backend_override: Backend | None = None,
    _clock: Callable[[], float] | None = None,
) -> Profile:
    """Run the full probe suite against one endpoint; return a Profile.

    ``backend`` forces the kind ("ollama" | "openai"); else auto-detect.
    ``record`` wraps the backend in a CallRecorder writing a transcript.
    ``window_cap`` is the user's context cap: it bounds both geometry's
    user_cap term and the ceiling ladder (spec §5).
    ``directives`` substitutes the consumer's own codec presentation for
    the built-in one; the profile's provenance and verdict lenses record
    which was used (landing is a property of the instrument, v1.1).
    """
    if mode not in MODE_PARAMS:
        raise ValueError(
            f"unknown mode: {mode!r} (expected 'quick', 'full', or 'thorough')")
    if tier is not None and emulated is None:
        # The marking rule (ruled 2026-08-13): a tier-labelled profile
        # must say whether the hardware was emulated. No default — an
        # unmarked emulated number could masquerade as real hardware.
        raise ValueError("a declared tier requires an explicit "
                         "emulated=True/False")
    params = MODE_PARAMS[mode]
    started = _utc_now()

    autodetected = backend is None and _backend_override is None
    live = (
        _backend_override
        if _backend_override is not None
        else detect_backend(base_url, model, forced=backend)
    )
    kind = _endpoint_kind(live, backend)
    active: Backend = CallRecorder(live, Path(record)) if record is not None else live
    # ``_clock`` is the meter's clock seam, injected only by tests: a
    # wall-clock ceiling that could only be exercised against real time
    # would be pinned by a sleeping suite, which is no pin at all.
    meter = (BudgetMeter(budget) if _clock is None
             else BudgetMeter(budget, clock=_clock))
    dropped: list[str] = []

    info = active.model_info()  # InfrastructureError propagates (spec §3)

    calibration: Calibration | None = None
    ceiling: Ceiling | None = None
    envelope = None
    codecs = None
    budget_death: BudgetExhausted | None = None

    try:
        calibration = calibrate(active, meter, seed=params.seeds[0])
    except BudgetExhausted as exc:
        budget_death = exc

    geometry = _read_geometry(active, info, calibration, window_cap, dropped)

    try:
        if budget_death is not None:
            raise budget_death
        ceiling = probe_ceiling(
            active,
            meter,
            cap_tokens=ceiling_cap_for(mode, info.training_ctx, window_cap),
            seeds=params.seeds,
            calibration=calibration,
        )
    except BudgetExhausted as exc:
        budget_death = exc
    if ceiling is None:
        dropped.append("ceiling: budget exhausted before any ladder call")
        if calibration is None:
            dropped.append("calibration: budget exhausted before completion")
    elif any(entry.signal == "budget" for entry in ceiling.evidence):
        # The partial ceiling is kept (it reports what it verified), but
        # the meter is dry: no further family may start.
        budget_death = BudgetExhausted("budget exhausted during the ceiling ladder")
    ceiling_shapes: tuple[ShapeCeiling, ...] | None = None
    if budget_death is None and active.caps.per_request_ctx:
        ceiling_shapes = probe_fixed_shapes(
            active, meter, calibration=calibration,
            shapes=params.shape_probes)
        if all(s.failure_mode == "unmeasured" for s in ceiling_shapes):
            ceiling_shapes = None
            dropped.append("ceiling_shapes: budget exhausted before any "
                           "shape probe completed")
    elif not active.caps.per_request_ctx:
        dropped.append("ceiling_shapes: per_request_ctx unavailable — "
                       "fixed request shapes cannot be pinned")
    else:
        dropped.append("ceiling_shapes: skipped, budget exhausted earlier")

    if ceiling is not None and not active.caps.per_request_ctx:
        # The evidence class is weaker and must be stated (spec §5/§11):
        # without options.num_ctx the ladder measured the server's own
        # configured context window, not a per-probe widened one.
        dropped.append(
            "ceiling: per_request_ctx unavailable — ladder measured the "
            "server's configured context window"
        )

    if budget_death is not None:
        dropped.append("envelope: skipped, budget exhausted earlier")
    else:
        envelope = probe_envelope(active, meter, n=params.envelope_n)
        if envelope.n < params.envelope_n:
            budget_death = BudgetExhausted("budget exhausted during envelope probes")
        if envelope.n == 0:
            # Zero completed probes is no measurement at all: None, never
            # a fidelity that looks measured (spec §8).
            envelope = None
            dropped.append("envelope: budget exhausted before any probe completed")

    speed: Speed | None = None
    loop: Loop | None = None
    long_output: LongOutput | None = None
    tools: Tools | None = None
    parallel: Parallel | None = None

    if budget_death is not None:
        dropped.append("codecs: skipped, budget exhausted earlier")
    else:
        codecs = probe_codecs(active, meter,
                              n_per_cell=params.codecs_n_per_cell,
                              directives=directives,
                              look_schedule=params.codec_look_schedule)
        if all(
            cell.n == 0 for grades in codecs.values() for cell in grades.values()
        ):
            codecs = None
            dropped.append("codecs: budget exhausted before any probe completed")
        else:
            # Spec §8 None rule: every UNMEASURED cell (n == 0) is named
            # in dropped — a Landing(None, 0) with an empty dropped list
            # would hide the budget death from consumers. Cells with
            # n > 0 are measurements at their honest n and stay.
            for codec, grades in codecs.items():
                for grade, cell in grades.items():
                    if cell.n == 0:
                        dropped.append(
                            f"codecs: {codec}.{grade} budget exhausted "
                            "before any probe completed"
                        )
            if _codecs_were_cut_off(codecs, params):
                budget_death = BudgetExhausted("budget exhausted during codec probes")

    if budget_death is not None:
        dropped.append("loop: skipped, budget exhausted earlier")
    else:
        loop = probe_loop(active, meter, runs=params.loop_runs)
        if loop.n_turns == 0:
            loop = None
            dropped.append("loop: budget exhausted before any turn completed")

    if budget_death is not None:
        dropped.append("speed: skipped, budget exhausted earlier")
    else:
        speed = probe_speed(active, meter, calibration=calibration,
                            decode_calls=params.speed_decode_calls)
        if speed.n_decode == 0 and speed.n_prefill == 0:
            speed = None
            dropped.append("speed: budget exhausted before any probe completed")

    # parallel (v1.7) runs IMMEDIATELY after speed, and full mode only.
    # Both halves are measurement decisions, not scheduling taste. The
    # ratio's denominator is this run's own single-lane decode rate, so
    # the family cannot start before speed has one and must not start at
    # all without one — a degradation ratio against a baseline from
    # another run is a comparison of two boxes wearing one number. And
    # quick is the mode an operator reaches for in a hurry: six lanes buy
    # a concurrency finding it did not ask for, so quick NAMES the family
    # instead of running it.
    if budget_death is not None:
        dropped.append("parallel: skipped, budget exhausted earlier")
    elif mode == "quick":
        dropped.append("parallel: quick mode — full mode measures concurrency")
    elif speed is None or speed.decode_tps is None:
        dropped.append("parallel: no single-lane baseline (speed unmeasured)")
    else:
        try:
            parallel = probe_parallel(active, meter,
                                      baseline_decode_tps=speed.decode_tps)
        except BudgetExhausted as exc:
            # Raised only when the meter refused the FIRST k, so nothing
            # was measured: the family is None, named, and the meter is
            # dry for everyone after it.
            budget_death = exc
            dropped.append("parallel: budget exhausted before any lane ran")

    if budget_death is not None:
        dropped.append("long_output: skipped, budget exhausted earlier")
    else:
        # ceiling_max is the ceiling probe's largest VERIFIED size, or
        # None when it measured nothing — ignorance, not a cap of zero,
        # and probe_long_output reads it that way (no rung is skipped
        # for a ceiling nobody measured).
        long_output = probe_long_output(
            active, meter,
            ceiling_max=ceiling.max_verified if ceiling else None)
        if not long_output.rungs:
            reason = "; ".join(long_output.skipped) or "no rung was attempted"
            long_output = None
            dropped.append(f"long_output: no rung ran ({reason})")
        elif all(rung.degenerate is None for rung in long_output.rungs):
            # The rungs are kept — they record what was asked and what
            # came back — but nothing was scored, so the profile says so
            # rather than letting an unflagged ladder read as a clean
            # one (ruled 2026-08-14).
            dropped.append(
                "long_output: rungs ran but no reply was scorable — "
                "the ladder measured nothing")

    # tools runs LAST (v1.6). Order is a budget decision: on a meter that
    # cannot pay for everything, the families that have carried verdicts
    # since v1 keep their calls and the newest one goes hungry, visibly.
    if budget_death is not None:
        dropped.append("tools: skipped, budget exhausted earlier")
    else:
        # A ToolsUnsupported is classified and absorbed INSIDE
        # probe_tools, which returns supported=False — a MEASURED
        # capability. Nothing here catches it, and nothing here treats
        # that value as a gap.
        tools = probe_tools(active, meter,
                            look_schedule=params.tools_look_schedule)
        if tools.supported is None:
            # Never attempted: the meter refused the first charge. Not a
            # refusal (that reads False) and not a partial (that keeps
            # its honest n) — it is unmeasured, and it is named.
            tools = None
            dropped.append("tools: budget exhausted before any turn completed")

    return _finish_profile(
        base_url=base_url,
        kind=kind,
        autodetected=autodetected,
        info=info,
        mode=mode,
        params=params,
        budget=budget,
        meter=meter,
        started=started,
        directives=directives,
        tier=tier,
        emulated=emulated,
        calibration=calibration,
        geometry=geometry,
        ceiling=ceiling,
        ceiling_shapes=ceiling_shapes,
        envelope=envelope,
        codecs=codecs,
        speed=speed,
        loop=loop,
        long_output=long_output,
        tools=tools,
        parallel=parallel,
        dropped=dropped,
        budget_death=budget_death,
    )
