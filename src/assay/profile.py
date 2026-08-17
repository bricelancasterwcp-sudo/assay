"""Versioned capability profile, verdicts, render (spec §8).

One JSON document; every field a measurement, a None named in
``provenance.dropped``, or provenance. The schema is self-policing:
constructing a Profile with a None family that ``dropped`` does not
name is a ValueError — unmeasured must always be named, never silent.

None-vs-zero at the verdict layer: unmeasured inputs yield
``"unmeasured"``, never ``"unusable"`` — a consumer must be able to
tell "assay could not measure this" from "the model failed".
"""

import dataclasses
import json
from dataclasses import dataclass

from assay.ceiling import CallEvidence, Ceiling, ShapeCeiling
from assay.codecs import Landing
from assay.fixtures import FIXTURE_SET
from assay.long_output import (DISTINCT_FLOOR, LONG_OUTPUT_TASK,
                               THRESHOLDS_PROVENANCE, ZLIB_FLOOR, LongOutput,
                               LongRung)
from assay.loop import LOOP_INSTRUMENT, Loop
from assay.envelope import Envelope
from assay.geometry import Geometry
from assay.speed import Speed
# The verdict arithmetic lives in assay.stats (a leaf module) so that
# codecs can stop sequentially without importing profile back. The
# private aliases keep this module's long-standing surface intact:
# `from assay.profile import wilson95` still resolves, and the
# thresholds stay readable under their old names.
from assay.stats import READY_THRESHOLD as _READY_THRESHOLD  # noqa: F401
from assay.stats import RISKY_THRESHOLD as _RISKY_THRESHOLD  # noqa: F401
from assay.stats import VERDICT_LENS
from assay.stats import ladder as _ladder
from assay.stats import wilson95
from assay.tools import TOOLS_INSTRUMENT, TOOLSET_NAME, Tools

PROFILE_VERSION = 6

_FAMILIES = ("geometry", "ceiling", "ceiling_shapes", "envelope", "codecs",
             "speed", "loop", "long_output", "tools")
#: Families the v1 schema did not have. A document written before one of
#: them existed simply has no key for it — a different fact from a
#: modern document that writes ``null`` because the family measured
#: nothing, and ``from_json`` keeps the two apart (spec §4).
_POST_V1_FAMILIES = ("ceiling_shapes", "speed", "loop", "long_output",
                     "tools")
_GRADE_FOR_VERDICTS = "small"
#: The codecs ``patch_editing`` may be carried by — either can carry it.
_PATCH_CODECS = ("search_replace", "whole_file")
_LONG_CONTEXT_TOKENS = 16384
_TRUNCATION_GUARD_TOKENS = 4096
# Speed floors (v1.2): tok/s a verdict is judged against. Defaults are
# provisional until sanity-checked on live hardware; every speed verdict
# carries its floors in the lens, so a different operator's floors are
# always visible. Decode = chat usability; prefill = agent usability
# (agent loops are prefill-dominated: they re-read context constantly).
_CHAT_READY_TPS = 8.0
_CHAT_RISKY_TPS = 4.0
_AGENT_READY_TPS = 200.0
_AGENT_RISKY_TPS = 80.0

_HONEST_MODES = frozenset({"hard_error", "none_up_to_cap"})
_LYING_MODES = frozenset({"silent_truncation", "missing_stats"})


@dataclass(frozen=True)
class Profile:
    assay_profile_version: int  # == PROFILE_VERSION
    probe_version: str
    endpoint: dict  # {"kind", "base_url", "autodetected"}
    model: dict  # {"name", "quant", "weights_bytes", "training_ctx"}
    geometry: Geometry | None
    ceiling: Ceiling | None
    ceiling_shapes: tuple[ShapeCeiling, ...] | None
    envelope: Envelope | None
    codecs: dict[str, dict[str, Landing]] | None
    speed: Speed | None
    loop: Loop | None
    long_output: LongOutput | None
    tools: Tools | None
    verdicts: dict[str, dict]
    provenance: dict  # started/finished/mode/seeds/budget/spent/calibration/dropped

    def __post_init__(self) -> None:
        dropped = self.provenance.get("dropped") or []
        named = {str(entry).split(":", 1)[0].strip() for entry in dropped}
        for family in _FAMILIES:
            if getattr(self, family) is None and family not in named:
                raise ValueError(
                    f"{family} is None but provenance.dropped does not name it"
                )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, payload: dict) -> "Profile":
        """Parse any profile this project has ever written (spec §4).

        The v1 families and ``verdicts``/``provenance`` are REQUIRED: a
        document without them is not a profile, and pretending otherwise
        would let an arbitrary JSON object parse as one. The families
        added after v1 are read with ``.get()`` — absent means the schema
        predated them, which is not a malformed profile — and every
        family the old document could not carry is named in the parsed
        ``provenance.dropped`` so the None-vs-zero rule survives the
        version boundary (an unnamed None is exactly the silent gap the
        __post_init__ guard exists to refuse).
        """
        return cls(
            assay_profile_version=payload["assay_profile_version"],
            probe_version=payload["probe_version"],
            endpoint=payload["endpoint"],
            model=payload["model"],
            geometry=_geometry_from(payload["geometry"]),
            ceiling=_ceiling_from(payload["ceiling"]),
            ceiling_shapes=_shapes_from(payload.get("ceiling_shapes")),
            envelope=_envelope_from(payload["envelope"]),
            codecs=_codecs_from(payload["codecs"]),
            speed=_speed_from(payload.get("speed")),
            loop=_loop_from(payload.get("loop")),
            long_output=_long_output_from(payload.get("long_output")),
            tools=_tools_from(payload.get("tools")),
            verdicts=payload["verdicts"],
            provenance=_provenance_naming_absent_families(payload),
        )


def _provenance_naming_absent_families(payload: dict) -> dict:
    """The payload's provenance, plus a dropped line per absent family.

    Only families with NO key at all are named here. A family present
    and null was written by a schema that had it and measured nothing —
    if that run failed to name it, the guard must still refuse the
    profile rather than paper over it on the way in.

    The input payload is never mutated: a caller who re-serializes the
    dict it passed in must get back what it read from disk.
    """
    provenance = payload["provenance"]
    absent = [family for family in _POST_V1_FAMILIES if family not in payload]
    if not absent:
        return provenance
    version = payload.get("assay_profile_version")
    dropped = list(provenance.get("dropped") or [])
    dropped += [
        f"{family}: not present in the parsed profile (schema v{version})"
        for family in absent
    ]
    return {**provenance, "dropped": dropped}


def _geometry_from(payload: dict | None) -> Geometry | None:
    return None if payload is None else Geometry(**payload)


def _ceiling_from(payload: dict | None) -> Ceiling | None:
    if payload is None:
        return None
    data = dict(payload)
    data["evidence"] = tuple(CallEvidence(**entry) for entry in data["evidence"])
    return Ceiling(**data)


def _envelope_from(payload: dict | None) -> Envelope | None:
    return None if payload is None else Envelope(**payload)


def _speed_from(payload: dict | None) -> Speed | None:
    if payload is None:
        return None
    data = dict(payload)
    # JSON has no tuples: the samples come back as lists (or absent, for
    # a profile written before v1.5). Absent stays absent — None means
    # "not recorded", and coercing it to () would claim a sampling run
    # that never happened.
    for key in ("decode_samples", "prefill_samples"):
        if data.get(key) is not None:
            data[key] = tuple(data[key])
    return Speed(**data)


def _shapes_from(payload) -> tuple[ShapeCeiling, ...] | None:
    if payload is None:
        return None
    return tuple(ShapeCeiling(**entry) for entry in payload)


def _loop_from(payload: dict | None) -> Loop | None:
    # A pre-v1.6 loop payload has no recovery/doom/error-run keys at all;
    # Loop defaults them to None, which is "this schema had no such
    # field" — never a zero that would read as a model that never
    # recovered.
    return None if payload is None else Loop(**payload)


def _tools_from(payload: dict | None) -> Tools | None:
    return None if payload is None else Tools(**payload)


def _long_output_from(payload: dict | None) -> LongOutput | None:
    if payload is None:
        return None
    # JSON has no tuples: rungs and skip reasons come back as lists and
    # must be coerced, or a round-tripped profile stops comparing equal
    # to itself.
    return LongOutput(
        rungs=tuple(LongRung(**rung) for rung in payload["rungs"]),
        skipped=tuple(payload["skipped"]),
    )


def _codecs_from(
    payload: dict | None,
) -> dict[str, dict[str, Landing]] | None:
    if payload is None:
        return None
    return {
        codec: {
            grade: Landing(
                lands=cell["lands"],
                # v1 cells have no applies-and-parses column: that lens
                # did not exist when they were written, so it reads
                # None — unmeasured. Copying `lands` across would
                # fabricate a measurement under an instrument that never
                # ran (the 2026-08-12 finding: the same model scores 0%
                # and 100% depending on which lens is asked).
                lands_applies=cell.get("lands_applies"),
                n=cell["n"],
            )
            for grade, cell in grades.items()
        }
        for codec, grades in payload.items()
    }


def _small_landing(
    codecs: dict[str, dict[str, Landing]] | None, codec: str,
    *, lens: str = "byte_equality",
) -> float | None:
    """The .small landing rate under the named lens, or None wherever it
    was not measured."""
    if codecs is None:
        return None
    cell = codecs.get(codec, {}).get(_GRADE_FOR_VERDICTS)
    if cell is None:
        return None
    if lens == "applies_and_parses":
        return cell.lands_applies  # None when the cell exists but n == 0
    return cell.lands


def _truncates_below_4k(ceiling: Ceiling | None) -> bool:
    return (
        ceiling is not None
        and ceiling.failure_mode == "silent_truncation"
        and ceiling.first_failure is not None
        and ceiling.first_failure < _TRUNCATION_GUARD_TOKENS
    )


def _loop_lens(loop: Loop | None) -> dict:
    """What a loop verdict was judged by — instrument, fixtures, sampler,
    and the ERROR-RUN denominator.

    ``n_error_runs`` is in the lens because ``recovery_rate`` alone does
    not say how much evidence is behind it: a budget-truncated 1/1 and a
    complete 5/5 both read 1.0, and one of them demotes a verdict. None
    where the family is unmeasured or was written by a schema that had
    no error script — never 0, which would claim an error script that
    ran and completed nothing.
    """
    return {"instrument": LOOP_INSTRUMENT, "fixtures": FIXTURE_SET,
            "temperature": 0.2,
            "n_error_runs": None if loop is None else loop.n_error_runs}


def _loop_verdict(loop: Loop | None) -> dict:
    """Turn discipline under the scripted loop: ready needs high action
    fidelity AND a landed patch AND, where it was measured, a way out of
    a failed patch. A model that follows the envelope but never advances
    (the 14B shape: fidelity 1.0, 0/940) is risky at best, and so is one
    that answers a rejected patch by re-sending it. Wilson over scored
    turns; provisional like the codec verdicts."""
    if loop is None or loop.action_fidelity is None:
        return {"verdict": "unmeasured", "provisional": False,
                "interval95": None, "lens": _loop_lens(loop)}
    lo, hi = wilson95(round(loop.action_fidelity * loop.n_turns),
                      loop.n_turns)
    verdict = _ladder(loop.action_fidelity)
    if verdict == "ready" and (loop.patch_rate or 0.0) < 0.5:
        verdict = "risky"  # follows the loop, does not advance it
    if (verdict == "ready" and loop.recovery_rate is not None
            and loop.recovery_rate < 0.5):
        # v1.6, mirroring the patch-rate rule one line up. The guard is
        # `is not None`, not truthiness: recovery_rate 0.0 is a measured
        # model that never recovered and MUST demote, while None is an
        # error script that never ran and must demote nothing.
        verdict = "risky"  # follows the loop, cannot get out of a failure
    provisional = _ladder(lo) != _ladder(hi)
    return {"verdict": verdict, "provisional": provisional,
            "interval95": [round(lo, 3), round(hi, 3)],
            "lens": _loop_lens(loop)}


def _tools_lens(tools: Tools | None) -> dict:
    """What a tool_calling verdict was judged by.

    The four metric rates ride in the lens rather than only in the
    family because the verdict ladders on the COMPOSITE, and a composite
    of 0.0 reached by never calling anything is a different finding from
    one reached by calling the right tool with wrong arguments. Every
    rate is None-safe: ``right_tool_rate`` and ``args_valid_rate`` are
    None whenever no T1 emitted a call at all — there was no tool name
    and no arguments to judge — and writing 0.0 there would claim a
    measurement of "called badly" that never happened.

    ``n_used`` is the COMPOSITE's denominator (``n_tasks``), not
    ``n_turns``: the T2 turns score result-use, not the composite, and
    quoting them would claim evidence the verdict never saw. It is
    ABSENT when nothing was scored, the same rule the codec lenses
    follow — ``n_used: 0`` reads as "graded on zero samples".

    These are rates of INSTRUCTED behavior: the instrument's system line
    announces the rubric it scores (call exactly one tool, use the
    arguments the request names, quote the result token). See
    ``assay.tools``' module docstring — a reader comparing them to a
    harness that does not spell the rules out should expect them high.
    """
    lens = {"instrument": TOOLS_INSTRUMENT, "toolset": TOOLSET_NAME,
            "stopping_rule": "fixed-n"}
    if tools is not None and tools.n_tasks:
        lens["n_used"] = tools.n_tasks
    lens["temperature"] = 0.2
    for rate in ("call_rate", "right_tool_rate", "args_valid_rate",
                 "result_use_rate"):
        lens[rate] = None if tools is None else getattr(tools, rate)
    return lens


def _tool_calling_verdict(tools: Tools | None) -> dict:
    """Native tool calling, laddered on the composite (spec v1.6 §1).

    Three outcomes that must never be confused:

    - **unmeasured** — the family never ran (budget death, or a profile
      from a schema that had no tools family).
    - **unsupported** — the endpoint REFUSED the tools parameter. A
      capability fact: not "unmeasured" (the refusal is a measurement)
      and not "unusable" (the model was never asked to do the task).
      ``interval95`` is None because there is no proportion to bound.
    - the ladder — ready/risky/unusable on the composite, Wilson-95 over
      ``round(composite * n_tasks), n_tasks``, provisional exactly as
      the codec verdicts: True when the interval's endpoints ladder to
      different rungs, because then the data cannot tell the achieved
      rung from its neighbours.
    """
    lens = _tools_lens(tools)
    if tools is None or tools.supported is None:
        return {"verdict": "unmeasured", "provisional": False,
                "interval95": None, "lens": lens}
    if tools.supported is False:
        return {"verdict": "unsupported", "provisional": False,
                "interval95": None, "lens": lens}
    if tools.composite is None:
        # Defensive: the probe cannot report supported=True with nothing
        # scored today. If it ever could, an unscored family is
        # unmeasured — never a rung it did not earn.
        return {"verdict": "unmeasured", "provisional": False,
                "interval95": None, "lens": lens}
    lo, hi = wilson95(round(tools.composite * tools.n_tasks), tools.n_tasks)
    verdict, provisional = _ladder_provisional(tools.composite, lo, hi)
    return {"verdict": verdict, "provisional": provisional,
            "interval95": [round(lo, 3), round(hi, 3)], "lens": lens}


def _long_output_lens(scorable: list[LongRung]) -> dict:
    """What a long_output verdict was judged by — floors and EXTENT.

    The floors are MIXED: ``ZLIB_FLOOR`` was derived 2026-08-15 from the
    captured anchor, ``DISTINCT_FLOOR`` is still assumed because the
    clusters overlap on it (see long_output's module docstring). The lens
    carries the provenance string, which says both halves, so no reader
    has to go looking for it: a verdict quoted without its thresholds is
    not a model property.

    ``rungs_scored`` and ``deepest_scored_tokens`` are the extent (ruled
    2026-08-15): "ready" on a ladder the ceiling cut off at 1024 and
    "ready" verified clean to 4096 are the same word for different
    findings, and report.py/diff.py read this entry, not the rendered
    table. The keys are present in every shape — 0 and None when nothing
    was scored — so a consumer parses one lens, not two.
    """
    return {
        "metrics": "distinct4gram+zlib",
        "distinct_floor": DISTINCT_FLOOR,
        "zlib_floor": ZLIB_FLOOR,
        "thresholds": THRESHOLDS_PROVENANCE,
        "task": LONG_OUTPUT_TASK,
        "temperature": 0.2,
        "rungs_scored": len(scorable),
        "deepest_scored_tokens": (max(rung.target_tokens for rung in scorable)
                                  if scorable else None),
    }


def _long_output_verdict(long_output: LongOutput | None) -> dict:
    """Where a long generation stops holding together, if it does.

    Only rungs with a MEASURED degenerate (True or False) count. A rung
    whose reply was too short to score is neither a measurement nor a
    skipped rung — it spent a call and learned nothing (ruled
    2026-08-14) — so a ladder with no scorable rung reads "unmeasured",
    never "ready": nothing was found healthy, it was merely not found
    degenerate.

    Among the scorable rungs: none degenerate -> "ready"; the FIRST
    scorable rung degenerate -> "unusable" (the smallest target this
    ladder could measure was already looping); a later one ->
    "degrades-at-<target>", naming the rung. Unscorable rungs never
    stand in for healthy ones, which is why the index is taken over the
    scorable rungs rather than over the ladder.

    ``provisional`` is LADDER COMPLETENESS (ruled 2026-08-15): True
    whenever the instrument did not finish the ladder it was configured
    to climb — any rung skipped (ceiling or budget) or any attempted rung
    that came back unscorable. A four-rung ladder scored end to end is
    settled; "ready" on a two-rung ladder the ceiling cut off at 1024 is
    not the same finding and must not wear the same badge. Extent is in
    the lens for a reader who looks; this flag is for the one who does
    not.

    The threshold cap ORs in: while the floors are assumed rather than
    derived, every measured verdict is provisional whatever the ladder
    did. Task 12 derived ``ZLIB_FLOOR`` and the cap is currently
    released, but it re-applies by itself if the provenance ever returns
    to an ``assumed`` prefix.

    "unmeasured" is False on both counts — it claims nothing that a
    finished ladder or a better threshold could revise.
    """
    scorable = [] if long_output is None else [
        rung for rung in long_output.rungs if rung.degenerate is not None
    ]
    if not scorable:
        return {"verdict": "unmeasured", "provisional": False,
                "lens": _long_output_lens(scorable)}
    first_bad = next(
        (i for i, rung in enumerate(scorable) if rung.degenerate), None)
    if first_bad is None:
        verdict = "ready"
    elif first_bad == 0:
        verdict = "unusable"
    else:
        verdict = f"degrades-at-{scorable[first_bad].target_tokens}"
    ladder_unfinished = (bool(long_output.skipped)
                         or len(scorable) != len(long_output.rungs))
    return {"verdict": verdict,
            "provisional": (ladder_unfinished
                            or THRESHOLDS_PROVENANCE.startswith("assumed")),
            "lens": _long_output_lens(scorable)}


def _long_context(ceiling: Ceiling | None) -> str:
    if ceiling is None:
        return "unmeasured"
    if ceiling.failure_mode in _LYING_MODES:
        return "risky"  # the daemon lies past the edge
    if (
        ceiling.max_verified is not None
        and ceiling.max_verified >= _LONG_CONTEXT_TOKENS
        and ceiling.failure_mode in _HONEST_MODES
    ):
        return "ready"
    return "unmeasured"


def _ladder_provisional(lands: float | None, lo: float, hi: float,
                        *, ready_blocked: bool = False) -> tuple[str, bool]:
    """(verdict, provisional): provisional when the interval endpoints
    ladder to different rungs than each other — the data cannot tell
    the achieved rung from its neighbours."""
    verdict = _ladder(lands, ready_blocked=ready_blocked)
    if lands is None:
        return verdict, False
    provisional = (_ladder(lo, ready_blocked=ready_blocked)
                   != _ladder(hi, ready_blocked=ready_blocked))
    return verdict, provisional


def _speed_ladder(rate: float | None, ready: float, risky: float) -> str:
    if rate is None:
        return "unmeasured"
    if rate >= ready:
        return "ready"
    if rate >= risky:
        return "risky"
    return "unusable"


def verdict_cell(
    codecs: dict[str, dict[str, Landing]] | None, codec: str
) -> Landing | None:
    """The cell a codec verdict is read from: the ``small`` grade."""
    if codecs is None:
        return None
    return codecs.get(codec, {}).get(_GRADE_FOR_VERDICTS)


def _verdict_rate(cell: Landing | None, codec: str) -> float | None:
    """The rate a codec's verdict is read from — the lens named in the
    shared registry (``stats.VERDICT_LENS``), which is the same entry
    ``codecs``' sequential stop test counts. None wherever the cell is
    missing or that lens was never measured (a v1 cell has no
    applies-and-parses column at all)."""
    if cell is None:
        return None
    return getattr(cell, VERDICT_LENS.get(codec, "lands_applies"))


def _best_patch(
    codecs: dict[str, dict[str, Landing]] | None
) -> tuple[Landing | None, float | None]:
    """(cell, rate) for the patch codec that lands best under its own
    registered lens. Returned together because the caller needs both and
    re-deriving the rate from the cell would mean naming the lens a
    second time — the duplication this consolidation removes."""
    best: Landing | None = None
    best_rate: float | None = None
    for codec in _PATCH_CODECS:
        cell = verdict_cell(codecs, codec)
        rate = _verdict_rate(cell, codec)
        if rate is not None and (best_rate is None or rate > best_rate):
            best, best_rate = cell, rate
    return best, best_rate


def best_patch_cell(
    codecs: dict[str, dict[str, Landing]] | None
) -> Landing | None:
    """The cell ``patch_editing`` is judged on: whichever patch codec
    lands best under applies-and-parses (either codec can carry the
    verdict). Public because the orchestrator stamps this cell's n into
    the lens — reading it from anywhere else could name an n that
    belongs to a cell no verdict used."""
    return _best_patch(codecs)[0]


def _codec_lens(landing: str, presentation: str, stopping_rule: str,
                n_used: dict[str, int] | None, verdict_name: str,
                sampler: dict) -> dict:
    """One codec verdict's lens: what landed, under what presentation,
    ended by what rule, at what n. ``n_used`` is ABSENT for a cell that
    was never measured — a zero there would read as "graded on zero
    samples", which is a different (and false) claim."""
    lens = {"landing": landing, "presentation": presentation,
            "stopping_rule": stopping_rule}
    n = (n_used or {}).get(verdict_name)
    if n is not None:
        lens["n_used"] = n
    lens.update(sampler)
    return lens


def compute_verdicts(
    geometry: Geometry | None,
    ceiling: Ceiling | None,
    envelope: Envelope | None,
    codecs: dict[str, dict[str, Landing]] | None,
    speed: Speed | None = None,
    loop: Loop | None = None,
    long_output: LongOutput | None = None,
    tools: Tools | None = None,
    *,
    presentation: str = "default-v1",
    stopping_rule: str = "fixed-n",
    n_used: dict[str, int] | None = None,
) -> dict[str, dict]:
    """Spec §8 verdict rules, v2: every verdict NAMES ITS LENS.

    The 2026-08-12 live validation measured the same model at 0% and
    100% edit landing under two instruments; a verdict quoted without
    its lens is not a model property. Each entry is
    ``{"verdict": ..., "lens": {...}}`` where the lens states the
    landing definition, the presentation (``default-v1`` or the
    consumer's ``custom`` directive), the pinned sampler, and — for
    long_context — the evidence class.

    The two codec lenses also carry ``stopping_rule`` and ``n_used``
    (v1.5): under sequential sampling a cell that stopped at n=5 and a
    fixed-n=5 cell hold the same number for different reasons, and the
    caller who ran the sampling is the one who knows which.

    ``patch_editing`` is judged under the **applies-and-parses** lens:
    an application accepting a patch validates the result by running
    it, so byte-equality's compliance-with-incidentals is the wrong
    predictor there. The raw byte-equality column stays in
    ``codecs`` for consumers who want the stricter number. Which lens
    each codec is graded under is registered once, in
    ``stats.VERDICT_LENS`` — the same entry ``codecs``' sequential stop
    test counts, so a cell can never stop on a lens its verdict ignores.

    Unmeasured inputs -> "unmeasured", never worse. ``geometry`` and
    ``envelope`` inform no verdict but are part of the stable signature.
    """
    del geometry, envelope  # no verdict consumes them
    sampler = {"temperature": 0.2, "fixtures": FIXTURE_SET}

    jo = verdict_cell(codecs, "json_object")
    jo_rate = _verdict_rate(jo, "json_object")
    jo_lo, jo_hi = (0.0, 1.0)
    if jo_rate is not None:
        jo_lo, jo_hi = wilson95(round(jo_rate * jo.n), jo.n)
    jo_verdict, jo_prov = _ladder_provisional(
        jo_rate, jo_lo, jo_hi,
        ready_blocked=_truncates_below_4k(ceiling))

    best_patch, patch_rate = _best_patch(codecs)
    p_lo, p_hi = (0.0, 1.0)
    if patch_rate is not None:
        p_lo, p_hi = wilson95(round(patch_rate * best_patch.n), best_patch.n)
    patch_verdict, patch_prov = _ladder_provisional(patch_rate, p_lo, p_hi)

    counts = None if ceiling is None else ceiling.counts_available
    return {
        "structured_extraction": {
            "verdict": jo_verdict,
            "provisional": jo_prov,
            "interval95": ([round(jo_lo, 3), round(jo_hi, 3)]
                           if jo_rate is not None else None),
            "lens": _codec_lens("json_valid_required_keys", presentation,
                                stopping_rule, n_used,
                                "structured_extraction", sampler),
        },
        "patch_editing": {
            "verdict": patch_verdict,
            "provisional": patch_prov,
            "interval95": ([round(p_lo, 3), round(p_hi, 3)]
                           if patch_rate is not None else None),
            "lens": _codec_lens("applies_and_parses(python)", presentation,
                                stopping_rule, n_used,
                                "patch_editing", sampler),
        },
        "long_context": {
            "verdict": _long_context(ceiling),
            "lens": {"evidence": ("counts+canary" if counts
                                  else "canary_only" if counts is not None
                                  else "unmeasured")},
        },
        "loop_discipline": _loop_verdict(loop),
        "chat_speed": {
            "verdict": _speed_ladder(
                None if speed is None else speed.decode_tps,
                _CHAT_READY_TPS, _CHAT_RISKY_TPS),
            "lens": {"metric": "decode_tps",
                     "floor_ready": _CHAT_READY_TPS,
                     "floor_risky": _CHAT_RISKY_TPS,
                     "evidence": ("unmeasured" if speed is None
                                  else speed.evidence)},
        },
        "agent_speed": {
            "verdict": _speed_ladder(
                None if speed is None else speed.prefill_tps,
                _AGENT_READY_TPS, _AGENT_RISKY_TPS),
            "lens": {"metric": "prefill_tps",
                     "floor_ready": _AGENT_READY_TPS,
                     "floor_risky": _AGENT_RISKY_TPS,
                     "evidence": ("unmeasured" if speed is None
                                  else speed.evidence)},
        },
        "long_output": _long_output_verdict(long_output),
        "tool_calling": _tool_calling_verdict(tools),
    }


def _show(value: object) -> str:
    return "unmeasured" if value is None else str(value)


def _moe_marker(geometry: Geometry) -> str:
    """`` | MoE <used>-of-<count>`` — only when BOTH counts are measured.

    A dense model carries None on both and gets no marker: it is not a
    0-expert MoE. One-sided metadata gets none either — "MoE 8-of-None"
    would print an unmeasured half as though it had been measured, which
    is the overclaim this whole schema exists to refuse.
    """
    if geometry.expert_count is None or geometry.expert_used_count is None:
        return ""
    return f" | MoE {geometry.expert_used_count}-of-{geometry.expert_count}"


def _render_geometry(geometry: Geometry | None) -> str:
    if geometry is None:
        return "geometry   unmeasured"
    vram = (
        f"{geometry.vram_free_mib} MiB free"
        if geometry.vram_free_mib is not None
        else "vram unmeasured"
    )
    return (
        f"geometry   {geometry.kv_kib_per_token} KiB/token"
        f" | usable window {geometry.usable_window}"
        f" (limited by {geometry.limited_by}) | {vram}"
        + _moe_marker(geometry)
    )


def _render_ceiling(ceiling: Ceiling | None) -> str:
    if ceiling is None:
        return "ceiling    unmeasured"
    return (
        f"ceiling    max verified {_show(ceiling.max_verified)}"
        f" | first failure {_show(ceiling.first_failure)}"
        f" | mode {ceiling.failure_mode}"
    )


def _render_envelope(envelope: Envelope | None) -> str:
    if envelope is None or envelope.fidelity is None:
        return "envelope   unmeasured"
    failures = envelope.failures
    return (
        f"envelope   fidelity {envelope.fidelity:.2f} (n={envelope.n})"
        f" | prose {failures.get('prose', 0)}"
        f" shape {failures.get('shape', 0)}"
        f" refusal {failures.get('refusal', 0)}"
    )


def _render_codecs(codecs: dict[str, dict[str, Landing]] | None) -> list[str]:
    if codecs is None:
        return ["codecs     unmeasured"]
    lines = ["codecs           " + "".join(g.ljust(12) for g in ("tiny", "small", "medium"))]
    for codec, grades in codecs.items():
        cells = []
        for grade in ("tiny", "small", "medium"):
            cell = grades.get(grade)
            if cell is None or cell.lands is None:
                cells.append("-".ljust(12))
            else:
                cells.append(f"{cell.lands:.2f} (n={cell.n})".ljust(12))
        lines.append(f"  {codec.ljust(15)}{''.join(cells)}")
    return lines


def _render_long_output(long_output: LongOutput | None) -> str:
    if long_output is None:
        return "long_output unmeasured"
    scorable = [rung for rung in long_output.rungs
                if rung.degenerate is not None]
    if not scorable:
        # Calls were spent and nothing was scored: say unmeasured, and
        # say how many rungs were asked, so the spend is visible.
        return ("long_output unmeasured"
                f" ({len(long_output.rungs)} rungs attempted, none scorable)")
    first_bad = next((rung for rung in scorable if rung.degenerate), None)
    state = (
        f"healthy through {scorable[-1].target_tokens} tokens"
        if first_bad is None
        else f"degenerate from {first_bad.target_tokens} tokens"
    )
    line = f"long_output {state} (rungs scored {len(scorable)})"
    if long_output.skipped:
        line += " | skipped " + "; ".join(long_output.skipped)
    return line


def _rate(value: float | None) -> str:
    """A rate for the human table: two places, or the word."""
    return "unmeasured" if value is None else f"{value:.2f}"


def _render_tools(tools: Tools | None) -> str:
    """The tools line, with the three outcomes kept apart.

    ``unsupported`` prints as itself rather than as "unmeasured": the
    endpoint was asked and said no, which is a fact about the endpoint
    and not a gap in the run.
    """
    if tools is None:
        return "tools      unmeasured"
    if tools.supported is False:
        return "tools      unsupported (the endpoint refused the tools parameter)"
    if tools.composite is None:
        return "tools      unmeasured"
    return (
        f"tools      composite {tools.composite:.2f} (n={tools.n_tasks})"
        f" | call {_rate(tools.call_rate)}"
        f" right-tool {_rate(tools.right_tool_rate)}"
        f" args {_rate(tools.args_valid_rate)}"
        f" result-use {_rate(tools.result_use_rate)}"
    )


def _verdict_word(entry: object) -> str:
    """The verdict word, from either schema.

    v1 wrote each verdict as a BARE STRING; v2 onward writes a dict that
    carries the verdict plus its lens. Indexing the string as a dict is
    what made the human view of an archived profile raise TypeError —
    report.py's ``_badge`` already drew this distinction, render_table
    did not.
    """
    if isinstance(entry, dict):
        return str(entry.get("verdict", "unmeasured"))
    return str(entry)


def _lens_line(verdicts: dict) -> str:
    """One ``name: k=v,...`` per verdict THAT HAS a lens.

    A bare-string (v1) verdict carries no lens, and a made-up one would
    be the exact overclaim the lens exists to prevent; a profile with no
    lenses at all says ``unmeasured``, which is what it is.
    """
    parts = [
        f"{name}: "
        + ",".join(f"{k}={_show(v)}" for k, v in entry["lens"].items())
        for name, entry in verdicts.items()
        if isinstance(entry, dict) and entry.get("lens")
    ]
    return "lenses     " + (" | ".join(parts) if parts else "unmeasured")


def render_table(profile: Profile) -> str:
    """Human view of a profile, of ANY schema version. Unmeasured is
    SAID, never shown as 0 — and never shown as Python's ``None``."""
    endpoint = profile.endpoint
    model = profile.model
    detected = "autodetected" if endpoint.get("autodetected") else "forced"
    lines = [
        f"assay profile v{profile.assay_profile_version}"
        f" (probe {profile.probe_version})",
        f"endpoint   {endpoint.get('kind')} {endpoint.get('base_url')} ({detected})",
        f"model      {model.get('name')}"
        f"  quant={_show(model.get('quant'))}"
        f"  training_ctx={_show(model.get('training_ctx'))}",
        "",
        _render_geometry(profile.geometry),
        _render_ceiling(profile.ceiling),
        _render_envelope(profile.envelope),
        *_render_codecs(profile.codecs),
        "",
        ("speed      unmeasured" if profile.speed is None else
         f"speed      decode {_show(profile.speed.decode_tps)} tok/s | "
         f"prefill {_show(profile.speed.prefill_tps)} tok/s "
         f"({profile.speed.evidence})"),
        _render_long_output(profile.long_output),
        _render_tools(profile.tools),
        "verdicts   "
        + " | ".join(f"{name}: {_verdict_word(entry)}"
                     for name, entry in profile.verdicts.items()),
        # ``_show`` inside the lens line too, and for the same reason as
        # every other line: a lens field is as unmeasured as a metric can
        # be — a capped ladder scores nothing, so
        # ``deepest_scored_tokens`` is None — and printing Python's
        # ``None`` beside a table that says "unmeasured" everywhere else
        # asks the reader to know Python.
        _lens_line(profile.verdicts),
    ]
    dropped = profile.provenance.get("dropped") or []
    if dropped:
        lines.append("dropped    " + "; ".join(str(entry) for entry in dropped))
    return "\n".join(lines)
