"""Degeneracy metrics and probe for long generations (spec §3).

Two cheap, orthogonal views of the same failure. ``distinct_n_ratio``
catches a model that loops phrases; ``zlib_ratio`` catches any
compressible collapse the n-gram window is too short to see (a single
character repeated forever has perfect 4-gram diversity of one gram and
a floor-scraping compression ratio). Either one dipping below its floor
is enough to call the output degenerate.

ONE FLOOR IS DERIVED, THE OTHER IS STILL ASSUMED. Task 12 captured live
long generations from seven models on the enumeration task
(``docs/superpowers/evidence/degenerate-anchor/``, 2026-08-15, ollama
0.32.13) and put both floors to the same rule — the midpoint between the
degenerate cluster's best value and the healthy cluster's worst. Only one
of them survived it:

- ``ZLIB_FLOOR`` is DERIVED. Degenerate best 0.2362, healthy worst
  0.2752 (a hermes3 code reply), midpoint 0.2557. The old assumed 0.20
  missed two genuinely degenerate replies outright and caught two more
  at 0.1976 and 0.1997 — its sensitivity was luck, not calibration.
- ``DISTINCT_FLOOR`` is NOT derived and stays assumed at 0.30, because
  the clusters OVERLAP on it: a 1.5b reply whose items 7-20 are the same
  sentence scores 0.6127, ABOVE the worst healthy code reply at 0.5952.
  A looping model that renumbers each repeated line keeps its 4-gram
  diversity up — which is precisely the collapse ``zlib_ratio`` is here
  to catch instead. No single genre-agnostic distinct floor separates
  the two, so none was invented.

``THRESHOLDS_PROVENANCE`` carries that mixed state verbatim into every
profile's verdict lens. It no longer starts with ``assumed``, so the
Task 9 forced-provisional cap releases: all ten captured degenerate
samples are flagged by the DERIVED floor alone, so the verdict's
sensitivity rests on measured data, and the assumed distinct floor is a
redundant backstop (the only sample below it is flagged by zlib too).

The committed transcripts under ``docs/evidence-transcripts/`` are code
and JSON — legitimately repetitive, and the reason a distinct floor
cannot be fitted to prose alone. They are the false-positive guard
(spec §3 amendment): healthy output of a repetitive genre that must not
flag, and their worst case is what caps how high a floor may go. Across
all 23 files (248 replies of >= 50 words) nothing flags under either
floor, but the margin is thin and thinner than before: worst zlib 0.2752
is 1.076x the derived ``ZLIB_FLOOR``, worst distinct 0.5952 is 1.984x
``DISTINCT_FLOOR``. Both edges are pinned in
``tests/test_long_output.py``. Should a later guard fail, the floors
yield: 0.2557 is derived from 276 samples, not from a population.

``probe_long_output`` is the other half: one call per rung up an
escalating ladder of generation targets, scoring each reply with the
functions above. A model that holds together for 512 tokens and loops
at 2048 is the failure this ladder exists to locate, and it can only be
located by asking for both. Every rung that does NOT run says why —
above the measured ceiling, or out of budget — because a missing rung
and a healthy rung are not the same finding.
"""

import zlib
from dataclasses import dataclass

from assay.backends.base import Backend
from assay.budget import BudgetMeter
from assay.errors import BudgetExhausted

# ASSUMED. The anchor's clusters overlap on this metric (degenerate
# 0.6127 > healthy 0.5952), so no floor was derivable; 0.30 is the
# original conservative pick, kept as a redundant backstop.
DISTINCT_FLOOR = 0.30
# DERIVED 2026-08-15: midpoint of (0.236194, 0.275208), the gap between
# the anchor's best degenerate reply and the worst healthy one.
ZLIB_FLOOR = 0.2557
THRESHOLDS_PROVENANCE = "derived-2026-08-15 (zlib only; distinct still assumed)"


def distinct_n_ratio(text: str, n: int = 4) -> float | None:
    """Unique n-grams / total n-grams over whitespace-split words.

    ``None`` when the text holds fewer than ``n`` words: there is no
    n-gram to be diverse or repetitive, so the metric is unmeasurable
    rather than 0. Short output is a separate complaint from degenerate
    output, and conflating them would let an empty reply read as the
    worst possible repetition.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    words = text.split()
    if len(words) < n:
        return None
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def zlib_ratio(text: str) -> float | None:
    """Compressed size / raw size over UTF-8 bytes.

    ``None`` for empty text — dividing by zero bytes is unmeasurable,
    not incompressible. The ratio can exceed 1.0 on very short strings
    (zlib's header outweighs anything it saves), which is why the floor
    is only ever read as a lower bound.
    """
    raw = text.encode("utf-8")
    if not raw:
        return None
    return len(zlib.compress(raw)) / len(raw)


def is_degenerate(distinct: float | None, z: float | None) -> bool | None:
    """True when EITHER metric sits below its floor.

    ``None`` only when both inputs are None — nothing was measured, so
    there is no verdict to give (never False, which would claim the
    output was checked and found healthy). A single None defers to the
    metric that did measure.
    """
    if distinct is None and z is None:
        return None
    return (distinct is not None and distinct < DISTINCT_FLOOR) or (
        z is not None and z < ZLIB_FLOOR
    )


# --- the probe --------------------------------------------------------------

RUNGS = (512, 1024, 2048, 4096)
LONG_OUTPUT_TASK = "enumeration-v1"
"""Names the task the rungs were measured on. Degeneracy is a property
of model x task, not of the model alone: an enumeration prompt invites
list-shaped repetition that a summarisation prompt would not. A reader
comparing two profiles needs to know both ran the same task."""

_PROMPT = (
    "Write a numbered list of distinct, specific facts, each about a "
    "different everyday object or place. One fact per line. Do not "
    "repeat a fact or an object. Continue the list until you are "
    "stopped."
)
_LONG_SEED = 1100
# Charge sizing ONLY, never reported as a measurement. The prompt is
# ~40 tokens however it is counted, while the generation target it sits
# beside is 512-4096 — the rough term is two orders of magnitude below
# the one that matters, so it is left rough on purpose.
_EST_CHARS_PER_TOKEN = 5


@dataclass(frozen=True)
class LongRung:
    """One attempted rung. Every metric is None when unmeasurable."""

    target_tokens: int
    generated_tokens: int | None  # reply.tokens_out; None when unreported
    distinct_ratio: float | None
    zlib_ratio: float | None
    degenerate: bool | None       # None = nothing scorable came back


@dataclass(frozen=True)
class LongOutput:
    rungs: tuple[LongRung, ...]   # attempted rungs, in ladder order
    # Why each unattempted rung did not run, naming itself:
    # "4096: above measured ceiling", "2048: budget exhausted".
    skipped: tuple[str, ...]


def _score(text: str) -> tuple[float | None, float | None, bool | None]:
    """(distinct, zlib, degenerate) for one reply body.

    A reply too short for the n-gram window — fewer than 4 words, which
    includes the empty and whitespace-only cases — is unmeasurable on
    all three, ``degenerate=None`` rather than False. ``zlib_ratio``
    alone would not stop there: it scores ``"ok"`` at 5.0 and
    ``"Sorry, I cannot."`` at 1.5, both far ABOVE ``ZLIB_FLOOR``,
    because compressing a handful of bytes measures zlib's own header
    and not the output. Deferring to that number would report a
    three-word refusal against a 4096-token target as checked-and-
    healthy — a value that looks like a measurement on text the
    instrument cannot measure.

    Task 7's pure functions are unchanged: ``is_degenerate`` correctly
    defers to whichever metric measured something, and this honesty
    rule is the probe's, where the target the reply fell short of is
    known. ``generated_tokens`` still passes through on the rung, so a
    consumer can see the target-vs-generated gap for itself.
    """
    distinct = distinct_n_ratio(text)
    if distinct is None:
        return None, None, None
    z = zlib_ratio(text)
    return distinct, z, is_degenerate(distinct, z)


def probe_long_output(
    backend: Backend,
    meter: BudgetMeter,
    *,
    ceiling_max: int | None,
    rungs: tuple[int, ...] = RUNGS,
    seed: int = _LONG_SEED,
) -> LongOutput:
    """Climb the rung ladder, scoring each reply for degeneracy.

    One call per rung, ``max_tokens`` set to the rung. Each call is
    charged ``max(1, len(_PROMPT) // 5) + target``: generation shares
    the context window with the prompt, so a 4096-token rung is not the
    same load on the endpoint as a 512-token one and the meter must not
    price them alike (the codec probes' sizing-proxy philosophy).

    Rungs that do not run are named in ``skipped`` rather than dropped
    silently. ``ceiling_max`` (the ceiling probe's largest VERIFIED
    size, ``None`` when it measured nothing — ignorance, not a cap of
    zero) skips any strictly larger rung; budget exhaustion skips the
    rung it hit and every later one. A rung that is BOTH above the
    ceiling and past the budget is named by the ceiling: that reason is
    a property of the model and holds whatever the budget did, where
    the budget reason would shift with how the run was ordered.

    Infrastructure errors propagate — a rung whose call failed at the
    transport is not a rung that produced healthy text (spec §3).
    """
    attempted: list[LongRung] = []
    skipped: list[str] = []
    budget_dead = False
    for i, target in enumerate(rungs):
        if ceiling_max is not None and target > ceiling_max:
            skipped.append(f"{target}: above measured ceiling")
            continue
        if budget_dead:
            skipped.append(f"{target}: budget exhausted")
            continue
        try:
            meter.charge(max(1, len(_PROMPT) // _EST_CHARS_PER_TOKEN) + target)
        except BudgetExhausted:
            budget_dead = True
            skipped.append(f"{target}: budget exhausted")
            continue
        reply = backend.generate(_PROMPT, seed=seed + i, max_tokens=target)
        distinct, z, degenerate = _score(reply.text)
        attempted.append(
            LongRung(
                target_tokens=target,
                generated_tokens=reply.tokens_out,
                distinct_ratio=distinct,
                zlib_ratio=z,
                degenerate=degenerate,
            )
        )
    return LongOutput(rungs=tuple(attempted), skipped=tuple(skipped))
