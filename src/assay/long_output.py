"""Degeneracy metrics for long generations (spec §3).

Two cheap, orthogonal views of the same failure. ``distinct_n_ratio``
catches a model that loops phrases; ``zlib_ratio`` catches any
compressible collapse the n-gram window is too short to see (a single
character repeated forever has perfect 4-gram diversity of one gram and
a floor-scraping compression ratio). Either one dipping below its floor
is enough to call the output degenerate.

THRESHOLDS ARE ASSUMED, NOT DERIVED. ``DISTINCT_FLOOR`` and
``ZLIB_FLOOR`` were picked to sit far below anything healthy output has
been observed to produce, not fitted to a measured distribution. Task 12
captures live long generations, derives real floors, and restamps
``THRESHOLDS_PROVENANCE`` as ``derived-<date>``; the verdict layer
(Task 9) caps its verdict provisional for as long as this string starts
with ``assumed``. Treat a flag from this module as a smoke alarm, not a
measurement.

The committed transcripts under ``docs/evidence-transcripts/`` are code
and JSON, which is the wrong genre for CALIBRATING prose degeneracy —
code is legitimately repetitive. They serve instead as a false-positive
guard (spec §3 amendment): healthy output of a repetitive genre that
must not flag. Measured against
``qwen2.5-coder-7b-instruct-q8_0-quick.jsonl`` (all 10 replies of >= 50
words), the worst healthy case scores distinct=0.961 and zlib=0.400 —
clear of both assumed floors by a wide margin, so no downward adjustment
was needed. Should a later guard fail, the floors yield: they are
assumed, and the transcripts are real data.
"""

import zlib

DISTINCT_FLOOR = 0.30   # assumed, not derived — see THRESHOLDS_PROVENANCE
ZLIB_FLOOR = 0.20       # assumed, not derived
THRESHOLDS_PROVENANCE = "assumed-not-derived-2026-08-14"


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
