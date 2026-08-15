"""Task 7 tests: long-output degeneracy metrics.

The floors these tests exercise are ASSUMED, not derived (Task 12 does
the deriving), so the tests deliberately pin two different kinds of
thing: synthetic extremes that any sane floor must separate, and one
guard against REAL committed output. The guard is the load-bearing
one — code replies are healthy output of a repetitive genre, and if the
assumed floors flagged them the floors would be too hot.
"""

import json
import pathlib

import pytest

from assay.long_output import (
    DISTINCT_FLOOR,
    THRESHOLDS_PROVENANCE,
    ZLIB_FLOOR,
    distinct_n_ratio,
    is_degenerate,
    zlib_ratio,
)

TRANSCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "docs/evidence-transcripts"

# ~200 words of varied prose: distinct factual sentences, no filler
# generation. Written out so the "healthy" side of every threshold is
# real text rather than a shuffled template.
VARIED_PROSE = (
    "Basalt cools quickly enough that its crystals stay too small to see. "
    "The harpsichord plucks a string where the piano strikes one with a felt hammer. "
    "Iceland runs most of its grid on geothermal steam and glacial meltwater. "
    "A octopus tastes whatever its arms touch, since chemoreceptors line every sucker. "
    "Cuneiform began as accounting marks pressed into wet clay near the Euphrates. "
    "Mercury has no meaningful atmosphere, so its daytime rock roasts and its night side freezes. "
    "Sourdough rises on wild yeast and lactobacilli that sour the crumb as they work. "
    "The Antikythera mechanism modelled eclipse cycles with bronze gearing two millennia ago. "
    "Peregrine falcons dive faster than any other animal, folding their wings into a teardrop. "
    "Portland cement hardens by hydration rather than by drying, which is why it cures underwater. "
    "Old growth redwood pulls moisture from coastal fog through needles high above the trunk. "
    "The Turing test sidesteps defining thought by asking only whether an examiner can tell. "
    "Saffron costs so much because each crocus yields three stigmas, all picked by hand. "
    "Greenland sharks may live four centuries, growing barely a centimetre a year. "
    "A violin bow needs rosin because bare horsehair slides across the string without catching. "
    "Tin pest turns solder brittle and grey when the metal spends long enough below thirteen degrees. "
    "Migrating monarchs navigate by a sun compass corrected against their circadian clock. "
    "Vulcanised rubber owes its resilience to sulphur bridges knitted between polymer chains. "
    "The Pantheon dome lightens toward its oculus by swapping heavy aggregate for pumice. "
    "Bees convert nectar to honey partly by fanning the comb until the water content drops. "
    "Neutron stars spin down as their magnetic fields drag against the surrounding plasma. "
    "Woodblock printers in Edo registered each colour with a carved corner notch called a kento. "
)

PATHOLOGICAL = "the same four words " * 200


def test_thresholds_are_registered_and_marked_assumed():
    assert (DISTINCT_FLOOR, ZLIB_FLOOR) == (0.30, 0.20)
    assert THRESHOLDS_PROVENANCE.startswith("assumed")


def test_pathological_repetition_flags():
    assert distinct_n_ratio(PATHOLOGICAL) < DISTINCT_FLOOR
    assert zlib_ratio(PATHOLOGICAL) < ZLIB_FLOOR
    assert is_degenerate(distinct_n_ratio(PATHOLOGICAL), zlib_ratio(PATHOLOGICAL)) is True


def test_varied_prose_passes():
    assert len(VARIED_PROSE.split()) >= 200
    assert distinct_n_ratio(VARIED_PROSE) > DISTINCT_FLOOR
    assert zlib_ratio(VARIED_PROSE) > ZLIB_FLOOR
    assert is_degenerate(distinct_n_ratio(VARIED_PROSE), zlib_ratio(VARIED_PROSE)) is False


def test_short_text_is_unmeasurable_not_degenerate():
    assert distinct_n_ratio("one two three") is None
    assert distinct_n_ratio("") is None
    assert zlib_ratio("") is None
    assert is_degenerate(None, None) is None


def test_exactly_n_words_is_one_gram_not_none():
    # The boundary between unmeasurable and measurable: n words make
    # exactly one n-gram, which is measurable (and trivially unique).
    assert distinct_n_ratio("one two three four") == 1.0


def test_a_single_none_defers_to_the_other_metric():
    assert is_degenerate(None, 0.05) is True
    assert is_degenerate(0.05, None) is True
    assert is_degenerate(None, 0.9) is False
    assert is_degenerate(0.9, None) is False


def test_either_metric_below_its_floor_flags():
    assert is_degenerate(0.05, 0.9) is True
    assert is_degenerate(0.9, 0.05) is True
    assert is_degenerate(0.9, 0.9) is False


def test_the_floor_is_an_exclusive_bound():
    # A value sitting exactly on the floor is not below it.
    assert is_degenerate(DISTINCT_FLOOR, ZLIB_FLOOR) is False


def test_ngram_windows_overlap_rather_than_chunk():
    # The suite otherwise cannot tell a sliding window from a chunked
    # one: "one two three four" scores 1.0 under both. Here they
    # diverge. Eight words give five OVERLAPPING 4-grams — (a b a b),
    # (b a b a), (a b a b), (b a b a), (a b a b) — of which 2 are
    # unique, so 2/5 = 0.4. A chunked implementation would cut two
    # non-overlapping grams, 1 unique of 2, and score 0.5.
    assert distinct_n_ratio("a b a b a b a b") == pytest.approx(0.4)


def test_zlib_ratio_measures_bytes_not_characters():
    # Non-ASCII text is where the denominator choice shows. Ten
    # repetitions of "ünïcödé " are 80 characters but 120 UTF-8 bytes;
    # the byte denominator scores 0.2 where a character denominator
    # would score 0.3. Bytes are correct — compression works on bytes,
    # and a character denominator would flatter multibyte text.
    text = "ünïcödé " * 10
    assert len(text) == 80 and len(text.encode("utf-8")) == 120
    assert zlib_ratio(text) == pytest.approx(0.2, abs=1e-4)


def test_a_nonsense_gram_width_is_refused_not_silently_answered():
    # n=0 would window empty tuples and report a ratio near zero — a
    # nonsense number that reads exactly like total collapse.
    for bad in (0, -1):
        try:
            distinct_n_ratio("one two three four", n=bad)
        except ValueError:
            continue
        raise AssertionError(f"n={bad} should have raised ValueError")


def _replies(name: str) -> list[str]:
    path = TRANSCRIPTS / name
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row["text"] for row in rows if row.get("outcome") == "reply" and row.get("text")]


def test_committed_code_replies_do_not_false_positive():
    """False-positive guard (spec §3 amendment).

    Code replies from the committed transcripts are healthy output of a
    repetitive GENRE — docstrings, imports, repeated ``def`` lines. If
    the assumed thresholds flagged them the thresholds would be too hot,
    and the floors (assumed) would have to yield to this data (real).
    """
    texts = [t for t in _replies("qwen2.5-coder-7b-instruct-q8_0-quick.jsonl")
             if len(t.split()) >= 50]
    assert texts, "guard needs real replies; transcript yielded none >= 50 words"
    for text in texts:
        distinct = distinct_n_ratio(text)
        z = zlib_ratio(text)
        assert is_degenerate(distinct, z) is not True, (
            f"healthy code reply flagged: distinct={distinct:.3f} zlib={z:.3f}"
        )


def test_no_committed_transcript_reply_false_positives():
    """The same guard widened to every committed transcript.

    248 replies of >= 50 words across 23 files, and the assumed floors
    clear all of them — but the margin is not uniform. The worst zlib is
    0.275 (``sweep-hermes3-latest.jsonl``), only 1.38x the assumed
    ZLIB_FLOOR of 0.20; the worst distinct is 0.5952, 1.98x
    DISTINCT_FLOOR. Neither floor has 2x headroom against real committed
    output, which is why the provenance string still says assumed: prose
    from a model that hedges repetitively could plausibly sit lower.
    """
    worst_distinct = worst_zlib = None
    seen = 0
    for path in sorted(TRANSCRIPTS.glob("*.jsonl")):
        for text in _replies(path.name):
            if len(text.split()) < 50:
                continue
            seen += 1
            distinct = distinct_n_ratio(text)
            z = zlib_ratio(text)
            assert is_degenerate(distinct, z) is not True, (
                f"{path.name}: distinct={distinct:.3f} zlib={z:.3f}"
            )
            worst_distinct = distinct if worst_distinct is None else min(worst_distinct, distinct)
            worst_zlib = z if worst_zlib is None else min(worst_zlib, z)
    assert seen == 248
    # Both headrooms pinned, so a future floor raise that eats the
    # observed margin fails here instead of silently flagging healthy
    # committed output. Neither floor clears 2x.
    assert worst_distinct == pytest.approx(0.5952, abs=1e-4)
    assert worst_zlib == pytest.approx(0.2752, abs=1e-4)
    assert worst_distinct / DISTINCT_FLOOR < 2.0
    assert worst_zlib / ZLIB_FLOOR < 2.0
