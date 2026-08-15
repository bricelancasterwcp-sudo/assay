"""Tasks 7 and 8 tests: long-output degeneracy metrics and the probe.

The floors these tests exercise are ASSUMED, not derived (Task 12 does
the deriving), so the tests deliberately pin two different kinds of
thing: synthetic extremes that any sane floor must separate, and one
guard against REAL committed output. The guard is the load-bearing
one — code replies are healthy output of a repetitive genre, and if the
assumed floors flagged them the floors would be too hot.

The Task 8 half below tests the escalating-rung probe: what it asks
for, what it charges, and — the part worth the most — what it refuses
to claim when a rung never ran or came back empty.
"""

import dataclasses
import json
import pathlib

import pytest
from fakes import ScriptedBackend

from assay.backends.base import Reply
from assay.budget import Budget, BudgetMeter
from assay.errors import BudgetExhausted, InfrastructureError
from assay.long_output import (
    DISTINCT_FLOOR,
    LONG_OUTPUT_TASK,
    RUNGS,
    THRESHOLDS_PROVENANCE,
    ZLIB_FLOOR,
    LongOutput,
    LongRung,
    _LONG_SEED,
    _PROMPT,
    distinct_n_ratio,
    is_degenerate,
    probe_long_output,
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


# --- Task 8: probe_long_output ----------------------------------------------

# len(_PROMPT) // 5, pinned as a literal: a test that recomputed the
# divisor from the module would agree with any divisor the module chose.
CHARGE_PROMPT_TOKENS = 38


class LongFake:
    """Backend replying with scripted texts in call order.

    A call past the end of the script raises IndexError, so a probe that
    takes one call too many fails loudly instead of silently reusing the
    last reply.
    """

    model = "long-fake"

    def __init__(self, texts, *, reports_counts: bool = True) -> None:
        self._texts = list(texts)
        self.reports_counts = reports_counts
        self.calls: list[tuple[str, int, int]] = []  # prompt, seed, max_tokens

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
        self.calls.append((prompt, seed, max_tokens))
        text = self._texts[len(self.calls) - 1]
        return Reply(
            text=text,
            tokens_in=None,
            tokens_out=len(text.split()) if self.reports_counts else None,
            stop_reason="length",
            raw={},
        )


class ExplodingFake:
    """Every call fails at the transport layer."""

    model = "exploding-fake"

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
        raise InfrastructureError("transport failure: connection refused (scripted)")


def long_meter(max_calls: int = 99, max_prompt_tokens: int = 10**9) -> BudgetMeter:
    return BudgetMeter(Budget(max_calls=max_calls, max_prompt_tokens=max_prompt_tokens))


def test_the_ladder_the_task_name_and_the_seed_base_are_registered():
    assert RUNGS == (512, 1024, 2048, 4096)
    assert LONG_OUTPUT_TASK == "enumeration-v1"
    # The seed base travels in provenance; a run that quietly moved it
    # would not be comparable with the profiles already recorded.
    assert _LONG_SEED == 1100


def test_each_rung_is_scored_from_its_own_reply():
    backend = LongFake([VARIED_PROSE, PATHOLOGICAL, VARIED_PROSE, PATHOLOGICAL])
    out = probe_long_output(backend, long_meter(), ceiling_max=None)
    assert isinstance(out, LongOutput)
    assert out.skipped == ()
    assert tuple(r.target_tokens for r in out.rungs) == RUNGS
    assert [r.degenerate for r in out.rungs] == [False, True, False, True]
    # The metrics are Task 7's, run on the reply text of THAT rung.
    assert out.rungs[0].distinct_ratio == pytest.approx(distinct_n_ratio(VARIED_PROSE))
    assert out.rungs[0].zlib_ratio == pytest.approx(zlib_ratio(VARIED_PROSE))
    assert out.rungs[1].distinct_ratio == pytest.approx(distinct_n_ratio(PATHOLOGICAL))
    assert out.rungs[0].generated_tokens == len(VARIED_PROSE.split())


def test_one_call_per_rung_asks_for_the_rung_and_steps_the_seed():
    backend = LongFake([VARIED_PROSE] * 4)
    probe_long_output(backend, long_meter(), ceiling_max=None, seed=7000)
    assert [c[0] for c in backend.calls] == [_PROMPT] * 4
    assert [c[1] for c in backend.calls] == [7000, 7001, 7002, 7003]
    assert [c[2] for c in backend.calls] == list(RUNGS)


def test_the_charge_is_the_prompt_plus_the_generation_target():
    # Generation shares the window, so the target is charged too — a
    # 4096-token rung is not the same load as a 512-token one and the
    # meter must not pretend otherwise.
    meter = long_meter()
    probe_long_output(LongFake([VARIED_PROSE] * 4), meter, ceiling_max=None)
    assert meter.spent.calls == 4
    assert meter.spent.prompt_tokens == 4 * CHARGE_PROMPT_TOKENS + sum(RUNGS)


def test_a_caller_supplied_ladder_is_honoured():
    backend = LongFake([VARIED_PROSE, PATHOLOGICAL])
    out = probe_long_output(
        backend, long_meter(), ceiling_max=None, rungs=(128, 256)
    )
    assert tuple(r.target_tokens for r in out.rungs) == (128, 256)
    assert [c[2] for c in backend.calls] == [128, 256]


def test_the_ceiling_caps_the_ladder_and_names_each_skipped_rung():
    backend = LongFake([VARIED_PROSE])
    out = probe_long_output(backend, long_meter(), ceiling_max=1000)
    assert len(backend.calls) == 1  # the capped rungs were never asked for
    assert tuple(r.target_tokens for r in out.rungs) == (512,)
    assert out.skipped == (
        "1024: above measured ceiling",
        "2048: above measured ceiling",
        "4096: above measured ceiling",
    )


def test_a_rung_sitting_exactly_at_the_ceiling_still_runs():
    # The ceiling is the largest VERIFIED size, so a rung equal to it is
    # inside what was measured; only a strictly larger rung is outside.
    backend = LongFake([VARIED_PROSE, VARIED_PROSE])
    out = probe_long_output(backend, long_meter(), ceiling_max=1024)
    assert tuple(r.target_tokens for r in out.rungs) == (512, 1024)
    assert out.skipped == (
        "2048: above measured ceiling",
        "4096: above measured ceiling",
    )


def test_an_unmeasured_ceiling_is_not_a_cap_of_zero():
    # ceiling_max None means the ceiling probe landed no number. That is
    # ignorance, not a limit — the ladder runs and the budget is the
    # only brake.
    out = probe_long_output(LongFake([VARIED_PROSE] * 4), long_meter(), ceiling_max=None)
    assert len(out.rungs) == 4
    assert out.skipped == ()


def test_budget_death_stops_the_ladder_and_names_every_later_rung():
    backend = LongFake([VARIED_PROSE])
    out = probe_long_output(backend, long_meter(max_calls=1), ceiling_max=None)
    assert len(backend.calls) == 1
    assert tuple(r.target_tokens for r in out.rungs) == (512,)
    assert out.skipped == (
        "1024: budget exhausted",
        "2048: budget exhausted",
        "4096: budget exhausted",
    )


def test_budget_exhaustion_is_reported_not_raised():
    # The probe swallows BudgetExhausted into `skipped`; a caller
    # running the family last must still get the earlier rungs back.
    try:
        out = probe_long_output(
            LongFake([VARIED_PROSE]), long_meter(max_calls=1), ceiling_max=None
        )
    except BudgetExhausted as exc:
        raise AssertionError(f"probe raised instead of reporting: {exc}")
    assert len(out.rungs) == 1


def test_the_token_budget_can_kill_a_rung_before_the_call_budget():
    # One token short of what rung 1 costs, with calls to spare.
    meter = long_meter(
        max_prompt_tokens=2 * CHARGE_PROMPT_TOKENS + 512 + 1024 - 1
    )
    backend = LongFake([VARIED_PROSE])
    out = probe_long_output(backend, meter, ceiling_max=None)
    assert len(backend.calls) == 1
    assert meter.spent.calls == 1
    assert meter.spent.prompt_tokens == CHARGE_PROMPT_TOKENS + 512
    assert out.skipped[0] == "1024: budget exhausted"


def test_a_capped_rung_is_named_by_the_ceiling_even_after_the_budget_died():
    # Both rules apply to rungs 2 and 3 here. The ceiling is a property
    # of the model and holds whatever the budget did, so it wins; the
    # budget reason would otherwise change with call ordering.
    backend = LongFake([VARIED_PROSE])
    out = probe_long_output(backend, long_meter(max_calls=1), ceiling_max=1500)
    assert out.skipped == (
        "1024: budget exhausted",
        "2048: above measured ceiling",
        "4096: above measured ceiling",
    )


def test_an_empty_reply_is_unmeasurable_never_healthy():
    backend = LongFake(["", VARIED_PROSE, VARIED_PROSE, VARIED_PROSE])
    out = probe_long_output(backend, long_meter(), ceiling_max=None)
    rung = out.rungs[0]
    assert rung.target_tokens == 512
    assert (rung.distinct_ratio, rung.zlib_ratio) == (None, None)
    # `is None`, not falsy: False would claim the output was checked and
    # found healthy.
    assert rung.degenerate is None
    # The rung still happened, and the backend's reported count stands.
    assert rung.generated_tokens == 0
    assert out.rungs[1].degenerate is False


def test_a_whitespace_only_reply_is_unmeasurable_too():
    # zlib on "   \n\n  " compresses to more bytes than it started with,
    # scoring far above the floor — a reply that said nothing would read
    # as checked-and-healthy if whitespace counted as content.
    backend = LongFake(["   \n\n  "] + [VARIED_PROSE] * 3)
    out = probe_long_output(backend, long_meter(), ceiling_max=None)
    assert zlib_ratio("   \n\n  ") > ZLIB_FLOOR  # the trap this avoids
    assert out.rungs[0].zlib_ratio is None
    assert out.rungs[0].degenerate is None


def test_a_reply_too_short_to_measure_is_unmeasurable_never_healthy():
    """A three-word refusal against a 4096-token target is not health.

    zlib on a handful of bytes measures zlib's own header, not the
    output: "ok" scores 5.0 and "Sorry, I cannot." scores 1.5, both far
    above ZLIB_FLOOR, and distinct-n has no window at all. Deferring to
    that number would stamp `degenerate=False` — checked and found
    healthy — on text the instrument cannot measure.
    """
    assert distinct_n_ratio("Sorry, I cannot.") is None
    assert zlib_ratio("Sorry, I cannot.") > ZLIB_FLOOR  # the trap
    assert zlib_ratio("ok") > ZLIB_FLOOR
    backend = LongFake(["Sorry, I cannot.", "ok"] + [VARIED_PROSE] * 2)
    out = probe_long_output(backend, long_meter(), ceiling_max=None)
    for rung in out.rungs[:2]:
        assert (rung.distinct_ratio, rung.zlib_ratio) == (None, None)
        assert rung.degenerate is None
    # The counts still pass through, so a consumer can see the gap
    # between what was asked for and what came back.
    assert out.rungs[0].target_tokens == 512
    assert out.rungs[0].generated_tokens == 3
    assert out.rungs[2].degenerate is False


def test_the_short_reply_cut_is_the_n_gram_window_not_a_new_threshold():
    # Exactly 4 words is where Task 7's distinct-n becomes measurable
    # (one gram, trivially unique), and that is the same boundary this
    # probe uses — no second, probe-only threshold to keep in sync.
    backend = LongFake(["one two three four"] + [VARIED_PROSE] * 3)
    out = probe_long_output(backend, long_meter(), ceiling_max=None)
    assert out.rungs[0].distinct_ratio == 1.0
    assert out.rungs[0].degenerate is False


def test_generated_tokens_is_none_when_the_backend_reports_nothing():
    backend = LongFake([VARIED_PROSE] * 4, reports_counts=False)
    out = probe_long_output(backend, long_meter(), ceiling_max=None)
    assert all(r.generated_tokens is None for r in out.rungs)
    # An unreported count does not stop the text itself being scored.
    assert all(r.degenerate is False for r in out.rungs)


def test_infrastructure_errors_propagate_and_are_never_scored():
    with pytest.raises(InfrastructureError):
        probe_long_output(ExplodingFake(), long_meter(), ceiling_max=None)


def test_the_result_objects_are_frozen():
    out = probe_long_output(LongFake([VARIED_PROSE] * 4), long_meter(), ceiling_max=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.rungs[0].degenerate = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.skipped = ()
    assert isinstance(out.rungs[0], LongRung)


def test_the_house_fake_answers_the_real_prompt():
    # The orchestrator's shared fake raises on any unscripted prompt, so
    # wiring this family into run() (Task 9) fails here first if the
    # fake has no answer for it.
    out = probe_long_output(ScriptedBackend(), long_meter(), ceiling_max=None)
    assert len(out.rungs) == len(RUNGS)
    assert out.skipped == ()
    assert all(r.degenerate is False for r in out.rungs)
