"""Tasks 7, 8 and 12 tests: long-output degeneracy metrics and the probe.

Three kinds of pin, in ascending order of how much they are worth:
synthetic extremes that any sane floor must separate; a false-positive
guard against REAL committed code replies, which are healthy output of a
repetitive genre and would be flagged by a floor set too hot; and the
Task 12 anchor — 28 live enumeration replies, labelled by reading them,
which is where ZLIB_FLOOR was actually derived from and which the
derivation arithmetic is recomputed against here rather than quoted.
DISTINCT_FLOOR is still assumed, and the reply that made it underivable
is pinned as a live counterexample.

The Task 8 half tests the escalating-rung probe: what it asks for, what
it charges, and — the part worth the most — what it refuses to claim
when a rung never ran or came back empty.

Everything is offline: the anchor tests read committed transcripts, and
the probe-level ones replay them through CallReplayer. No daemon, no GPU.
"""

import dataclasses
import json
import pathlib

import pytest
from fakes import ScriptedBackend

from assay.backends.base import BackendCaps, Reply
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
from assay.replay import CallReplayer

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


def test_thresholds_are_registered_and_the_mixed_provenance_is_explicit():
    # Task 12 derived ZLIB_FLOOR from the anchor capture and could NOT
    # derive DISTINCT_FLOOR (the clusters overlap on it). The provenance
    # string has to say both halves, because it is what every profile's
    # verdict lens quotes — a bare "derived-<date>" would claim the whole
    # instrument was calibrated when half of it was not.
    assert (DISTINCT_FLOOR, ZLIB_FLOOR) == (0.30, 0.2557)
    assert THRESHOLDS_PROVENANCE.startswith("derived-2026-08-15")
    assert "distinct still assumed" in THRESHOLDS_PROVENANCE


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
    the floors flagged them they would be too hot, and they would have
    to yield to this data: a derived floor comes from 276 samples, a
    transcript reply is real output that a real model really produced.
    That is not hypothetical for the zlib floor, which Task 12 derived
    2026-08-15 with only 1.076x clearance over the worst of these; the
    distinct floor is still assumed and clears them by 1.98x.
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

    248 replies of >= 50 words across 23 files, and the floors clear all
    of them — but the margin is not uniform, and Task 12's derivation
    made the zlib one TIGHTER, deliberately. The worst zlib is 0.2752
    (``sweep-hermes3-latest.jsonl``), now only 1.076x the derived
    ZLIB_FLOOR of 0.2557 where it was 1.38x the assumed 0.20; the worst
    distinct is 0.5952, 1.98x the still-assumed DISTINCT_FLOOR. That
    same hermes3 reply is the healthy edge the derivation used, so this
    guard is not an independent check of it — it is the pin that stops
    the margin being eaten silently by a later floor raise.
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
    # committed output. Neither floor clears 2x, and zlib now clears
    # barely 1.07x — the price of deriving it.
    assert worst_distinct == pytest.approx(0.5952, abs=1e-4)
    assert worst_zlib == pytest.approx(0.2752, abs=1e-4)
    assert worst_distinct / DISTINCT_FLOOR == pytest.approx(1.984, abs=1e-3)
    assert worst_zlib / ZLIB_FLOOR == pytest.approx(1.076, abs=1e-3)
    # The readable claim the exact ratios encode: neither floor has 2x
    # headroom against real committed output, and zlib now has barely 1.1x.
    assert worst_distinct / DISTINCT_FLOOR < 2.0
    assert worst_zlib / ZLIB_FLOOR < 1.1


# --- Task 12: the degeneracy anchor -----------------------------------------
#
# The floors above are no longer guesses on the zlib side: they were fitted
# to 28 live enumeration replies captured 2026-08-15 (ollama 0.32.13, seven
# models, one call per rung) and committed under docs/superpowers/evidence/
# degenerate-anchor/ with human labels. These tests read the COMMITTED text
# — no daemon, no GPU, no network — and hold the derivation to it.


ANCHOR = pathlib.Path(__file__).resolve().parents[1] / (
    "docs/superpowers/evidence/degenerate-anchor")


def _anchor_samples() -> list[dict]:
    """Labelled anchor replies, each carrying its recorded text.

    The label comes from ``labels.json`` (human judgement, recorded once);
    the text comes from the transcript. Pairing them here means a
    transcript edited out from under its label fails the suite rather
    than quietly re-deriving the floors from different data.
    """
    labels = json.loads((ANCHOR / "labels.json").read_text())
    texts: dict[tuple[str, int], str] = {}
    for path in sorted(ANCHOR.glob("*-longoutput.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("outcome") == "reply":
                texts[(path.name, row["seed"])] = row["text"]
    out = []
    for sample in labels["samples"]:
        text = texts[(sample["transcript"], sample["seed"])]
        out.append({**sample, "text": text})
    return out


def test_the_anchor_capture_is_committed_whole():
    samples = _anchor_samples()
    assert len(samples) == 28  # 7 models x 4 rungs
    assert len({s["model"] for s in samples}) == 7
    assert sum(s["label"] == "degenerate" for s in samples) == 10
    # The metrics recorded beside each label still describe the committed
    # text: labels.json is evidence, not a cache that may drift.
    for sample in samples:
        assert distinct_n_ratio(sample["text"]) == pytest.approx(
            sample["distinct_ratio"], abs=1e-5), sample["transcript"]
        assert zlib_ratio(sample["text"]) == pytest.approx(
            sample["zlib_ratio"], abs=1e-5), sample["transcript"]


def test_every_labelled_degenerate_anchor_reply_flags():
    """The acceptance test the floors exist to pass.

    Ten replies that a human read and called degenerate — models emitting
    one sentence over and over under a fresh number. If the instrument
    does not flag these it does not work.
    """
    degenerate = [s for s in _anchor_samples() if s["label"] == "degenerate"]
    assert len(degenerate) == 10
    for sample in degenerate:
        verdict = is_degenerate(distinct_n_ratio(sample["text"]),
                                zlib_ratio(sample["text"]))
        assert verdict is True, (
            f"{sample['model']} seed {sample['seed']} "
            f"({100 * sample['duplicate_item_fraction']:.0f}% duplicate items) "
            f"read as healthy: distinct={sample['distinct_ratio']:.4f} "
            f"zlib={sample['zlib_ratio']:.4f}")


def test_every_labelled_healthy_anchor_reply_passes():
    """The other half: same task, same prompt, healthy output.

    18 replies from the same enumeration prompt that repeat nothing. A
    floor that flags these is measuring the task, not the model.
    """
    healthy = [s for s in _anchor_samples() if s["label"] == "healthy"]
    assert len(healthy) == 18
    for sample in healthy:
        verdict = is_degenerate(distinct_n_ratio(sample["text"]),
                                zlib_ratio(sample["text"]))
        assert verdict is False, (
            f"{sample['model']} seed {sample['seed']} flagged: "
            f"distinct={sample['distinct_ratio']:.4f} "
            f"zlib={sample['zlib_ratio']:.4f}")


def test_the_derived_zlib_floor_is_the_midpoint_of_the_measured_band():
    """The derivation arithmetic, recomputed rather than quoted.

    Floor = midpoint between the degenerate cluster's best value and the
    healthy cluster's worst. The healthy cluster is every real healthy
    reply available — the anchor's enumeration replies AND the committed
    code corpus, whose worst case (0.2752) is lower than any healthy
    enumeration and is therefore what actually caps the floor.

    LOAD-BEARING BEYOND THE ARITHMETIC: ``degenerate_best < ZLIB_FLOOR``
    IS the claim that released Task 9's forced-provisional cap — the
    derived floor alone catches every degenerate sample, 10/10, which is
    why a still-assumed DISTINCT_FLOOR does not make the verdict
    unsettled. Weaken this assertion and that justification goes with it.
    """
    samples = _anchor_samples()
    degenerate_best = max(zlib_ratio(s["text"])
                          for s in samples if s["label"] == "degenerate")
    healthy_worst_enum = min(zlib_ratio(s["text"])
                             for s in samples if s["label"] == "healthy")
    healthy_worst_code = min(
        zlib_ratio(text)
        for path in sorted(TRANSCRIPTS.glob("*.jsonl"))
        for text in _replies(path.name)
        if len(text.split()) >= 50)
    assert degenerate_best == pytest.approx(0.236194, abs=1e-5)
    assert healthy_worst_enum == pytest.approx(0.454239, abs=1e-5)
    assert healthy_worst_code == pytest.approx(0.275208, abs=1e-5)
    # The code genre binds, not the enumeration genre.
    healthy_worst = min(healthy_worst_enum, healthy_worst_code)
    assert healthy_worst == healthy_worst_code
    assert degenerate_best < healthy_worst  # separated, so derivable
    assert ZLIB_FLOOR == round((degenerate_best + healthy_worst) / 2, 4)
    # And the shipped constant sits strictly inside the band it came from.
    assert degenerate_best < ZLIB_FLOOR < healthy_worst


def test_the_distinct_clusters_overlap_so_that_floor_could_not_be_derived():
    """Why DISTINCT_FLOOR is still assumed, kept as a live counterexample.

    A qwen2.5-coder:1.5b reply whose items 7-20 are one repeated sentence
    scores 0.6127 on distinct-n — ABOVE the worst healthy committed code
    reply at 0.5952. Renumbering each looped line keeps the 4-gram window
    fed, so no single genre-agnostic distinct floor separates degenerate
    prose from healthy code. Anyone tempted to fit one later has to get
    past this test first.
    """
    samples = _anchor_samples()
    degenerate_best = max(distinct_n_ratio(s["text"])
                          for s in samples if s["label"] == "degenerate")
    healthy_worst_code = min(
        distinct_n_ratio(text)
        for path in sorted(TRANSCRIPTS.glob("*.jsonl"))
        for text in _replies(path.name)
        if len(text.split()) >= 50)
    assert degenerate_best == pytest.approx(0.612745, abs=1e-5)
    assert healthy_worst_code == pytest.approx(0.595238, abs=1e-5)
    assert degenerate_best > healthy_worst_code  # overlap: no gap to halve
    assert DISTINCT_FLOOR == 0.30  # unchanged, and still assumed
    # The overlapping degenerate reply is caught anyway — by zlib, which
    # is the whole reason two orthogonal metrics are read.
    overlapper = max((s for s in samples if s["label"] == "degenerate"),
                     key=lambda s: distinct_n_ratio(s["text"]))
    assert distinct_n_ratio(overlapper["text"]) > DISTINCT_FLOOR
    assert zlib_ratio(overlapper["text"]) < ZLIB_FLOOR


def test_the_derived_floor_catches_what_the_assumed_floor_missed():
    """What deriving bought: two real misses fixed, no new false positive.

    Under the old assumed 0.20 the anchor's degenerate cluster scored
    8/10 — and two of those eight passed at 0.1976 and 0.1997, which is
    luck, not calibration. Under 0.2557 it scores 10/10 while every
    healthy sample still clears.
    """
    assumed_zlib_floor = 0.20  # the pre-Task-12 value, pinned as history
    samples = _anchor_samples()
    missed = [s for s in samples
              if s["label"] == "degenerate"
              and distinct_n_ratio(s["text"]) >= DISTINCT_FLOOR
              and zlib_ratio(s["text"]) >= assumed_zlib_floor]
    assert {(s["model"], s["seed"]) for s in missed} == {
        ("qwen2.5-coder:0.5b-instruct-q8_0", 1103),
        ("qwen2.5-coder:1.5b", 1101),
    }
    for sample in missed:
        assert is_degenerate(distinct_n_ratio(sample["text"]),
                             zlib_ratio(sample["text"])) is True


def _anchor_replay(transcript: str):
    """Re-run probe_long_output over a committed anchor transcript.

    The strongest form of the acceptance test: not the metrics in
    isolation but the whole probe — ladder, scoring, rung assembly —
    against text a real daemon really produced. CallReplayer is strict,
    so this also proves the transcript is complete and correctly keyed.
    """
    path = ANCHOR / transcript
    model = json.loads(path.read_text().splitlines()[0])["model"]
    backend = CallReplayer(path, model=model, caps=BackendCaps(
        reports_counts=True, per_request_ctx=True,
        truncate_control=True, metadata_access=True))
    return probe_long_output(backend, long_meter(), ceiling_max=None)


def test_the_anchor_degenerate_transcript_replays_and_every_rung_flags():
    out = _anchor_replay("qwen2.5-coder-0.5b-instruct-q8_0-longoutput.jsonl")
    assert tuple(r.target_tokens for r in out.rungs) == RUNGS
    assert [r.degenerate for r in out.rungs] == [True, True, True, True]
    # The 4096 rung is the one the assumed 0.20 floor let through.
    assert out.rungs[3].zlib_ratio == pytest.approx(0.2362, abs=1e-4)
    assert out.rungs[3].distinct_ratio == pytest.approx(0.5431, abs=1e-4)


def test_the_anchor_healthy_transcript_replays_and_no_rung_flags():
    out = _anchor_replay("gemma2-9b-longoutput.jsonl")
    assert tuple(r.target_tokens for r in out.rungs) == RUNGS
    assert [r.degenerate for r in out.rungs] == [False] * 4


def test_the_anchor_mixed_transcript_locates_where_the_model_degrades():
    # The ladder shape the family exists to find, from real output: a q4
    # 1.5b that is clean at 512, loops at 1024 and 2048, and comes back
    # clean at 4096 (where it happened to stop after 138 tokens).
    # Degeneracy is per-generation, not a switch thrown once at a size.
    out = _anchor_replay("qwen2.5-coder-1.5b-longoutput.jsonl")
    assert [r.degenerate for r in out.rungs] == [False, True, True, False]


def test_the_anchor_labels_never_came_near_their_own_boundary():
    """The labels are human, so their credibility is worth pinning.

    Samples were sorted by reading them; ``duplicate_item_fraction`` (the
    share of numbered items restating an earlier one verbatim) is the
    recorded audit trail, not a third metric — nothing in src/ computes
    it. The clusters do not touch on it, so no label was a coin flip.
    """
    samples = _anchor_samples()
    healthy = [s["duplicate_item_fraction"]
               for s in samples if s["label"] == "healthy"]
    degenerate = [s["duplicate_item_fraction"]
                  for s in samples if s["label"] == "degenerate"]
    assert max(healthy) == 0.0
    assert min(degenerate) >= 0.50


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


def test_a_family_admitted_on_calls_can_still_die_on_the_token_meter():
    """CARRIED-DEBT item 29. "What starts, finishes" is a CALL-meter
    guarantee — the README, the CHANGELOG and `run.py`'s docstring all
    say so after ruling 14 qualified them — and nothing tested the
    other half: an admitted family CAN still be cut mid-run by the
    token meter, and when it is, it keeps what it measured.

    A claim three documents make and no test exercises is a claim, not
    a property. The charges here are derived, not guessed: each rung
    costs `38 + target` tokens, so 550 / 1612 / 3698 cumulative over
    the first three rungs, and a 2000-token budget cuts at 2048 while
    `max_calls` stays far out of reach — the cut is the TOKEN meter's.
    """
    meter = BudgetMeter(Budget(max_calls=99, max_prompt_tokens=2000))
    result = probe_long_output(ScriptedBackend(), meter, ceiling_max=None)

    measured = [rung.target_tokens for rung in result.rungs]
    assert measured == [512, 1024], "the rungs it paid for must survive"
    assert any("2048" in entry for entry in result.skipped), (
        "the cut is NAMED, not inferred from an absent row")
    assert any("4096" in entry for entry in result.skipped)
    # Nothing was recorded for a call that never launched.
    assert meter.spent.prompt_tokens == 1612
    assert meter.spent.calls == 2
