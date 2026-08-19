"""Task 5 tests: ``assay diff`` core — identity gate, drift vs noise.

Every payload here is a RAW dict, the way ``json.loads`` hands one
back, because that is what ``diff`` reads: a v1 profile on disk has
plain-string verdicts, no ``ceiling_shapes`` key, and codec cells
without ``lands_applies``, and ``Profile.from_json`` would raise on all
three. The suite therefore pins the old shapes as well as the new ones.

The Wilson intervals quoted in the codec tests were computed with
``assay.stats.wilson95`` before being written down:
4/5 = [0.376, 0.964], 5/5 = [0.566, 1.000] (overlap, noise);
2/35 = [0.016, 0.186], 33/35 = [0.814, 0.984] (disjoint, drift).
"""

import json
import pathlib

import pytest

from assay.diff import Change, DiffResult, diff_profiles, identity_gate, render_diff

_MISSING = object()

EVIDENCE = pathlib.Path(__file__).resolve().parents[1] / "docs/superpowers/evidence"


def codec_cell(lands=1.0, lands_applies=1.0, n=5):
    """One codec × grade cell. Pass ``lands_applies=_MISSING`` for the
    v1 shape, which had no applies-and-parses column at all."""
    cell = {"lands": lands, "n": n}
    if lands_applies is not _MISSING:
        cell["lands_applies"] = lands_applies
    return cell


def make_codecs(**overrides):
    codecs = {
        "json_object": {grade: codec_cell() for grade in ("tiny", "small", "medium")},
        "whole_file": {grade: codec_cell(0.8, 0.8) for grade in ("tiny", "small", "medium")},
    }
    codecs.update(overrides)
    return codecs


def make_speed(**overrides):
    speed = {
        "decode_tps": 10.0,
        "prefill_tps": 300.0,
        "evidence": "server_timings",
        "n_decode": 3,
        "n_prefill": 1,
        "decode_samples": [10.0, 10.0, 10.0],
        "prefill_samples": [300.0],
    }
    speed.update(overrides)
    return speed


def make_verdicts(**overrides):
    verdicts = {
        "structured_extraction": {"verdict": "ready", "provisional": False, "lens": {}},
        "patch_editing": {"verdict": "risky", "provisional": False, "lens": {}},
    }
    verdicts.update(overrides)
    return verdicts


def make_profile(**overrides):
    """A small valid payload dict; every override is one flat keyword so
    a test states only the thing it is varying.

    ``assay_profile_version`` stays 5 DELIBERATELY through the v6 bump:
    ``diff`` compares documents and never reads the schema number, and a
    fixture that tracked PROFILE_VERSION would quietly stop covering the
    case an operator actually has — last month's profile against
    today's.
    """
    model = {
        "name": overrides.pop("model_name", "qwen2.5-coder:7b"),
        "quant": overrides.pop("quant", "Q8_0"),
        "weights_bytes": overrides.pop("weights_bytes", 8_000_000_000),
        "training_ctx": 32768,
    }
    ceiling = overrides.pop("ceiling", _MISSING)
    if ceiling is _MISSING:
        ceiling = {
            "max_verified": overrides.pop("max_verified", 15872),
            "first_failure": 16384,
            "failure_mode": overrides.pop("failure_mode", "hard_error"),
            "counts_available": True,
            "evidence": [],
        }
    payload = {
        "assay_profile_version": 5,
        "probe_version": overrides.pop("probe_version", "0.5.0"),
        "endpoint": {"kind": "ollama", "base_url": "http://x", "autodetected": True},
        "model": model,
        "geometry": None,
        "ceiling": ceiling,
        "ceiling_shapes": overrides.pop("shapes", None),
        "envelope": None,
        "codecs": overrides.pop("codecs", make_codecs()),
        "speed": overrides.pop("speed", make_speed()),
        "loop": None,
        "verdicts": overrides.pop("verdicts", make_verdicts()),
        "provenance": {
            "tier": overrides.pop("tier", "average-gamer-8gb"),
            "emulated": overrides.pop("emulated", False),
            "mode": "quick",
            "dropped": [],
        },
    }
    if overrides:
        raise TypeError(f"unknown make_profile overrides: {sorted(overrides)}")
    return payload


# --- identity gate ---------------------------------------------------


def test_identical_profiles_no_drift():
    p = make_profile()
    result = diff_profiles(p, p)
    assert result.comparable and result.changes == ()


def test_identity_mismatch_is_fatal_even_when_cells_match():
    result = diff_profiles(make_profile(), make_profile(model_name="other-model"))
    assert result.comparable is False


def test_fatal_identity_mismatch_scores_nothing():
    """Fatal means the cells are not comparable, so none are reported —
    a rung 'improvement' across two different models is not a fact."""
    result = diff_profiles(
        make_profile(max_verified=4096),
        make_profile(model_name="other-model", max_verified=32768),
    )
    assert result.changes == () and result.within_noise == () and result.dropped == ()


def test_identity_gate_names_the_mismatched_field():
    comparable, notes = identity_gate(
        make_profile(), make_profile(model_name="other-model"))
    assert comparable is False
    assert any("model.name" in note for note in notes)


def test_quant_known_on_one_side_only_is_a_warning_not_fatal():
    comparable, notes = identity_gate(make_profile(quant=None), make_profile())
    assert comparable is True
    assert any("model.quant" in note for note in notes)


def test_differing_quant_is_fatal():
    comparable, _ = identity_gate(make_profile(), make_profile(quant="Q4_K_M"))
    assert comparable is False


def test_weights_bytes_known_on_one_side_only_is_a_warning_not_fatal():
    comparable, notes = identity_gate(
        make_profile(weights_bytes=None), make_profile())
    assert comparable is True
    assert any("model.weights_bytes" in note for note in notes)


def test_differing_weights_bytes_is_fatal():
    comparable, _ = identity_gate(
        make_profile(), make_profile(weights_bytes=4_000_000_000))
    assert comparable is False


def test_tier_declared_on_one_side_only_is_fatal():
    """Unlike quant, an undeclared tier is not a benign unknown: the
    number's meaning depends on the hardware it was measured on."""
    comparable, _ = identity_gate(make_profile(tier=None), make_profile())
    assert comparable is False


def test_both_tiers_absent_is_comparable():
    comparable, notes = identity_gate(
        make_profile(tier=None, emulated=None),
        make_profile(tier=None, emulated=None))
    assert comparable is True and notes == ()


def test_emulated_mismatch_is_fatal():
    comparable, _ = identity_gate(make_profile(), make_profile(emulated=True))
    assert comparable is False


def test_an_unmarked_profile_reads_as_not_recorded_not_as_a_disagreement():
    """A v1 profile has no tier keys at all. That is still fatal — an
    undeclared tier is unknown hardware — but the note must not claim
    the two runs disagreed about the hardware."""
    old = make_profile()
    del old["provenance"]["tier"]
    del old["provenance"]["emulated"]
    comparable, notes = identity_gate(old, make_profile())
    assert comparable is False
    assert any("provenance.tier not recorded on one side" in note for note in notes)
    assert any("provenance.emulated not recorded on one side" in note
               for note in notes)
    assert not any("differs" in note for note in notes)


def test_a_null_tier_reads_as_not_recorded_too():
    """Present-with-null is the same claim as absent: nobody declared it."""
    comparable, notes = identity_gate(
        make_profile(tier=None, emulated=None), make_profile())
    assert comparable is False
    assert all("not recorded on one side" in note for note in notes)


def test_a_real_tier_disagreement_still_says_differs():
    comparable, notes = identity_gate(make_profile(),
                                      make_profile(tier="enthusiast-16gb"))
    assert comparable is False
    assert any("provenance.tier differs: 'average-gamer-8gb'"
               " -> 'enthusiast-16gb'" in note for note in notes)


def test_a_real_emulated_disagreement_still_says_differs():
    _, notes = identity_gate(make_profile(), make_profile(emulated=True))
    assert any("provenance.emulated differs: False -> True" in note
               for note in notes)


def test_identity_notes_survive_into_the_result():
    result = diff_profiles(make_profile(quant=None), make_profile())
    assert result.comparable is True
    assert any("model.quant" in note for note in result.identity_notes)


# --- ceiling ---------------------------------------------------------


def test_ceiling_shrink_is_regression():
    result = diff_profiles(make_profile(max_verified=15872),
                           make_profile(max_verified=11520))
    (change,) = [c for c in result.changes if c.family == "ceiling"]
    assert change.direction == "regression"
    assert (change.cell, change.basis) == ("max_verified", "rung-change")
    assert (change.old, change.new) == (15872, 11520)


def test_ceiling_growth_is_improvement():
    result = diff_profiles(make_profile(max_verified=11520),
                           make_profile(max_verified=15872))
    (change,) = [c for c in result.changes if c.family == "ceiling"]
    assert change.direction == "improvement"


def test_equal_ceiling_is_named_in_within_noise():
    result = diff_profiles(make_profile(), make_profile())
    assert "ceiling.max_verified" in result.within_noise


def test_ceiling_measured_on_one_side_only_is_dropped():
    result = diff_profiles(make_profile(), make_profile(ceiling=None))
    assert "ceiling.max_verified" in result.dropped
    assert [c for c in result.changes if c.family == "ceiling"] == []


def test_move_into_a_lying_failure_mode_is_regression():
    result = diff_profiles(make_profile(failure_mode="hard_error"),
                           make_profile(failure_mode="silent_truncation"))
    (change,) = [c for c in result.changes if c.cell == "failure_mode"]
    assert change.direction == "regression" and change.basis == "flip"


def test_move_out_of_a_lying_failure_mode_is_improvement():
    result = diff_profiles(make_profile(failure_mode="missing_stats"),
                           make_profile(failure_mode="hard_error"))
    (change,) = [c for c in result.changes if c.cell == "failure_mode"]
    assert change.direction == "improvement"


def test_honest_to_honest_failure_mode_is_neutral():
    result = diff_profiles(make_profile(failure_mode="hard_error"),
                           make_profile(failure_mode="none_up_to_cap"))
    (change,) = [c for c in result.changes if c.cell == "failure_mode"]
    assert change.direction == "neutral"


# --- ceiling shapes --------------------------------------------------


def _shape(shape, max_verified, mode="ok_to_shape"):
    return {"shape": shape, "max_verified": max_verified, "failure_mode": mode}


def test_shapes_are_matched_by_shape_value_not_position():
    old = make_profile(shapes=[_shape(4096, 3900), _shape(8192, 7800)])
    new = make_profile(shapes=[_shape(8192, 7800), _shape(4096, 1800)])
    (change,) = [c for c in result_changes(old, new) if c.family == "ceiling_shapes"]
    assert change.cell == "shape:4096"
    assert (change.old, change.new) == (3900, 1800)
    assert change.direction == "regression"


def result_changes(old, new):
    return diff_profiles(old, new).changes


def test_unmatched_shape_is_dropped():
    old = make_profile(shapes=[_shape(4096, 3900)])
    new = make_profile(shapes=[_shape(4096, 3900), _shape(16384, 15000)])
    result = diff_profiles(old, new)
    assert any("16384" in name for name in result.dropped)
    assert [c for c in result.changes if c.family == "ceiling_shapes"] == []


def test_shape_failure_mode_flip_is_scored_separately():
    old = make_profile(shapes=[_shape(4096, 3900, "ok_to_shape")])
    new = make_profile(shapes=[_shape(4096, 3900, "silent_truncation")])
    (change,) = [c for c in result_changes(old, new) if c.family == "ceiling_shapes"]
    assert change.cell == "shape:4096.failure_mode"
    assert change.direction == "regression" and change.basis == "flip"


def test_unmeasured_shape_mode_is_dropped_never_scored():
    old = make_profile(shapes=[_shape(4096, None, "unmeasured")])
    new = make_profile(shapes=[_shape(4096, None, "ok_to_shape")])
    result = diff_profiles(old, new)
    assert [c for c in result.changes if c.family == "ceiling_shapes"] == []
    assert "ceiling_shapes.shape:4096.failure_mode" in result.dropped


def test_shape_present_on_only_one_side_but_wholly_unmeasured_is_not_dropped():
    """I5: a shape's `shape` KEY can be present in the entries list
    while nothing about it was measured — `failure_mode: "unmeasured"`,
    `max_verified: null`. The other four comparators (`_exact`,
    `_diff_codec_cell`, `_diff_speed_cell`, `_diff_verdicts`) all treat
    "unmeasured on both sides" as no cell at all, never `dropped`
    (`dropped` means measured on exactly one side — the rule v1.8's
    exit 3 reads directly). `_diff_shapes` must keep the same rule even
    when the unmeasured placeholder's key is present on only one side
    and wholly absent on the other: the shape was still not measured
    on either side, so it must not read as a comparison that happened
    on one side and vanished on the other.

    Not reachable through a real campaign today (every `MODE_PARAMS`
    entry shares one `shape_probes` list and `run.py` normalizes an
    all-unmeasured shape family to `None` before it ever reaches
    `diff`), but the list is not CLI-exposed as fixed, and a future
    configurable shape list turns this into a spurious dropped cell —
    or a spurious exit 3 under `--gate`.
    """
    old = make_profile(shapes=[_shape(4096, None, "unmeasured")])
    new = make_profile(shapes=None)
    result = diff_profiles(old, new)
    assert [c for c in result.changes if c.family == "ceiling_shapes"] == []
    assert not any("4096" in name for name in result.dropped)


# --- verdicts --------------------------------------------------------


def test_verdict_fall_down_the_ladder_is_regression():
    result = diff_profiles(
        make_profile(),
        make_profile(verdicts=make_verdicts(
            structured_extraction={"verdict": "risky", "provisional": False,
                                   "lens": {}})))
    (change,) = [c for c in result.changes if c.family == "verdict"]
    assert change.cell == "structured_extraction"
    assert change.direction == "regression" and change.basis == "flip"
    assert (change.old, change.new) == ("ready", "risky")


def test_verdict_climb_up_the_ladder_is_improvement():
    result = diff_profiles(
        make_profile(verdicts=make_verdicts(
            patch_editing={"verdict": "unusable", "provisional": False,
                           "lens": {}})),
        make_profile())
    (change,) = [c for c in result.changes if c.family == "verdict"]
    assert change.direction == "improvement"


def test_unmeasured_verdict_is_dropped_never_scored():
    result = diff_profiles(
        make_profile(verdicts=make_verdicts(
            patch_editing={"verdict": "unmeasured", "lens": {}})),
        make_profile())
    assert [c for c in result.changes if c.family == "verdict"] == []
    assert "verdict.patch_editing" in result.dropped


def test_verdict_present_on_one_side_only_is_dropped():
    thin = make_verdicts()
    del thin["patch_editing"]
    result = diff_profiles(make_profile(verdicts=thin), make_profile())
    assert "verdict.patch_editing" in result.dropped


def test_verdicts_unmeasured_on_both_sides_are_not_dropped():
    """`dropped` means "measured on exactly one side" — the rule the
    other four families already follow (`_exact`, `_diff_codec_cell`,
    `_diff_speed_cell` all return no cell when both sides are absent).

    Reproduced live against 0.9.0 by bloomery's drift watch: two
    byte-equal profiles reported `dropped: verdict.long_context`.
    Nothing vanished between them, and after v1.8 the exit code reads
    `dropped` directly — so a display-layer lie here would become an
    exit-code lie.
    """
    both = make_verdicts(long_context={"verdict": "unmeasured", "lens": {}})
    result = diff_profiles(make_profile(verdicts=both),
                           make_profile(verdicts=json.loads(json.dumps(both))))
    assert result.dropped == ()
    assert [c for c in result.changes if c.family == "verdict"] == []
    # Not "checked and clean" either: an unmeasured cell was not checked.
    assert "verdict.long_context" not in result.within_noise


def test_verdicts_keyed_but_unmeasurable_on_both_sides_are_not_dropped():
    """The same rule for a key BOTH documents carry with nothing in it.

    `_verdict_of(None)` yields `(None, None)`, so this name does reach
    the guard — unlike a key neither side carries, which never enters
    the iteration at all and would pass this test for the wrong reason.
    """
    both = make_verdicts(patch_editing=None)
    result = diff_profiles(make_profile(verdicts=both),
                           make_profile(verdicts=json.loads(json.dumps(both))))
    assert result.dropped == ()
    assert [c for c in result.changes if c.family == "verdict"] == []
    assert "verdict.patch_editing" not in result.within_noise


def _long_output(verdict):
    return {"verdict": verdict, "provisional": True, "lens": {}}


def _long_output_change(old_verdict, new_verdict):
    result = diff_profiles(
        make_profile(verdicts=make_verdicts(
            long_output=_long_output(old_verdict))),
        make_profile(verdicts=make_verdicts(
            long_output=_long_output(new_verdict))))
    (change,) = [c for c in result.changes if c.cell == "long_output"]
    return change


@pytest.mark.parametrize(("old", "new", "expected"), [
    # ready > risky > degrades-at-N > unusable (ruled 2026-08-15).
    ("ready", "degrades-at-2048", "regression"),
    ("degrades-at-2048", "ready", "improvement"),
    ("risky", "degrades-at-2048", "regression"),
    ("degrades-at-2048", "risky", "improvement"),
    ("degrades-at-1024", "unusable", "regression"),
    ("unusable", "degrades-at-512", "improvement"),
    # ...and between two degrades-at rungs, the LARGER N is better:
    # holding together to 2048 before looping beats looping at 1024.
    ("degrades-at-1024", "degrades-at-2048", "improvement"),
    ("degrades-at-2048", "degrades-at-1024", "regression"),
    ("degrades-at-512", "degrades-at-4096", "improvement"),
])
def test_the_degrades_at_rung_takes_its_place_on_the_ladder(old, new, expected):
    change = _long_output_change(old, new)
    assert change.direction == expected
    assert change.basis == "flip"
    assert (change.old, change.new) == (old, new)


def test_degrades_at_is_its_own_rung_not_a_flavour_of_unusable():
    # The extent term alone would ORDER these correctly by accident
    # (any degrades-at-N outranks unusable on N), which is why the rung
    # term is asserted directly: degrades-at is a position on the
    # ladder, strictly between unusable and risky, and a model that
    # loops from 512 is not the same finding as one that never held
    # together at all.
    from assay.diff import _rung_rank

    order = ["unusable", "degrades-at-512", "degrades-at-4096",
             "risky", "ready"]
    ranks = [_rung_rank(value) for value in order]
    assert ranks == sorted(ranks), "the ladder must be strictly ordered"
    assert len(set(ranks)) == len(ranks)
    assert _rung_rank("unusable")[0] < _rung_rank("degrades-at-512")[0]
    assert _rung_rank("degrades-at-4096")[0] < _rung_rank("risky")[0]
    # Rungs that carry no extent all sit at extent 0, so the second
    # term never reorders them against each other.
    assert {_rung_rank(v)[1] for v in ("unusable", "risky", "ready")} == {0}


def test_a_rung_name_this_comparator_cannot_read_is_not_ranked():
    # Something moved and the diff says so, but an unparsable rung is
    # not scored as a regression: ranking it would be a guess, and a
    # guessed direction is what a gate would act on.
    change = _long_output_change("ready", "degrades-at-soon")
    assert change.direction == "neutral"


def test_provisional_flip_alone_is_neutral():
    result = diff_profiles(
        make_profile(),
        make_profile(verdicts=make_verdicts(
            structured_extraction={"verdict": "ready", "provisional": True,
                                   "lens": {}})))
    (change,) = [c for c in result.changes if c.family == "verdict"]
    assert change.cell == "structured_extraction.provisional"
    assert change.direction == "neutral" and change.basis == "flip"


# --- v1.6: unsupported, the rung below unusable ----------------------


def _tool_calling(verdict):
    """A ``tool_calling`` verdict entry as ``profile.py`` writes the
    unsupported one: no interval, because there is no proportion to
    bound."""
    return {"verdict": verdict, "provisional": False,
            "interval95": None, "lens": {}}


def _tool_calling_change(old_verdict, new_verdict):
    result = diff_profiles(
        make_profile(verdicts=make_verdicts(
            tool_calling=_tool_calling(old_verdict))),
        make_profile(verdicts=make_verdicts(
            tool_calling=_tool_calling(new_verdict))))
    (change,) = [c for c in result.changes if c.cell == "tool_calling"]
    return change


@pytest.mark.parametrize(("old", "new", "expected"), [
    # ready > risky > degrades-at-N > unusable > unsupported.
    # An endpoint that used to take the tools parameter and now refuses
    # it lost a capability: every rung above falls to unsupported.
    ("ready", "unsupported", "regression"),
    ("risky", "unsupported", "regression"),
    ("degrades-at-2048", "unsupported", "regression"),
    ("unusable", "unsupported", "regression"),
    # ...and any measured rung is an improvement on a refusal, INCLUDING
    # unusable: being asked and failing is more than never being asked.
    ("unsupported", "ready", "improvement"),
    ("unsupported", "risky", "improvement"),
    ("unsupported", "degrades-at-512", "improvement"),
    ("unsupported", "unusable", "improvement"),
])
def test_unsupported_is_the_bottom_rung_of_the_ladder(old, new, expected):
    change = _tool_calling_change(old, new)
    assert change.direction == expected
    assert change.basis == "flip"
    assert (change.old, change.new) == (old, new)


def test_unsupported_is_a_rung_below_unusable_not_a_flavour_of_it():
    # Asserted on the rank directly, not only through directions: a
    # comparator that gave unsupported the SAME rank as unusable would
    # pass every ordering pair above that does not involve the two of
    # them, and would then score unusable -> unsupported as neutral —
    # the flip a gate exists to catch.
    from assay.diff import _rung_rank

    order = ["unsupported", "unusable", "degrades-at-512",
             "degrades-at-4096", "risky", "ready"]
    ranks = [_rung_rank(value) for value in order]
    assert ranks == sorted(ranks), "the ladder must be strictly ordered"
    assert len(set(ranks)) == len(ranks)
    assert _rung_rank("unsupported")[0] < _rung_rank("unusable")[0]
    # It carries no extent, like every other named rung: the second term
    # must never reorder it against unusable.
    assert _rung_rank("unsupported")[1] == 0


def test_unsupported_is_scored_not_dropped_the_way_unmeasured_is():
    """The refusal IS a measurement. ``unmeasured`` drops (nothing was
    observed); ``unsupported`` is a fact about the endpoint and has to
    reach the gate."""
    result = diff_profiles(
        make_profile(verdicts=make_verdicts(
            tool_calling=_tool_calling("ready"))),
        make_profile(verdicts=make_verdicts(
            tool_calling=_tool_calling("unsupported"))))
    assert "verdict.tool_calling" not in result.dropped
    assert [c.cell for c in result.changes if c.family == "verdict"] == [
        "tool_calling"]


def test_two_unsupported_sides_are_checked_and_clean_not_a_change():
    result = diff_profiles(
        make_profile(verdicts=make_verdicts(
            tool_calling=_tool_calling("unsupported"))),
        make_profile(verdicts=make_verdicts(
            tool_calling=_tool_calling("unsupported"))))
    assert [c for c in result.changes if c.cell == "tool_calling"] == []
    assert "verdict.tool_calling" in result.within_noise


def test_a_committed_v4_profile_still_diffs_against_itself():
    """The v1.6 rung must not change how a pre-v1.6 document reads: a
    real v4 profile — no tools family, no tool_calling verdict — against
    itself is comparable, clean, and non-vacuous."""
    payload = json.loads(
        (EVIDENCE / "tier-enthusiast/qwen3-14b.json").read_text(
            encoding="utf-8"))
    assert payload["assay_profile_version"] == 4
    assert "tools" not in payload and "tool_calling" not in payload["verdicts"]

    result = diff_profiles(payload, payload)

    assert result.comparable
    assert result.changes == ()
    assert "verdict.tool_calling" not in result.dropped
    assert len(result.within_noise) >= 13, result.within_noise


def test_old_schema_verdicts_do_not_manufacture_a_provisional_cell():
    """Neither v1 profile HAS a provisional flag, so there is nothing to
    drop — a cell absent from both sides is not a cell."""
    old = make_profile(verdicts={"structured_extraction": "ready"}, ceiling=None)
    new = make_profile(verdicts={"structured_extraction": "ready"}, ceiling=None)
    result = diff_profiles(old, new)
    assert not [name for name in result.dropped if "provisional" in name]
    assert not [name for name in result.dropped if name.startswith("ceiling.")]


def test_old_schema_string_verdicts_still_compare():
    """A v1 profile stored ``"ready"``, not ``{"verdict": "ready"}``."""
    old = make_profile(verdicts={"structured_extraction": "ready"})
    new = make_profile(verdicts={"structured_extraction": "unusable"})
    (change,) = [c for c in diff_profiles(old, new).changes
                 if c.family == "verdict"]
    assert (change.old, change.new) == ("ready", "unusable")
    assert change.direction == "regression"


# --- codecs ----------------------------------------------------------


def _one_cell_codecs(lands, n, lands_applies=_MISSING):
    return {"json_object": {"small": codec_cell(lands, lands_applies, n)}}


def test_overlapping_codec_intervals_are_within_noise():
    # 4/5 = [0.376, 0.964] vs 5/5 = [0.566, 1.000]: overlap -> no Change.
    result = diff_profiles(make_profile(codecs=_one_cell_codecs(0.8, 5)),
                           make_profile(codecs=_one_cell_codecs(1.0, 5)))
    assert [c for c in result.changes if c.family == "codec"] == []
    assert "codec.json_object.small.lands" in result.within_noise


def test_disjoint_codec_intervals_flag():
    # 2/35 = [0.016, 0.186] vs 33/35 = [0.814, 0.984]: disjoint -> Change.
    result = diff_profiles(make_profile(codecs=_one_cell_codecs(2 / 35, 35)),
                           make_profile(codecs=_one_cell_codecs(33 / 35, 35)))
    (change,) = [c for c in result.changes if c.family == "codec"]
    assert change.cell == "json_object.small.lands"
    assert change.basis == "disjoint-intervals"
    assert change.direction == "improvement"


def test_disjoint_codec_drop_is_regression():
    result = diff_profiles(make_profile(codecs=_one_cell_codecs(33 / 35, 35)),
                           make_profile(codecs=_one_cell_codecs(2 / 35, 35)))
    (change,) = [c for c in result.changes if c.family == "codec"]
    assert change.direction == "regression"


def test_both_codec_lenses_are_compared():
    old = {"whole_file": {"small": codec_cell(0.0, 2 / 35, 35)}}
    new = {"whole_file": {"small": codec_cell(0.0, 33 / 35, 35)}}
    result = diff_profiles(make_profile(codecs=old), make_profile(codecs=new))
    (change,) = [c for c in result.changes if c.family == "codec"]
    assert change.cell == "whole_file.small.lands_applies"
    assert "codec.whole_file.small.lands" in result.within_noise


def test_zero_n_codec_cell_is_dropped_never_scored():
    old = {"json_object": {"small": codec_cell(None, None, 0)}}
    new = {"json_object": {"small": codec_cell(1.0, 1.0, 5)}}
    result = diff_profiles(make_profile(codecs=old), make_profile(codecs=new))
    assert [c for c in result.changes if c.family == "codec"] == []
    assert "codec.json_object.small.lands" in result.dropped


def test_a_stated_rate_over_zero_samples_is_still_dropped():
    """Defensive: a cell claiming ``lands: 0.0`` on ``n: 0`` must not be
    graded — wilson95(0, 0) is the whole unit interval, which overlaps
    everything and would report the cell as checked-and-clean."""
    old = {"json_object": {"small": codec_cell(0.0, 0.0, 0)}}
    new = {"json_object": {"small": codec_cell(1.0, 1.0, 5)}}
    result = diff_profiles(make_profile(codecs=old), make_profile(codecs=new))
    assert [c for c in result.changes if c.family == "codec"] == []
    assert "codec.json_object.small.lands" in result.dropped
    assert "codec.json_object.small.lands" not in result.within_noise


def test_lens_missing_on_one_side_is_dropped():
    """A v1 cell has no ``lands_applies`` column; absence is not zero."""
    old = {"whole_file": {"small": codec_cell(1.0, _MISSING, 5)}}   # v1 shape
    new = {"whole_file": {"small": codec_cell(0.0, 0.0, 5)}}        # v5 shape
    result = diff_profiles(make_profile(codecs=old), make_profile(codecs=new))
    assert "codec.whole_file.small.lands_applies" in result.dropped


def test_json_objects_coinciding_lenses_report_one_change_not_two():
    """``json_object``'s two lenses are the SAME measurement.

    Validation IS the application for that codec (codecs.py says so
    where it picks the verdict lens), so ``lands`` and ``lands_applies``
    are written from one count. Diffing both turned one measured move
    into two Change rows and two within-noise names — a reader counting
    the report would double every json_object finding.
    """
    old = {"json_object": {"small": codec_cell(2 / 35, 2 / 35, 35)}}
    new = {"json_object": {"small": codec_cell(33 / 35, 33 / 35, 35)}}
    result = diff_profiles(make_profile(codecs=old), make_profile(codecs=new))

    (change,) = [c for c in result.changes if c.family == "codec"]
    assert change.cell == "json_object.small.lands"
    assert not [name for name in result.within_noise + result.dropped
                if name.startswith("codec.json_object")
                and name.endswith(".lands_applies")]


def test_the_other_codecs_keep_both_lenses():
    """The skip is json_object's alone: for the patch codecs the two
    lenses are different instruments and disagreed by 100 points live."""
    cells = {"small": codec_cell(1.0, 1.0, 5)}
    result = diff_profiles(
        make_profile(codecs={"search_replace": cells, "whole_file": cells}),
        make_profile(codecs={"search_replace": cells, "whole_file": cells}))
    for codec in ("search_replace", "whole_file"):
        for lens in ("lands", "lands_applies"):
            assert f"codec.{codec}.small.{lens}" in result.within_noise


def test_codec_absent_on_one_side_is_dropped():
    result = diff_profiles(make_profile(codecs=make_codecs()),
                           make_profile(codecs={"json_object": {}}))
    assert "codec.whole_file.small.lands" in result.dropped
    assert [c for c in result.changes if c.family == "codec"] == []


# --- speed -----------------------------------------------------------


def test_absent_cell_is_dropped_never_scored():
    result = diff_profiles(make_profile(), make_profile(speed=None))
    assert result.changes == ()
    assert "speed.decode_tps" in result.dropped
    assert "speed.prefill_tps" in result.dropped


def test_a_cell_absent_from_both_sides_is_not_reported_at_all():
    """Two pre-speed profiles are not "two dropped speed cells" — there
    is no cell. Dropped means *one* side measured it."""
    result = diff_profiles(make_profile(speed=None), make_profile(speed=None))
    assert not [name for name in result.dropped if name.startswith("speed.")]
    assert not [name for name in result.within_noise if name.startswith("speed.")]


def test_speed_welch_beyond_two_se_flags():
    old = make_speed(decode_tps=10.0, decode_samples=[10.0, 10.2, 9.8])
    new = make_speed(decode_tps=5.0, decode_samples=[5.0, 5.2, 4.8])
    result = diff_profiles(make_profile(speed=old), make_profile(speed=new))
    (change,) = [c for c in result.changes if c.cell == "decode_tps"]
    assert change.basis == "beyond-2se" and change.direction == "regression"


def test_faster_decode_is_improvement():
    old = make_speed(decode_tps=5.0, decode_samples=[5.0, 5.2, 4.8])
    new = make_speed(decode_tps=10.0, decode_samples=[10.0, 10.2, 9.8])
    result = diff_profiles(make_profile(speed=old), make_profile(speed=new))
    (change,) = [c for c in result.changes if c.cell == "decode_tps"]
    assert change.direction == "improvement"


def test_noisy_samples_beat_the_20pct_rule_of_thumb():
    """30% apart, but the samples are wide enough that 2 SE covers it:
    with real spread on the page the assumed threshold does not rule."""
    old = make_speed(decode_tps=10.0, decode_samples=[5.0, 15.0])
    new = make_speed(decode_tps=13.0, decode_samples=[6.0, 20.0])
    result = diff_profiles(make_profile(speed=old), make_profile(speed=new))
    assert [c for c in result.changes if c.cell == "decode_tps"] == []
    assert "speed.decode_tps" in result.within_noise


def test_tight_samples_flag_a_change_the_20pct_rule_would_miss():
    """10% apart — under the rule of thumb — but the runs were tight:
    2 SE is 0.33 tok/s and the gap is 1.0, so the gap is real."""
    old = make_speed(decode_tps=10.0, decode_samples=[10.0, 10.2, 9.8])
    new = make_speed(decode_tps=11.0, decode_samples=[11.0, 11.2, 10.8])
    result = diff_profiles(make_profile(speed=old), make_profile(speed=new))
    (change,) = [c for c in result.changes if c.cell == "decode_tps"]
    assert change.basis == "beyond-2se" and change.direction == "improvement"


def test_zero_variance_samples_still_flag_a_real_gap():
    old = make_speed(decode_tps=10.0, decode_samples=[10.0, 10.0])
    new = make_speed(decode_tps=11.0, decode_samples=[11.0, 11.0])
    result = diff_profiles(make_profile(speed=old), make_profile(speed=new))
    (change,) = [c for c in result.changes if c.cell == "decode_tps"]
    assert change.basis == "beyond-2se"


def test_speed_fallback_threshold_is_named_assumed():
    # Both sides one call, no samples on the page, 30% apart.
    old = make_speed(decode_tps=10.0, n_decode=1, decode_samples=None)
    new = make_speed(decode_tps=7.0, n_decode=1, decode_samples=None)
    result = diff_profiles(make_profile(speed=old), make_profile(speed=new))
    (change,) = [c for c in result.changes if c.cell == "decode_tps"]
    assert change.basis == "threshold-20pct-assumed"
    assert change.direction == "regression"


def test_speed_fallback_under_the_threshold_is_within_noise():
    old = make_speed(decode_tps=10.0, decode_samples=None)
    new = make_speed(decode_tps=11.0, decode_samples=None)
    result = diff_profiles(make_profile(speed=old), make_profile(speed=new))
    assert [c for c in result.changes if c.cell == "decode_tps"] == []
    assert "speed.decode_tps" in result.within_noise


def test_empty_sample_list_falls_back_to_the_assumed_threshold():
    """``[]`` means sampling ran and accepted nothing — it is not two
    samples, so Welch has nothing to work with."""
    old = make_speed(decode_tps=10.0, decode_samples=[])
    new = make_speed(decode_tps=13.0, decode_samples=[])
    result = diff_profiles(make_profile(speed=old), make_profile(speed=new))
    (change,) = [c for c in result.changes if c.cell == "decode_tps"]
    assert change.basis == "threshold-20pct-assumed"


def test_one_sided_samples_fall_back_to_the_assumed_threshold():
    old = make_speed(decode_tps=10.0, decode_samples=[10.0, 10.0, 10.0])
    new = make_speed(decode_tps=13.0, decode_samples=None)
    result = diff_profiles(make_profile(speed=old), make_profile(speed=new))
    (change,) = [c for c in result.changes if c.cell == "decode_tps"]
    assert change.basis == "threshold-20pct-assumed"


def test_speed_rate_measured_on_one_side_only_is_dropped():
    old = make_speed(prefill_tps=None, prefill_samples=[], n_prefill=0)
    result = diff_profiles(make_profile(speed=old), make_profile())
    assert "speed.prefill_tps" in result.dropped
    assert [c for c in result.changes if c.cell == "prefill_tps"] == []


# --- render ----------------------------------------------------------


def test_render_says_no_drift_when_nothing_changed():
    p = make_profile()
    assert render_diff(diff_profiles(p, p)).splitlines()[0] == "no drift beyond noise"


def test_render_names_family_cell_direction_and_basis():
    result = diff_profiles(make_profile(max_verified=15872),
                           make_profile(max_verified=11520))
    assert ("ceiling.max_verified: 15872 -> 11520 (regression, rung-change)"
            in render_diff(result))


def test_render_keeps_rates_looking_like_rates():
    """``1 -> 0`` would read as a count; a landing rate must not shed
    its decimal point on the way to the page."""
    result = diff_profiles(make_profile(codecs=_one_cell_codecs(1.0, 35)),
                           make_profile(codecs=_one_cell_codecs(0.0, 35)))
    assert "codec.json_object.small.lands: 1.0 -> 0.0" in render_diff(result)


def test_render_preserves_a_fractional_speed_value():
    """The rendered number IS the measured number: 3456.78 tok/s must
    not reach the page as 3457.0, which would assert a precision the
    measurement never had."""
    old = make_speed(prefill_tps=3456.78, prefill_samples=None)
    new = make_speed(prefill_tps=1234.5, prefill_samples=None)
    text = render_diff(diff_profiles(make_profile(speed=old),
                                     make_profile(speed=new)))
    assert ("speed.prefill_tps: 3456.78 -> 1234.5"
            " (regression, threshold-20pct-assumed)" in text)


def test_render_never_falls_back_to_scientific_notation():
    old = make_speed(prefill_tps=12345.6, prefill_samples=None)
    new = make_speed(prefill_tps=1000.0, prefill_samples=None)
    text = render_diff(diff_profiles(make_profile(speed=old),
                                     make_profile(speed=new)))
    assert "speed.prefill_tps: 12345.6 -> 1000.0" in text
    assert "e+" not in text


def test_render_prints_the_assumed_basis_verbatim():
    old = make_speed(decode_tps=10.0, decode_samples=None)
    new = make_speed(decode_tps=13.0, decode_samples=None)
    text = render_diff(diff_profiles(make_profile(speed=old),
                                     make_profile(speed=new)))
    assert "threshold-20pct-assumed" in text


def test_render_lists_within_noise_and_dropped():
    result = diff_profiles(make_profile(), make_profile(speed=None))
    text = render_diff(result)
    assert "within noise: " in text
    assert "dropped: " in text and "speed.decode_tps" in text


def test_render_names_the_incomplete_comparison(capsys):
    """The prose and the exit code tell ONE story. A reader who sees
    exit 3 must find the reason on the page without re-running with
    --json."""
    thin = make_verdicts()
    del thin["patch_editing"]
    result = diff_profiles(make_profile(verdicts=thin), make_profile())
    page = render_diff(result)
    assert "incomplete: 1 cell(s) measured on one side only" in page
    assert "dropped: verdict.patch_editing" in page


def test_the_render_names_incomparable_separately_from_dropped():
    """A reader must be able to act on the difference. Re-running fixes
    a one-sided cell; nothing fixes a cell the two instruments defined
    differently."""
    thin = make_verdicts(parallel={"verdict": "risky", "lens": {}})
    del thin["patch_editing"]
    old = make_profile(probe_version="0.10.0", verdicts=thin)
    new = make_profile(probe_version="0.11.0", verdicts=make_verdicts(
        parallel={"verdict": "ready", "lens": {}}))
    page = render_diff(diff_profiles(old, new))

    assert "incomparable: 1 cell(s) measured under a different rule" in page
    assert "verdict.parallel" in page
    assert "incomplete: 1 cell(s) measured on one side only" in page
    assert "dropped: verdict.patch_editing" in page


def test_render_of_an_incomparable_pair_says_why():
    text = render_diff(diff_profiles(make_profile(),
                                     make_profile(model_name="other-model")))
    assert text.splitlines()[0] == "not comparable"
    assert "model.name" in text


def test_render_shows_warning_notes_on_a_comparable_pair():
    text = render_diff(diff_profiles(make_profile(quant=None), make_profile()))
    assert "note: " in text and "model.quant" in text


def test_a_one_sided_none_is_dropped_not_rendered_as_a_change():
    """The None-vs-zero rule at the render layer: an unmeasured ceiling
    on one side shows up under ``dropped``, never as a rung change."""
    old = make_profile(ceiling={"max_verified": None, "first_failure": None,
                                "failure_mode": "hard_error",
                                "counts_available": False, "evidence": []})
    text = render_diff(diff_profiles(old, make_profile()))
    assert "ceiling.max_verified" in text.split("dropped: ")[1]
    assert "ceiling.max_verified:" not in text


# --- shapes of the public types --------------------------------------


def test_change_and_result_are_frozen():
    change = Change(family="ceiling", cell="max_verified", direction="neutral",
                    old=1, new=1, basis="rung-change")
    with pytest.raises(Exception):
        change.family = "speed"
    result = DiffResult(comparable=True, identity_notes=(), changes=(),
                        within_noise=(), dropped=())
    with pytest.raises(Exception):
        result.comparable = False


# --- real-data acceptance: the committed same-day rerun pairs --------

_RERUN_PAIRS = ("qwen2.5-coder-7b-instruct-q8_0-quick.json",
                "codegemma-7b-instruct-q8_0-quick.json",
                "granite-code-8b-instruct-q8_0-quick.json")


def _rerun_pair(name: str) -> tuple[dict, dict]:
    return (json.loads((EVIDENCE / "live" / name).read_text(encoding="utf-8")),
            json.loads((EVIDENCE / "live-run2" / name).read_text(encoding="utf-8")))


@pytest.mark.parametrize("name", _RERUN_PAIRS)
def test_live_rerun_pairs_read_within_noise(name):
    """The committed anchor (spec §2): ``live/`` and ``live-run2/`` are
    the same three models, same day, same unrestarted Ollama daemon —
    the recorded example of sampler-level variation that must NOT be
    read as drift.

    These are ``assay_profile_version: 1`` files: bare-string verdicts,
    no ``ceiling_shapes``, no ``speed``, codec cells with no
    ``lands_applies`` column and no per-call samples. Reading them at
    all is half of what this asserts.
    """
    old, new = _rerun_pair(name)
    result = diff_profiles(old, new)
    assert result.comparable, name
    regressions = [change for change in result.changes
                   if change.direction == "regression"]
    assert regressions == [], (name, regressions)
    # Nothing moved beyond noise in EITHER direction, so the pair also
    # exits 0 without --gate. Stronger than the regression clause, and
    # true of the committed files as measured on 2026-08-12.
    assert result.changes == (), (name, result.changes)
    # ...and it is clean because the comparator absorbed the movement,
    # not because it compared nothing: 13-14 cells were checked, the
    # ceiling and the whole codec matrix among them.
    assert "ceiling.max_verified" in result.within_noise, name
    assert len(result.within_noise) >= 13, (name, result.within_noise)


def test_the_rerun_codec_cells_that_actually_moved_are_absorbed_as_noise():
    """Non-vacuity, stated as the evidence README states it: granite's
    "stray landings moving 0.2 -> 0.0 between runs shows the n=5 noise
    scale". Five codec cells genuinely differ across the two runs —
    codegemma ``search_replace.tiny`` 0.2 -> 0.0 and granite's
    ``whole_file.tiny`` 0.2 -> 0.0, ``whole_file.medium`` 0.4 -> 0.0,
    ``json_object.tiny`` 0.2 -> 0.0, ``json_object.small`` 0.2 -> 0.0.
    Every one is one or two probes out of five; Wilson-95 at n=5 must
    swallow all of them, and a comparator that flagged them would make
    the tool unreadable on its own recorded data.
    """
    moved = []
    for name in _RERUN_PAIRS:
        old, new = _rerun_pair(name)
        within_noise = diff_profiles(old, new).within_noise
        for codec, grades in old["codecs"].items():
            for grade, cell in grades.items():
                if cell["lands"] == new["codecs"][codec][grade]["lands"]:
                    continue
                moved.append((name, codec, grade))
                assert f"codec.{codec}.{grade}.lands" in within_noise, (
                    name, codec, grade)
    assert len(moved) == 5, moved


# --- cross-version: today's shape against a committed v4 profile -----
#
# These are REGRESSION PINS, not a fix: the comparator was written to
# read every schema this project has ever written, and it already
# handles the v1.7 additions by the None-is-never-zero rule. What was
# never pinned is the whole cross-version path end to end — an operator
# diffing last month's committed profile against a run made today —
# and the families v1.7 adds are exactly the shape of gap that reads as
# a regression if absence is ever scored.

_COMMITTED_V4 = EVIDENCE / "tier-enthusiast/qwen2-5-coder-1-5b-instruct-q8_0.json"
#: The three json SHAPE grades v1.7 added (``codecs.JSON_DEEP_GRADES``).
#: Named here rather than imported so the pin states the strings a v4
#: file cannot contain, whatever the probe later calls them.
_DEEP_GRADES = ("nested", "tabular", "constrained")


def _committed_v4() -> dict:
    payload = json.loads(_COMMITTED_V4.read_text(encoding="utf-8"))
    assert payload["assay_profile_version"] == 4, "the fixture is the old shape"
    assert "tools" not in payload
    assert set(payload["codecs"]["json_object"]) == set(_GRADES_V4)
    return payload


_GRADES_V4 = ("tiny", "small", "medium")


def _current_shape_for(old: dict, **overrides) -> dict:
    """A profile in TODAY's shape, from the profile factories: the tools
    family, a ``tool_calling`` verdict, and ``json_object`` measured at
    six grades.

    Its identity is pinned to the old file's — same model, quant,
    weights, tier — because the question here is what happens across a
    SCHEMA boundary, and an identity mismatch would answer a different
    one (that path is pinned separately, below).
    """
    from assay.profile import PROFILE_VERSION
    from test_profile import make_codecs, make_profile, make_tools

    profile = make_profile(
        assay_profile_version=PROFILE_VERSION,
        model=dict(old["model"]),
        codecs=make_codecs(deep=True),
        tools=make_tools(),
        verdicts={
            "structured_extraction": {"verdict": "ready", "provisional": False,
                                      "interval95": None, "lens": {}},
            "patch_editing": {"verdict": "ready", "provisional": False,
                              "interval95": None, "lens": {}},
            "long_context": {"verdict": "ready", "provisional": False,
                             "interval95": None, "lens": {}},
            "tool_calling": _tool_calling("ready"),
        },
        **overrides)
    payload = json.loads(profile.to_json())
    # The gate reads tier and emulated off provenance; the factory's
    # provenance is a fixture, so the hardware half of the identity is
    # taken from the file being compared against.
    payload["provenance"]["tier"] = old["provenance"]["tier"]
    payload["provenance"]["emulated"] = old["provenance"]["emulated"]
    return payload


def test_current_shape_against_v4_drops_the_families_v4_could_not_carry():
    """The families v1.7 added are absent from a v4 file because the
    schema had no place for them, not because the model failed at them.
    Every one must reach ``dropped`` and NONE may reach ``changes``: a
    comparator that scored them would report a regression on the day the
    schema grew, on every archived profile at once.
    """
    old = _committed_v4()
    new = _current_shape_for(old)

    result = diff_profiles(old, new)

    assert result.comparable, result.identity_notes
    assert "verdict.tool_calling" in result.dropped
    for grade in _DEEP_GRADES:
        assert f"codec.json_object.{grade}.lands" in result.dropped, grade
    absent = {"tool_calling", *_DEEP_GRADES}
    assert not [change for change in result.changes
                if any(name in change.cell for name in absent)], result.changes
    # Non-vacuity: the cells both files DO carry were compared, so the
    # clean half of this is a comparison and not an empty run.
    assert "codec.json_object.small.lands" in (
        result.within_noise + tuple(c.cell for c in result.changes))


def test_current_shape_against_v4_renders_the_absent_cells_as_unmeasured():
    """...and the reader is told. ``render_diff`` names every dropped
    cell in full — the shape grades and the tools verdict included — so
    "not compared" is on the page rather than being silence the reader
    has to notice."""
    old = _committed_v4()
    rendered = render_diff(diff_profiles(old, _current_shape_for(old)))

    dropped_line = next(line for line in rendered.splitlines()
                        if line.startswith("dropped: "))
    assert "verdict.tool_calling" in dropped_line
    for grade in _DEEP_GRADES:
        assert f"codec.json_object.{grade}.lands" in dropped_line, grade
    assert "not comparable" not in rendered


def test_cli_diff_across_the_version_boundary_takes_the_documented_exits(
        tmp_path, capsys):
    """The documented codes on this path, from the CLI docstring:
    0 nothing moved beyond noise, 1 drift found, 2 not comparable, 3
    incomplete.

    The v4 file against itself is 0 — the version boundary alone
    manufactures nothing. Against a factory profile of the same model it
    is 3, not 1: the schema grew between v4 and the current shape, so
    tool_calling and the deep json grades were measured on the new side
    only, and under v1.8 that incompleteness outranks the real drift
    this fixture's ceiling and speed values also carry (this pair reads
    3 because the newer schema actually measured cells — tool_calling,
    the deep json grades — that the older side lacks, not merely
    because the schemas differ; the instrument-changed rule enforcing
    itself). Never 2, which would mean the schema bump had broken the
    identity gate, and never 4, which would mean the current shape no
    longer parses as a profile document.
    """
    from assay.cli import main

    old = _committed_v4()
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps(old), encoding="utf-8")
    new_path = tmp_path / "new.json"
    new_path.write_text(json.dumps(_current_shape_for(old)), encoding="utf-8")
    other_path = tmp_path / "other.json"
    other = _current_shape_for(old)
    other["model"] = {**other["model"], "name": "some-other-model:14b"}
    other_path.write_text(json.dumps(other), encoding="utf-8")

    assert main(["diff", str(old_path), str(old_path)]) == 0
    assert "no drift beyond noise" in capsys.readouterr().out

    assert main(["diff", str(old_path), str(new_path)]) == 3
    capsys.readouterr()

    # The identity gate is unchanged by the new families: a different
    # model is still fatal, still reports nothing, still exits 2 — 2
    # outranks 3, so this pair never even reaches the incompleteness
    # question.
    assert main(["diff", str(old_path), str(other_path)]) == 2
    out = capsys.readouterr().out
    assert out.startswith("not comparable")
    assert "dropped" not in out


def test_version_ordering_is_parsed_not_lexical():
    """The trap this parser exists for. `"0.9.0" < "0.11.0"` is False
    under string comparison, because "9" sorts after "1" — so a lexical
    check would decide a 0.9.0-vs-0.11.0 pair does NOT straddle a
    0.11.0 break and would score it. The failure direction is the
    dangerous one: silently comparing two numbers that answer different
    questions.
    """
    from assay.diff import _parse_version

    assert _parse_version("0.9.0") < _parse_version("0.11.0")
    assert _parse_version("0.11.0") > _parse_version("0.10.0")
    assert _parse_version("1.0.0") > _parse_version("0.99.0")
    # The literal string comparison this replaces, pinned so nobody
    # "simplifies" the parser away.
    assert ("0.9.0" < "0.11.0") is False


def test_an_unparseable_version_is_not_a_version():
    from assay.diff import _parse_version

    for value in (None, "", "not-a-version", "0.x.0", 11, []):
        assert _parse_version(value) is None, repr(value)


def test_a_non_ascii_digit_returns_none_instead_of_raising():
    """`str.isdigit()` is True for superscripts and subscripts, which
    `int()` rejects — so `_parse_version("1.².0")` RAISED where it
    promises None. `main()` catches only `BudgetExhausted` and
    `InfrastructureError`, so the ValueError escaped the documented
    0/1/2/3/4 exit taxonomy as an uncaught traceback, and §6's
    unparseable-straddles rule never got the chance to run.
    `isdecimal()` is the predicate that means what this parser needs.
    """
    from assay.diff import _parse_version, _straddles

    for value in ("1.².0", "1.₂.0"):
        assert _parse_version(value) is None, repr(value)
    # And the value it should have had all along: an instrument we
    # cannot identify is not a comparable one.
    assert _straddles("verdict.parallel", "1.².0", "0.11.0")
    # The boundary `isdecimal` actually draws, stated rather than
    # implied: it promises `int()` will not raise, NOT that the digits
    # are ASCII. Arabic-Indic digits are decimal and `int()` reads them,
    # so this parses to a real version instead of being rejected. That
    # is the honest consequence of the predicate, and harmless — assay
    # writes `probe_version` from `__version__`, which is ASCII.
    assert _parse_version("١.٢.٣") == (1, 2, 3)


def test_a_version_that_is_not_three_components_is_not_a_version():
    """The silent-sort trap. `(0, 11)` compares BELOW `(0, 11, 0)`, so a
    two-component `"0.11"` — a document written under the NEW rule — was
    filed on the OLD side of a 0.11.0 break and scored. Rejected rather
    than padded: assay writes `probe_version` from `__version__`, which
    is always three components, so anything else was not written by this
    instrument, and padding would assert a patch level the document
    never stated. `None` routes to §6's discipline — unknown instrument,
    not comparable — which fails toward refusing to score rather than
    toward scoring a pair that must not be scored.
    """
    from assay.diff import _parse_version

    for value in ("0.11", "0", "0.11.0.1"):
        assert _parse_version(value) is None, repr(value)
    assert _parse_version("0.11.0") == (0, 11, 0)


def test_a_two_component_version_does_not_sort_below_its_own_break():
    """The defect end to end, in the direction that publishes a claim.
    A `"0.11"` document was measured under v1.9's rule; scoring it
    against a 0.10.0 baseline reports an instrument change as an
    endpoint improvement — plain exit 1, `--gate` exit 0, the exact
    headline this wave exists to stop.
    """
    old = make_profile(probe_version="0.10.0", verdicts=make_verdicts(
        parallel={"verdict": "risky", "lens": {}}))
    new = make_profile(probe_version="0.11", verdicts=make_verdicts(
        parallel={"verdict": "ready", "lens": {}}))

    result = diff_profiles(old, new)

    assert result.incomparable == ("verdict.parallel",)
    assert [c for c in result.changes if c.cell == "parallel"] == []


def test_a_two_component_version_is_incomparable_against_its_own_era():
    """The other direction. `"0.11"` against `"0.11.0"` is the same era
    by eye, and under the tuple-length bug it straddled for an incorrect
    REASON (a bogus ordering) while its sibling above did not straddle
    at all — two mutually inconsistent answers from one rule. Both now
    resolve the same honest way: the version is not one assay writes, so
    which rule produced it is not established, and §6 refuses to score.
    """
    old = make_profile(probe_version="0.11", verdicts=make_verdicts(
        parallel={"verdict": "risky", "lens": {}}))
    new = make_profile(probe_version="0.11.0", verdicts=make_verdicts(
        parallel={"verdict": "ready", "lens": {}}))

    assert diff_profiles(old, new).incomparable == ("verdict.parallel",)


def test_a_pair_straddling_a_break_is_named_incomparable():
    from assay.diff import _straddles

    assert _straddles("verdict.parallel", "0.10.0", "0.11.0")
    # Order does not matter: an upgrade and a downgrade are equally
    # incomparable, because the two documents still answer different
    # questions.
    assert _straddles("verdict.parallel", "0.11.0", "0.10.0")


def test_a_pair_on_one_side_of_a_break_compares_normally():
    from assay.diff import _straddles

    assert not _straddles("verdict.parallel", "0.10.0", "0.10.0")
    assert not _straddles("verdict.parallel", "0.11.0", "0.11.0")
    assert not _straddles("verdict.parallel", "0.11.0", "0.12.0")


def test_a_cell_with_no_registered_break_never_straddles():
    """The registry is an allowlist of KNOWN breaks. A cell nobody
    registered compares exactly as it did before this wave — which is
    what keeps additive schema bumps working unchanged."""
    from assay.diff import _straddles

    assert not _straddles("ceiling.max_verified", "0.9.0", "0.11.0")
    assert not _straddles("speed.decode_tps", "0.1.0", "0.11.0")


def test_an_unknown_instrument_straddles_every_break():
    """Absence is not evidence of sameness. "We could not establish
    which rule produced this" must resolve to NOT comparable, never to
    comparable-by-default — the discipline the rest of the instrument
    applies to an absent measurement."""
    from assay.diff import _straddles

    assert _straddles("verdict.parallel", None, "0.11.0")
    assert _straddles("verdict.parallel", "0.11.0", None)
    assert _straddles("verdict.parallel", "garbage", "0.11.0")
    assert _straddles("verdict.parallel", None, None)


# --- Task 2: DiffResult.incomparable ----------------------------------


def _ceiling(max_verified):
    """Mirrors ``test_cli.py``'s helper of the same name — only
    ``max_verified`` and ``failure_mode`` are diffed, so a bare
    two-field ceiling is a complete fixture."""
    return {"ceiling": {"max_verified": max_verified,
                        "failure_mode": "hard_error"}}


def test_a_straddled_verdict_is_incomparable_not_scored():
    """v1.9 changed what `verdict.parallel` MEANS — `classify_mode`
    moved from an absolute-seconds overlap test to a fraction of the
    shorter span — so the same endpoint can read `risky` under one rule
    and `ready` under the other. Scoring that as an improvement
    publishes an instrument change as a fact about the endpoint.
    """
    old = make_profile(probe_version="0.10.0", verdicts=make_verdicts(
        parallel={"verdict": "risky", "lens": {}}))
    new = make_profile(probe_version="0.11.0", verdicts=make_verdicts(
        parallel={"verdict": "ready", "lens": {}}))
    result = diff_profiles(old, new)

    assert "verdict.parallel" in result.incomparable
    assert [c for c in result.changes if c.cell == "parallel"] == []
    assert "verdict.parallel" not in result.within_noise
    # NOT dropped: v1.8 pinned `dropped` to mean measured on exactly one
    # side, and both sides measured this. Different fact, different
    # field.
    assert "verdict.parallel" not in result.dropped


def test_the_same_pair_within_one_era_still_scores():
    """The guard must not become always-on — that would be its own kind
    of useless. Two documents from the same side of the break compare
    exactly as they did before this wave."""
    old = make_profile(probe_version="0.11.0", verdicts=make_verdicts(
        parallel={"verdict": "risky", "lens": {}}))
    new = make_profile(probe_version="0.11.0", verdicts=make_verdicts(
        parallel={"verdict": "ready", "lens": {}}))
    result = diff_profiles(old, new)

    assert result.incomparable == ()
    (change,) = [c for c in result.changes if c.cell == "parallel"]
    assert (change.old, change.new) == ("risky", "ready")


def test_a_straddle_does_not_stop_the_other_cells_comparing():
    """The break is per-CELL, not per-document. `ceiling` and `speed`
    rules did not change at 0.11.0, so a v1.9 boundary must not blind
    the whole comparison."""
    old = make_profile(probe_version="0.10.0",
                       verdicts=make_verdicts(
                           parallel={"verdict": "risky", "lens": {}}),
                       **_ceiling(8192))
    new = make_profile(probe_version="0.11.0",
                       verdicts=make_verdicts(
                           parallel={"verdict": "ready", "lens": {}}),
                       **_ceiling(4096))
    result = diff_profiles(old, new)

    assert result.incomparable == ("verdict.parallel",)
    (change,) = [c for c in result.changes if c.family == "ceiling"]
    assert (change.old, change.new) == (8192, 4096)


def test_dropped_and_incomparable_stay_distinct():
    """Two different reasons a comparison can be incomplete, and a
    reader must be able to act on the difference: a one-sided cell is
    fixed by re-running, an incomparable one never can be."""
    thin = make_verdicts(parallel={"verdict": "risky", "lens": {}})
    del thin["patch_editing"]
    old = make_profile(probe_version="0.10.0", verdicts=thin)
    new = make_profile(probe_version="0.11.0", verdicts=make_verdicts(
        parallel={"verdict": "ready", "lens": {}}))
    result = diff_profiles(old, new)

    assert "verdict.patch_editing" in result.dropped
    assert "verdict.parallel" in result.incomparable
    assert "verdict.parallel" not in result.dropped
    assert "verdict.patch_editing" not in result.incomparable


def test_a_straddling_cell_measured_on_one_side_is_dropped_not_incomparable():
    """`dropped` means measured on exactly one side, full stop (v1.8) —
    a cell that ALSO straddles a registered break is no exception.
    Re-running the missing side is exactly what fixes a `dropped` cell,
    which is not true of an `incomparable` one, so filing it as
    `incomparable` tells the operator the wrong story: that the gap can
    never close. Checked in both directions, and BOTH fields asserted
    each time — a build that files the cell as both `dropped` and
    `incomparable` must still fail this.
    """
    old = make_profile(probe_version="0.9.0", verdicts=make_verdicts())
    new = make_profile(probe_version="0.11.0", verdicts=make_verdicts(
        parallel={"verdict": "ready", "lens": {}}))

    old_side_missing = diff_profiles(old, new)
    assert "verdict.parallel" in old_side_missing.dropped
    assert old_side_missing.incomparable == ()

    new_side_missing = diff_profiles(new, old)
    assert "verdict.parallel" in new_side_missing.dropped
    assert new_side_missing.incomparable == ()


def test_an_explicit_unmeasured_straddling_cell_is_also_dropped():
    """`_verdict_of` treats an explicit "unmeasured" verdict the same as
    an absent key — both mean nothing was measured on that side — so a
    straddling cell where one side reads "unmeasured" must land in
    `dropped`, exactly like the absent-key case above, not
    `incomparable`."""
    old = make_profile(probe_version="0.9.0", verdicts=make_verdicts(
        parallel={"verdict": "unmeasured", "lens": {}}))
    new = make_profile(probe_version="0.11.0", verdicts=make_verdicts(
        parallel={"verdict": "ready", "lens": {}}))

    result = diff_profiles(old, new)

    assert "verdict.parallel" in result.dropped
    assert result.incomparable == ()


def test_neither_side_measured_a_straddling_cell_is_nothing_at_all():
    """A third outcome for a straddling cell, distinct from both above:
    neither side measured it at all. `dropped` doesn't apply — v1.8
    pinned that word to mean measured on exactly one side, and nothing
    was measured here on EITHER side — and `incomparable` doesn't apply
    either: there is no pair of answers to be incompatible when neither
    rule ever ran. The cell appears in nothing: not `dropped`, not
    `incomparable`, not `changes`, not `within_noise`.

    This falls out of the fixed ordering for free — the both-unmeasured
    `continue` runs before the straddle check ever looks at this cell —
    but it is cheap to pin so a future "simplification" that moves the
    guard back above the measured determination gets caught here too.
    """
    old = make_profile(probe_version="0.9.0", verdicts=make_verdicts(
        parallel={"verdict": "unmeasured", "lens": {}}))
    new = make_profile(probe_version="0.11.0", verdicts=make_verdicts())

    result = diff_profiles(old, new)

    assert result.dropped == ()
    assert result.incomparable == ()
    assert "parallel" not in [c.cell for c in result.changes]
    assert "verdict.parallel" not in result.within_noise
