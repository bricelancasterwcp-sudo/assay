"""Cover: the crossed-model coverage check (spec: docs/superpowers/
specs/2026-08-19-assay-v1.11-cover-design.md)."""

from assay.cover import (CoverResult, _cover_exit_code, cover_identity_gate,
                         cover_profiles, render_cover)
from assay.diff import Change


def _profile(*, name="qwen-a", quant="Q4_K_M", weights=1000,
             tier="enthusiast-16gb", emulated=False,
             probe_version="0.13.0", schema=10, **families):
    """Minimal profile payload. Family payloads (verdicts=, speed=,
    ceiling=...) merge in as top-level keys, matching what
    `assay probe --json` writes and what the diff walkers read."""
    doc = {
        "assay_profile_version": schema,
        "probe_version": probe_version,
        "model": {"name": name, "quant": quant, "weights_bytes": weights},
        "provenance": {"tier": tier, "emulated": emulated},
    }
    doc.update(families)
    return doc


def _verdicts(**names):
    """{"patch_editing": "ready"} -> the profile's verdicts payload."""
    return {"verdicts": {name: {"verdict": verdict}
                         for name, verdict in names.items()}}


def test_crossed_model_names_pass_the_cover_gate():
    """The inversion this gate exists for: two different models is the
    point of the command, not a refusal."""
    comparable, notes = cover_identity_gate(
        _profile(name="qwen-a", quant="Q4_K_M", weights=1000),
        _profile(name="qwen-b", quant="Q8_0", weights=2000))
    assert comparable
    # Recorded, never decisive — a reader must still see what differed.
    assert any("model.name" in note for note in notes)
    assert any("model.quant" in note for note in notes)
    assert any("model.weights_bytes" in note for note in notes)


def test_tier_mismatch_refuses():
    comparable, notes = cover_identity_gate(
        _profile(tier="enthusiast-16gb"), _profile(tier="basic-8gb"))
    assert not comparable
    assert any("provenance.tier" in note for note in notes)


def test_one_sided_emulated_refuses():
    """Absent and present-with-null make the same claim: nobody
    declared the hardware. Same absent-is-fatal rule as diff's gate."""
    comparable, _ = cover_identity_gate(
        _profile(emulated=None), _profile(emulated=False))
    assert not comparable


def test_absent_tier_on_both_sides_refuses():
    """Deliberately stricter than `diff`'s gate, which passes this
    pair: a coverage claim "for this box" with no box declared on
    either side is a silent pass, not an agreement."""
    comparable, notes = cover_identity_gate(
        _profile(tier=None), _profile(tier=None))
    assert not comparable
    assert any("provenance.tier not declared on either side" in note
               for note in notes)


def test_absent_emulated_on_both_sides_refuses():
    comparable, notes = cover_identity_gate(
        _profile(emulated=None), _profile(emulated=None))
    assert not comparable
    assert any("provenance.emulated not declared on either side" in note
               for note in notes)


def test_probe_version_inequality_refuses():
    comparable, notes = cover_identity_gate(
        _profile(probe_version="0.12.0"), _profile(probe_version="0.13.0"))
    assert not comparable
    assert any("probe_version" in note for note in notes)


def test_absent_instrument_on_both_sides_refuses():
    """An undeclared instrument is an unknown instrument — equality of
    two Nones is not an identity."""
    comparable, _ = cover_identity_gate(
        _profile(probe_version=None), _profile(probe_version=None))
    assert not comparable


def test_schema_inequality_refuses():
    comparable, notes = cover_identity_gate(
        _profile(schema=9), _profile(schema=10))
    assert not comparable
    assert any("assay_profile_version" in note for note in notes)


def test_identical_identity_passes_with_no_notes():
    comparable, notes = cover_identity_gate(_profile(), _profile())
    assert comparable
    assert notes == ()


def test_reinterpretation_table():
    """The core rule, enumerated. Floor measures two verdict cells and
    a speed cell; each candidate wrinkle maps to exactly one bucket."""
    floor = _profile(**_verdicts(patch_editing="ready",
                                 structured_extraction="risky"),
                     speed={"decode_tps": 100.0})
    table = [
        # (candidate families, expected uncovered names,
        #  expected incomplete, why)
        (dict(**_verdicts(patch_editing="ready",
                          structured_extraction="risky"),
              speed={"decode_tps": 100.0}),
         (), (),
         "identical candidate covers"),
        (dict(**_verdicts(patch_editing="risky",
                          structured_extraction="risky"),
              speed={"decode_tps": 100.0}),
         ("verdict.patch_editing",), (),
         "a rung below on the ladder is uncovered"),
        (dict(**_verdicts(patch_editing="ready",
                          structured_extraction="ready"),
              speed={"decode_tps": 100.0}),
         (), (),
         "a rung ABOVE is covered — improvements are not evidence against"),
        (dict(**_verdicts(patch_editing="ready"),
              speed={"decode_tps": 100.0}),
         (), ("verdict.structured_extraction",),
         "a floor cell the candidate did not measure is incomplete"),
        (dict(**_verdicts(patch_editing="ready",
                          structured_extraction="risky",
                          tool_calling="ready"),
              speed={"decode_tps": 100.0}),
         (), (),
         "a candidate-only cell is ignored, not evidence"),
        (dict(**_verdicts(patch_editing="ready",
                          structured_extraction="risky"),
              speed={"decode_tps": 60.0}),
         ("speed.decode_tps",), (),
         "slower beyond the assumed 20% threshold is uncovered"),
        (dict(**_verdicts(patch_editing="ready",
                          structured_extraction="risky"),
              speed={"decode_tps": 140.0}),
         (), (),
         "faster is covered — one-directional"),
    ]
    for families, expect_uncovered, expect_incomplete, why in table:
        result = cover_profiles(floor, _profile(name="qwen-b", **families))
        assert result.comparable, why
        got_uncovered = tuple(f"{c.family}.{c.cell}" for c in result.uncovered)
        assert got_uncovered == expect_uncovered, why
        assert result.incomplete == expect_incomplete, why


def test_incomplete_never_passes_even_fully_covered_elsewhere():
    """Load-bearing: every measured cell covered, one floor cell
    unmeasured — the result must carry incomplete, and Task 3 pins
    that incomplete outranks covered in the exit code."""
    floor = _profile(**_verdicts(patch_editing="ready",
                                 structured_extraction="ready"))
    candidate = _profile(name="qwen-b",
                         **_verdicts(patch_editing="ready"))
    result = cover_profiles(floor, candidate)
    assert result.incomplete == ("verdict.structured_extraction",)
    assert result.uncovered == ()


def test_candidate_only_cells_are_named_ignored():
    floor = _profile(**_verdicts(patch_editing="ready"))
    candidate = _profile(name="qwen-b",
                         **_verdicts(patch_editing="ready",
                                     tool_calling="unusable"))
    result = cover_profiles(floor, candidate)
    assert result.ignored == ("verdict.tool_calling",)
    assert result.uncovered == ()
    assert result.incomplete == ()


def test_refused_pair_reports_nothing_beyond_notes():
    """Same rule as diff_profiles: cells printed beside a refusal
    invite reading them anyway."""
    result = cover_profiles(_profile(probe_version="0.12.0"),
                            _profile(probe_version="0.13.0"))
    assert not result.comparable
    assert result.uncovered == ()
    assert result.covered == ()
    assert result.incomplete == ()


def test_equal_but_unparseable_instrument_refuses_naming_the_cells():
    """The `incomparable` branch is LIVE, not defense-in-depth. A
    `probe_version` that is not three decimal components parses to
    None, so `_straddles` fail-safes it to straddling every registered
    break — while the gate's equality check passes it, both sides being
    the same unparseable string. Cover refuses, naming the cells."""
    floor = _profile(probe_version="0.13",
                     verdicts={"parallel": {"verdict": "ready"}})
    candidate = _profile(name="qwen-b", probe_version="0.13",
                         verdicts={"parallel": {"verdict": "ready"}})
    result = cover_profiles(floor, candidate)
    assert result.comparable is False
    assert result.incomparable == ("verdict.parallel",)
    assert result.uncovered == ()
    assert result.covered == ()
    assert result.incomplete == ()
    assert result.ignored == ()


def test_floor_only_provisional_rider_is_never_decisive():
    """Spec §2: provisional flags ride along in the evidence and are
    never decisive. A floor that recorded the flag against a candidate
    that did not still has its verdict covered — spending exit 3 on the
    rider would let a flag decide a pair whose verdict itself covers."""
    floor = _profile(verdicts={"a": {"verdict": "ready",
                                     "provisional": True}})
    candidate = _profile(name="qwen-b",
                         verdicts={"a": {"verdict": "ready"}})
    result = cover_profiles(floor, candidate)
    assert result.covered == ("verdict.a",)
    assert result.incomplete == ()
    assert not any(name.endswith(".provisional") for name in
                   result.incomplete + result.ignored + result.covered)


def test_candidate_only_provisional_rider_is_not_ignored_noise():
    """The mirror: a rider the candidate alone carries is not an
    ignored candidate-only CELL either. Same rule, same reason."""
    floor = _profile(verdicts={"a": {"verdict": "ready"}})
    candidate = _profile(name="qwen-b",
                         verdicts={"a": {"verdict": "ready",
                                         "provisional": True}})
    result = cover_profiles(floor, candidate)
    assert result.covered == ("verdict.a",)
    assert result.ignored == ()
    assert "verdict.a.provisional" not in result.ignored
    assert "verdict.a.provisional" not in result.incomplete


def test_floor_covers_itself():
    floor = _profile(**_verdicts(patch_editing="ready"),
                     speed={"decode_tps": 100.0},
                     ceiling={"max_verified": 8192})
    result = cover_profiles(floor, floor)
    assert result.comparable
    assert result.uncovered == ()
    assert result.incomplete == ()
    assert result.covered  # every measured cell, named


def _change(cell="patch_editing", family="verdict"):
    return Change(family=family, cell=cell, direction="regression",
                  old="ready", new="risky", basis="flip")


def test_exit_code_precedence_table():
    """2 > 3 > 1 > 0, enumerated — the contract bloomery reads."""
    table = [
        (CoverResult(comparable=False, identity_notes=("x",)), 2,
         "refused outranks everything"),
        (CoverResult(comparable=False, identity_notes=(),
                     incomparable=("verdict.parallel",)), 2,
         "a straddled cell is a refusal, not a score"),
        (CoverResult(comparable=True, identity_notes=(),
                     uncovered=(_change(),),
                     incomplete=("verdict.tool_calling",)), 3,
         "incomplete outranks not-covered: the unmeasured cell may "
         "hide a worse answer than the measured one"),
        (CoverResult(comparable=True, identity_notes=(),
                     uncovered=(_change(),)), 1, "not covered"),
        (CoverResult(comparable=True, identity_notes=(),
                     covered=("verdict.patch_editing",),
                     ignored=("verdict.tool_calling",)), 0,
         "ignored cells decide nothing"),
    ]
    for result, expected, why in table:
        assert _cover_exit_code(result) == expected, why


def test_render_names_the_evidence():
    """Every bucket reaches the page. The headline is the SAME
    precedence the exit code uses (2 > 3 > 1 > 0), so this result —
    uncovered cells AND an unmeasured floor cell — headlines
    "incomplete" while still naming the uncovered cell below it: a
    headline that disagreed with the exit code beside it would send a
    reader looking for a different answer than the machine got."""
    result = CoverResult(
        comparable=True,
        identity_notes=("model.name (informational): 'a' -> 'b'",),
        uncovered=(_change(),),
        covered=("verdict.structured_extraction",),
        incomplete=("verdict.tool_calling",),
        ignored=("speed.prefill_tps",))
    text = render_cover(result)
    assert text.startswith("cover: incomplete")
    assert _cover_exit_code(result) == 3   # headline == exit code
    assert "verdict.patch_editing" in text and "'ready' -> 'risky'" in text
    assert "verdict.tool_calling" in text
    assert "1 cell(s)" in text          # covered count
    assert "ignored" in text
    assert "model.name" in text


def test_render_headlines_not_covered_when_that_is_the_answer():
    """The other side of the precedence: with nothing incomplete, a
    single uncovered cell IS the headline, in the words the exit-1
    contract uses."""
    result = CoverResult(comparable=True, identity_notes=(),
                         uncovered=(_change(),),
                         covered=("verdict.structured_extraction",))
    text = render_cover(result)
    assert text.startswith("cover: not covered")
    assert _cover_exit_code(result) == 1
    assert "uncovered verdict.patch_editing" in text


def test_render_headlines_covered_when_nothing_is_missing():
    result = CoverResult(comparable=True, identity_notes=(),
                         covered=("verdict.patch_editing",),
                         ignored=("verdict.tool_calling",))
    text = render_cover(result)
    assert text.startswith("cover: covered")
    assert "ignored (candidate-only): 1 cell(s)" in text


def test_render_refusal_shows_notes_only():
    result = CoverResult(comparable=False,
                         identity_notes=("probe_version must match "
                                         "exactly: '0.12.0' -> '0.13.0'",))
    text = render_cover(result)
    assert "not comparable" in text
    assert "probe_version" in text
