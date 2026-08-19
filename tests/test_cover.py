"""Cover: the crossed-model coverage check (spec: docs/superpowers/
specs/2026-08-19-assay-v1.11-cover-design.md)."""

from assay.cover import CoverResult, cover_identity_gate


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
