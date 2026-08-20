"""Coverage: does a candidate profile provide what a floor measured?

`diff` asks whether one endpoint moved between two measurements, and
refuses crossed model pairs because a drift statement about two
different models describes neither. `cover` is the crossed-pair
comparison that IS supported: one-directional — for every cell the
floor measured, the candidate must rank at least as high on that
cell's own scale. Spec:
docs/superpowers/specs/2026-08-19-assay-v1.11-cover-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

from .diff import Change, _differs, _families


@dataclass(frozen=True)
class CoverResult:
    comparable: bool
    identity_notes: tuple[str, ...]
    #: Floor cells the candidate ranks below, beyond that cell's own
    #: noise discipline. Full `Change` objects: the render and the
    #: `--json` document need both values, not just the name.
    uncovered: tuple[Change, ...] = ()
    covered: tuple[str, ...] = ()
    #: Floor-measured cells the candidate did not measure. Never a
    #: pass: the unmeasured cell may hide exactly the regression the
    #: check exists to catch (the exit-3 discipline).
    incomplete: tuple[str, ...] = ()
    #: Candidate-only cells. Counted, named, decisive of nothing —
    #: coverage is one-directional.
    ignored: tuple[str, ...] = ()
    #: Cells straddling a registered semantic break, or measured by an
    #: instrument nobody can identify. Reachable through the gate: two
    #: sides carrying the SAME unparseable `probe_version` satisfy its
    #: equality check, and `_straddles` fail-safes an unparseable
    #: version to straddling every registered break. Refused, never
    #: assumed comparable.
    incomparable: tuple[str, ...] = ()


def cover_identity_gate(floor: dict, candidate: dict) -> tuple[bool, tuple[str, ...]]:
    """(comparable, notes). Model identity may differ — that is the
    point of the command. Hardware class and instrument must not:
    a floor measured on different hardware is not a floor for this
    box, and a floor measured by a different instrument was measured
    under different rules (strict equality per the spec's ruling —
    the semantic-break registry is not a complete inventory, so
    version tolerance here would trust an incomplete table).
    Undeclared is unknown on both counts: hardware missing from BOTH
    sides is fatal here too (spec §1's 2026-08-19 amendment) where
    `diff`'s gate passes that pair."""
    floor_model = floor.get("model") or {}
    cand_model = candidate.get("model") or {}
    floor_prov = floor.get("provenance") or {}
    cand_prov = candidate.get("provenance") or {}
    notes: list[str] = []
    fatal = False
    for field in ("name", "quant", "weights_bytes"):
        old_value, new_value = floor_model.get(field), cand_model.get(field)
        if old_value != new_value:
            notes.append(f"model.{field} (informational): "
                         f"{old_value!r} -> {new_value!r}")
    for field in ("tier", "emulated"):
        old_value, new_value = floor_prov.get(field), cand_prov.get(field)
        if old_value is None and new_value is None:
            # A deliberate deviation from `diff`'s gate, which passes
            # this pair (two Nones compare equal there). Cover's own
            # instrument loop below already holds that undeclared is
            # unknown, and a coverage claim "for this box" with no box
            # declared on EITHER side is exactly the silent pass this
            # gate exists to refuse. Checked before the equality test,
            # which two Nones would otherwise satisfy.
            notes.append(f"provenance.{field} not declared on either side: "
                         "unknown hardware")
            fatal = True
            continue
        if old_value == new_value:
            continue
        if old_value is None or new_value is None:
            notes.append(f"provenance.{field} not recorded on one side: "
                         f"{old_value!r} -> {new_value!r}")
        else:
            notes.append(_differs(f"provenance.{field}", old_value, new_value))
        fatal = True
    for field in ("probe_version", "assay_profile_version"):
        old_value, new_value = floor.get(field), candidate.get(field)
        if old_value is None or new_value is None or old_value != new_value:
            notes.append(f"{field} must match exactly: "
                         f"{old_value!r} -> {new_value!r}")
            fatal = True
    return (not fatal), tuple(notes)


def cover_profiles(floor: dict, candidate: dict) -> CoverResult:
    """One-directional coverage of `candidate` against `floor`.

    A refused pair reports NOTHING beyond the notes — the
    `diff_profiles` rule: cells printed beside a refusal invite
    reading them anyway.
    """
    comparable, notes = cover_identity_gate(floor, candidate)
    if not comparable:
        return CoverResult(comparable=False, identity_notes=notes)
    pair = _families(floor, candidate)
    if pair.incomparable:
        # Reachable today, in exactly the case the gate's equality
        # check does not close: a `probe_version` equal on both sides
        # but unparseable (not three decimal components). Equality
        # passes the gate, `_parse_version` returns None, and
        # `_straddles` fail-safes an unidentifiable version to
        # straddling every REGISTERED break. Refusing here is that
        # fail-safe's other half — the rule that produced those cells
        # was never established, so they are not comparable.
        return CoverResult(comparable=False, identity_notes=notes,
                           incomparable=pair.incomparable)
    # A self-diff of the floor puts exactly its measured cells into
    # within_noise, under the walkers' own names — the floor-side
    # membership test, byte-for-byte consistent with `pair` by
    # construction.
    floor_measured = frozenset(_families(floor, floor).within_noise)
    uncovered = tuple(change for change in pair.changes
                      if change.direction == "regression")
    covered = pair.within_noise + tuple(
        f"{change.family}.{change.cell}" for change in pair.changes
        if change.direction != "regression")
    # A `verdict.<name>.provisional` sub-cell is an evidence RIDER, not
    # a cell: spec §2 holds the provisional flags are never decisive —
    # a floor cell is the floor whether or not its Wilson interval had
    # decided. Left in the partition, a floor that recorded the flag
    # against a candidate that did not would land the rider in
    # `incomplete` and spend exit 3 on a pair whose verdict itself
    # covers, letting the flag decide after all. Both-measured flips
    # still ride along in `covered` (§3), where they decide nothing.
    dropped = tuple(cell for cell in pair.dropped
                    if not cell.endswith(".provisional"))
    incomplete = tuple(cell for cell in dropped
                       if cell in floor_measured)
    ignored = tuple(cell for cell in dropped
                    if cell not in floor_measured)
    return CoverResult(comparable=True, identity_notes=notes,
                       uncovered=uncovered, covered=covered,
                       incomplete=incomplete, ignored=ignored)


def _cover_exit_code(result: CoverResult) -> int:
    """0 covered, 1 not covered, 2 refused, 3 incomplete — precedence
    2 > 3 > 1 > 0, mirroring `diff --gate` so a consumer's four-code
    reading maps over unchanged. Incomplete outranks not-covered for
    the same reason exit 3 outranks 1 in diff: an unmeasured floor
    cell may hide a worse answer than any measured one."""
    if not result.comparable:
        return 2
    if result.incomplete:
        return 3
    if result.uncovered:
        return 1
    return 0


def render_cover(result: CoverResult) -> str:
    """Plain text in `render_diff`'s style: the answer, then every cell
    that produced it.

    The headline word is the exit code's own word, chosen by the SAME
    precedence (`_cover_exit_code`): a pair that is both not-covered
    and incomplete headlines "incomplete" while still naming its
    uncovered cells below. A headline that disagreed with the number
    beside it in CI would send a reader looking for a different answer
    than the machine acted on. Note what "covered: N cell(s)" counts:
    every cell the candidate did not rank BELOW the floor on —
    improvements and neutral flips among them — not cells that held
    still.
    """
    if not result.comparable:
        lines = ["not comparable:"]
        lines.extend(f"  {note}" for note in result.identity_notes)
        if result.incomparable:
            # "not established", never "measured under different rules"
            # — `render_diff`'s ruled wording, and cover needs it MORE
            # than diff does. Diff has two routes here and the weaker
            # word is the one true for both; cover has exactly one, and
            # it is the route the stronger word is false about. The gate
            # demands EQUAL `probe_version`s and two equal parseable
            # versions cannot straddle a break, so a cell only arrives
            # here when both sides named the same UNREADABLE version:
            # nobody can say the rules differed, only that nobody
            # established they agree.
            lines.append("  cells not established to share a measurement "
                         "rule: " + ", ".join(result.incomparable))
        return "\n".join(lines)
    if result.incomplete:
        verdict = "incomplete"
    elif result.uncovered:
        verdict = "not covered"
    else:
        verdict = "covered"
    lines = [f"cover: {verdict}"]
    lines.extend(f"  note: {note}" for note in result.identity_notes)
    lines.extend(
        f"  uncovered {change.family}.{change.cell}: "
        f"{change.old!r} -> {change.new!r} ({change.basis})"
        for change in result.uncovered)
    lines.extend(
        f"  incomplete {cell}: the floor measured it, the candidate "
        "did not" for cell in result.incomplete)
    lines.append(f"  covered: {len(result.covered)} cell(s)")
    if result.ignored:
        lines.append("  ignored (candidate-only): "
                     f"{len(result.ignored)} cell(s)")
    return "\n".join(lines)
