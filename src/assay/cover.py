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
    #: Cells straddling a registered semantic break. Unreachable while
    #: the gate demands instrument equality (equal versions cannot
    #: straddle); refused rather than assumed if a future caller
    #: loosens the gate.
    incomparable: tuple[str, ...] = ()


def cover_identity_gate(floor: dict, candidate: dict) -> tuple[bool, tuple[str, ...]]:
    """(comparable, notes). Model identity may differ — that is the
    point of the command. Hardware class and instrument must not:
    a floor measured on different hardware is not a floor for this
    box, and a floor measured by a different instrument was measured
    under different rules (strict equality per the spec's ruling —
    the semantic-break registry is not a complete inventory, so
    version tolerance here would trust an incomplete table)."""
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
