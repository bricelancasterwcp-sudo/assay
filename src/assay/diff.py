"""Two profiles, one question: did anything real change? (spec §2)

This module compares RAW payload dicts — ``json.loads`` of a profile
file — never ``Profile`` objects. A v1 profile on disk stores plain
strings for verdicts, has no ``ceiling_shapes`` key at all, and carries
codec cells without ``lands_applies``; ``Profile.from_json`` indexes
those keys and would raise before a single cell got compared. Diff has
to read every schema this project has ever written, so it reads dicts.

**The identity gate.** The spec asks for a "blob digest" so two
profiles of *different weights* are never subtracted from each other.
Profiles do not carry a digest of the weights file — nothing in the
probe path ever hashes it — so the gate is realized as the fields
profiles actually do carry: model name, quant, and ``weights_bytes``,
plus the hardware tier and its emulated marking. Two profiles agreeing
on all five are the same model on the same class of machine as far as
the record can tell; quant or size known on only one side is a warning
(the older file simply did not record it), a *disagreement* is fatal.
Tier is stricter than quant: an undeclared tier is not a benign unknown
because the number's meaning depends on the hardware under it — though
the note for it says "not recorded on one side", never "differs", so a
schema-era gap is not read as two machines disagreeing.

**Drift vs noise.** A diff that reports every wobble is a diff nobody
reads. Each family is judged by the strongest evidence its cells carry:

- ceiling / shapes / verdicts: exact values, so any move is real
  (``rung-change`` and ``flip``). The verdict ladder is
  ready > risky > degrades-at-N > unusable > unsupported, and between
  two ``degrades-at`` rungs the LARGER N is the better one — degrading
  later is an improvement, not a regression;
- codecs: Wilson-95 per side, flagged only when the intervals are
  DISJOINT (``disjoint-intervals``) — 4/5 vs 5/5 is not a finding;
- speed: Welch 2-SE over the per-call samples (``beyond-2se``) when
  both sides recorded ≥2, and otherwise a 20% rule of thumb whose
  basis is named ``threshold-20pct-assumed`` in the output, because a
  threshold nobody measured should say so on the page.

**None is never zero.** A cell measured on one side only goes to
``dropped`` and is never scored — as does a verdict that is
``unmeasured`` on exactly ONE side, and (same rule, extended from the
spec's verdict clause) a ``failure_mode`` whose literal value is
``"unmeasured"``. Absence is not a regression. Absence on BOTH sides
is not even a drop: nothing was compared and nothing vanished, so
``dropped`` means precisely "measured on exactly one side" — which is
what v1.8's exit 3 reads. ``unsupported`` is the one word that looks
like absence and is not: the endpoint was asked and said no, so it
ranks (bottom rung) instead of dropping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from assay.stats import wilson95

UNMEASURED = "unmeasured"
#: The assumed speed threshold, printed with its basis so the
#: assumption is visible at the point of use.
SPEED_ASSUMED_THRESHOLD = 0.20
BASIS_ASSUMED = "threshold-20pct-assumed"
BASIS_RUNG = "rung-change"
BASIS_FLIP = "flip"
BASIS_DISJOINT = "disjoint-intervals"
BASIS_2SE = "beyond-2se"

# ready > risky > degrades-at-N > unusable > unsupported. ready/risky/
# unusable come from assay.stats.ladder; degrades-at-N is the
# long_output family's own rung (v1.5), and it sits above unusable — a
# model that holds together for a while is better than one that never
# does — and below risky. "unsupported" is the tools family's rung
# (v1.6): the endpoint REFUSED the tools parameter, so the model was
# never asked to do the task. That is the bottom of the ladder and not
# a tie with unusable — being asked and failing every task is more than
# never being asked — which is what makes ready -> unsupported a
# regression a --gate must fail on, and unsupported -> anything measured
# an improvement it must not.
#
# "unmeasured" is the ladder's remaining value and deliberately absent
# here: it is dropped, never ranked. unsupported is NOT dropped, because
# the refusal is a measurement.
_LADDER_RANK = {"unsupported": 0, "unusable": 1, "degrades-at": 2,
                "risky": 3, "ready": 4}
_DEGRADES_PREFIX = "degrades-at-"
_LYING_MODES = frozenset({"silent_truncation", "missing_stats"})
_GRADES = ("tiny", "small", "medium")
_LENSES = ("lands", "lands_applies")
#: Codecs whose two lens columns are one measurement (see ``_lenses_for``).
_COINCIDING_LENS_CODECS = frozenset({"json_object"})
_SPEED_CELLS = (("decode_tps", "decode_samples"),
                ("prefill_tps", "prefill_samples"))


@dataclass(frozen=True)
class Change:
    family: str      # ceiling | ceiling_shapes | verdict | codec | speed
    cell: str        # e.g. max_verified, shape:4096, patch_editing,
                     # whole_file.small.lands_applies, decode_tps
    direction: str   # regression | improvement | neutral
    old: object
    new: object
    basis: str       # rung-change | flip | disjoint-intervals |
                     # beyond-2se | threshold-20pct-assumed


@dataclass(frozen=True)
class DiffResult:
    comparable: bool
    identity_notes: tuple[str, ...]   # mismatches (fatal) or warnings
    changes: tuple[Change, ...]
    within_noise: tuple[str, ...]     # cells checked and clean, named
    dropped: tuple[str, ...]          # cells present on one side only


@dataclass(frozen=True)
class _Cells:
    """One family's contribution. Immutable, merged by concatenation —
    no shared accumulator to mutate."""
    changes: tuple[Change, ...] = ()
    within_noise: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()


def _merge(*parts: _Cells) -> _Cells:
    return _Cells(
        changes=tuple(c for part in parts for c in part.changes),
        within_noise=tuple(n for part in parts for n in part.within_noise),
        dropped=tuple(d for part in parts for d in part.dropped),
    )


# --- identity gate ---------------------------------------------------


def identity_gate(old: dict, new: dict) -> tuple[bool, tuple[str, ...]]:
    """(comparable, notes). Notes list every mismatch and every
    one-sided unknown; comparability turns on the fatal ones only."""
    old_model, new_model = old.get("model") or {}, new.get("model") or {}
    old_prov, new_prov = old.get("provenance") or {}, new.get("provenance") or {}
    notes: list[str] = []
    fatal = False

    if old_model.get("name") != new_model.get("name"):
        notes.append(_differs("model.name", old_model.get("name"),
                              new_model.get("name")))
        fatal = True

    for field in ("quant", "weights_bytes"):
        old_value, new_value = old_model.get(field), new_model.get(field)
        if old_value is None and new_value is None:
            continue
        if old_value is None or new_value is None:
            # Warning, not fatal: the older profile did not record it.
            notes.append(f"model.{field} known on one side only: "
                         f"{old_value!r} -> {new_value!r}")
            continue
        if old_value != new_value:
            notes.append(_differs(f"model.{field}", old_value, new_value))
            fatal = True

    for field in ("tier", "emulated"):
        old_value, new_value = old_prov.get(field), new_prov.get(field)
        if old_value == new_value:
            continue
        # Still fatal — an undeclared tier is unknown hardware, which is
        # what this gate exists to catch — but a v1 profile that predates
        # the marking did not DISAGREE about the hardware, and a note
        # saying it did would send a reader looking for a machine that
        # changed. Absent and present-with-null make the same claim:
        # nobody declared it.
        if old_value is None or new_value is None:
            notes.append(f"provenance.{field} not recorded on one side: "
                         f"{old_value!r} -> {new_value!r}")
        else:
            notes.append(_differs(f"provenance.{field}", old_value, new_value))
        fatal = True

    return (not fatal), tuple(notes)


def _differs(field: str, old: object, new: object) -> str:
    return f"{field} differs: {old!r} -> {new!r}"


# --- direction rules -------------------------------------------------


def _numeric_direction(old: object, new: object) -> str:
    return "regression" if new < old else "improvement"


def _mode_direction(old: object, new: object) -> str:
    """A daemon that starts lying past its edge is a regression however
    the numbers moved; honest-to-honest is a fact, not a grade."""
    if new in _LYING_MODES and old not in _LYING_MODES:
        return "regression"
    if old in _LYING_MODES and new not in _LYING_MODES:
        return "improvement"
    return "neutral"


def _rung_rank(value: object) -> tuple[int, int] | None:
    """(rung, extent) for one verdict string, or None when it is not a
    rung this comparator can read.

    The extent term only ever breaks ties INSIDE ``degrades-at``, where
    the larger N is better: a model that holds together to 2048 before
    it loops is better than one that loops at 1024, and a gate told the
    opposite would fail a build for an improvement. Every named rung
    carries extent 0, so the ordering among them is unchanged.

    An unparsable ``degrades-at-<something>`` returns None rather than a
    guessed position: a direction nobody can derive is not one a CI gate
    should act on.
    """
    if not isinstance(value, str):
        return None
    if value.startswith(_DEGRADES_PREFIX):
        extent = value[len(_DEGRADES_PREFIX):]
        if not extent.isdigit():
            return None
        return _LADDER_RANK["degrades-at"], int(extent)
    rank = _LADDER_RANK.get(value)
    return None if rank is None else (rank, 0)


def _ladder_direction(old: object, new: object) -> str:
    old_rank, new_rank = _rung_rank(old), _rung_rank(new)
    if old_rank is None or new_rank is None:
        return "neutral"  # an unknown rung is not a rung
    if new_rank == old_rank:
        # Two spellings of one position (the caller only asks when the
        # strings differ): nothing moved on the ladder.
        return "neutral"
    return "regression" if new_rank < old_rank else "improvement"


def _neutral(old: object, new: object) -> str:
    return "neutral"


def _exact(family: str, cell: str, old: object, new: object,
           *, basis: str, direction) -> _Cells:
    """One exact-valued cell. One-sided None -> dropped (never scored),
    equal -> named in within_noise, different -> a Change."""
    name = f"{family}.{cell}"
    if old is None or new is None:
        if old is None and new is None:
            return _Cells()
        return _Cells(dropped=(name,))
    if old == new:
        return _Cells(within_noise=(name,))
    return _Cells(changes=(Change(family=family, cell=cell,
                                  direction=direction(old, new),
                                  old=old, new=new, basis=basis),))


def _measured_mode(mode: object) -> object:
    """A literal ``"unmeasured"`` failure mode is an absent cell, not a
    value to flip against (the verdict rule, applied to modes)."""
    return None if mode == UNMEASURED else mode


# --- families --------------------------------------------------------


def _diff_ceiling(old: dict, new: dict) -> _Cells:
    old_cell, new_cell = old.get("ceiling") or {}, new.get("ceiling") or {}
    return _merge(
        _exact("ceiling", "max_verified", old_cell.get("max_verified"),
               new_cell.get("max_verified"), basis=BASIS_RUNG,
               direction=_numeric_direction),
        _exact("ceiling", "failure_mode",
               _measured_mode(old_cell.get("failure_mode")),
               _measured_mode(new_cell.get("failure_mode")),
               basis=BASIS_FLIP, direction=_mode_direction),
    )


def _by_shape(entries: object) -> dict:
    return {entry["shape"]: entry
            for entry in (entries or [])
            if isinstance(entry, dict) and entry.get("shape") is not None}


def _shape_measured(entry: dict) -> bool:
    """A shape entry counts as measured when it carries a real value on
    at least one axis — the same test `_measured_mode` applies to a
    verdict. An entry that made it into the list with
    ``failure_mode: "unmeasured"`` and ``max_verified: null`` was never
    actually measured; the ``shape`` key's mere presence is not
    evidence of that."""
    return (entry.get("max_verified") is not None
            or _measured_mode(entry.get("failure_mode")) is not None)


def _diff_shapes(old: dict, new: dict) -> _Cells:
    """Shapes are matched by their ``shape`` value, never by position:
    a run that probed one fewer shape would otherwise silently compare
    4096 against 8192.

    A shape's key can be present in the entries list without the shape
    being MEASURED — an unmeasured placeholder still carries a
    ``shape`` field. Dropping on key presence alone, before looking at
    the values, would call that "measured on one side", which is the
    same invariant break `_exact`, `_diff_codec_cell`, `_diff_speed_cell`
    and `_diff_verdicts` all refuse elsewhere in this module: a cell
    unmeasured on both sides produces no cell, dropped or otherwise.
    """
    old_shapes = _by_shape(old.get("ceiling_shapes"))
    new_shapes = _by_shape(new.get("ceiling_shapes"))
    parts = []
    for shape in sorted(set(old_shapes) | set(new_shapes)):
        cell = f"shape:{shape}"
        old_entry, new_entry = old_shapes.get(shape), new_shapes.get(shape)
        old_measured = old_entry is not None and _shape_measured(old_entry)
        new_measured = new_entry is not None and _shape_measured(new_entry)
        if not old_measured and not new_measured:
            continue  # unmeasured (or absent) on both sides: no cell
        if old_entry is None or new_entry is None:
            parts.append(_Cells(dropped=(f"ceiling_shapes.{cell}",)))
            continue
        parts.append(_exact("ceiling_shapes", cell,
                            old_entry.get("max_verified"),
                            new_entry.get("max_verified"),
                            basis=BASIS_RUNG, direction=_numeric_direction))
        parts.append(_exact("ceiling_shapes", f"{cell}.failure_mode",
                            _measured_mode(old_entry.get("failure_mode")),
                            _measured_mode(new_entry.get("failure_mode")),
                            basis=BASIS_FLIP, direction=_mode_direction))
    return _merge(*parts)


def _verdict_of(entry: object) -> tuple[object, object]:
    """(verdict, provisional) from either schema: v1 stored the bare
    string, v2+ stores a dict with the lens beside it."""
    if isinstance(entry, str):
        return entry, None
    if isinstance(entry, dict):
        return entry.get("verdict"), entry.get("provisional")
    return None, None


def _diff_verdicts(old: dict, new: dict) -> _Cells:
    old_verdicts = old.get("verdicts") or {}
    new_verdicts = new.get("verdicts") or {}
    parts = []
    for name in sorted(set(old_verdicts) | set(new_verdicts)):
        old_value, old_prov = _verdict_of(old_verdicts.get(name))
        new_value, new_prov = _verdict_of(new_verdicts.get(name))
        old_measured = old_value is not None and old_value != UNMEASURED
        new_measured = new_value is not None and new_value != UNMEASURED
        if not old_measured and not new_measured:
            # Absent (or unmeasured) on BOTH sides: nothing was
            # compared and nothing vanished. `dropped` means measured
            # on exactly one side — the rule `_exact`,
            # `_diff_codec_cell` and `_diff_speed_cell` already keep,
            # and the rule v1.8's exit 3 reads directly.
            continue
        if not (old_measured and new_measured):
            parts.append(_Cells(dropped=(f"verdict.{name}",)))
            continue
        scored = _exact("verdict", name, old_value, new_value,
                        basis=BASIS_FLIP, direction=_ladder_direction)
        parts.append(scored)
        if scored.changes:
            # The rung move is the story; the provisional flag rides
            # along with it rather than being reported twice.
            continue
        parts.append(_exact("verdict", f"{name}.provisional",
                            old_prov, new_prov,
                            basis=BASIS_FLIP, direction=_neutral))
    return _merge(*parts)


def _codec_rate(cell: object, lens: str) -> tuple[float | None, int]:
    """The cell's rate under one lens, or None wherever it was not
    measured: n == 0, the lens column absent (a v1 cell has no
    ``lands_applies``), or the whole cell missing."""
    if not isinstance(cell, dict):
        return None, 0
    n = cell.get("n") or 0
    rate = cell.get(lens)
    if n <= 0 or rate is None:
        return None, 0
    return rate, n


def _diff_codec_cell(codec: str, grade: str, lens: str,
                     old_cell: object, new_cell: object) -> _Cells:
    name = f"{codec}.{grade}.{lens}"
    old_rate, old_n = _codec_rate(old_cell, lens)
    new_rate, new_n = _codec_rate(new_cell, lens)
    if old_rate is None or new_rate is None:
        if old_rate is None and new_rate is None:
            return _Cells()
        return _Cells(dropped=(f"codec.{name}",))
    # Exact reconstruction: the stored rate is an unrounded count/n.
    old_lo, old_hi = wilson95(round(old_rate * old_n), old_n)
    new_lo, new_hi = wilson95(round(new_rate * new_n), new_n)
    if old_hi < new_lo or new_hi < old_lo:
        direction = "improvement" if new_rate > old_rate else "regression"
        return _Cells(changes=(Change(family="codec", cell=name,
                                      direction=direction, old=old_rate,
                                      new=new_rate, basis=BASIS_DISJOINT),))
    return _Cells(within_noise=(f"codec.{name}",))


def _lenses_for(codec: str) -> tuple[str, ...]:
    """The lenses that are separate MEASUREMENTS for this codec.

    ``json_object``'s two are not: validation IS the application there
    (codecs.py, ``_verdict_lens``), so the probe writes both columns
    from one count and they move together always. Diffing both turned
    one measured change into two Change rows — ``json_object.small.lands``
    and ``.lands_applies``, same numbers, same basis — and a reader
    counting the report would double every json_object finding. One
    cell, one Change. The patch codecs keep both: there the lenses are
    different instruments that disagreed by 100 points on one live model.
    """
    return ("lands",) if codec in _COINCIDING_LENS_CODECS else _LENSES


def _ordered_grades(old_grades: dict, new_grades: dict) -> list[str]:
    seen = set(old_grades) | set(new_grades)
    known = [grade for grade in _GRADES if grade in seen]
    return known + sorted(seen - set(_GRADES))


def _diff_codecs(old: dict, new: dict) -> _Cells:
    old_codecs = old.get("codecs") or {}
    new_codecs = new.get("codecs") or {}
    parts = []
    for codec in sorted(set(old_codecs) | set(new_codecs)):
        old_grades = old_codecs.get(codec) or {}
        new_grades = new_codecs.get(codec) or {}
        for grade in _ordered_grades(old_grades, new_grades):
            for lens in _lenses_for(codec):
                parts.append(_diff_codec_cell(codec, grade, lens,
                                              old_grades.get(grade),
                                              new_grades.get(grade)))
    return _merge(*parts)


def _variance(samples) -> float:
    """n-1 sample variance (the samples are a sample of the machine's
    behaviour, not its population)."""
    mean = sum(samples) / len(samples)
    return sum((value - mean) ** 2 for value in samples) / (len(samples) - 1)


def _welch_se(old_samples, new_samples) -> float:
    return math.sqrt(_variance(old_samples) / len(old_samples)
                     + _variance(new_samples) / len(new_samples))


def _relative_change(old_value: float, new_value: float) -> float:
    if old_value == 0:
        return 0.0 if new_value == old_value else math.inf
    return abs(new_value - old_value) / abs(old_value)


def _diff_speed_cell(cell: str, sample_key: str,
                     old_speed: dict, new_speed: dict) -> _Cells:
    name = f"speed.{cell}"
    old_value, new_value = old_speed.get(cell), new_speed.get(cell)
    if old_value is None or new_value is None:
        if old_value is None and new_value is None:
            return _Cells()
        return _Cells(dropped=(name,))
    old_samples = old_speed.get(sample_key) or ()
    new_samples = new_speed.get(sample_key) or ()
    if len(old_samples) >= 2 and len(new_samples) >= 2:
        basis = BASIS_2SE
        flagged = abs(new_value - old_value) > 2 * _welch_se(old_samples,
                                                             new_samples)
    else:
        # Either side is pre-v5 (no samples), stopped at one call, or
        # sampled and accepted nothing. No spread on the page means the
        # threshold is an assumption, and the basis says so.
        basis = BASIS_ASSUMED
        flagged = _relative_change(old_value, new_value) > SPEED_ASSUMED_THRESHOLD
    if not flagged:
        return _Cells(within_noise=(name,))
    direction = "regression" if new_value < old_value else "improvement"
    return _Cells(changes=(Change(family="speed", cell=cell,
                                  direction=direction, old=old_value,
                                  new=new_value, basis=basis),))


def _diff_speed(old: dict, new: dict) -> _Cells:
    old_speed, new_speed = old.get("speed") or {}, new.get("speed") or {}
    return _merge(*[_diff_speed_cell(cell, sample_key, old_speed, new_speed)
                    for cell, sample_key in _SPEED_CELLS])


# --- public entry points ---------------------------------------------


def diff_profiles(old: dict, new: dict) -> DiffResult:
    """Compare two raw profile payloads.

    A fatal identity mismatch reports NOTHING beyond the notes: a rung
    "improvement" measured across two different models is not a fact,
    and printing it beside the mismatch would invite reading it anyway.
    """
    comparable, notes = identity_gate(old, new)
    if not comparable:
        return DiffResult(comparable=False, identity_notes=notes,
                          changes=(), within_noise=(), dropped=())
    cells = _merge(
        _diff_ceiling(old, new),
        _diff_shapes(old, new),
        _diff_verdicts(old, new),
        _diff_codecs(old, new),
        _diff_speed(old, new),
    )
    return DiffResult(comparable=True, identity_notes=notes,
                      changes=cells.changes, within_noise=cells.within_noise,
                      dropped=cells.dropped)


def _show(value: object) -> str:
    """Unmeasured is SAID (``render_table``'s rule) and the rendered
    number IS the measured number.

    Plain ``str``, the same rule ``profile._show`` uses, so a field
    prints identically in ``render_table`` and here. Python's float
    ``str`` is the shortest text that round-trips: it neither rounds
    3456.78 tok/s to 3457 — a precision the measurement never had,
    asserted by a diff that exists to catch exactly that kind of drift
    — nor drops into scientific notation at report magnitudes, and it
    always keeps the decimal point, so a landing rate never reaches the
    page as ``1 -> 0`` and reads like a count of something.
    """
    return UNMEASURED if value is None else str(value)


def render_diff(result: DiffResult) -> str:
    """Plain text, in ``render_table``'s style: what moved, what was
    checked and clean, what could not be compared at all."""
    if not result.comparable:
        return "\n".join(["not comparable",
                          *(f"  {note}" for note in result.identity_notes)])
    lines = [f"note: {note}" for note in result.identity_notes]
    if result.changes:
        lines.extend(
            f"{change.family}.{change.cell}:"
            f" {_show(change.old)} -> {_show(change.new)}"
            f" ({change.direction}, {change.basis})"
            for change in result.changes)
    else:
        lines.append("no drift beyond noise")
    if result.within_noise:
        lines.append("within noise: " + ", ".join(result.within_noise))
    if result.dropped:
        # Said BEFORE the list, because the list alone reads as a
        # footnote and this is the headline: part of the comparison
        # did not happen. Exit 3 says the same thing to a machine.
        lines.append(f"incomplete: {len(result.dropped)} cell(s)"
                     " measured on one side only")
        lines.append("dropped: " + ", ".join(result.dropped))
    return "\n".join(lines)
