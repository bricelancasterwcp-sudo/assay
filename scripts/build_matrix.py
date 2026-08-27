#!/usr/bin/env python3
"""Build the published capability matrix from a directory of profiles.

Run from the repository root, with assay installed in the environment
(``pip install -e .``)::

    python scripts/build_matrix.py

The import below is a plain one and this file does NO ``sys.path``
surgery. That is deliberate: a script that rewrites the import path can
pick up a different assay than the one the tests ran against, and the
page it writes is the one artifact a reader has no way to check. If the
import fails, the environment is wrong and saying so is the correct
outcome. The two default paths are relative for the same reason the
script has no clock — the build must be reproducible by anyone standing
in the repository root.

**Determinism is the law here, because the output is committed.** The
same profiles must rebuild the same bytes, so nothing on this page may
come from the build clock: every date is READ FROM PROFILE PROVENANCE,
the profiles are ordered by path (never by directory order, which is a
filesystem detail that differs between machines), and every version and
tier list is sorted. A page that churned on every rebuild would make the
diff between two builds unreadable, and the diff is exactly where a real
change would have shown.

**Errata are annotations, never edits.** A profile is not rewritten
after the fact, so a figure a later fix superseded stays on this page —
and, until now, stayed on it unmarked. An optional
``errata/matrix-errata.json`` sidecar under the campaign directory
(``_load_errata``) names the model, the fields and the note, and the
renderer flags the value where it appears while leaving it exactly as
measured. The markdown ``ERRATA.md`` beside the profiles remains the
human record and is never parsed: prose changes shape whenever someone
improves it, and a build that scraped it would stop flagging things the
day a heading moved.

**The escape contract.** ``render_report``'s ``intro_html`` is
author-supplied markup and is inserted verbatim — it carries links,
which is why it is HTML at all. The trust stops at ``_intro_html``:
every value it quotes out of a profile (probe versions, tiers, capture
dates) is document text from an endpoint's replies and goes through
``html.escape`` before it reaches the paragraph.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from assay.report import MARKABLE_FIELDS, render_report

#: Where the campaign writes its profiles, and where Pages serves the
#: page from (master ``/docs``). Both are wired to these literals
#: elsewhere in the wave, so they are pinned by a test rather than left
#: to a help string.
_DEFAULT_PROFILES_DIR = Path(
    "docs/superpowers/evidence/tier-enthusiast-2026-08")
_DEFAULT_OUT = Path("docs/matrix/index.html")

_PAGE_TITLE = "assay capability matrix"
_REPO_URL = "https://github.com/bricelancasterwcp-sudo/assay"
#: Relative to ``docs/matrix/index.html``, which is where this page
#: lives: the errata sit beside the evidence they correct, and a link
#: that resolved only on github.com would be dead in the offline copy
#: this page is otherwise able to be.
_ERRATA_HREF = "../superpowers/evidence/tier-enthusiast-2026-08/ERRATA.md"

#: The machine-readable errata sidecar, in a SUBDIRECTORY of the
#: campaign rather than beside the profiles, and that placement is the
#: whole of what this constant has to get right.
#:
#: The evidence directory's contract with everything that reads it is
#: "every ``*.json`` here is a profile", and it is not this script's
#: contract to relax: ``assay report docs/.../tier-enthusiast-2026-08/
#: *.json`` is a supported thing to run, the CLI refuses a document that
#: does not declare ``assay_profile_version`` (exit 4), and a sidecar
#: sitting in that glob would have broken the published corpus for every
#: consumer to spare this one script a subdirectory. A test caught it;
#: the placement is the fix.
#:
#: The markdown ``ERRATA.md`` beside the profiles remains the human
#: record and is NOT parsed: prose is written for a reader and changes
#: shape whenever a human improves it, and a build that scraped it would
#: silently stop flagging things the day a heading moved.
#:
#: Its job is the one thing house discipline leaves open. Profiles are
#: never rewritten after the fact, so the published matrix keeps showing
#: a figure a later fix superseded — with nothing on the page saying so.
#: The sidecar makes the page say so WITHOUT touching the evidence: the
#: value stays exactly as measured and wears a flag pointing at the
#: correction. It never carries a replacement value, and there is no
#: field in its schema through which it could; substituting a number out
#: here would be the forbidden edit done one layer further out, where a
#: reader comparing the page against the profile cannot see it.
_SIDECAR_PATH = Path("errata") / "matrix-errata.json"

#: The keys an entry must carry, all of them non-empty. A flag with no
#: note and no pointer tells a reader something is wrong and nothing
#: about what or where to read it, which is a worse page than no flag.
_SIDECAR_TEXT_KEYS = ("id", "model", "note", "href")


class BuildError(Exception):
    """Anything that stops the page from being written at all.

    There is no partial build: a matrix is a published claim about which
    models were measured, so a directory this script cannot read whole
    produces no page rather than a page missing rows nobody would notice
    were missing.
    """


def _load_profile(path: Path) -> dict:
    """One raw profile document — ``assay report``'s gate, restated.

    ``{}`` parses, is an object, and renders a full row of "unmeasured"
    badges under whatever name the file happens to have: a published
    capability claim for a run nobody made. A profile must SAY it is one,
    and ``assay_profile_version`` is the key every schema this project
    has written carries.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BuildError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise BuildError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or "assay_profile_version" not in payload:
        raise BuildError(
            f"{path} is not a profile document: no assay_profile_version key "
            f"(it would publish a row of unmeasured badges under this file's "
            f"name — a capability claim for a run nobody made)")
    return payload


def _load_profiles(directory: Path) -> list[dict]:
    """Every ``*.json`` in the directory, ordered by path.

    ``sorted`` is load-bearing, not tidiness: ``glob`` yields whatever
    order the filesystem stores, which differs between machines and
    changes as files are rewritten, and the committed page would then
    depend on the box that built it.

    An empty match is a REFUSAL rather than an empty page. A matrix with
    a header and no rows publishes "a campaign ran and found nothing",
    which is the opposite of what a directory nobody has filled means.

    ``glob`` is not recursive, which is why the errata sidecar lives in a
    subdirectory: every ``*.json`` at this level is a profile, here and
    for every other consumer of the published evidence, and this function
    needs no exception to say so.
    """
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise BuildError(
            f"no profile JSON matched {directory}/*.json — refusing to "
            f"publish an empty matrix (a page with no rows reads as a "
            f"campaign that measured nothing)")
    return [_load_profile(path) for path in paths]


def _sidecar_entry_error(entry, index: int, known: set) -> str | None:
    """Why this entry cannot be published, or ``None``.

    Every branch here guards the same failure — a flag that does not
    appear. A typo'd model, an unmarkable field, a missing note: each one
    leaves the superseded figure on the page looking exactly as checked
    as its neighbours, and a build that shrugged would publish that. The
    refusals are the mechanism's honesty; without them the sidecar is a
    file that only LOOKS like it did something.
    """
    where = f"{_SIDECAR_PATH} entry {index}"
    if not isinstance(entry, dict):
        return f"{where} is not an object"
    for key in _SIDECAR_TEXT_KEYS:
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"{where} has no {key} (a non-empty string is required)"
    fields = entry.get("fields")
    if not isinstance(fields, list) or not fields:
        return (f"{where} names no fields — an erratum that marks nothing "
                f"is a file that only looks like it flagged something")
    for field in fields:
        if field not in MARKABLE_FIELDS:
            return (f"{where} names field {field!r}, which the renderer "
                    f"cannot mark (it knows "
                    f"{sorted(MARKABLE_FIELDS)}) — the flag would not appear "
                    f"anywhere on the page")
    if entry["model"] not in known:
        return (f"{where} names model {entry['model']!r}, which no profile "
                f"in this directory carries — the flag would not appear on "
                f"any row")
    return None


def _load_errata(directory: Path,
                 profiles: list[dict]) -> dict[str, list[dict]] | None:
    """The sidecar, keyed by model name — or ``None`` when there is none.

    ``None`` is not an empty mapping: it is the page as it was before
    this mechanism existed, down to the stylesheet, which is what makes
    the annotation auditable by diff.

    ``errata_sidecar_version`` is ``_load_profile``'s gate restated: a
    document must SAY what it is, or any JSON object in this directory
    is an errata sidecar that happens to annotate nothing.

    ``href`` is resolved by the BROWSER, relative to the published page —
    the same contract ``_ERRATA_HREF`` above already carries, and the
    reason both are page-relative literals rather than repository paths:
    this page is meant to work as an offline copy, where a link that only
    resolved on github.com is dead.
    """
    path = directory / _SIDECAR_PATH
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BuildError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise BuildError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or "errata_sidecar_version" not in payload:
        raise BuildError(
            f"{path} is not an errata sidecar: no errata_sidecar_version key")
    entries = payload.get("errata")
    if not isinstance(entries, list) or not entries:
        raise BuildError(
            f"{path} lists no errata — a sidecar with no entries says "
            f"'corrections were considered' and flags nothing; deleting the "
            f"file says the same thing more honestly")
    known = {(p.get("model") or {}).get("name") for p in profiles}
    by_model: dict[str, list[dict]] = {}
    for index, entry in enumerate(entries):
        problem = _sidecar_entry_error(entry, index, known)
        if problem is not None:
            raise BuildError(problem)
        by_model.setdefault(entry["model"], []).append(entry)
    return by_model


def _unique(values) -> list[str]:
    """The distinct non-empty values, sorted and escaped.

    Sorted so the sentence is the same on every machine; escaped because
    every one of these came out of a profile document, and the paragraph
    they land in is the one place on the page that is NOT escaped for
    the caller.
    """
    return sorted({html.escape(str(value)) for value in values if value})


def _capture_days(profile: dict) -> list[str]:
    """One profile's capture days, from its provenance timestamps.

    The date part only: a matrix says WHEN a campaign was measured, and
    an ISO second on a page that spans two days is precision nobody
    asked for. Both ends of the run are read, because a run that started
    before midnight and finished after it happened on both days.
    """
    prov = profile.get("provenance") or {}
    return [html.escape(str(prov[end])[:10])
            for end in ("started", "finished") if prov.get(end)]


def _field_phrase(values: list, one: str, many: str, none: str,
                  absent: str) -> str:
    """One field across the set: what was declared, and HOW MANY
    PROFILES DECLARED NOTHING.

    The count is the point. ``_unique`` drops the profiles that carry no
    value, so a sentence built from the survivors alone reads as a
    universal — "measured on hardware tier enthusiast-16gb" — over a
    table whose own rows say ``undeclared`` and ``probe=—``. Six of the
    twenty-three profiles committed today declare no tier, so this is
    not a hypothetical: the intro would have been contradicted by the
    matrix directly beneath it. A page may summarise what it has; it may
    not summarise away what it hasn't.

    One value, several, or none is said as itself for the same reason a
    dash is not a zero — a set of one is not a range, and a set of none
    is not a quiet omission.
    """
    present = _unique(values)
    missing = sum(1 for value in values if not value)
    if not present:
        return none
    phrase = (one.format(present[0]) if len(present) == 1
              else many.format(", ".join(present)))
    if not missing:
        return phrase
    return f"{phrase} {absent.format(missing, len(values))}"


def _errata_lede(errata: dict[str, list[dict]] | None) -> str:
    """The Corrections paragraph's extra sentence, on a page that has
    flags to explain.

    A marker a reader has not been told about is a marker they have to
    guess at, and the wrong guess — "this number has been fixed" — is the
    dangerous one, because it makes a superseded figure look safe to
    use. The whole sentence lives inside the span, leading space
    included, so removing the span leaves the unflagged paragraph
    byte-for-byte.
    """
    if not errata:
        return ""
    # FIGURES, not errata: one erratum can supersede several numbers (E2
    # moves both the kv charge and the window derived from it), and a
    # reader counting flags down the page must find as many as the
    # sentence promised. Counting entries here would have said "1" over
    # two visible marks.
    figures = sum(len(entry.get("fields") or [])
                  for entries in errata.values() for entry in entries)
    rows = len(errata)
    return (f'<span class="erratum-lede"> {figures} figure(s) on {rows} '
            f'row(s) here are superseded by an erratum. Each is flagged where '
            f'it appears and is shown exactly as measured, never replaced — '
            f'the flag links to the correction.</span>')


def _intro_html(profiles: list[dict],
                errata: dict[str, list[dict]] | None = None) -> str:
    """The paragraph above the matrix: what it is, what it does not say,
    what measured it, and where the corrections live.

    The tools caveat is not optional garnish. The tools probe's system
    line ANNOUNCES the rubric it scores (``assay/tools.py``), so every
    rate under that column is a rate of instructed behaviour; a reader
    who takes it for a model reaching for a tool on its own has been
    misled by this page, and the sentence that prevents that has to
    travel with the numbers.
    """
    tiers = _field_phrase(
        [(p.get("provenance") or {}).get("tier") for p in profiles],
        "Hardware tier {}", "Hardware tiers {}",
        "No profile here declares a hardware tier",
        "({0} of {1} declare none)")
    probes = _field_phrase(
        [p.get("probe_version") for p in profiles],
        "Measured by probe version {}", "Measured by probe versions {}",
        "No profile here records the probe version that measured it",
        "({0} of {1} record none)")
    # The days are a SPAN, not a list: a fifteen-model campaign that ran
    # over two nights has a first day and a last one, and naming every
    # day between them would be a fact about scheduling, not about the
    # measurements. The undated profiles are still counted — a span read
    # off the profiles that carry stamps says nothing about the ones
    # that do not, and must not appear to.
    per_profile = [_capture_days(p) for p in profiles]
    day_list = sorted({day for days in per_profile for day in days})
    undated = sum(1 for days in per_profile if not days)
    days = ("No profile here records when it was captured" if not day_list
            else f"Captured on {day_list[0]}" if len(day_list) == 1
            else f"Captured between {day_list[0]} and {day_list[-1]}")
    if day_list and undated:
        days += f" ({undated} of {len(profiles)} record no date)"
    return (
        "<p><strong>What this is.</strong> One row per locally-served "
        f"model, measured by <a href=\"{_REPO_URL}\">assay</a> and published "
        "exactly as measured. Each badge is a verdict with its lens "
        "attached — hover it for the numbers that decided it — and a dashed "
        "badge marked &#8224; is provisional: that sample could not separate "
        "the verdict from its neighbours. Unmeasured is said in words, never "
        "shown as a zero.</p>"

        "<p><strong>The tool-calling rates are rates of instructed "
        "behaviour.</strong> The probe's system line announces the rubric it "
        "scores — call exactly one tool, use the arguments the request names, "
        "quote the result token — so a rate here says the model can follow "
        "that protocol when it is told to. Nothing on this page may be read "
        "as a model reaching for a tool on its own.</p>"

        # Three sentences rather than one, so each fact carries its own
        # missing count instead of hanging a single "(and N declare
        # none)" off a clause nobody can tell it belongs to.
        f"<p><strong>Provenance.</strong> {tiers}. {probes}. {days}. "
        "Every row's detail block names the probe version that "
        "measured it, and the line under the heading names the schema "
        "version(s) the documents carry — a row measured by an older "
        "instrument is not comparable to its neighbour just because they "
        "share a page.</p>"

        "<p><strong>Corrections.</strong> Profiles are never rewritten after "
        "the fact; evidence is not edited to suit a later fix. Where a fix "
        "changed what a number means, it is filed in "
        f"<a href=\"{_ERRATA_HREF}\">the errata</a>.{_errata_lede(errata)}</p>"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="render every profile in a directory as the published "
                    "capability matrix (deterministic: no clock is read)")
    parser.add_argument("--profiles-dir", type=Path,
                        default=_DEFAULT_PROFILES_DIR,
                        help="directory of profile JSONs "
                             "(default: %(default)s)")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                        help="page to write (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        profiles = _load_profiles(args.profiles_dir)
        errata = _load_errata(args.profiles_dir, profiles)
    except BuildError as error:
        print(f"build_matrix: {error}", file=sys.stderr)
        return 1
    page = render_report(profiles, page_title=_PAGE_TITLE,
                         intro_html=_intro_html(profiles, errata),
                         errata=errata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # ``newline="\n"`` so the committed bytes do not depend on the
    # platform that built them, which is the same law as the clock.
    args.out.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(profiles)} profile(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
