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

from assay.report import render_report

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
_ERRATA_HREF = "../superpowers/evidence/tier-enthusiast/ERRATA.md"


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
    """
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise BuildError(
            f"no profile JSON matched {directory}/*.json — refusing to "
            f"publish an empty matrix (a page with no rows reads as a "
            f"campaign that measured nothing)")
    return [_load_profile(path) for path in paths]


def _unique(values) -> list[str]:
    """The distinct non-empty values, sorted and escaped.

    Sorted so the sentence is the same on every machine; escaped because
    every one of these came out of a profile document, and the paragraph
    they land in is the one place on the page that is NOT escaped for
    the caller.
    """
    return sorted({html.escape(str(value)) for value in values if value})


def _dates(profiles: list[dict]) -> list[str]:
    """The distinct capture days, sorted, from provenance timestamps.

    The date part only: a matrix says WHEN a campaign was measured, and
    an ISO second on a page that spans two days is precision nobody
    asked for. Both ends of each run are read, because a run that
    started before midnight and finished after it happened on both days.
    """
    stamps = [(profile.get("provenance") or {}).get(end)
              for profile in profiles for end in ("started", "finished")]
    return sorted({html.escape(str(stamp)[:10]) for stamp in stamps if stamp})


def _spans(items: list[str], one: str, many: str, none: str) -> str:
    """One item, several, or none — said as itself in all three cases.

    A set of one is not a range, a set of none is not a set of one, and
    "probes 0.9.0" or a silently omitted clause would each be the page
    guessing on the reader's behalf.
    """
    if not items:
        return none
    if len(items) == 1:
        return one.format(items[0])
    return many.format(", ".join(items))


def _intro_html(profiles: list[dict]) -> str:
    """The paragraph above the matrix: what it is, what it does not say,
    what measured it, and where the corrections live.

    The tools caveat is not optional garnish. The tools probe's system
    line ANNOUNCES the rubric it scores (``assay/tools.py``), so every
    rate under that column is a rate of instructed behaviour; a reader
    who takes it for a model reaching for a tool on its own has been
    misled by this page, and the sentence that prevents that has to
    travel with the numbers.
    """
    tiers = _spans(_unique((p.get("provenance") or {}).get("tier")
                           for p in profiles),
                   "hardware tier {}", "hardware tiers {}",
                   "an undeclared hardware tier")
    probes = _spans(_unique(p.get("probe_version") for p in profiles),
                    "probe version {}", "probe versions {}",
                    "an unrecorded probe version")
    # The days are a SPAN, not a list: a fifteen-model campaign that ran
    # over two nights has a first day and a last one, and naming every
    # day between them would be a fact about scheduling, not about the
    # measurements.
    day_list = _dates(profiles)
    days = ("on dates the profiles do not record" if not day_list
            else f"on {day_list[0]}" if len(day_list) == 1
            else f"between {day_list[0]} and {day_list[-1]}")
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

        f"<p><strong>Provenance.</strong> Measured on {tiers} by {probes} "
        f"{days}. Every row's detail block names the probe version that "
        "measured it, and the line under the heading names the schema "
        "version(s) the documents carry — a row measured by an older "
        "instrument is not comparable to its neighbour just because they "
        "share a page.</p>"

        "<p><strong>Corrections.</strong> Profiles are never rewritten after "
        "the fact; evidence is not edited to suit a later fix. Where a fix "
        "changed what a number means, it is filed in "
        f"<a href=\"{_ERRATA_HREF}\">the errata</a>.</p>"
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
    except BuildError as error:
        print(f"build_matrix: {error}", file=sys.stderr)
        return 1
    page = render_report(profiles, page_title=_PAGE_TITLE,
                         intro_html=_intro_html(profiles))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # ``newline="\n"`` so the committed bytes do not depend on the
    # platform that built them, which is the same law as the clock.
    args.out.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(profiles)} profile(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
