"""`scripts/build_matrix.py`: the published capability matrix (v1.7).

The page this script writes is COMMITTED, so the tests here are mostly
about determinism: the same profiles must produce the same bytes, or the
diff between two builds stops being readable and a real change hides
inside the churn. Nothing on the page may come from the build clock —
every date is read from profile provenance — and the profiles are
ordered by path rather than by whatever order the filesystem hands back.

The script is loaded from its path rather than imported as a module,
which is also a pin: `scripts/` is not a package and never will be, and
the script has to keep working as `python scripts/build_matrix.py`.
"""

import importlib.util
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

from assay import report
from test_profile import make_profile

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "build_matrix.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_matrix", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_matrix = _load_script()

#: Deliberately nowhere near today: every date assertion below is also a
#: statement that the build clock was not consulted, and a fixture dated
#: "recently" would make that statement true only until tomorrow.
_STARTED = "2020-01-02T03:04:05Z"
_FINISHED = "2020-01-03T06:07:08Z"


def write_profile(directory: Path, filename: str, *, model: str,
                  probe_version: str | None = "0.9.0",
                  started: str | None = _STARTED,
                  finished: str | None = _FINISHED,
                  tier: str | None = "enthusiast-16gb") -> dict:
    """One profile on disk. ``None`` DELETES the field rather than
    writing a null — which is the state the committed evidence is
    actually in: six of the twenty-three profiles in this repository
    have no ``provenance.tier`` key at all."""
    directory.mkdir(parents=True, exist_ok=True)
    doc = json.loads(make_profile().to_json())
    doc["model"]["name"] = model
    if probe_version is None:
        doc.pop("probe_version")
    else:
        doc["probe_version"] = probe_version
    for key, value in (("tier", tier), ("started", started),
                       ("finished", finished)):
        if value is None:
            doc["provenance"].pop(key, None)
        else:
            doc["provenance"][key] = value
    (directory / filename).write_text(json.dumps(doc, indent=2),
                                      encoding="utf-8")
    return doc


def make_erratum(**overrides) -> dict:
    """One machine-readable erratum entry, as the sidecar carries it.

    ``fields`` names the profile paths the correction lands on, and every
    one of them must be a path the renderer knows how to mark — a
    correction the page silently failed to flag is the exact failure this
    mechanism exists to prevent."""
    entry = dict(
        id="E9",
        model="model-a",
        fields=["geometry.kv_kib_per_token"],
        note="charged the wrong layer count — see ERRATA.md E9",
        href="../evidence/ERRATA.md",
    )
    entry.update(overrides)
    return entry


def write_sidecar(directory: Path, entries=None, *, version=1,
                  key: str = "errata") -> Path:
    """The errata sidecar, in the subdirectory the build reads it from.

    NOT beside the profiles, and the placement is load-bearing rather
    than tidy: the evidence directory's contract with every consumer is
    that each ``*.json`` in it is a profile, and `assay report` refuses
    one that is not (exit 4)."""
    path = directory / "errata" / "matrix-errata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: dict = {}
    if version is not None:
        doc["errata_sidecar_version"] = version
    doc[key] = [make_erratum()] if entries is None else entries
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def strip_errata_markup(page: str) -> str:
    """Everything the errata mechanism adds to a page, removed.

    Every addition is class-marked ``erratum`` precisely so this function
    can be surgical: what is left must be the page the build made before
    the mechanism existed, to the byte. The stylesheet block is stripped
    by identity against the module constant rather than by pattern —
    a regex over CSS would be the kind of approximate check that lets
    a real difference through."""
    page = page.replace(report._ERRATA_CSS, "")
    for pattern in (r'<sup class="erratum".*?</sup>',
                    r'<p class="erratum-note">.*?</p>',
                    r'<span class="erratum-lede">.*?</span>'):
        page = re.sub(pattern, "", page, flags=re.S)
    return page


def build(tmp_path: Path, out_name: str = "index.html") -> Path:
    out = tmp_path / "site" / out_name
    code = build_matrix.main(["--profiles-dir", str(tmp_path / "profiles"),
                              "--out", str(out)])
    assert code == 0
    return out


# --- determinism -----------------------------------------------------------


def test_two_builds_of_the_same_profiles_are_byte_identical(tmp_path):
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_profile(profiles, "b.json", model="model-b", probe_version="0.5.0")

    first = build(tmp_path, "one.html").read_bytes()
    second = build(tmp_path, "two.html").read_bytes()

    assert first == second
    assert first  # ...and it is a page, not an empty file both times


def test_every_date_on_the_page_comes_from_provenance_not_the_clock(tmp_path):
    """The one law that cannot be checked by rebuilding: a build that
    stamped "generated on <today>" would still be byte-identical to
    another build made the same minute, and would start lying at
    midnight. The fixtures are dated 2020 so the assertion below can
    never be satisfied by coincidence.

    Scoped to the head of the page — the title and the intro — because
    that is the whole of what this script writes. Below the matrix tag
    the text is profile content, and a profile is free to carry today's
    date in a field (the parallel family's tolerance provenance does);
    forbidding it there would be forbidding a measurement.
    """
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_profile(profiles, "b.json", model="model-b",
                  started="2020-01-05T00:00:00Z",
                  finished="2020-01-06T00:00:00Z")

    head = build(tmp_path).read_text(
        encoding="utf-8").split('<table class="matrix">')[0]

    assert "between 2020-01-02 and 2020-01-06" in head
    for today in {date.today().isoformat(),
                  datetime.now(timezone.utc).date().isoformat()}:
        assert today not in head, today


def test_rows_are_ordered_by_path_not_by_directory_order(tmp_path):
    """Sorted by filename, and the fixture proves it is that sort and not
    another: the alphabetically-first FILE holds the alphabetically-last
    MODEL, so a sort by model name — or the filesystem's own order — puts
    the rows the other way round."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "z-last-file.json", model="aaa-first-model")
    write_profile(profiles, "a-first-file.json", model="zzz-last-model")

    page = build(tmp_path).read_text(encoding="utf-8")

    assert page.index("zzz-last-model") < page.index("aaa-first-model")


# --- what the page says ----------------------------------------------------


def test_the_page_names_the_probe_version_of_every_row(tmp_path):
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a", probe_version="0.5.0")
    write_profile(profiles, "b.json", model="model-b", probe_version="0.9.0")

    page = build(tmp_path).read_text(encoding="utf-8")

    assert "probe=0.5.0" in page and "probe=0.9.0" in page
    # ...and the intro says the set spans two instruments, which is the
    # fact a reader needs before comparing one row against another.
    assert "0.5.0" in page.split('<table class="matrix">')[0]


def test_the_provenance_sentence_counts_the_profiles_that_declare_nothing(
        tmp_path):
    """A mixed set is the normal case, not the edge one: six of the
    twenty-three profiles committed in this repository carry no
    ``provenance.tier``. Summarising only the profiles that DO carry a
    field publishes a universal — "Hardware tier enthusiast-16gb" — over
    a table whose own rows say ``undeclared`` and ``probe=—`` two inches
    below. The count is what keeps the sentence true.
    """
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_profile(profiles, "b.json", model="model-b", tier=None,
                  probe_version=None, started=None, finished=None)

    head = build(tmp_path).read_text(
        encoding="utf-8").split('<table class="matrix">')[0]

    assert "Hardware tier enthusiast-16gb (1 of 2 declare none)" in head
    assert "Measured by probe version 0.9.0 (1 of 2 record none)" in head
    assert "(1 of 2 record no date)" in head
    # ...and the row the sentence is about says the same thing
    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "undeclared" in page and "probe=—" in page


def test_a_set_where_every_profile_declares_everything_carries_no_count(
        tmp_path):
    """The counterpart, and the reason the clause is conditional: a
    complete set is the campaign's goal, and a page that appended
    "(0 of 15 declare none)" to every sentence would be noise standing
    where a real gap should stand out."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_profile(profiles, "b.json", model="model-b")

    head = build(tmp_path).read_text(
        encoding="utf-8").split('<table class="matrix">')[0]

    assert "Hardware tier enthusiast-16gb." in head
    assert "of 2" not in head


def test_a_set_where_nobody_declares_a_field_says_so_outright(tmp_path):
    """Not "0 of 2 declared it" and not silence: no value to summarise is
    its own statement, and the sentence makes it."""
    write_profile(tmp_path / "profiles", "a.json", model="model-a",
                  tier=None, probe_version=None, started=None, finished=None)

    head = build(tmp_path).read_text(
        encoding="utf-8").split('<table class="matrix">')[0]

    assert "No profile here declares a hardware tier" in head
    assert "No profile here records the probe version that measured it" in head
    assert "No profile here records when it was captured" in head
    assert "of 1" not in head


def test_the_intro_carries_the_instructed_behaviour_caveat(tmp_path):
    write_profile(tmp_path / "profiles", "a.json", model="model-a")

    intro = build(tmp_path).read_text(encoding="utf-8")
    intro = intro.split('<table class="matrix">')[0]

    assert "instructed" in intro
    # the caveat is worth nothing if it does not say what was instructed
    assert "one tool" in intro


def test_the_intro_links_the_repo_and_the_errata(tmp_path):
    write_profile(tmp_path / "profiles", "a.json", model="model-a")

    page = build(tmp_path).read_text(encoding="utf-8")

    assert "https://github.com/bricelancasterwcp-sudo/assay" in page
    # relative to docs/matrix/index.html, which is where this page lives
    assert "../superpowers/evidence/tier-enthusiast-2026-08/ERRATA.md" in page


def test_the_intro_escapes_the_profile_sourced_values_it_quotes(tmp_path):
    """``intro_html`` is trusted markup by contract — so the trust stops
    here, at the one place profile text enters it. The probe version, the
    tier and the dates are all document fields."""
    write_profile(tmp_path / "profiles", "a.json", model="model-a",
                  probe_version="<script>alert(1)</script>",
                  tier="<img src=x onerror=alert(2)>")

    page = build(tmp_path).read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in page
    assert "<img src=x" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_the_page_is_titled_as_a_matrix_not_as_a_report(tmp_path):
    write_profile(tmp_path / "profiles", "a.json", model="model-a")

    page = build(tmp_path).read_text(encoding="utf-8")

    assert "<title>assay capability matrix</title>" in page
    assert "<h1>assay capability matrix</h1>" in page


# --- refusals --------------------------------------------------------------


def test_an_empty_profiles_dir_exits_non_zero_and_writes_no_page(tmp_path,
                                                                 capsys):
    """Never an empty matrix. A page with a header and no rows is a
    published claim that a campaign was run and found nothing, which is
    the opposite of what an empty directory means."""
    (tmp_path / "profiles").mkdir()
    out = tmp_path / "site" / "index.html"

    code = build_matrix.main(["--profiles-dir", str(tmp_path / "profiles"),
                              "--out", str(out)])

    assert code != 0
    assert "no profile" in capsys.readouterr().err.lower()
    assert not out.exists()


def test_a_missing_profiles_dir_is_the_same_refusal(tmp_path, capsys):
    out = tmp_path / "site" / "index.html"

    code = build_matrix.main(["--profiles-dir", str(tmp_path / "nope"),
                              "--out", str(out)])

    assert code != 0
    assert str(tmp_path / "nope") in capsys.readouterr().err
    assert not out.exists()


def test_a_dir_whose_files_are_not_json_matches_nothing_and_refuses(tmp_path,
                                                                    capsys):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "README.md").write_text("not a profile", encoding="utf-8")
    out = tmp_path / "site" / "index.html"

    assert build_matrix.main(["--profiles-dir", str(profiles),
                              "--out", str(out)]) != 0
    assert "no profile" in capsys.readouterr().err.lower()
    assert not out.exists()


def test_a_document_that_is_not_a_profile_stops_the_build(tmp_path, capsys):
    """`assay report`'s exit-4 lesson, on the published surface: `{}`
    renders a row of unmeasured badges under whatever name the file has,
    which is a capability claim for a run nobody made."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "good.json", model="model-a")
    (profiles / "bad.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "site" / "index.html"

    assert build_matrix.main(["--profiles-dir", str(profiles),
                              "--out", str(out)]) != 0
    err = capsys.readouterr().err
    assert "bad.json" in err and "assay_profile_version" in err
    assert not out.exists()


def test_unreadable_json_stops_the_build_before_anything_is_written(tmp_path,
                                                                    capsys):
    profiles = tmp_path / "profiles"
    write_profile(profiles, "good.json", model="model-a")
    (profiles / "truncated.json").write_text("{not json", encoding="utf-8")
    out = tmp_path / "site" / "index.html"

    assert build_matrix.main(["--profiles-dir", str(profiles),
                              "--out", str(out)]) != 0
    assert "truncated.json" in capsys.readouterr().err
    assert not out.exists()


# --- the wiring the campaign depends on ------------------------------------


def test_the_defaults_are_the_campaign_dir_and_the_published_path():
    """Task 12-13 write the campaign into that directory and Pages serves
    that path; both are wired to these two literals, so they are pinned
    here rather than left to a help string."""
    args = build_matrix.parse_args([])

    assert args.profiles_dir == Path(
        "docs/superpowers/evidence/tier-enthusiast-2026-08")
    assert args.out == Path("docs/matrix/index.html")


def test_the_out_directory_is_created_if_it_does_not_exist(tmp_path):
    write_profile(tmp_path / "profiles", "a.json", model="model-a")
    out = tmp_path / "brand" / "new" / "index.html"

    assert build_matrix.main(["--profiles-dir", str(tmp_path / "profiles"),
                              "--out", str(out)]) == 0
    assert out.exists()


# --- errata: the published number stays, and wears a flag -------------------
#
# House discipline is that a profile is NEVER rewritten after the fact:
# evidence is not edited to suit a later fix. That leaves the published
# matrix showing a figure a later fix has superseded, with nothing on the
# page saying so — which is the worst of both worlds, because a reader
# has no way to know the correction exists. The sidecar closes that gap
# WITHOUT touching the evidence: the wrong value stays visible, and a
# marker beside it says it is corrected by erratum and where to read the
# correction. A silent value replacement would be the same edit the
# discipline forbids, done one layer further out where nobody can see it.


def test_the_sidecar_flags_the_field_it_names_beside_the_published_value(
        tmp_path):
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles)

    page = build(tmp_path).read_text(encoding="utf-8")

    # the published number is STILL THERE, and the flag is attached to it
    assert '56<sup class="erratum"' in page
    assert "KiB/token" in page
    # the note and the pointer both reach the reader
    assert "charged the wrong layer count" in page
    assert "../evidence/ERRATA.md" in page
    assert "E9" in page


def test_the_note_says_the_flagged_figure_is_left_as_measured(tmp_path):
    """The one thing a marker alone cannot say. A reader who takes a
    flagged number for a quietly-corrected one has been misled in the
    dangerous direction — they would then trust it — so the block spells
    out that the value below is the measurement, unchanged."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles)

    detail = build(tmp_path).read_text(
        encoding="utf-8").split("<summary>model-a</summary>")[1]

    assert 'class="erratum-note"' in detail
    assert "left exactly as measured" in detail
    # ...and it stands ABOVE the figure it qualifies, not after it
    assert detail.index("erratum-note") < detail.index("KiB/token")


def test_an_erratum_marks_every_field_it_names_and_no_other(tmp_path):
    """Two fields named, two flags — and the third value on the same line
    carries none. A marker that spread to the whole line would say the
    ``limited_by`` classification was corrected too, which no erratum
    here claims."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles, [make_erratum(fields=["geometry.kv_kib_per_token",
                                                  "geometry.usable_window"])])

    page = build(tmp_path).read_text(encoding="utf-8")

    assert '56<sup class="erratum"' in page
    assert '32768<sup class="erratum"' in page
    assert '(limited by training_ctx)<sup' not in page


def test_a_field_the_sidecar_does_not_name_is_left_unflagged(tmp_path):
    """The counterpart: naming one field flags one field. A mechanism
    that flagged the whole row's geometry would make the sidecar's
    ``fields`` list decorative."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles, [make_erratum(fields=["geometry.usable_window"])])

    page = build(tmp_path).read_text(encoding="utf-8")

    assert '32768<sup class="erratum"' in page
    assert '56<sup class="erratum"' not in page


def test_the_flagged_row_is_marked_in_the_matrix_not_only_in_its_detail(
        tmp_path):
    """The corrected fields live inside a collapsed ``<details>``. A flag
    that only appears once that block is opened is a flag a reader
    scanning the matrix never sees, so the row carries one too."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles)

    matrix = build(tmp_path).read_text(encoding="utf-8").split("</table>")[0]

    assert 'class="erratum"' in matrix
    assert "model-a" in matrix


def test_only_the_named_model_is_flagged(tmp_path):
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_profile(profiles, "b.json", model="model-b")
    write_sidecar(profiles)

    page = build(tmp_path).read_text(encoding="utf-8")
    b_detail = page.split("<summary>model-b</summary>")[1]

    assert "erratum" not in b_detail


def test_the_intro_says_the_page_carries_flags_and_what_they_mean(tmp_path):
    """The count is of FIGURES, not of errata. One erratum can supersede
    several numbers — the real E2 moves both the kv charge and the window
    derived from it — and a reader who counts the marks down the page
    must find as many as the sentence promised."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_profile(profiles, "b.json", model="model-b")
    write_sidecar(profiles, [make_erratum(fields=["geometry.kv_kib_per_token",
                                                  "geometry.usable_window"])])

    page = build(tmp_path).read_text(encoding="utf-8")
    head = page.split('<table class="matrix">')[0]

    assert "2 figure(s) on 1 row(s)" in head
    assert head.count("2 figure(s)") == 1
    # the sentence must say the value was NOT replaced, or a reader will
    # assume a flagged number has quietly been corrected in place
    assert "never replaced" in head


def test_the_sidecar_stays_out_of_the_profile_glob(tmp_path):
    """The placement, pinned — and it was a bug first.

    The sidecar started out beside the profiles as ``ERRATA.json``, and
    `test_cli_still_loads_the_committed_evidence_profiles` failed
    immediately: `assay report <dir>/*.json` is a supported thing to run
    against the published corpus, and the CLI refuses any document
    without `assay_profile_version` (exit 4). One script's convenience
    would have broken the evidence directory for every consumer of it,
    so the sidecar lives one level down. `glob` is not recursive, which
    is the whole mechanism.
    """
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles)

    assert [p.name for p in sorted(profiles.glob("*.json"))] == ["a.json"]

    page = build(tmp_path).read_text(encoding="utf-8")

    assert "1 profile(s)" in page
    # ...and it is READ, not merely out of the way: the flag is there
    assert 'class="erratum"' in page


# --- errata: no sidecar changes nothing ------------------------------------


def test_a_directory_with_no_sidecar_renders_no_erratum_markup(tmp_path):
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")

    page = build(tmp_path).read_text(encoding="utf-8")

    assert "erratum" not in page


def test_the_sidecar_adds_the_flags_and_nothing_else_to_the_page(tmp_path):
    """The byte-identity law, stated as a difference: build the same
    profiles with and without a sidecar, remove every erratum-classed
    addition from the annotated page, and what is left must be the
    unannotated page EXACTLY. Anything else — a reordered row, a changed
    count, a stray wrapper — would mean the mechanism edits the page it
    is only supposed to annotate."""
    plain_dir = tmp_path / "plain"
    flagged_dir = tmp_path / "flagged"
    for directory in (plain_dir, flagged_dir):
        write_profile(directory, "a.json", model="model-a")
        write_profile(directory, "b.json", model="model-b")
    write_sidecar(flagged_dir, [make_erratum(
        fields=["geometry.kv_kib_per_token", "geometry.usable_window"])])

    def render(directory: Path, name: str) -> str:
        out = tmp_path / name
        assert build_matrix.main(["--profiles-dir", str(directory),
                                  "--out", str(out)]) == 0
        return out.read_text(encoding="utf-8")

    plain = render(plain_dir, "plain.html")
    flagged = render(flagged_dir, "flagged.html")

    assert flagged != plain
    assert strip_errata_markup(flagged) == plain


def test_two_builds_with_the_same_sidecar_are_byte_identical(tmp_path):
    """Determinism is not suspended for annotations: the sidecar's own
    order is the page's order, and nothing about a flag may come from a
    dict iteration that differs between runs."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles)

    assert (build(tmp_path, "one.html").read_bytes()
            == build(tmp_path, "two.html").read_bytes())


# --- errata: the refusals that keep a flag from silently not appearing ------


def test_a_sidecar_naming_a_model_no_profile_carries_stops_the_build(
        tmp_path, capsys):
    """The quiet failure this mechanism is most exposed to: a model name
    that does not match any row annotates nothing, and the page publishes
    the superseded figure looking exactly as checked as its neighbours.
    A typo must be loud."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles, [make_erratum(model="model-typo")])
    out = tmp_path / "site" / "index.html"

    assert build_matrix.main(["--profiles-dir", str(profiles),
                              "--out", str(out)]) != 0
    assert "model-typo" in capsys.readouterr().err
    assert not out.exists()


def test_a_sidecar_naming_a_field_the_page_cannot_mark_stops_the_build(
        tmp_path, capsys):
    """Same failure, other axis. The renderer marks the field paths it
    knows; a path it does not know renders no flag at all, so the build
    refuses rather than publishing a correction nobody can see."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles, [make_erratum(fields=["codecs.whole_file.tiny"])])
    out = tmp_path / "site" / "index.html"

    assert build_matrix.main(["--profiles-dir", str(profiles),
                              "--out", str(out)]) != 0
    assert "codecs.whole_file.tiny" in capsys.readouterr().err
    assert not out.exists()


def test_an_entry_without_a_note_or_a_pointer_stops_the_build(tmp_path,
                                                              capsys):
    """A bare marker is a worse page than no marker: it tells a reader
    something is wrong and nothing about what or where to read it."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    entry = make_erratum()
    entry.pop("note")
    write_sidecar(profiles, [entry])
    out = tmp_path / "site" / "index.html"

    assert build_matrix.main(["--profiles-dir", str(profiles),
                              "--out", str(out)]) != 0
    assert "note" in capsys.readouterr().err
    assert not out.exists()


def test_an_unreadable_sidecar_stops_the_build(tmp_path, capsys):
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    sidecar = profiles / "errata" / "matrix-errata.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("{not json", encoding="utf-8")
    out = tmp_path / "site" / "index.html"

    assert build_matrix.main(["--profiles-dir", str(profiles),
                              "--out", str(out)]) != 0
    assert "matrix-errata.json" in capsys.readouterr().err
    assert not out.exists()


def test_a_sidecar_that_declares_no_version_stops_the_build(tmp_path, capsys):
    """``_load_profile``'s gate, restated for the sidecar: a document
    must SAY what it is. Any JSON object would otherwise be accepted as
    an errata sidecar that annotates nothing."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles, version=None)
    out = tmp_path / "site" / "index.html"

    assert build_matrix.main(["--profiles-dir", str(profiles),
                              "--out", str(out)]) != 0
    assert "errata_sidecar_version" in capsys.readouterr().err
    assert not out.exists()


def test_a_sidecar_with_no_entries_stops_the_build(tmp_path, capsys):
    """An empty sidecar is a file that says "corrections were considered"
    and flags nothing. Deleting it says the same thing more honestly."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles, [])
    out = tmp_path / "site" / "index.html"

    assert build_matrix.main(["--profiles-dir", str(profiles),
                              "--out", str(out)]) != 0
    assert "errata" in capsys.readouterr().err
    assert not out.exists()


def test_the_sidecar_is_escaped_like_every_other_document_it_annotates(
        tmp_path):
    """The sidecar is author-supplied and repo-committed, which is the
    same trust class as ``intro_html`` — and it is escaped anyway. It is
    read from a directory a campaign writes into, and the one input that
    is trusted "because we wrote it" is the one that ships the markup."""
    profiles = tmp_path / "profiles"
    write_profile(profiles, "a.json", model="model-a")
    write_sidecar(profiles, [make_erratum(
        id="<script>alert(1)</script>",
        note='"><img src=x onerror=alert(2)>',
        href='" onmouseover=alert(3) x="')])

    page = build(tmp_path).read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in page
    assert "<img src=x" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    # the href is an ATTRIBUTE value: what matters is that the quote
    # cannot close it. The payload survives as inert text inside the
    # attribute, which is exactly what escaping is supposed to do.
    assert '" onmouseover=' not in page
    assert "&quot; onmouseover=alert(3) x=&quot;" in page


# --- the committed page ----------------------------------------------------


def test_the_committed_page_is_what_the_defaults_build_today(tmp_path):
    """Spec §5's acceptance: the published matrix is a COPY of a build,
    so it is the one artifact in this repository that can go stale in
    silence. A renderer change or an edit to the campaign's evidence
    leaves `docs/matrix/index.html` publishing the old numbers while
    every other test in this file — all of which build into `tmp_path` —
    stays green. This rebuilds the page from the script's REAL defaults
    and compares the bytes, so drift on either side fails here.

    It passed on arrival: the committed page is what the current
    renderer makes of the current evidence. That makes it a guard rather
    than a fix, and when it does fail the remedy is to rerun
    ``python scripts/build_matrix.py`` and commit the result — never to
    loosen the comparison, which would retire the only check that the
    published page and the repository still agree.

    The defaults are read from `parse_args` rather than retyped, so this
    builds exactly what the bare command builds; the literals themselves
    are pinned by the test above.
    """
    defaults = build_matrix.parse_args([])
    out = tmp_path / "index.html"

    code = build_matrix.main(
        ["--profiles-dir", str(_REPO / defaults.profiles_dir),
         "--out", str(out)])

    assert code == 0
    assert out.read_bytes() == (_REPO / defaults.out).read_bytes()
