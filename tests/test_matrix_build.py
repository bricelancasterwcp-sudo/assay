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
from datetime import date, datetime, timezone
from pathlib import Path

from test_profile import make_profile

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_matrix.py"


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
    assert "../superpowers/evidence/tier-enthusiast/ERRATA.md" in page


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
