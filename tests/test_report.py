"""The report GUI (v1.4): one self-contained HTML page from N profiles."""

import json
from pathlib import Path

from assay.cli import main
from assay.report import VERDICT_ORDER, render_report

from test_profile import make_profile  # the canonical full-profile builder
from test_profile import make_long_output, unscorable_rung

_LIVE = Path(__file__).resolve().parents[1] / "docs/superpowers/evidence/live"
_V1_PROFILE = _LIVE / "qwen2.5-coder-7b-instruct-q8_0-quick.json"


def profile_dict(**overrides):
    return json.loads(make_profile(**overrides).to_json())


def test_report_renders_matrix_with_lensed_badges():
    p = profile_dict()
    html = render_report([p])
    assert "<!doctype html" in html
    assert "qwen2.5-coder:7b-instruct-q8_0" in html
    # every verdict column present
    for v in ("structured_extraction", "patch_editing", "loop_discipline",
              "chat_speed"):
        assert v in html
    # lenses surface as hover titles
    assert "landing=" in html


def test_provisional_verdicts_render_dashed_with_interval():
    p = profile_dict()
    p["verdicts"]["patch_editing"] = {
        "verdict": "ready", "provisional": True,
        "interval95": [0.566, 1.0], "lens": {"landing": "test"},
    }
    html = render_report([p])
    # the marker must be ON the rendered badge, not merely in the CSS
    assert 'class="badge b-ready provisional"' in html
    assert "[0.57, 1.00]" in html


def test_emulated_tier_is_labelled():
    p = profile_dict()
    p["provenance"]["tier"] = "average-gamer-8gb"
    p["provenance"]["emulated"] = True
    html = render_report([p])
    assert "average-gamer-8gb" in html
    assert "emulated" in html


def test_dropped_lines_print_in_full():
    p = profile_dict()
    p["provenance"]["dropped"] = ["speed: budget exhausted before any probe"]
    html = render_report([p])
    assert "speed: budget exhausted before any probe" in html


def test_report_escapes_hostile_model_names():
    p = profile_dict()
    p["model"]["name"] = "<script>alert(1)</script>"
    html = render_report([p])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_escapes_every_profile_sourced_cell_not_just_the_name():
    """The model name was escaped; the grids around it were not.

    A profile is an untrusted document — it can arrive by email, or be
    written by the endpoint under test — and the module's own comments
    claim a hostile one cannot write markup through the page. That claim
    has to hold for every cell that carries profile text, not the one
    cell somebody remembered.
    """
    payload = "<script>alert(1)</script>"
    p = profile_dict()
    p["ceiling_shapes"][0]["shape"] = payload            # shapes grid
    p["ceiling_shapes"][0]["max_verified"] = payload
    p["codecs"]["json_object"]["small"]["n"] = payload   # codec grid n=
    p["loop"]["patch_rate"] = payload                    # loop line
    p["geometry"]["kv_kib_per_token"] = payload          # geometry line
    p["ceiling"]["max_verified"] = payload               # ceiling line
    p["envelope"]["fidelity"] = payload                  # envelope line

    html = render_report([p])

    assert "<script>" not in html
    # ...and every one of them still reached the page, escaped: dropping
    # the cell would hide the finding instead of neutering the markup.
    assert html.count("&lt;script&gt;alert(1)&lt;/script&gt;") == 7


# --- v1.5: the long_output column and its rung grid ------------------------


def test_long_output_is_a_matrix_column_next_to_long_context():
    # Position is the point: long_context is how much it can READ and
    # long_output is how much it can WRITE, and a reader comparing them
    # should not have to hunt across the table.
    assert VERDICT_ORDER.index("long_output") == (
        VERDICT_ORDER.index("long_context") + 1)
    html = render_report([profile_dict()])
    assert "<th>long_output</th>" in html


def test_degrades_at_verdict_is_styled_risky_and_says_where():
    # The verdict is dynamic (degrades-at-2048), so a class built from
    # it lands on no CSS rule at all and the badge renders uncoloured —
    # the one verdict that names a limit would be the one nobody sees.
    p = profile_dict()
    p["verdicts"]["long_output"] = {
        "verdict": "degrades-at-2048", "provisional": True,
        "lens": {"rungs_scored": 3, "deepest_scored_tokens": 2048},
    }
    html = render_report([p])
    assert 'class="badge b-risky provisional"' in html
    # ...and the extent survives verbatim in the label: "risky" alone
    # does not say where it broke.
    assert "degrades-at-2048" in html
    assert "b-degrades-at" not in html


def test_rung_grid_prints_every_metric_and_dashes_what_was_not_measured():
    lo = make_long_output(degenerate_from=2048)
    lo = type(lo)(rungs=lo.rungs + (unscorable_rung(4096),), skipped=lo.skipped)
    html = render_report([profile_dict(long_output=lo)])
    for header in ("target", "generated", "distinct", "zlib", "degenerate"):
        assert header in html
    assert "512" in html and "504" in html      # target / generated
    assert "0.94" in html and "0.41" in html    # the healthy 512 rung
    assert "0.04" in html and "0.03" in html    # the degenerate 2048 rung
    # a rung that spent a call and scored nothing reads as dashes, never
    # as zeros — 0.00 distinct would be the worst possible reading.
    grid = html.split("<th>long output target</th>")[1].split("</table>")[0]
    row = next(r for r in grid.split("<tr>") if ">4096</td>" in r)
    assert row.count("—") == 3
    assert "0.00" not in html


def test_rung_grid_names_the_rungs_that_never_ran():
    lo = make_long_output(skipped=("4096: budget exhausted",))
    html = render_report([profile_dict(long_output=lo)])
    assert "4096: budget exhausted" in html


def test_profiles_without_the_family_render_unmeasured_not_a_crash():
    p = profile_dict()
    del p["long_output"]
    html = render_report([p])
    assert 'class="badge b-unmeasured"' in html
    assert "long_output unmeasured" in html


def test_v1_payloads_still_render_through_the_v15_report():
    # v1 wrote verdicts as BARE STRINGS and knew no long_output family;
    # the committed live evidence is the real thing, not a fixture.
    payload = json.loads(_V1_PROFILE.read_text(encoding="utf-8"))
    assert payload["assay_profile_version"] == 1
    assert isinstance(payload["verdicts"]["structured_extraction"], str)

    html = render_report([payload])

    # Every column unmeasured (bare strings carry no lens, and v1 has no
    # long_output at all) — and nothing raised on the way there.
    assert html.count('class="badge b-unmeasured"') == len(VERDICT_ORDER)
    assert "long_output unmeasured" in html


def test_a_none_in_a_lens_reads_unmeasured_in_the_badge_tooltip():
    """The tooltip is a lens, and a lens obeys the None-vs-zero rule.

    A ladder the ceiling capped scores nothing, so its lens carries
    ``deepest_scored_tokens: None``; the tooltip interpolated that
    straight and hovered as ``deepest_scored_tokens=None`` — Python's
    word for unmeasured, beside a page that says "unmeasured"
    everywhere else (``render_table`` was fixed for this in v1.5; the
    HTML view was not).
    """
    p = profile_dict()
    p["verdicts"]["long_output"] = {
        "verdict": "unmeasured",
        "lens": {"rungs_scored": 0, "deepest_scored_tokens": None},
    }
    html = render_report([p])
    assert "deepest_scored_tokens=unmeasured" in html
    assert "deepest_scored_tokens=None" not in html
    assert "rungs_scored=0" in html  # a measured zero is still a zero


def test_a_verdict_the_stylesheet_does_not_know_borrows_no_colour():
    # The class attribute is no longer built from profile text: an
    # unknown verdict renders uncoloured (it earned no colour) and a
    # hostile one cannot break out of the attribute.
    p = profile_dict()
    p["verdicts"]["long_context"] = {
        "verdict": '"><script>alert(1)</script>', "lens": {}}
    html = render_report([p])
    assert "<script>" not in html
    assert 'class="badge"' in html


def test_v1_codec_cells_dash_the_column_v1_never_wrote():
    # v1 measured `lands` and not `lands_applies`. Formatting the absent
    # half crashed the whole page; printing it as 0.00 would be worse
    # still — that reads as "nothing applied", which v1 never claimed.
    p = profile_dict()
    p["codecs"]["search_replace"]["tiny"] = {"lands": 0.6, "n": 5}
    html = render_report([p])
    assert "0.60 / —" in html


def test_cli_report_rejects_files_that_are_not_profiles(tmp_path, capsys):
    """`report` gets `diff`'s exit 4, for the same reason: a rendered
    matrix row asserts a model was measured, and `{}` measured nothing."""
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json at all", encoding="utf-8")
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    out = tmp_path / "report.html"

    assert main(["report", str(garbage), "--out", str(out)]) == 4
    err = capsys.readouterr().err
    assert "infrastructure" in err.lower() and "garbage.json" in err

    assert main(["report", str(empty), "--out", str(out)]) == 4
    assert "assay_profile_version" in capsys.readouterr().err

    # and no half-written report was left behind claiming to be one.
    assert not out.exists()


def test_cli_report_renders_the_committed_live_profiles(tmp_path, capsys):
    out = tmp_path / "report.html"
    files = sorted(str(path) for path in _LIVE.glob("*.json"))
    assert files

    assert main(["report", *files, "--out", str(out)]) == 0

    html = out.read_text(encoding="utf-8")
    assert "<th>long_output</th>" in html
    assert "granite-code" in html
    capsys.readouterr()


def test_cli_report_writes_the_file(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    f1.write_text(json.dumps(profile_dict()))
    p2 = profile_dict()
    p2["model"]["name"] = "second-model"
    f2.write_text(json.dumps(p2))
    out = tmp_path / "report.html"
    assert main(["report", str(f1), str(f2), "--out", str(out)]) == 0
    html = out.read_text()
    assert "second-model" in html and "qwen2.5-coder" in html
