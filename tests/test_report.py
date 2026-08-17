"""The report GUI (v1.4): one self-contained HTML page from N profiles."""

import json
from pathlib import Path

from assay.cli import main
from assay.report import VERDICT_ORDER, render_report

from test_profile import make_profile  # the canonical full-profile builder
from test_profile import (_DEEP_GRADES, make_codecs, make_geometry,
                          make_long_output, make_loop, make_tools,
                          unscorable_rung)

_EVIDENCE = Path(__file__).resolve().parents[1] / "docs/superpowers/evidence"
_LIVE = _EVIDENCE / "live"
_V1_PROFILE = _LIVE / "qwen2.5-coder-7b-instruct-q8_0-quick.json"
#: A v4-era profile: it has speed, loop and ceiling_shapes, and none of
#: the v1.6 families. Real recorded evidence, not a hand-thinned fixture.
_V4_PROFILE = _EVIDENCE / "tier-enthusiast/qwen3-14b.json"


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
    # Scoped to the GRID, not the page: v1.6 gave the loop line a
    # `repeat_rate` of a measured 0.00, which is an honest zero and a
    # different claim from an unscorable rung's missing metric.
    assert "0.00" not in grid


def test_rung_grid_names_the_rungs_that_never_ran():
    lo = make_long_output(skipped=("4096: budget exhausted",))
    html = render_report([profile_dict(long_output=lo)])
    assert "4096: budget exhausted" in html


# --- v1.6: the MoE marker in the geometry detail ---------------------------


def test_detail_marks_a_moe_model_with_used_of_count():
    html = render_report([profile_dict(
        geometry=make_geometry(expert_count=128, expert_used_count=8))])
    assert "MoE 8-of-128" in html


def test_detail_omits_the_moe_marker_unless_both_counts_are_measured():
    # A dense model is not a 0-expert MoE, and one measured half is not
    # an MoE fact — neither gets a marker.
    assert "MoE" not in render_report([profile_dict()])
    assert "MoE" not in render_report([
        profile_dict(geometry=make_geometry(expert_count=128))])
    assert "MoE" not in render_report([
        profile_dict(geometry=make_geometry(expert_used_count=8))])


def test_detail_geometry_survives_a_profile_predating_the_expert_fields():
    # The report reads raw dicts off disk, including profiles written
    # before the expert keys existed: a missing key must not crash the
    # page it is only a marker on.
    p = profile_dict()
    del p["geometry"]["expert_count"]
    del p["geometry"]["expert_used_count"]
    html = render_report([p])
    assert "KiB/token" in html
    assert "MoE" not in html


# --- v1.6: the tool_calling column, its rates, and the loop's recovery ----


def _detail_line(html: str, label: str) -> str:
    """The one detail paragraph whose key is ``label``.

    Matched on the closing tag, not the opening one, so a key that also
    carries a hover (``tools``) is found by the same helper as one that
    does not (``loop``).
    """
    return html.split(f">{label}</span>")[1].split("</p>")[0]


def test_tool_calling_is_a_matrix_column_after_loop_discipline():
    # Position is the point, as with long_output: loop_discipline is
    # whether it can follow a scripted loop and tool_calling is whether
    # the endpoint takes tools at all — a reader deciding whether to
    # wire the model into an agent reads them together.
    assert VERDICT_ORDER.index("tool_calling") == (
        VERDICT_ORDER.index("loop_discipline") + 1)
    assert "<th>tool_calling</th>" in render_report([profile_dict()])


def test_unsupported_badge_keeps_its_word_and_borrows_the_unmeasured_grey():
    """A refusal is not a gap, so the label stays verbatim; but it is
    also not a rung the model earned, so it takes no rung colour. The
    class attribute is bucketed, never built from the verdict string."""
    p = profile_dict()
    p["verdicts"]["tool_calling"] = {
        "verdict": "unsupported", "provisional": False,
        "interval95": None, "lens": {"toolset": "assay-tools-v1"}}
    html = render_report([p])
    assert ('<span class="badge b-unmeasured" title="toolset=assay-tools-v1">'
            "unsupported</span>") in html
    # ...and no class was invented for it: b-unsupported styles nothing.
    assert "b-unsupported" not in html
    # No interval to print, and none invented.
    assert '<span class="interval">' not in html


def test_detail_prints_the_four_tool_rates_with_the_composite_and_its_n():
    html = render_report([profile_dict(tools=make_tools(
        call_rate=0.8, right_tool_rate=0.75, args_valid_rate=0.5,
        result_use_rate=0.6, composite=0.4, n_tasks=5))])
    line = _detail_line(html, "tools")
    assert "composite 0.40" in line
    assert "n=5" in line
    assert "call 0.80" in line
    assert "right-tool 0.75" in line
    assert "args 0.50" in line
    assert "result-use 0.60" in line


def test_detail_dashes_the_tool_rates_a_run_could_not_measure():
    """``right_tool_rate`` and ``args_valid_rate`` are over the T1s that
    called at ALL, so both are None when nothing called — while the
    composite beside them is a measured 0.0. Printing 0.00 there would
    claim a "called the wrong tool with bad arguments" finding the run
    never made."""
    html = render_report([profile_dict(tools=make_tools(
        call_rate=0.0, right_tool_rate=None, args_valid_rate=None,
        result_use_rate=0.0, composite=0.0, n_tasks=5))])
    line = _detail_line(html, "tools")
    assert "composite 0.00" in line     # a measured zero is still a zero
    assert "call 0.00" in line
    assert "right-tool —" in line
    assert "args —" in line
    assert "result-use 0.00" in line


def test_detail_says_the_endpoint_refused_rather_than_dashing_it():
    html = render_report([profile_dict(tools=make_tools(
        supported=False, call_rate=None, right_tool_rate=None,
        args_valid_rate=None, result_use_rate=None, composite=None,
        n_tasks=0, n_turns=0))])
    line = _detail_line(html, "tools")
    assert "unsupported" in line
    assert "unmeasured" not in line


def test_detail_names_the_tool_rates_as_instructed_behaviour():
    """The lens fact, on the page that publishes the numbers: the probe's
    system line ANNOUNCES the rubric it scores, so these are rates of
    instructed behaviour and not of a model reaching for tools on its
    own. A reader comparing them to a harness that does not spell the
    rules out should expect them high."""
    html = render_report([profile_dict()])
    assert "instructed behaviour" in html.lower()
    assert "reaches for" not in html.lower()


def test_detail_tools_absent_from_the_schema_reads_unmeasured_not_a_crash():
    p = profile_dict()
    del p["tools"]
    html = render_report([p])
    assert "unmeasured" in _detail_line(html, "tools")


def test_detail_loop_line_shows_recovery_doom_and_the_error_run_count():
    # v1.6's second loop script. recovery_rate alone does not say how
    # much evidence is behind it, so the denominator rides beside it.
    html = render_report([profile_dict(loop=make_loop(
        recovery_rate=0.5, doom_loop_rate=0.25, n_error_runs=4))])
    line = _detail_line(html, "loop")
    assert "recovery 0.50" in line
    assert "doom 0.25" in line
    assert "error runs=4" in line


def test_detail_loop_line_dashes_a_schema_that_had_no_error_script():
    # A pre-v1.6 loop payload has no recovery/doom/error-run keys at
    # all. 0.00 recovery would read as a model that never got out of a
    # failed patch — a finding that schema could not make.
    p = profile_dict()
    for key in ("recovery_rate", "doom_loop_rate", "n_error_runs"):
        del p["loop"][key]
    line = _detail_line(render_report([p]), "loop")
    assert "recovery —" in line
    assert "doom —" in line
    assert "error runs=—" in line
    assert "None" not in line


def test_detail_loop_rates_that_were_never_measured_read_as_dashes():
    # Every loop rate is `float | None`; the line used to interpolate
    # them raw, so an unmeasured one hovered as Python's `None` on a
    # page that says "unmeasured" or "—" everywhere else.
    p = profile_dict()
    p["loop"]["action_fidelity"] = None
    p["loop"]["patch_rate"] = None
    line = _detail_line(render_report([p]), "loop")
    assert "None" not in line
    assert line.count("—") == 2


def test_detail_dashes_the_pre_v16_ceiling_and_envelope_nulls_too():
    """The two lines the v1.6 ``_num`` sweep did not reach.

    ``ceiling.max_verified`` is None whenever the ladder died before it
    verified anything, and ``envelope.fidelity`` is None whenever the
    budget died at n == 0 (the field's own comment says so). Both were
    interpolated raw, so the oldest lines on the page printed Python's
    ``None`` while every family added since printed a dash.
    """
    p = profile_dict()
    p["ceiling"]["max_verified"] = None
    p["envelope"]["fidelity"] = None

    html = render_report([p])

    assert "max verified —" in _detail_line(html, "ceiling")
    assert "None" not in _detail_line(html, "ceiling")
    assert "fidelity —" in _detail_line(html, "envelope")
    assert "None" not in _detail_line(html, "envelope")


def test_detail_keeps_a_measured_ceiling_a_token_count_not_a_rate():
    # The dash fix must not reformat the number: max_verified is a token
    # count, and "16384.00" would read as a rate that overflowed.
    line = _detail_line(render_report([profile_dict()]), "ceiling")
    assert "max verified 11500 " in line


def test_v4_era_payloads_render_through_the_v16_report(tmp_path):
    """A real v4 profile: speed, loop and shapes, but no tools family,
    no long_output, and a loop with no error script. Every v1.6 addition
    has to read as unmeasured rather than raise."""
    payload = json.loads(_V4_PROFILE.read_text(encoding="utf-8"))
    assert payload["assay_profile_version"] == 4
    assert "tools" not in payload
    assert "recovery_rate" not in payload["loop"]

    html = render_report([payload])

    assert 'class="badge b-unmeasured"' in html
    assert "unmeasured" in _detail_line(html, "tools")
    assert "recovery —" in _detail_line(html, "loop")
    out = tmp_path / "report.html"
    assert main(["report", str(_V4_PROFILE), "--out", str(out)]) == 0


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


def _codec_grid_html(html: str) -> str:
    """The codec matrix table, alone."""
    start = html.index('<table class="grid"><tr><th>codec ')
    return html[start:html.index("</table>", start) + len("</table>")]


def test_codec_grid_carries_every_grade_that_was_measured():
    """The HTML matrix had the text table's hardcoded triple and the
    same consequence: v1.7 writes six ``json_object`` cells and the page
    showed three, so the shape grades were paid for, recorded, and
    invisible to every reader of the report."""
    p = profile_dict(codecs=make_codecs(deep=True))
    grid = _codec_grid_html(render_report([p]))

    for grade in _DEEP_GRADES:
        assert f"<th>{grade}</th>" in grid, grade
    # The measured rates reach the page, both lenses (json's two
    # coincide by construction), and the patch codecs — never asked at
    # those shapes — dash rather than borrow a number.
    assert "0.60 / 0.60" in grid and "0.40 / 0.40" in grid
    sr_row = grid[grid.index("<td>search_replace</td>"):]
    sr_row = sr_row[:sr_row.index("</tr>")]
    assert sr_row.count('<td class="k">—</td>') == len(_DEEP_GRADES), sr_row


def test_a_grade_name_from_the_document_cannot_write_markup():
    """The grade columns are now read OFF the profile, so a header cell
    carries document text for the first time. It goes through the same
    escaper every other profile string does — a derived column must not
    become the one unescaped hole in the page."""
    p = profile_dict()
    p["codecs"]["json_object"]["<script>alert(1)</script>"] = {
        "lands": 1.0, "lands_applies": 1.0, "n": 5}
    html = render_report([p])
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_codec_grid_of_a_three_grade_profile_is_unchanged():
    """The regression pin: a v4 profile — and every other committed one
    — keeps exactly the three columns it always had, in order, with no
    shape column invented for a run that never measured one."""
    payload = json.loads(_V4_PROFILE.read_text(encoding="utf-8"))
    grid = _codec_grid_html(render_report([payload]))

    assert ("</span></th><th>tiny</th><th>small</th><th>medium</th></tr>"
            in grid)
    assert grid.count("<th>") == 4
    for grade in _DEEP_GRADES:
        assert grade not in grid, grade


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
