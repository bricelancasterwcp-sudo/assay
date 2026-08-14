"""The report GUI (v1.4): one self-contained HTML page from N profiles."""

import json

from assay.cli import main
from assay.report import render_report

from test_profile import make_profile  # the canonical full-profile builder


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
