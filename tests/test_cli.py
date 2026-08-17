"""Task 11 tests: the assay CLI (plan Task 11, spec §9).

Exit codes: 0 = profile produced (whatever it says); 2 = budget
exhausted before ANY family completed; 4 = infrastructure failure
before any measurement. No test here touches a real socket: the
backend factory and the VRAM reader are always replaced.

``diff`` measures nothing and so reads 1 and 2 differently — 1 = drift
found, 2 = the pair is not comparable — which is why its tests live in
their own section at the bottom rather than beside probe's.
"""

import json
from pathlib import Path

import pytest
from fakes import (CodecFailingBackend, MetadataFreeBackend, ScriptedBackend,
                   UnreachableBackend)

from assay import cli

_URL = "http://fake-host:11434"


@pytest.fixture(autouse=True)
def _no_real_world(monkeypatch):
    """Tests never shell out to nvidia-smi and never detect over HTTP."""
    monkeypatch.setattr("assay.run.free_vram_mib", lambda: 14558)
    monkeypatch.setattr("assay.cli.free_vram_mib", lambda: 14558)


def _use_backend(monkeypatch, backend) -> None:
    def fake_detect(base_url, model, *, forced=None):
        return backend

    monkeypatch.setattr("assay.run.detect_backend", fake_detect)
    monkeypatch.setattr("assay.cli.detect_backend", fake_detect)


def test_exit_0_with_profile_json_written(tmp_path, monkeypatch, capsys):
    _use_backend(monkeypatch, ScriptedBackend())
    out = tmp_path / "profile.json"
    record = tmp_path / "transcript.jsonl"

    code = cli.main(
        [
            "probe",
            _URL,
            "--model",
            "fake-model",
            "--quick",
            "--json",
            str(out),
            "--record",
            str(record),
        ]
    )

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["assay_profile_version"] == 6
    assert payload["ceiling"]["failure_mode"] == "none_up_to_cap"
    assert payload["verdicts"]["long_context"]["verdict"] == "ready"
    # The documented quick default budget was applied...
    assert payload["provenance"]["budget"] == {
        "max_calls": 130,
        "max_prompt_tokens": 220_000,
    }
    # ...and it COVERS the whole quick suite (2 calibration + 5 ladder +
    # 9 shapes + 10 envelope + 45 codecs + 15 loop + 2 speed + 4
    # long-output rungs + 10 tools = 102): the run must never exhaust the
    # default budget on a well-behaved endpoint (spec §12 criterion 1).
    # The loop family went 9 -> 15 BY DESIGN in v1.6 (scripted-loop-v2
    # plays a two-turn error script alongside each three-turn golden run)
    # and the tools family is new in v1.6.
    assert payload["provenance"]["spent"]["calls"] == 102
    assert (payload["provenance"]["spent"]["prompt_tokens"]
            < payload["provenance"]["budget"]["max_prompt_tokens"])
    for codec in ("search_replace", "whole_file", "json_object"):
        for grade in ("tiny", "small", "medium"):
            assert payload["codecs"][codec][grade]["n"] == 5, (codec, grade)
    assert payload["provenance"]["dropped"] == []

    # The v1.6 families survive the document round trip end to end, with
    # NON-None recovery/doom values: the error script is measured, not
    # merely defaulted (Task 5 carry — these had never been pinned
    # through the CLI).
    assert payload["loop"]["recovery_rate"] == 1.0
    assert payload["loop"]["doom_loop_rate"] == 0.0
    assert payload["loop"]["n_error_runs"] == 3
    assert payload["tools"]["supported"] is True
    assert payload["tools"]["composite"] == 1.0
    assert payload["tools"]["n_tasks"] == 5
    assert payload["verdicts"]["tool_calling"]["verdict"] == "ready"
    assert (payload["verdicts"]["loop_discipline"]["lens"]["instrument"]
            == "scripted-loop-v2")
    assert payload["verdicts"]["loop_discipline"]["lens"]["n_error_runs"] == 3

    # Human table on stdout; recording actually wrote a transcript.
    assert "assay profile" in capsys.readouterr().out
    rows = record.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 102  # one recorded call per spent call
    assert json.loads(rows[0])["outcome"] == "reply"
    # ...and the tool turns are recorded as tool turns, not as generates.
    assert sum(json.loads(row).get("kind") == "chat_tools" for row in rows) == 10


def test_exit_2_when_nothing_completed(monkeypatch, capsys):
    # No metadata (geometry unmeasurable) and a zero-call budget: the
    # first calibration charge dies with no family completed.
    _use_backend(monkeypatch, MetadataFreeBackend())

    code = cli.main(
        ["probe", _URL, "--model", "fake-model", "--max-calls", "0"]
    )

    assert code == 2
    assert "budget" in capsys.readouterr().err.lower()


def test_exit_4_on_unreachable_endpoint(monkeypatch, capsys):
    _use_backend(monkeypatch, UnreachableBackend())

    code = cli.main(["probe", _URL, "--model", "fake-model"])

    assert code == 4
    assert "infrastructure" in capsys.readouterr().err.lower()


def test_envelope_subcommand_prints_single_family_slice(monkeypatch, capsys):
    _use_backend(monkeypatch, ScriptedBackend())

    code = cli.main(["envelope", _URL, "--model", "fake-model"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"envelope"}
    assert payload["envelope"]["fidelity"] == 1.0
    # v1.5: the default mode is full, so the family subcommand runs
    # full's 30 envelope probes (it was quick's 10 before the remap).
    assert payload["envelope"]["n"] == 30


def test_cli_default_mode_is_full():
    # v1.5: sequential stopping made the honest mode affordable, so it
    # is what an operator gets without asking (spec §1 amendment).
    args = cli._build_parser().parse_args(
        ["probe", "http://x", "--model", "m"])
    assert args.mode == "full"
    quick = cli._build_parser().parse_args(
        ["probe", "http://x", "--model", "m", "--quick"])
    assert quick.mode == "quick"


def test_default_budget_for_the_default_mode_covers_the_worst_case():
    # The default mode's worst case is now thorough's old worst case
    # (a codec matrix that never decides early runs to the 315-call
    # cap); the default budget must cover it, not the old full suite.
    # Full's worst case is 411 of 500 in v1.6 (+10 tools, +10 error-script
    # turns), so it still has headroom and does not move.
    assert cli.DEFAULT_BUDGETS["full"].max_calls == 500
    assert cli.DEFAULT_BUDGETS["full"].max_prompt_tokens == 1_000_000
    assert cli.DEFAULT_BUDGETS["full"] == cli.DEFAULT_BUDGETS["thorough"]


def test_the_quick_default_budget_was_raised_for_the_v1_6_families():
    # PRE-REGISTERED (v1.6 plan): quick's worst case went 93 -> 109 with
    # the tools family (+10) and the loop error script (+6), which is one
    # call short of the old 110 default — a mid-family death on any run
    # that hits the worst case. The raise is deliberate and pinned here
    # so it cannot drift back.
    assert cli.DEFAULT_BUDGETS["quick"].max_calls == 130
    assert cli.DEFAULT_BUDGETS["quick"].max_prompt_tokens == 220_000


@pytest.mark.parametrize("flags", [[], ["--full"], ["--thorough"]])
def test_codecs_subcommand_stops_sequentially_like_the_probe_command(
    flags, monkeypatch, capsys
):
    # The family subcommand shares the mode table, so it must share the
    # stopping rule too: without the schedule this default-mode run
    # would spend 315 fixed calls on cells decided at n=5.
    backend = CodecFailingBackend()
    _use_backend(monkeypatch, backend)

    code = cli.main(["codecs", _URL, "--model", "fake-model", *flags])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)["codecs"]
    for codec, grades in payload.items():
        for grade, cell in grades.items():
            assert cell["n"] == 5, (codec, grade)
    assert backend.calls == 45


# --- diff subcommand -------------------------------------------------
#
# Exit codes here are a DIFFERENT taxonomy from probe's: 1 = drift
# found (with --gate: a regression), 2 = the two files are not
# comparable at all. 4 keeps its meaning — a file we could not read is
# infrastructure, never a finding.


def _diff_payload(**overrides):
    """A minimal raw profile payload: enough identity for the gate,
    plus one exact-valued cell for a doctored copy to move.

    The version stays 5 DELIBERATELY through the v6 bump: ``diff``
    compares documents, never schema numbers, and a fixture that tracked
    PROFILE_VERSION would quietly stop covering the case an operator
    actually has — last month's profile against today's.
    """
    payload = {
        "assay_profile_version": 5,
        "model": {"name": "fake-model", "quant": "Q8_0",
                  "weights_bytes": 8_000_000_000},
        "provenance": {"tier": "average-gamer-8gb", "emulated": False},
        "ceiling": {"max_verified": 8192, "failure_mode": "hard_error"},
    }
    payload.update(overrides)
    return payload


def _write_profile(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _ceiling(max_verified):
    return {"ceiling": {"max_verified": max_verified,
                        "failure_mode": "hard_error"}}


def test_cli_diff_exit_codes(tmp_path, capsys):
    old = _write_profile(tmp_path / "old.json", _diff_payload())
    same = _write_profile(tmp_path / "same.json", _diff_payload())
    worse = _write_profile(tmp_path / "worse.json",
                           _diff_payload(**_ceiling(4096)))
    other = _write_profile(tmp_path / "other.json",
                           _diff_payload(model={"name": "other-model",
                                                "quant": "Q8_0",
                                                "weights_bytes": 8_000_000_000}))

    assert cli.main(["diff", old, same]) == 0
    assert "no drift beyond noise" in capsys.readouterr().out

    assert cli.main(["diff", old, worse]) == 1
    assert "8192 -> 4096" in capsys.readouterr().out

    # Not comparable is its own answer, not "no changes" and not a
    # regression: nothing was subtracted, so nothing is reported.
    assert cli.main(["diff", old, other]) == 2
    assert "not comparable" in capsys.readouterr().out


def test_cli_diff_gate_fails_only_on_regressions(tmp_path, capsys):
    old = _write_profile(tmp_path / "old.json", _diff_payload())
    better = _write_profile(tmp_path / "better.json",
                            _diff_payload(**_ceiling(16384)))
    worse = _write_profile(tmp_path / "worse.json",
                           _diff_payload(**_ceiling(4096)))
    other = _write_profile(tmp_path / "other.json",
                           _diff_payload(model={"name": "other-model"}))

    # Bare diff answers "did anything move"; --gate answers "did
    # anything get WORSE", which is the CI question.
    assert cli.main(["diff", old, better]) == 1
    assert cli.main(["diff", old, better, "--gate"]) == 0
    assert cli.main(["diff", old, worse, "--gate"]) == 1
    # --gate must not launder incomparability into a pass.
    assert cli.main(["diff", old, other, "--gate"]) == 2
    capsys.readouterr()


def test_cli_diff_gate_fails_when_long_output_starts_degrading(tmp_path, capsys):
    # A model that used to hold together to the top of the ladder and
    # now loops at 2048 is a REGRESSION the CI gate must fail on —
    # before the rung ordering existed, "degrades-at-2048" was an
    # unknown string, scored neutral, and sailed through --gate.
    def payload(verdict):
        return _diff_payload(verdicts={"long_output": {
            "verdict": verdict, "provisional": True, "lens": {}}})

    ready = _write_profile(tmp_path / "ready.json", payload("ready"))
    at_2048 = _write_profile(tmp_path / "at2048.json",
                             payload("degrades-at-2048"))
    at_4096 = _write_profile(tmp_path / "at4096.json",
                             payload("degrades-at-4096"))
    unusable = _write_profile(tmp_path / "unusable.json", payload("unusable"))

    assert cli.main(["diff", ready, at_2048, "--gate"]) == 1
    assert "regression" in capsys.readouterr().out
    # Degrading LATER is an improvement: it moved, so a bare diff says
    # 1, but the gate must not fail a model that got better.
    assert cli.main(["diff", at_2048, at_4096, "--gate"]) == 0
    assert cli.main(["diff", at_2048, at_4096]) == 1
    assert cli.main(["diff", at_2048, unusable, "--gate"]) == 1
    capsys.readouterr()


def test_cli_diff_json_writes_the_whole_result(tmp_path, capsys):
    old = _write_profile(tmp_path / "old.json", _diff_payload())
    worse = _write_profile(tmp_path / "worse.json",
                           _diff_payload(**_ceiling(4096)))
    out = tmp_path / "diff.json"

    code = cli.main(["diff", old, worse, "--json", str(out)])

    assert code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["comparable"] is True
    assert payload["changes"] == [{
        "family": "ceiling", "cell": "max_verified", "direction": "regression",
        "old": 8192, "new": 4096, "basis": "rung-change",
    }]
    # The clean cells and the identity notes travel with it: a machine
    # reader needs to know what WAS checked, not only what moved.
    assert payload["within_noise"] == ["ceiling.failure_mode"]
    assert payload["dropped"] == []
    assert payload["identity_notes"] == []
    # stdout still carries the human rendering.
    assert "8192 -> 4096" in capsys.readouterr().out


def test_cli_diff_json_records_an_incomparable_pair_too(tmp_path, capsys):
    old = _write_profile(tmp_path / "old.json", _diff_payload())
    other = _write_profile(tmp_path / "other.json",
                           _diff_payload(model={"name": "other-model"}))
    out = tmp_path / "diff.json"

    assert cli.main(["diff", old, other, "--json", str(out)]) == 2

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["comparable"] is False
    assert payload["changes"] == []
    assert any("model.name" in note for note in payload["identity_notes"])
    capsys.readouterr()


def test_cli_diff_unreadable_file_is_infrastructure_not_a_finding(
    tmp_path, capsys
):
    """Exit 4, never 1: a file we could not read has told us nothing
    about the model, and a CI gate reading 1 would report drift that
    was never measured."""
    old = _write_profile(tmp_path / "old.json", _diff_payload())
    missing = str(tmp_path / "absent.json")
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json at all", encoding="utf-8")

    assert cli.main(["diff", old, missing]) == 4
    assert "infrastructure" in capsys.readouterr().err.lower()

    assert cli.main(["diff", old, str(garbage)]) == 4
    err = capsys.readouterr().err
    assert "infrastructure" in err.lower() and "garbage.json" in err

    # ...and with --gate too: the gate must not read a broken file as
    # "no regressions".
    assert cli.main(["diff", old, missing, "--gate"]) == 4
    capsys.readouterr()


def test_cli_diff_rejects_valid_json_that_is_not_a_profile(tmp_path, capsys):
    """Valid JSON is not the bar — a profile document is an object. A
    list parses fine and then blows up inside the comparator, which
    would surface as a traceback instead of exit 4."""
    old = _write_profile(tmp_path / "old.json", _diff_payload())
    listy = tmp_path / "list.json"
    listy.write_text(json.dumps([_diff_payload()]), encoding="utf-8")

    assert cli.main(["diff", old, str(listy)]) == 4
    assert "not a profile document" in capsys.readouterr().err


def test_cli_diff_rejects_an_object_that_is_not_a_profile(tmp_path, capsys):
    """The SILENT half of the same bug, and the worse half.

    ``{}`` is an object, so an isinstance check waves it through; then
    every identity field reads ``None == None``, which the gate treats
    as non-fatal, and every family finds nothing to compare. The result
    is exit 0 on "no drift beyond noise" — a green CI check for a file
    nobody measured. Exit-1-on-unread is loud and gets investigated;
    exit-0-on-unparsed is silent and never does. A profile is
    identified by the key every profile has carried since v1.
    """
    old = _write_profile(tmp_path / "old.json", _diff_payload())
    versionless = {key: value for key, value in _diff_payload().items()
                   if key != "assay_profile_version"}
    for name, payload in (("empty", {}),
                          ("errorish", {"error": "model not found"}),
                          ("versionless", versionless)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        # Against itself (where identity trivially "matches")...
        assert cli.main(["diff", str(path), str(path)]) == 4, name
        assert "assay_profile_version" in capsys.readouterr().err, name
        # ...and as the NEW side of a real pair, gated: the CI shape.
        assert cli.main(["diff", old, str(path), "--gate"]) == 4, name
        assert "not a profile document" in capsys.readouterr().err, name


def test_cli_rejects_a_version_key_with_no_model_behind_it(tmp_path, capsys):
    """The version key alone does not make a document a profile.

    ``{"assay_profile_version": 5}`` satisfies the key check and then
    behaves exactly like the ``{}`` the key check was added to catch:
    the identity gate reads ``None == None`` on the model name and calls
    the pair comparable, no family finds a cell, and a self-diff exits
    0 — a green CI check for a file nobody measured. What a profile
    must actually SAY is which model was measured.
    """
    old = _write_profile(tmp_path / "old.json", _diff_payload())
    for name, payload in (
        ("stub", {"assay_profile_version": 5}),
        ("nameless", {"assay_profile_version": 5, "model": {"quant": "Q8_0"}}),
        ("blank", {"assay_profile_version": 5, "model": {"name": "  "}}),
        ("notadict", {"assay_profile_version": 5, "model": "granite"}),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert cli.main(["diff", str(path), str(path)]) == 4, name
        assert "model.name" in capsys.readouterr().err, name
        assert cli.main(["diff", old, str(path), "--gate"]) == 4, name
        capsys.readouterr()
        out = tmp_path / "report.html"
        assert cli.main(["report", str(path), "--out", str(out)]) == 4, name
        capsys.readouterr()
        assert not out.exists(), name


def test_cli_still_loads_the_committed_evidence_profiles(tmp_path, capsys):
    """The guard must not reject the real thing: every committed profile
    names its model, v1 included."""
    evidence = Path(__file__).resolve().parents[1] / "docs/superpowers/evidence"
    files = sorted(path
                   for folder in ("live", "live-run2", "tier-enthusiast")
                   for path in (evidence / folder).glob("*.json"))
    assert len(files) >= 20
    out = tmp_path / "report.html"
    assert cli.main(["report", *[str(f) for f in files], "--out", str(out)]) == 0
    capsys.readouterr()


def test_cli_diff_reads_the_committed_live_rerun_pair(tmp_path, capsys):
    """End to end on real files: the v1 profiles under
    ``docs/superpowers/evidence`` are same-day same-daemon reruns, so
    the CLI answer is 0 — bare and gated."""
    evidence = Path(__file__).resolve().parents[1] / "docs/superpowers/evidence"
    name = "granite-code-8b-instruct-q8_0-quick.json"
    old = str(evidence / "live" / name)
    new = str(evidence / "live-run2" / name)
    out = tmp_path / "diff.json"

    assert cli.main(["diff", old, new, "--json", str(out)]) == 0
    assert cli.main(["diff", old, new, "--gate"]) == 0
    assert "no drift beyond noise" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["comparable"] is True
