"""Task 11 tests: the assay CLI (plan Task 11, spec §9).

Exit codes: 0 = profile produced (whatever it says); 2 = budget
exhausted before ANY family completed; 4 = infrastructure failure
before any measurement. No test here touches a real socket: the
backend factory and the VRAM reader are always replaced.
"""

import json

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
    assert payload["assay_profile_version"] == 4
    assert payload["ceiling"]["failure_mode"] == "none_up_to_cap"
    assert payload["verdicts"]["long_context"]["verdict"] == "ready"
    # The documented quick default budget was applied...
    assert payload["provenance"]["budget"] == {
        "max_calls": 110,
        "max_prompt_tokens": 200_000,
    }
    # ...and it COVERS the whole quick suite (2 calibration + 5 ladder +
    # 10 envelope + 45 codecs + 2 speed = 64): the run must never exhaust the
    # default budget on a well-behaved endpoint (spec §12 criterion 1).
    assert payload["provenance"]["spent"]["calls"] == 82
    for codec in ("search_replace", "whole_file", "json_object"):
        for grade in ("tiny", "small", "medium"):
            assert payload["codecs"][codec][grade]["n"] == 5, (codec, grade)
    assert payload["provenance"]["dropped"] == []

    # Human table on stdout; recording actually wrote a transcript.
    assert "assay profile" in capsys.readouterr().out
    rows = record.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 82
    assert json.loads(rows[0])["outcome"] == "reply"


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
    assert cli.DEFAULT_BUDGETS["full"].max_calls == 500
    assert cli.DEFAULT_BUDGETS["full"].max_prompt_tokens == 1_000_000
    assert cli.DEFAULT_BUDGETS["full"] == cli.DEFAULT_BUDGETS["thorough"]


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
