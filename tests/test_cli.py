"""Task 11 tests: the assay CLI (plan Task 11, spec §9).

Exit codes: 0 = profile produced (whatever it says); 2 = budget
exhausted before ANY family completed; 4 = infrastructure failure
before any measurement. No test here touches a real socket: the
backend factory and the VRAM reader are always replaced.
"""

import json

import pytest
from fakes import MetadataFreeBackend, ScriptedBackend, UnreachableBackend

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
    assert payload["envelope"]["n"] == 10
