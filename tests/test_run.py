"""Task 11 tests: the probe() orchestrator (plan Task 11, spec §9)."""

import json

import pytest
from fakes import MetadataFreeBackend, ScriptedBackend

from assay import Budget, Profile, probe

_URL = "http://fake-host:11434"
_VRAM_MIB = 14558

# Quick mode against ScriptedBackend, call by call: 2 calibration,
# 5 ladder sizes (1024..16384, one seed, no bisection when clean),
# 10 envelope, 3 codecs x 3 grades x 5 = 45 codec probes.
_QUICK_CALLS_TOTAL = 2 + 5 + 10 + 45
_CALLS_THROUGH_CEILING = 2 + 5


@pytest.fixture(autouse=True)
def _fixed_vram(monkeypatch):
    """Pin the VRAM reading: tests never shell out to nvidia-smi."""
    monkeypatch.setattr("assay.run.free_vram_mib", lambda: _VRAM_MIB)


def test_full_pipeline_produces_complete_profile():
    backend = ScriptedBackend()
    profile = probe(
        _URL,
        "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000),
        mode="quick",
        _backend_override=backend,
    )

    assert isinstance(profile, Profile)
    assert profile.endpoint["base_url"] == _URL
    assert profile.endpoint["autodetected"] is False
    assert profile.model["name"] == "fake-model"
    assert profile.model["training_ctx"] == 32768

    # Geometry: the known qwen-shaped arithmetic, training_ctx binding.
    assert profile.geometry is not None
    assert profile.geometry.kv_kib_per_token == 56
    assert profile.geometry.vram_free_mib == _VRAM_MIB
    assert profile.geometry.limited_by == "training_ctx"

    # Ceiling: clean up to the quick cap, honestly bounded.
    assert profile.ceiling is not None
    assert profile.ceiling.max_verified == 16384
    assert profile.ceiling.first_failure is None
    assert profile.ceiling.failure_mode == "none_up_to_cap"

    # Envelope and codecs fully measured.
    assert profile.envelope is not None
    assert profile.envelope.fidelity == 1.0
    assert profile.envelope.n == 10
    assert profile.codecs is not None
    for codec in ("search_replace", "whole_file", "json_object"):
        for grade in ("tiny", "small", "medium"):
            cell = profile.codecs[codec][grade]
            assert cell.lands == 1.0, (codec, grade)
            assert cell.n == 5

    assert profile.verdicts == {
        "structured_extraction": "ready",
        "patch_editing": "ready",
        "long_context": "ready",
    }

    # Provenance carries the real meter, not a value that merely looks
    # like a measurement.
    spent = profile.provenance["spent"]
    assert spent["calls"] == _QUICK_CALLS_TOTAL == backend.calls
    assert spent["prompt_tokens"] > 0
    assert profile.provenance["mode"] == "quick"
    assert profile.provenance["dropped"] == []
    calibration = profile.provenance["calibration"]
    assert calibration["counts_available"] is True
    assert calibration["deterministic"] is True

    # The produced profile survives its own serialization contract.
    assert Profile.from_json(json.loads(profile.to_json())) == profile


def test_budget_death_mid_pipeline_yields_named_nones():
    # Exactly enough calls for calibration + the quick ladder; the first
    # envelope charge dies. Ceiling completes, envelope measures nothing.
    profile = probe(
        _URL,
        "fake-model",
        budget=Budget(max_calls=_CALLS_THROUGH_CEILING, max_prompt_tokens=100_000),
        mode="quick",
        _backend_override=ScriptedBackend(),
    )

    assert profile.ceiling is not None
    assert profile.ceiling.failure_mode == "none_up_to_cap"
    assert profile.envelope is None
    assert profile.codecs is None

    dropped = profile.provenance["dropped"]
    assert any(entry.startswith("envelope:") for entry in dropped)
    assert any(entry.startswith("codecs:") for entry in dropped)

    # Verdicts partially unmeasured: measured ceiling still speaks.
    assert profile.verdicts["long_context"] == "ready"
    assert profile.verdicts["structured_extraction"] == "unmeasured"
    assert profile.verdicts["patch_editing"] == "unmeasured"

    assert profile.provenance["spent"]["calls"] == _CALLS_THROUGH_CEILING


def test_budget_death_in_ceiling_stops_later_families_from_spending():
    # The token limit dies mid-ladder (after calibration 1000 + sizes
    # 1024 + 2048 = 4072, the 4096 charge is refused) while calls and
    # ~900 tokens of headroom remain. Small envelope/codec calls WOULD
    # still be admitted — the orchestrator must not attempt them.
    backend = ScriptedBackend()
    profile = probe(
        _URL,
        "fake-model",
        budget=Budget(max_calls=100, max_prompt_tokens=5000),
        mode="quick",
        _backend_override=backend,
    )

    # The partial ceiling reports what it verified, honestly bounded.
    assert profile.ceiling is not None
    assert profile.ceiling.max_verified == 2048
    assert profile.ceiling.first_failure is None
    assert profile.ceiling.failure_mode == "budget"

    # No later family ran or spent anything: 2 calibration + 2 ladder.
    assert backend.calls == 4
    assert profile.provenance["spent"]["calls"] == 4
    assert profile.envelope is None
    assert profile.codecs is None
    dropped = profile.provenance["dropped"]
    assert "envelope: skipped, budget exhausted earlier" in dropped
    assert "codecs: skipped, budget exhausted earlier" in dropped


def test_budget_death_mid_codecs_names_unmeasured_cells_in_dropped():
    # 17 calls reach codecs (2 calibration + 5 ladder + 10 envelope);
    # 20 codec calls complete 4 cells, then the meter dies. Spec §8:
    # every UNMEASURED cell (Landing(None, 0)) must be NAMED in
    # dropped — a None measurement with dropped == [] tells a consumer
    # nothing was dropped.
    profile = probe(
        _URL,
        "fake-model",
        budget=Budget(max_calls=_CALLS_THROUGH_CEILING + 10 + 20, max_prompt_tokens=1_000_000),
        mode="quick",
        _backend_override=ScriptedBackend(),
    )

    assert profile.codecs is not None
    cells = {
        (codec, grade): cell
        for codec, grades in profile.codecs.items()
        for grade, cell in grades.items()
    }
    unmeasured = sorted(key for key, cell in cells.items() if cell.n == 0)
    assert unmeasured, "the budget must really have died mid-matrix"
    dropped = profile.provenance["dropped"]
    for codec, grade in unmeasured:
        assert (
            f"codecs: {codec}.{grade} budget exhausted before any probe completed"
            in dropped
        )
    # Measured cells (n > 0) are measurements, never named as dropped.
    for (codec, grade), cell in cells.items():
        if cell.n > 0:
            assert not any(f"{codec}.{grade}" in entry for entry in dropped)


def test_ceiling_without_per_request_ctx_is_stated_in_dropped():
    # A backend that cannot widen num_ctx per request (openai_compat
    # shape): the ceiling still measures, but the profile must state
    # that the ladder ran in the server's configured window.
    profile = probe(
        _URL,
        "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=1_000_000),
        mode="quick",
        _backend_override=MetadataFreeBackend(),
    )

    assert profile.ceiling is not None
    assert any(
        entry.startswith("ceiling: per_request_ctx unavailable")
        for entry in profile.provenance["dropped"]
    )


def test_budget_is_required():
    with pytest.raises(TypeError):
        probe(_URL, "fake-model", _backend_override=ScriptedBackend())
