"""Task 10 tests: profile schema, verdicts, render (spec §8)."""

import dataclasses
import json

import pytest

from assay.ceiling import CallEvidence, Ceiling
from assay.codecs import Landing
from assay.envelope import Envelope
from assay.geometry import Geometry
from assay.profile import Profile, compute_verdicts, render_table

_FAMILIES = ("geometry", "ceiling", "envelope", "codecs")
_GRADES = ("tiny", "small", "medium")
_CODECS = ("search_replace", "whole_file", "json_object")


def make_geometry() -> Geometry:
    return Geometry(
        kv_kib_per_token=56,
        vram_free_mib=14558,
        usable_window=32768,
        limited_by="training_ctx",
        source="api_show",
    )


def make_ceiling(
    *,
    max_verified: int | None = 11500,
    first_failure: int | None = 11800,
    mode: str = "missing_stats",
) -> Ceiling:
    return Ceiling(
        max_verified=max_verified,
        first_failure=first_failure,
        failure_mode=mode,
        counts_available=True,
        evidence=(
            CallEvidence(est_tokens=1024, seed=0, signal="ok", detail="tokens_in=1010"),
            CallEvidence(
                est_tokens=11800,
                seed=0,
                signal="missing_stats",
                detail="ContractViolation: no counts",
            ),
        ),
    )


def honest_ceiling() -> Ceiling:
    return make_ceiling(max_verified=16384, first_failure=None, mode="none_up_to_cap")


def make_envelope() -> Envelope:
    return Envelope(fidelity=0.97, n=30, failures={"prose": 1, "shape": 0, "refusal": 0})


def make_codecs(
    *, sr_small: float = 0.9, wf_small: float = 0.8, jo_small: float = 0.95
) -> dict[str, dict[str, Landing]]:
    smalls = {"search_replace": sr_small, "whole_file": wf_small, "json_object": jo_small}
    return {
        codec: {
            "tiny": Landing(lands=1.0, n=5),
            "small": Landing(lands=smalls[codec], n=5),
            "medium": Landing(lands=0.5, n=5),
        }
        for codec in _CODECS
    }


def make_profile(*, provenance_dropped: tuple[str, ...] = (), **overrides) -> Profile:
    kwargs = dict(
        assay_profile_version=1,
        probe_version="0.1.0",
        endpoint={
            "kind": "ollama",
            "base_url": "http://gpu-box:11434",
            "autodetected": True,
        },
        model={
            "name": "qwen2.5-coder:7b-instruct-q8_0",
            "quant": "q8_0",
            "weights_bytes": 8100000000,
            "training_ctx": 32768,
        },
        geometry=make_geometry(),
        ceiling=make_ceiling(),
        envelope=make_envelope(),
        codecs=make_codecs(),
        verdicts={
            "structured_extraction": "ready",
            "patch_editing": "ready",
            "long_context": "risky",
        },
        provenance={
            "started": "2026-08-12T21:00:00Z",
            "finished": "2026-08-12T21:02:00Z",
            "mode": "quick",
            "seeds": [0, 1],
            "budget": {"max_calls": 60, "max_prompt_tokens": 120000},
            "spent": {"calls": 43, "prompt_tokens": 91000},
            "calibration": {
                "chars_per_token": 5.9,
                "counts_available": True,
                "deterministic": True,
            },
            "dropped": list(provenance_dropped),
        },
    )
    kwargs.update(overrides)
    return Profile(**kwargs)


# --- serialization ---------------------------------------------------------


def test_round_trips_through_json_with_dataclass_equality():
    profile = make_profile()
    restored = Profile.from_json(json.loads(profile.to_json()))
    assert restored == profile
    # The nested dataclass TYPES are restored, not just equal shapes.
    assert isinstance(restored.geometry, Geometry)
    assert isinstance(restored.ceiling, Ceiling)
    assert isinstance(restored.ceiling.evidence[0], CallEvidence)
    assert isinstance(restored.envelope, Envelope)
    assert isinstance(restored.codecs["json_object"]["small"], Landing)


def test_every_profile_field_is_wired_into_the_payload():
    payload = json.loads(make_profile().to_json())
    field_names = {field.name for field in dataclasses.fields(Profile)}
    assert set(payload) == field_names


# --- verdicts --------------------------------------------------------------


@pytest.mark.parametrize(
    ("jo_small", "expected"),
    [
        (0.91, "ready"),
        (0.90, "ready"),
        (0.89, "risky"),
        (0.61, "risky"),
        (0.60, "risky"),
        (0.59, "unusable"),
    ],
)
def test_verdict_boundaries_structured_extraction(jo_small, expected):
    codecs = make_codecs(jo_small=jo_small)
    verdicts = compute_verdicts(None, honest_ceiling(), None, codecs)
    assert verdicts["structured_extraction"] == expected


def test_structured_extraction_ready_blocked_by_truncation_below_4k():
    codecs = make_codecs(jo_small=0.95)
    low = make_ceiling(max_verified=1024, first_failure=2048, mode="silent_truncation")
    assert compute_verdicts(None, low, None, codecs)["structured_extraction"] == "risky"
    high = make_ceiling(max_verified=8000, first_failure=8192, mode="silent_truncation")
    assert compute_verdicts(None, high, None, codecs)["structured_extraction"] == "ready"


@pytest.mark.parametrize(
    ("sr_small", "wf_small", "expected"),
    [
        (0.90, 0.30, "ready"),
        (0.30, 0.90, "ready"),  # max() semantics: either codec can carry it
        (0.89, 0.30, "risky"),
        (0.10, 0.60, "risky"),
        (0.59, 0.10, "unusable"),
    ],
)
def test_verdict_boundaries_patch_editing(sr_small, wf_small, expected):
    codecs = make_codecs(sr_small=sr_small, wf_small=wf_small)
    verdicts = compute_verdicts(None, honest_ceiling(), None, codecs)
    assert verdicts["patch_editing"] == expected


@pytest.mark.parametrize(
    ("max_verified", "first_failure", "mode", "expected"),
    [
        (16384, 17000, "hard_error", "ready"),
        (16383, 17000, "hard_error", "unmeasured"),
        (16385, None, "none_up_to_cap", "ready"),
        (20000, 21000, "missing_stats", "risky"),  # a lying mode is never "ready"
        (11500, 11800, "silent_truncation", "risky"),
        (20000, 21000, "canary_loss", "unmeasured"),
    ],
)
def test_verdict_boundaries_long_context(max_verified, first_failure, mode, expected):
    ceiling = make_ceiling(
        max_verified=max_verified, first_failure=first_failure, mode=mode
    )
    verdicts = compute_verdicts(None, ceiling, None, None)
    assert verdicts["long_context"] == expected


def test_unmeasured_inputs_yield_unmeasured_not_unusable():
    verdicts = compute_verdicts(None, None, None, None)
    assert verdicts == {
        "structured_extraction": "unmeasured",
        "patch_editing": "unmeasured",
        "long_context": "unmeasured",
    }
    # A codec matrix whose cells were never measured is just as None.
    unmeasured_cells = {
        codec: {grade: Landing(lands=None, n=0) for grade in _GRADES}
        for codec in _CODECS
    }
    verdicts = compute_verdicts(None, None, None, unmeasured_cells)
    assert verdicts["structured_extraction"] == "unmeasured"
    assert verdicts["patch_editing"] == "unmeasured"


# --- dropped naming --------------------------------------------------------


@pytest.mark.parametrize("family", _FAMILIES)
def test_dropped_names_every_none_family(family):
    with pytest.raises(ValueError, match=family):
        make_profile(**{family: None})  # dropped stays empty: must refuse
    named = make_profile(
        **{family: None},
        provenance_dropped=(f"{family}: budget exhausted",),
    )
    assert getattr(named, family) is None


def test_all_none_families_construct_when_all_named():
    profile = make_profile(
        geometry=None,
        ceiling=None,
        envelope=None,
        codecs=None,
        verdicts={
            "structured_extraction": "unmeasured",
            "patch_editing": "unmeasured",
            "long_context": "unmeasured",
        },
        provenance_dropped=tuple(f"{family}: budget exhausted" for family in _FAMILIES),
    )
    assert all(getattr(profile, family) is None for family in _FAMILIES)


# --- render ----------------------------------------------------------------


def test_render_table_names_unmeasured_not_zero():
    bare = make_profile(
        geometry=None,
        ceiling=None,
        envelope=None,
        codecs=None,
        verdicts={
            "structured_extraction": "unmeasured",
            "patch_editing": "unmeasured",
            "long_context": "unmeasured",
        },
        provenance_dropped=tuple(f"{family}: budget exhausted" for family in _FAMILIES),
    )
    rendered = render_table(bare)
    assert "unmeasured" in rendered
    # None is never rendered as a number that looks like a measurement.
    assert "0.0" not in rendered
    assert "None" not in rendered
    # Measured values DO render.
    assert "0.97" in render_table(make_profile())
