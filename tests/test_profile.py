"""Task 10 tests: profile schema, verdicts, render (spec §8)."""

import dataclasses
import json

import pytest

from assay.ceiling import CallEvidence, Ceiling
from assay.codecs import Landing
from assay.envelope import Envelope
from assay.geometry import Geometry
from assay.profile import Profile, compute_verdicts, render_table

_FAMILIES = ("geometry", "ceiling", "ceiling_shapes", "envelope", "codecs", "speed", "loop")
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


def make_shapes():
    from assay.ceiling import ShapeCeiling
    return (ShapeCeiling(shape=4096, max_verified=3712,
                         failure_mode="ok_to_shape"),)


def make_loop():
    from assay.loop import Loop
    return Loop(action_fidelity=1.0, patch_rate=1.0, finish_rate=1.0,
                repeat_rate=0.0, anchor_violations=0, n_runs=3, n_turns=9)


def make_speed():
    from assay.speed import Speed
    return Speed(decode_tps=16.0, prefill_tps=1024.0,
                 evidence="server_timings", n_decode=1, n_prefill=1)


def make_codecs(
    *, sr_small: float = 0.9, wf_small: float = 0.8, jo_small: float = 0.95
) -> dict[str, dict[str, Landing]]:
    smalls = {"search_replace": sr_small, "whole_file": wf_small, "json_object": jo_small}
    return {
        codec: {
            "tiny": Landing(lands=1.0, lands_applies=1.0, n=5),
            "small": Landing(lands=smalls[codec], lands_applies=smalls[codec], n=5),
            "medium": Landing(lands=0.5, lands_applies=0.5, n=5),
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
        ceiling_shapes=make_shapes(),
        envelope=make_envelope(),
        codecs=make_codecs(),
        speed=make_speed(),
        loop=make_loop(),
        verdicts={
            "structured_extraction": {"verdict": "ready", "lens": {"landing": "test"}},
            "patch_editing": {"verdict": "ready", "lens": {"landing": "test"}},
            "long_context": {"verdict": "risky", "lens": {"landing": "test"}},
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
    assert verdicts["structured_extraction"]["verdict"] == expected


def test_structured_extraction_ready_blocked_by_truncation_below_4k():
    codecs = make_codecs(jo_small=0.95)
    low = make_ceiling(max_verified=1024, first_failure=2048, mode="silent_truncation")
    assert compute_verdicts(None, low, None, codecs)["structured_extraction"]["verdict"] == "risky"
    high = make_ceiling(max_verified=8000, first_failure=8192, mode="silent_truncation")
    assert compute_verdicts(None, high, None, codecs)["structured_extraction"]["verdict"] == "ready"


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
    assert verdicts["patch_editing"]["verdict"] == expected


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
    assert verdicts["long_context"]["verdict"] == expected


def test_unmeasured_inputs_yield_unmeasured_not_unusable():
    verdicts = compute_verdicts(None, None, None, None)
    assert {name: entry["verdict"] for name, entry in verdicts.items()} == {
        "structured_extraction": "unmeasured",
        "patch_editing": "unmeasured",
        "long_context": "unmeasured",
        "loop_discipline": "unmeasured",
        "chat_speed": "unmeasured",
        "agent_speed": "unmeasured",
    }
    # v1.1: every verdict names its lens, even when unmeasured.
    for entry in verdicts.values():
        assert "lens" in entry and entry["lens"]
    # A codec matrix whose cells were never measured is just as None.
    unmeasured_cells = {
        codec: {grade: Landing(lands=None, lands_applies=None, n=0) for grade in _GRADES}
        for codec in _CODECS
    }
    verdicts = compute_verdicts(None, None, None, unmeasured_cells)
    assert verdicts["structured_extraction"]["verdict"] == "unmeasured"
    assert verdicts["patch_editing"]["verdict"] == "unmeasured"


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
        ceiling_shapes=None,
        envelope=None,
        codecs=None,
        speed=None,
        loop=None,
        verdicts={
            "structured_extraction": {"verdict": "unmeasured", "lens": {"landing": "test"}},
            "patch_editing": {"verdict": "unmeasured", "lens": {"landing": "test"}},
            "long_context": {"verdict": "unmeasured", "lens": {"evidence": "unmeasured"}},
        },
        provenance_dropped=tuple(f"{family}: budget exhausted" for family in _FAMILIES),
    )
    assert all(getattr(profile, family) is None for family in _FAMILIES)


# --- render ----------------------------------------------------------------


def test_render_table_names_unmeasured_not_zero():
    bare = make_profile(
        geometry=None,
        ceiling=None,
        ceiling_shapes=None,
        envelope=None,
        codecs=None,
        speed=None,
        loop=None,
        verdicts={
            "structured_extraction": {"verdict": "unmeasured", "lens": {"landing": "test"}},
            "patch_editing": {"verdict": "unmeasured", "lens": {"landing": "test"}},
            "long_context": {"verdict": "unmeasured", "lens": {"evidence": "unmeasured"}},
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


def test_patch_verdict_is_judged_under_the_applies_lens():
    # v1.1: byte-equality says unusable (0.0), applies-and-parses says
    # ready (0.95) — patch_editing must follow the applies lens and SAY
    # so. The 2026-08-12 measurement (same model, 0% vs 100% under two
    # instruments) is why the lens is part of the verdict.
    codecs = {
        codec: {
            grade: Landing(lands=0.0, lands_applies=0.95, n=5)
            for grade in _GRADES
        }
        for codec in _CODECS
    }
    verdicts = compute_verdicts(None, None, None, codecs)
    assert verdicts["patch_editing"]["verdict"] == "ready"
    assert verdicts["patch_editing"]["lens"]["landing"] == "applies_and_parses(python)"


def test_custom_presentation_is_named_in_every_codec_lens():
    verdicts = compute_verdicts(None, None, None, None, presentation="custom")
    assert verdicts["structured_extraction"]["lens"]["presentation"] == "custom"
    assert verdicts["patch_editing"]["lens"]["presentation"] == "custom"


@pytest.mark.parametrize("tps,expected", [
    (8.0, "ready"), (7.99, "risky"), (4.0, "risky"), (3.99, "unusable"),
])
def test_chat_speed_floor_boundaries(tps, expected):
    from assay.speed import Speed
    speed = Speed(decode_tps=tps, prefill_tps=1000.0,
                  evidence="server_timings", n_decode=1, n_prefill=1)
    verdicts = compute_verdicts(None, None, None, None, speed)
    assert verdicts["chat_speed"]["verdict"] == expected
    assert verdicts["chat_speed"]["lens"]["floor_ready"] == 8.0


@pytest.mark.parametrize("tps,expected", [
    (200.0, "ready"), (199.0, "risky"), (80.0, "risky"), (79.0, "unusable"),
])
def test_agent_speed_floor_boundaries(tps, expected):
    from assay.speed import Speed
    speed = Speed(decode_tps=16.0, prefill_tps=tps,
                  evidence="wall_clock_counts", n_decode=1, n_prefill=1)
    verdicts = compute_verdicts(None, None, None, None, speed)
    assert verdicts["agent_speed"]["verdict"] == expected
    assert verdicts["agent_speed"]["lens"]["evidence"] == "wall_clock_counts"


def test_five_of_five_is_ready_but_provisional():
    # Wilson 95% on 5/5 spans ~[0.57, 1.0]: ready and risky are
    # indistinguishable at quick-mode n, and the verdict must SAY so
    # (external review, 2026-08-13).
    codecs = {
        codec: {grade: Landing(lands=1.0, lands_applies=1.0, n=5)
                for grade in _GRADES}
        for codec in _CODECS
    }
    verdicts = compute_verdicts(None, None, None, codecs)
    entry = verdicts["structured_extraction"]
    assert entry["verdict"] == "ready"
    assert entry["provisional"] is True
    lo, hi = entry["interval95"]
    assert 0.55 < lo < 0.58 and hi == 1.0


def test_zero_of_five_is_unusable_and_not_provisional():
    # 0/5 spans ~[0, 0.43], entirely below the risky floor: the rung is
    # decided even at this n, and saying "provisional" would be noise.
    codecs = {
        codec: {grade: Landing(lands=0.0, lands_applies=0.0, n=5)
                for grade in _GRADES}
        for codec in _CODECS
    }
    verdicts = compute_verdicts(None, None, None, codecs)
    entry = verdicts["structured_extraction"]
    assert entry["verdict"] == "unusable"
    assert entry["provisional"] is False


def test_codec_lens_declares_the_fixture_set():
    from assay.fixtures import FIXTURE_SET
    verdicts = compute_verdicts(None, None, None, None)
    assert verdicts["patch_editing"]["lens"]["fixtures"] == FIXTURE_SET
    assert FIXTURE_SET == "codec-fixtures-v2"


def test_codec_lens_names_its_stopping_rule_and_n_used():
    # v1.5: two cells reading 1.00 mean different things when one was
    # sampled 5 times under a fixed n and the other stopped at 35 under
    # a sequential rule. The lens carries both facts or the number is
    # not interpretable.
    verdicts = compute_verdicts(
        None, None, None, make_codecs(),
        stopping_rule="wilson95-looks-5-10-20-35",
        n_used={"structured_extraction": 20, "patch_editing": 35},
    )
    jo = verdicts["structured_extraction"]["lens"]
    patch = verdicts["patch_editing"]["lens"]
    assert jo["stopping_rule"] == "wilson95-looks-5-10-20-35"
    assert jo["n_used"] == 20
    assert patch["stopping_rule"] == "wilson95-looks-5-10-20-35"
    assert patch["n_used"] == 35


def test_codec_lens_stopping_rule_defaults_to_fixed_n():
    verdicts = compute_verdicts(None, None, None, make_codecs())
    assert verdicts["structured_extraction"]["lens"]["stopping_rule"] == "fixed-n"
    assert verdicts["patch_editing"]["lens"]["stopping_rule"] == "fixed-n"


def test_absent_n_used_is_absent_from_the_lens_never_zero():
    # None-vs-zero: a cell nobody measured has no n_used at all.
    verdicts = compute_verdicts(
        None, None, None, make_codecs(),
        n_used={"structured_extraction": 10},
    )
    assert verdicts["structured_extraction"]["lens"]["n_used"] == 10
    assert "n_used" not in verdicts["patch_editing"]["lens"]
    unlensed = compute_verdicts(None, None, None, None)
    assert "n_used" not in unlensed["structured_extraction"]["lens"]
    assert "n_used" not in unlensed["patch_editing"]["lens"]


def test_loop_verdict_downgrades_follow_without_advance():
    # The 14B shape: envelope discipline perfect, nothing ever advances.
    # High fidelity + patch_rate 0 must read risky, never ready.
    from assay.loop import Loop
    loop = Loop(action_fidelity=1.0, patch_rate=0.0, finish_rate=1.0,
                repeat_rate=0.4, anchor_violations=0, n_runs=5, n_turns=15)
    verdicts = compute_verdicts(None, None, None, None, None, loop)
    assert verdicts["loop_discipline"]["verdict"] == "risky"
    assert verdicts["loop_discipline"]["lens"]["instrument"] == "scripted-loop-v1"
