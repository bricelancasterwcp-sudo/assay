"""Task 10 tests: profile schema, verdicts, render (spec §8)."""

import dataclasses
import json
from pathlib import Path

import pytest

from assay.ceiling import CallEvidence, Ceiling
from assay.codecs import Landing
from assay.envelope import Envelope
from assay.geometry import Geometry
from assay.long_output import (DISTINCT_FLOOR, LONG_OUTPUT_TASK,
                               THRESHOLDS_PROVENANCE, ZLIB_FLOOR, LongOutput,
                               LongRung)
from assay.profile import (PROFILE_VERSION, Profile, compute_verdicts,
                           render_table)

_FAMILIES = ("geometry", "ceiling", "ceiling_shapes", "envelope", "codecs",
             "speed", "loop", "long_output")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_V1_PROFILE = (_REPO_ROOT / "docs/superpowers/evidence/live"
               / "qwen2.5-coder-7b-instruct-q8_0-quick.json")
_GRADES = ("tiny", "small", "medium")
_CODECS = ("search_replace", "whole_file", "json_object")


def make_geometry(**overrides) -> Geometry:
    """The canonical geometry: a DENSE model, so no expert metadata."""
    fields = dict(
        kv_kib_per_token=56,
        vram_free_mib=14558,
        usable_window=32768,
        limited_by="training_ctx",
        source="api_show",
    )
    fields.update(overrides)
    return Geometry(**fields)


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
    # Two decode samples whose mean is the reported rate: the round-trip
    # test then covers the tuple coercion, not just the scalars.
    return Speed(decode_tps=16.0, prefill_tps=1024.0,
                 evidence="server_timings", n_decode=2, n_prefill=1,
                 decode_samples=(15.0, 17.0), prefill_samples=(1024.0,))


def make_long_output(
    *, degenerate_from: int | None = None,
    targets: tuple[int, ...] = (512, 1024, 2048),
    skipped: tuple[str, ...] = ("4096: above measured ceiling",),
) -> LongOutput:
    """A measured ladder; ``degenerate_from`` is the first bad rung."""
    rungs = tuple(
        LongRung(
            target_tokens=target,
            generated_tokens=target - 8,
            distinct_ratio=0.04 if _bad(target, degenerate_from) else 0.94,
            zlib_ratio=0.03 if _bad(target, degenerate_from) else 0.41,
            degenerate=_bad(target, degenerate_from),
        )
        for target in targets
    )
    return LongOutput(rungs=rungs, skipped=skipped)


def _bad(target: int, degenerate_from: int | None) -> bool:
    return degenerate_from is not None and target >= degenerate_from


def unscorable_rung(target: int) -> LongRung:
    """A rung that spent a call and measured nothing: every metric None."""
    return LongRung(target_tokens=target, generated_tokens=3,
                    distinct_ratio=None, zlib_ratio=None, degenerate=None)


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
        long_output=make_long_output(),
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


def test_speed_samples_survive_the_round_trip_as_tuples():
    # asdict() writes tuples as JSON arrays; the parser must put them
    # back as tuples or the frozen dataclass no longer compares equal.
    restored = Profile.from_json(json.loads(make_profile().to_json()))
    assert restored.speed.decode_samples == (15.0, 17.0)
    assert isinstance(restored.speed.decode_samples, tuple)
    assert restored.speed.prefill_samples == (1024.0,)
    assert isinstance(restored.speed.prefill_samples, tuple)


def test_speed_payload_predating_samples_parses_as_none():
    # A profile written before v1.5 has no samples keys. It reloads with
    # None — "not recorded" — never with an empty tuple, which would
    # claim a sampling run that never happened.
    from assay.profile import _speed_from
    speed = _speed_from({"decode_tps": 66.0, "prefill_tps": 3765.0,
                         "evidence": "server_timings",
                         "n_decode": 1, "n_prefill": 1})
    assert speed.decode_samples is None
    assert speed.prefill_samples is None


def test_empty_speed_samples_parse_as_an_empty_TUPLE_not_a_list():
    # This payload is reachable: a probe whose budget dies between the
    # decode calls and the prefill call is KEPT (run.py only drops a
    # speed cell when both n are zero) and writes "prefill_samples": [].
    # A falsy-check in the parser would leave that as a list inside a
    # frozen dataclass, and the profile would stop comparing equal to
    # itself across a round trip — the exact bug the coercion prevents.
    from assay.profile import _speed_from
    speed = _speed_from({"decode_tps": 16.0, "prefill_tps": None,
                         "evidence": "server_timings",
                         "n_decode": 1, "n_prefill": 0,
                         "decode_samples": [16.0], "prefill_samples": []})
    assert speed.prefill_samples == ()
    assert isinstance(speed.prefill_samples, tuple)
    assert speed.decode_samples == (16.0,)


def test_speed_samples_explicitly_null_stay_none():
    # An unmeasured probe serialized with nulls must not be coerced into
    # tuple(None) — the parser has to guard the None case.
    from assay.profile import _speed_from
    speed = _speed_from({"decode_tps": None, "prefill_tps": None,
                         "evidence": "unmeasured",
                         "n_decode": 0, "n_prefill": 0,
                         "decode_samples": None, "prefill_samples": None})
    assert speed.decode_samples is None
    assert speed.prefill_samples is None


def test_every_profile_field_is_wired_into_the_payload():
    payload = json.loads(make_profile().to_json())
    field_names = {field.name for field in dataclasses.fields(Profile)}
    assert set(payload) == field_names


# --- v1.6: MoE metadata rides in the geometry family -----------------------


def test_every_geometry_field_is_wired_into_the_payload():
    # Same rule as the rungs: consumers read the ARTIFACT. A field that
    # exists on Geometry and not in the JSON is a measurement nobody
    # downstream can see.
    payload = json.loads(make_profile().to_json())
    assert set(payload["geometry"]) == {
        field.name for field in dataclasses.fields(Geometry)}
    # The canonical profile is a dense model: the expert cells are
    # written as null, never as 0.
    assert payload["geometry"]["expert_count"] is None
    assert payload["geometry"]["expert_used_count"] is None


def test_geometry_round_trips_the_expert_fields():
    profile = make_profile(
        geometry=make_geometry(expert_count=128, expert_used_count=8))

    restored = Profile.from_json(json.loads(profile.to_json()))

    assert restored == profile
    assert restored.geometry.expert_count == 128
    assert restored.geometry.expert_used_count == 8


def test_geometry_payload_predating_the_expert_fields_parses_as_none():
    # Every profile this project has ever written must still parse: a
    # pre-v1.6 geometry has no expert keys at all, and that reloads as
    # None ("not recorded"), never as a routing claim.
    from assay.profile import _geometry_from

    geometry = _geometry_from({"kv_kib_per_token": 56, "vram_free_mib": 14558,
                               "usable_window": 32768,
                               "limited_by": "training_ctx",
                               "source": "api_show"})

    assert geometry.expert_count is None
    assert geometry.expert_used_count is None


def test_render_marks_a_moe_model_with_used_of_count():
    profile = make_profile(
        geometry=make_geometry(expert_count=128, expert_used_count=8))

    assert "MoE 8-of-128" in render_table(profile)


def test_render_omits_the_moe_marker_for_a_dense_model():
    assert "MoE" not in render_table(make_profile())


@pytest.mark.parametrize("half", [
    pytest.param({"expert_count": 128}, id="count_only"),
    pytest.param({"expert_used_count": 8}, id="used_only"),
])
def test_render_omits_the_moe_marker_when_only_one_count_is_measured(half):
    # One-sided metadata is not an MoE fact. "MoE 8-of-None" would print
    # an unmeasured half as though it had been measured.
    profile = make_profile(geometry=make_geometry(**half))

    assert "MoE" not in render_table(profile)


# --- schema v5: the long_output family -------------------------------------


def test_schema_version_and_package_version_move_together():
    # The schema and the distribution version are one release, not two:
    # a profile that says v5 must have been written by a 0.6.0 probe.
    import assay

    assert PROFILE_VERSION == 5
    assert assay.__version__ == "0.6.0"
    assert 'version = "0.6.0"' in (
        _REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # The README states the schema version to a reader who will never
    # open profile.py. It sat two versions stale through a green suite
    # (it said 3 while PROFILE_VERSION was 4) because nothing pinned it.
    assert f"assay_profile_version: {PROFILE_VERSION}" in (
        _REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_long_output_round_trips_as_tuples_of_rungs():
    restored = Profile.from_json(json.loads(make_profile().to_json()))
    assert restored == make_profile()
    assert isinstance(restored.long_output, LongOutput)
    assert isinstance(restored.long_output.rungs, tuple)
    assert isinstance(restored.long_output.rungs[0], LongRung)
    assert isinstance(restored.long_output.skipped, tuple)
    assert restored.long_output.skipped == ("4096: above measured ceiling",)


def test_every_long_rung_field_is_wired_into_the_payload():
    # The robigo per_record lesson: consumers read the ARTIFACT, not the
    # in-memory object. A field that exists on LongRung and not in the
    # JSON is a measurement nobody downstream can see.
    payload = json.loads(make_profile().to_json())
    rung = payload["long_output"]["rungs"][0]
    assert set(rung) == {field.name for field in dataclasses.fields(LongRung)}
    assert set(payload["long_output"]) == {
        field.name for field in dataclasses.fields(LongOutput)}


@pytest.mark.parametrize(("degenerate_from", "expected"), [
    (None, "ready"),
    (512, "unusable"),      # the first rung already loops
    (1024, "degrades-at-1024"),
    (2048, "degrades-at-2048"),
])
def test_long_output_verdict_ladder(degenerate_from, expected):
    long_output = make_long_output(degenerate_from=degenerate_from)
    verdicts = compute_verdicts(None, None, None, None, None, None, long_output)
    assert verdicts["long_output"]["verdict"] == expected


def full_ladder(*, degenerate_from: int | None = None) -> LongOutput:
    """Every configured rung attempted and scored, nothing skipped."""
    return make_long_output(degenerate_from=degenerate_from,
                            targets=(512, 1024, 2048, 4096), skipped=())


def test_a_ladder_scored_end_to_end_is_not_provisional():
    # UPDATED BY TASK 12, deliberately, twice over. The old cap forced
    # True while both floors were guesses; the anchor capture derived
    # ZLIB_FLOOR and released it. What decides the flag now (ruled
    # 2026-08-15) is LADDER COMPLETENESS — and this ladder climbed every
    # rung it was configured to climb and scored all four, so there is
    # nothing left for a finished run to revise.
    assert not THRESHOLDS_PROVENANCE.startswith("assumed")
    for degenerate_from in (None, 512, 2048):
        entry = compute_verdicts(
            None, None, None, None, None, None,
            full_ladder(degenerate_from=degenerate_from))["long_output"]
        assert entry["provisional"] is False, degenerate_from


def test_a_ladder_the_ceiling_or_budget_cut_short_stays_provisional():
    # Ruled 2026-08-15. "ready" through 2048 because the ceiling stopped
    # the ladder there is NOT the finding "ready, verified to 4096", and
    # the badge must not read the same. The lens carries the extent for a
    # reader who looks; this flag is for the one who does not.
    for reason in ("4096: above measured ceiling", "4096: budget exhausted"):
        for degenerate_from in (None, 512, 2048):
            entry = compute_verdicts(
                None, None, None, None, None, None,
                make_long_output(degenerate_from=degenerate_from,
                                 skipped=(reason,)))["long_output"]
            assert entry["provisional"] is True, (reason, degenerate_from)
            # The verdict itself is unaffected — completeness is a
            # confidence claim, not a grade.
            assert entry["verdict"] != "unmeasured"


def test_an_unscorable_rung_leaves_the_ladder_unfinished_too():
    # A rung that spent a call and measured nothing did not climb: the
    # ladder has a hole in it even though nothing was skipped.
    ladder = LongOutput(
        rungs=full_ladder().rungs[:3] + (unscorable_rung(4096),), skipped=())
    entry = compute_verdicts(None, None, None, None, None, None,
                             ladder)["long_output"]
    assert entry["verdict"] == "ready"
    assert entry["provisional"] is True
    assert entry["lens"]["rungs_scored"] == 3
    assert entry["lens"]["deepest_scored_tokens"] == 2048


def test_the_forced_provisional_cap_still_works_if_a_floor_goes_back_to_assumed():
    # The cap was not deleted, only released: DISTINCT_FLOOR is still
    # assumed, and if a future change makes the provenance say so at the
    # front of the string, every measured verdict must go provisional
    # again without anyone re-implementing the rule. The ladder here is
    # COMPLETE on purpose — otherwise the flag would come back True from
    # the completeness rule and this test would pin nothing.
    import assay.profile as profile_module

    for degenerate_from in (None, 512, 2048):
        long_output = full_ladder(degenerate_from=degenerate_from)
        baseline = compute_verdicts(None, None, None, None, None, None,
                                    long_output)["long_output"]
        assert baseline["provisional"] is False  # the cap is what moves it
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(profile_module, "THRESHOLDS_PROVENANCE",
                          "assumed-not-derived-2099-01-01")
            entry = compute_verdicts(None, None, None, None, None, None,
                                     long_output)["long_output"]
        assert entry["provisional"] is True, degenerate_from
        assert entry["lens"]["thresholds"] == "assumed-not-derived-2099-01-01"


def test_long_output_lens_names_floors_task_and_threshold_provenance():
    entry = compute_verdicts(None, None, None, None, None, None,
                             make_long_output())["long_output"]
    assert entry["lens"] == {
        "metrics": "distinct4gram+zlib",
        "distinct_floor": DISTINCT_FLOOR,
        "zlib_floor": ZLIB_FLOOR,
        "thresholds": THRESHOLDS_PROVENANCE,
        "task": LONG_OUTPUT_TASK,
        "temperature": 0.2,
        "rungs_scored": 3,
        "deepest_scored_tokens": 2048,
    }


def test_the_lens_says_how_far_the_ladder_actually_got():
    # Ruled 2026-08-15: "ready" alone has no extent. A model whose
    # ceiling stopped the ladder at 1024 and a model verified clean to
    # 4096 both read "ready", and report.py/diff.py consume the verdict
    # entry, not the rendered table — so the extent must live in the
    # lens or it does not exist for them.
    shallow = compute_verdicts(
        None, None, None, None, None, None,
        make_long_output(targets=(512, 1024)))["long_output"]["lens"]
    assert shallow["rungs_scored"] == 2
    assert shallow["deepest_scored_tokens"] == 1024

    deep = compute_verdicts(
        None, None, None, None, None, None,
        make_long_output(targets=(512, 1024, 2048, 4096)))["long_output"]["lens"]
    assert deep["rungs_scored"] == 4
    assert deep["deepest_scored_tokens"] == 4096


def test_the_lens_counts_only_scored_rungs():
    # A rung that measured nothing adds no extent: it neither counts
    # nor deepens the reach the verdict claims.
    healthy = make_long_output(targets=(512,)).rungs[0]
    ladder = LongOutput(rungs=(healthy, unscorable_rung(4096)), skipped=())
    lens = compute_verdicts(None, None, None, None, None, None,
                            ladder)["long_output"]["lens"]
    assert lens["rungs_scored"] == 1
    assert lens["deepest_scored_tokens"] == 512


def test_an_unmeasured_lens_keeps_the_same_shape():
    # Same keys either way — a consumer reads one lens shape, and the
    # unmeasured case says 0 rungs and a None depth rather than going
    # silent on the question.
    measured = compute_verdicts(None, None, None, None, None, None,
                                make_long_output())["long_output"]["lens"]
    for long_output in (None, LongOutput(rungs=(), skipped=()),
                        LongOutput(rungs=(unscorable_rung(512),), skipped=())):
        lens = compute_verdicts(None, None, None, None, None, None,
                                long_output)["long_output"]["lens"]
        assert set(lens) == set(measured)
        assert lens["rungs_scored"] == 0
        assert lens["deepest_scored_tokens"] is None


def test_unmeasured_long_output_is_unmeasured_and_not_provisional():
    for long_output in (
        None,
        LongOutput(rungs=(), skipped=("512: budget exhausted",)),
    ):
        entry = compute_verdicts(None, None, None, None, None, None,
                                 long_output)["long_output"]
        assert entry["verdict"] == "unmeasured"
        assert entry["provisional"] is False
        assert entry["lens"]["thresholds"] == THRESHOLDS_PROVENANCE


def test_a_ladder_of_unscorable_rungs_measured_nothing():
    # Ruled 2026-08-14: an all-None rung is NOT a measurement and NOT a
    # skipped rung — it spent a call and scored nothing. A ladder whose
    # every attempted rung came back unscorable must read "unmeasured",
    # never "ready": no rung was found healthy, they were merely not
    # found degenerate, which is a different (and empty) claim.
    ladder = LongOutput(rungs=tuple(unscorable_rung(t)
                                    for t in (512, 1024, 2048)),
                        skipped=())
    entry = compute_verdicts(None, None, None, None, None, None,
                             ladder)["long_output"]
    assert entry["verdict"] == "unmeasured"
    assert entry["provisional"] is False


def test_unscorable_rungs_never_stand_in_for_healthy_ones():
    # A rung that measured nothing cannot certify the rungs below it.
    # 512 unscorable + 1024 degenerate is "unusable" — the smallest rung
    # this ladder actually measured was already looping — not
    # "degrades-at-1024", which would claim a healthy 512 nobody saw.
    ladder = LongOutput(
        rungs=(unscorable_rung(512),
               make_long_output(degenerate_from=1024,
                                targets=(1024,)).rungs[0]),
        skipped=())
    entry = compute_verdicts(None, None, None, None, None, None,
                             ladder)["long_output"]
    assert entry["verdict"] == "unusable"

    # ...and a measured-healthy rung below a measured-degenerate one
    # still names the rung it degraded at, unscorable rungs between them
    # notwithstanding.
    healthy = make_long_output(targets=(512,)).rungs[0]
    bad = make_long_output(degenerate_from=2048, targets=(2048,)).rungs[0]
    mixed = LongOutput(rungs=(healthy, unscorable_rung(1024), bad),
                       skipped=())
    entry = compute_verdicts(None, None, None, None, None, None,
                             mixed)["long_output"]
    assert entry["verdict"] == "degrades-at-2048"


# --- v1 back-compat (spec §4) ----------------------------------------------


def test_a_committed_v1_profile_still_parses():
    # Post-v1 families are ABSENT from a v1 document; the parser must
    # read one without exploding, and must NAME every family the old
    # schema could not carry (the None-vs-zero rule holds across the
    # version boundary too — an unnamed None is a silent gap).
    payload = json.loads(_V1_PROFILE.read_text(encoding="utf-8"))
    assert payload["assay_profile_version"] == 1
    profile = Profile.from_json(payload)

    assert profile.ceiling is not None  # a v1 family really parsed
    assert profile.geometry is not None
    for family in ("ceiling_shapes", "speed", "loop", "long_output"):
        assert getattr(profile, family) is None
        assert any(entry.startswith(f"{family}:")
                   for entry in profile.provenance["dropped"]), family
    # The parsed payload is not mutated by the upgrade note.
    assert payload["provenance"]["dropped"] == []


def test_a_v1_codec_cell_has_no_applies_lens_and_says_so():
    # v1 measured byte-equality only. Parsing one back must leave
    # lands_applies None — unmeasured under that lens — never a copy of
    # `lands`, which would fabricate a measurement that never ran.
    payload = json.loads(_V1_PROFILE.read_text(encoding="utf-8"))
    cell = Profile.from_json(payload).codecs["search_replace"]["tiny"]
    assert cell.n > 0
    assert cell.lands is not None
    assert cell.lands_applies is None


def test_a_present_but_null_family_still_has_to_be_named():
    # The back-compat .get() must not weaken the guard: a MODERN profile
    # that writes "speed": null with an empty dropped list is still a
    # silent None and still refused.
    payload = json.loads(make_profile().to_json())
    payload["speed"] = None
    with pytest.raises(ValueError, match="speed"):
        Profile.from_json(payload)


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
        "long_output": "unmeasured",
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
        long_output=None,
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
        long_output=None,
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


def test_render_table_carries_a_long_output_line():
    bare = make_profile(
        long_output=None,
        provenance_dropped=("long_output: budget exhausted",))
    assert "long_output unmeasured" in render_table(bare)

    healthy = render_table(make_profile())
    assert "long_output" in healthy
    assert "2048" in healthy  # the deepest rung it held together at

    degraded = render_table(
        make_profile(long_output=make_long_output(degenerate_from=1024)))
    assert "long_output" in degraded
    assert "1024" in degraded  # the rung it went degenerate at is NAMED


def test_render_table_says_unmeasured_for_a_ladder_that_scored_nothing():
    # Calls were spent, rungs exist, nothing was measured: the human
    # view must not show that as a clean ladder.
    nothing = make_profile(
        long_output=LongOutput(rungs=(unscorable_rung(512),), skipped=()))
    assert "long_output unmeasured" in render_table(nothing)


def test_the_lens_line_says_unmeasured_where_the_lens_holds_none():
    """``render_table``'s one rule, applied to the last line that broke it.

    A ladder the ceiling capped scores nothing, so its lens carries
    ``deepest_scored_tokens: None`` — and the lens line interpolated that
    straight, printing ``deepest_scored_tokens=None`` beside a column of
    fields that all say "unmeasured". A reader has to know Python to read
    the difference, and there isn't one.
    """
    capped = make_profile(
        long_output=LongOutput(rungs=(unscorable_rung(512),), skipped=()),
        verdicts=compute_verdicts(None, None, None, None, long_output=None),
    )
    rendered = render_table(capped)
    assert "deepest_scored_tokens=unmeasured" in rendered
    assert "None" not in rendered


def test_render_table_reads_a_v1_profile_without_raising():
    """v1 wrote verdicts as BARE STRINGS. ``render_table`` indexed them
    as dicts, so the one human view of an archived profile raised
    TypeError — on real committed evidence, not a hypothetical. A string
    carries no lens; it renders as the word it is and the lens line says
    so rather than inventing one."""
    payload = json.loads(_V1_PROFILE.read_text(encoding="utf-8"))
    assert isinstance(payload["verdicts"]["structured_extraction"], str)

    rendered = render_table(Profile.from_json(payload))

    assert "structured_extraction: ready" in rendered
    assert "patch_editing: unusable" in rendered
    assert "long_context: ready" in rendered
    assert "lenses     unmeasured" in rendered
    assert "None" not in rendered


def test_render_table_still_prints_lenses_for_a_modern_profile():
    # The v1 fix must not cost the v5 lens line: a dict verdict keeps
    # rendering its lens fields.
    rendered = render_table(make_profile())
    assert "lenses     " in rendered
    assert "landing=" in rendered
    assert "lenses     unmeasured" not in rendered


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


def test_codec_verdicts_read_the_shared_lens_registry(monkeypatch):
    """The verdict layer's lens choice is the SAME object codecs' stop
    test reads (v1.6 consolidation). Move the registry and both verdicts
    move with it — that is the whole guarantee, and it is why the rule
    is no longer spelled twice."""
    from assay import stats
    from assay.profile import VERDICT_LENS

    assert VERDICT_LENS is stats.VERDICT_LENS  # the object, not a copy

    codecs = {
        codec: {
            grade: Landing(lands=0.0, lands_applies=0.95, n=20)
            for grade in _GRADES
        }
        for codec in _CODECS
    }
    verdicts = compute_verdicts(None, None, None, codecs)
    assert verdicts["structured_extraction"]["verdict"] == "unusable"
    assert verdicts["patch_editing"]["verdict"] == "ready"

    monkeypatch.setitem(stats.VERDICT_LENS, "json_object", "lands_applies")
    monkeypatch.setitem(stats.VERDICT_LENS, "search_replace", "lands")
    monkeypatch.setitem(stats.VERDICT_LENS, "whole_file", "lands")
    flipped = compute_verdicts(None, None, None, codecs)
    assert flipped["structured_extraction"]["verdict"] == "ready"
    assert flipped["patch_editing"]["verdict"] == "unusable"


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
    # v1 -> v2 BY DESIGN (v1.6): the error script changed the instrument,
    # so the lens string must change with it — a profile scored under
    # scripted-loop-v2 may never read as a scripted-loop-v1 measurement.
    assert verdicts["loop_discipline"]["lens"]["instrument"] == "scripted-loop-v2"
