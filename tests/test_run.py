"""Task 11 tests: the probe() orchestrator (plan Task 11, spec §9)."""

import json

import pytest
from fakes import (CodecFailingBackend, LongOutputDegradingBackend,
                   LongOutputTerseBackend, MetadataFreeBackend,
                   ScriptedBackend)

from assay import Budget, Profile, probe

_URL = "http://fake-host:11434"
_VRAM_MIB = 14558

# Quick mode against ScriptedBackend, call by call: 2 calibration,
# 5 ladder sizes (1024..16384, one seed, no bisection when clean),
# 10 envelope, 3 codecs x 3 grades x 5 = 45 codec probes.
# +shapes +loop (v1.4), +4 long-output rungs (v1.5).
_QUICK_CALLS_TOTAL = 2 + 5 + 9 + 10 + 45 + 9 + 2 + 4
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

    assert {n: e["verdict"] for n, e in profile.verdicts.items()} == {
        "structured_extraction": "ready",
        "patch_editing": "ready",
        "long_context": "ready",
        "loop_discipline": "ready",
        "chat_speed": "ready",
        "agent_speed": "ready",
        "long_output": "ready",
    }
    assert profile.speed is not None
    assert profile.speed.decode_tps == 16.0
    assert profile.speed.evidence == "server_timings"
    # v1.1 lens contract: patch_editing is judged applies-and-parses,
    # the presentation is the default, and provenance records it.
    assert profile.verdicts["patch_editing"]["lens"]["landing"] == "applies_and_parses(python)"
    assert profile.verdicts["patch_editing"]["lens"]["presentation"] == "default-v1"
    assert profile.provenance["presentation"] == "default-v1"

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
    assert profile.verdicts["long_context"]["verdict"] == "ready"
    assert profile.verdicts["structured_extraction"]["verdict"] == "unmeasured"
    assert profile.verdicts["patch_editing"]["verdict"] == "unmeasured"

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


def test_geometry_reads_the_post_load_serving_state():
    # v1.1, live-validation finding 3: a cold model's pre-load VRAM
    # reading double-counts its weights (granite/codegemma read
    # "usable window 0"). Geometry must be computed from a model_info
    # fetched AFTER calibration's first live call. The fake reports a
    # different training_ctx once any generate has happened, so the
    # geometry value itself proves which reading was used.
    class ColdThenLoaded(ScriptedBackend):
        def __init__(self):
            super().__init__()
            self.generated = False
            self.info_calls = 0

        def generate(self, *a, **k):
            self.generated = True
            return super().generate(*a, **k)

        def model_info(self):
            self.info_calls += 1
            info = super().model_info()
            import dataclasses
            if self.generated:
                return dataclasses.replace(info, training_ctx=4096, loaded=True)
            return dataclasses.replace(info, loaded=False)

    backend = ColdThenLoaded()
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000),
        mode="quick", _backend_override=backend,
    )
    assert backend.info_calls == 2
    # 4096 can only have come from the post-calibration model_info.
    assert profile.geometry is not None
    assert profile.geometry.usable_window <= 4096
    # The identity fields still come from the pre-load info.
    assert profile.model["training_ctx"] == 32768


def test_declared_tier_requires_explicit_emulation_marking():
    # Ruled 2026-08-13: emulated tiers are allowed but ALWAYS marked.
    with pytest.raises(ValueError, match="emulated"):
        probe(_URL, "fake-model",
              budget=Budget(max_calls=200, max_prompt_tokens=200_000),
              mode="quick", tier="average-gamer-8gb",
              _backend_override=ScriptedBackend())


def test_tier_and_emulation_travel_in_provenance():
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000),
        mode="quick", tier="average-gamer-8gb", emulated=True,
        _backend_override=ScriptedBackend(),
    )
    assert profile.provenance["tier"] == "average-gamer-8gb"
    assert profile.provenance["emulated"] is True
    # Undeclared stays honestly undeclared, never defaulted.
    bare = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000),
        mode="quick", _backend_override=ScriptedBackend(),
    )
    assert bare.provenance["tier"] is None
    assert bare.provenance["emulated"] is None


def test_full_mode_verdicts_carry_sequential_lens():
    # v1.5: the default mode samples codec cells sequentially, so the
    # verdict lens must say HOW the sample ended — the stopping rule and
    # the n it actually reached. A cell stopped at n=5 and a fixed-n=5
    # cell are the same number under different instruments; only the
    # lens tells them apart.
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=500, max_prompt_tokens=2_000_000),
        mode="full", _backend_override=ScriptedBackend(),
    )
    for name in ("structured_extraction", "patch_editing"):
        lens = profile.verdicts[name]["lens"]
        assert lens["stopping_rule"] == "wilson95-looks-5-10-20-35", name
    assert (profile.verdicts["structured_extraction"]["lens"]["n_used"]
            == profile.codecs["json_object"]["small"].n)
    # A perfect cell is only decided at the terminal look (Wilson lower
    # on 35/35 is 0.9011): the schedule runs to the cap here.
    assert profile.codecs["json_object"]["small"].n == 35
    assert profile.verdicts["patch_editing"]["lens"]["n_used"] == 35


def test_quick_mode_lens_is_fixed_n():
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000),
        mode="quick", _backend_override=ScriptedBackend(),
    )
    for name in ("structured_extraction", "patch_editing"):
        lens = profile.verdicts[name]["lens"]
        assert lens["stopping_rule"] == "fixed-n", name
        assert lens["n_used"] == 5, name


def test_unmeasured_codecs_have_no_n_used_in_the_lens():
    # None-vs-zero at the lens layer: a cell that was never sampled has
    # NO n_used entry. `n_used: 0` would read as "measured zero times
    # and still graded", which is not what happened.
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=_CALLS_THROUGH_CEILING, max_prompt_tokens=100_000),
        mode="full", _backend_override=ScriptedBackend(),
    )
    assert profile.codecs is None
    for name in ("structured_extraction", "patch_editing"):
        lens = profile.verdicts[name]["lens"]
        assert "n_used" not in lens, name
        assert lens["stopping_rule"] == "wilson95-looks-5-10-20-35", name


def test_sequential_early_stop_is_not_read_as_budget_death():
    # Every codec cell decides "unusable" at its first look and stops at
    # n=5 with the meter still full. The orchestrator must not confuse a
    # cell that STOPPED with a cell that was CUT OFF: loop and speed
    # still run, and nothing claims the budget was exhausted.
    backend = CodecFailingBackend()
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=500, max_prompt_tokens=2_000_000),
        mode="full", _backend_override=backend,
    )
    assert profile.codecs is not None
    for codec, grades in profile.codecs.items():
        for grade, cell in grades.items():
            assert cell.n == 5, (codec, grade)
            assert cell.lands_applies == 0.0, (codec, grade)
    assert profile.verdicts["structured_extraction"]["verdict"] == "unusable"
    assert profile.verdicts["structured_extraction"]["provisional"] is False
    assert profile.loop is not None
    assert profile.speed is not None
    assert profile.provenance["dropped"] == []
    assert backend.calls < 500


def test_budget_death_mid_sequential_cell_still_reads_as_budget_death():
    # The other side of the early-stop rule: a cell cut off BETWEEN
    # looks (n=12 is neither a look point nor the cap) is a dead meter,
    # not a decision, and the families after codecs must be skipped and
    # named — never quietly attempted on an exhausted budget.
    pre_codec_calls = 2 + 12 + 9 + 30  # calibration, ladder x2 seeds, shapes, envelope
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=pre_codec_calls + 12,
                      max_prompt_tokens=2_000_000),
        mode="full", _backend_override=ScriptedBackend(),
    )
    first = profile.codecs["search_replace"]["tiny"]
    assert first.n == 12, "the meter must really die between looks"
    assert profile.loop is None and profile.speed is None
    dropped = profile.provenance["dropped"]
    assert "loop: skipped, budget exhausted earlier" in dropped
    assert "speed: skipped, budget exhausted earlier" in dropped


def test_n_used_names_the_cell_each_verdict_was_actually_graded_on():
    # Sequential sampling lets the cells of one run end at DIFFERENT n.
    # patch_editing is graded on whichever patch codec lands best under
    # applies-and-parses, so its n_used must come from that cell — not
    # from json's, and not from the patch codec that lost.
    from assay.codecs import Landing
    from assay.run import _codec_n_used

    def cell(applies, n):
        return Landing(lands=applies, lands_applies=applies, n=n)

    codecs = {
        "json_object": {"small": cell(0.0, 5)},
        "search_replace": {"small": cell(0.5, 35)},
        "whole_file": {"small": cell(0.9, 20)},  # the graded cell
    }
    assert _codec_n_used(codecs) == {"structured_extraction": 5,
                                     "patch_editing": 20}
    # Unmeasured cells contribute no entry at all, never a zero.
    unmeasured = {codec: {"small": Landing(lands=None, lands_applies=None, n=0)}
                  for codec in ("json_object", "search_replace", "whole_file")}
    assert _codec_n_used(unmeasured) == {}
    assert _codec_n_used(None) == {}


@pytest.mark.parametrize("mode,budget,expected", [
    # The literals are pinned here, not read off the table: one decode
    # call in quick (cheap, mean of one, spread unknowable) and three in
    # full/thorough (the smallest count whose samples show a spread).
    ("quick", Budget(max_calls=200, max_prompt_tokens=200_000), 1),
    ("full", Budget(max_calls=500, max_prompt_tokens=2_000_000), 3),
    ("thorough", Budget(max_calls=500, max_prompt_tokens=2_000_000), 3),
])
def test_speed_decode_calls_follow_the_mode(mode, budget, expected):
    # The mode table owns how many decode calls are affordable, and the
    # orchestrator must SPEND that number — a mode that says 3 and
    # probes once buys none of the spread the samples exist to show.
    from assay.run import MODE_PARAMS

    assert MODE_PARAMS[mode].speed_decode_calls == expected
    profile = probe(
        _URL, "fake-model", budget=budget, mode=mode,
        _backend_override=ScriptedBackend(),
    )
    assert profile.speed.n_decode == expected
    assert len(profile.speed.decode_samples) == expected
    assert len(profile.speed.prefill_samples) == profile.speed.n_prefill == 1


def test_long_output_family_climbs_the_full_ladder_on_a_healthy_endpoint():
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000),
        mode="quick", _backend_override=ScriptedBackend(),
    )
    assert profile.long_output is not None
    assert [rung.target_tokens for rung in profile.long_output.rungs] == [
        512, 1024, 2048, 4096]
    # The measured ceiling is 16384, so no rung is out of reach.
    assert profile.long_output.skipped == ()
    assert all(rung.degenerate is False for rung in profile.long_output.rungs)
    assert profile.verdicts["long_output"]["verdict"] == "ready"
    assert profile.verdicts["long_output"]["provisional"] is True
    assert profile.provenance["dropped"] == []


def test_long_output_degradation_is_located_at_the_rung_it_starts():
    # The family exists for exactly this shape: fine at 512 and 1024,
    # looping from 2048 up. A single-target probe cannot see it.
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000),
        mode="quick", _backend_override=LongOutputDegradingBackend(2048),
    )
    flags = {rung.target_tokens: rung.degenerate
             for rung in profile.long_output.rungs}
    assert flags == {512: False, 1024: False, 2048: True, 4096: True}
    assert profile.verdicts["long_output"]["verdict"] == "degrades-at-2048"


def test_long_output_rungs_above_the_measured_ceiling_are_skipped_by_name():
    # ceiling_max is the ladder's cap: a rung bigger than anything the
    # ceiling verified is not attempted, and says why it was not.
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000),
        mode="quick", window_cap=1024, _backend_override=ScriptedBackend(),
    )
    assert profile.ceiling.max_verified == 1024
    assert [rung.target_tokens for rung in profile.long_output.rungs] == [512, 1024]
    assert profile.long_output.skipped == ("2048: above measured ceiling",
                                           "4096: above measured ceiling")
    assert profile.verdicts["long_output"]["verdict"] == "ready"


def test_long_output_skipped_when_the_budget_died_earlier():
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=_CALLS_THROUGH_CEILING,
                      max_prompt_tokens=100_000),
        mode="quick", _backend_override=ScriptedBackend(),
    )
    assert profile.long_output is None
    assert "long_output: skipped, budget exhausted earlier" in (
        profile.provenance["dropped"])
    assert profile.verdicts["long_output"]["verdict"] == "unmeasured"


def test_a_ladder_that_ran_no_rung_is_none_and_named():
    # The meter runs dry inside the speed family (which does not itself
    # declare budget death), so long_output starts on an empty meter:
    # every rung is skipped, nothing is measured, and the family is
    # None — named, never a silently empty ladder.
    backend = ScriptedBackend()
    through_speed = _QUICK_CALLS_TOTAL - 4
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=through_speed, max_prompt_tokens=1_000_000),
        mode="quick", _backend_override=backend,
    )
    assert backend.calls == through_speed  # not one rung was attempted
    assert profile.speed is not None
    assert profile.long_output is None
    assert any(entry.startswith("long_output: no rung ran")
               for entry in profile.provenance["dropped"])
    assert profile.verdicts["long_output"]["verdict"] == "unmeasured"


def test_a_ladder_that_scored_nothing_is_named_in_dropped_and_kept():
    # Four calls spent, four rungs attempted, nothing scorable came
    # back. The rungs are real evidence (they say what was asked and
    # what came back), so the family is KEPT — but the profile must say
    # the ladder measured nothing, and the verdict must not read ready.
    backend = LongOutputTerseBackend()
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000),
        mode="quick", _backend_override=backend,
    )
    assert profile.long_output is not None
    assert len(profile.long_output.rungs) == 4
    assert all(rung.degenerate is None for rung in profile.long_output.rungs)
    assert any("long_output:" in entry and "scorable" in entry
               for entry in profile.provenance["dropped"])
    assert profile.verdicts["long_output"]["verdict"] == "unmeasured"


def test_thorough_params_equal_full_params():
    # v1.5: --thorough is an alias. Its old fixed n=35 is exactly the
    # sequential cap, so it buys nothing full does not already buy; the
    # key stays so the documented flag still parses.
    from assay.run import MODE_PARAMS

    assert MODE_PARAMS["thorough"] == MODE_PARAMS["full"]


def test_thorough_mode_buys_a_decidable_ready():
    # 35 = 7 reps x 5 tasks: the smallest n where 35/35 clears the 0.9
    # ready floor non-provisionally (Wilson lower 0.9011). quick/full
    # stay honest-but-provisional; thorough makes a decided ready
    # PURCHASABLE (external review follow-up, 2026-08-13).
    from assay.profile import wilson95
    from assay.run import MODE_PARAMS

    n = MODE_PARAMS["thorough"].codecs_n_per_cell
    assert n == 35
    assert wilson95(n, n)[0] > 0.9

    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=500, max_prompt_tokens=2_000_000),
        mode="thorough", _backend_override=ScriptedBackend(),
    )
    entry = profile.verdicts["structured_extraction"]
    assert entry["verdict"] == "ready"
    assert entry["provisional"] is False
    assert profile.codecs["json_object"]["small"].n == 35
