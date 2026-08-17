"""Task 11 tests: the probe() orchestrator (plan Task 11, spec §9)."""

import json

import pytest
from fakes import (CodecFailingBackend, LongOutputDegradingBackend,
                   LongOutputTerseBackend, MetadataFreeBackend,
                   ScriptedBackend, ToolsUnsupportedBackend)

from assay import Budget, Profile, probe
from assay.errors import BudgetExhausted

_URL = "http://fake-host:11434"
_VRAM_MIB = 14558

# Quick mode against ScriptedBackend, call by call: 2 calibration,
# 5 ladder sizes (1024..16384, one seed, no bisection when clean),
# 10 envelope, 60 codec probes (search_replace and whole_file at 3
# grades each, json_object at 6 since v1.7, x 5 tasks per cell).
# +shapes +loop (v1.4), +4 long-output rungs (v1.5), +10 tools (v1.6).
# The loop term is 15, not 9, BY DESIGN (v1.6): scripted-loop-v2 plays a
# 2-turn error script alongside each of the 3 golden 3-turn runs.
_QUICK_CALLS_LONG_OUTPUT = 4
_QUICK_CALLS_TOOLS = 10  # 5 scripted tasks x 2 turns
_QUICK_CALLS_CODECS = 60  # 12 cells x 5: json gained three deep grades
_QUICK_CALLS_TOTAL = (2 + 5 + 9 + 10 + _QUICK_CALLS_CODECS + 15 + 2
                      + _QUICK_CALLS_LONG_OUTPUT + _QUICK_CALLS_TOOLS)
_CALLS_THROUGH_CEILING = 2 + 5

# Headroom for a CLEAN full run against these fakes, which is 552 calls
# in v1.7 (12 codec cells to the 35-sample cap + 40 tools turns + 6
# parallel lanes + the rest). This is test headroom, above the default
# in cli.DEFAULT_BUDGETS — these tests are about the mode table and the
# families, not about what the default budget covers.
_CLEAN_FULL_RUN_HEADROOM = 700

#: The line quick mode names instead of measuring concurrency (v1.7).
#: Quick's `dropped` is not empty any more and never can be: the
#: parallel family is None there, and an unmeasured family is named
#: whatever the reason — a mode decision as much as a dead meter.
_QUICK_PARALLEL_DROP = "parallel: quick mode — full mode measures concurrency"


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
    from assay.codecs import GRADES_FOR
    for codec in ("search_replace", "whole_file", "json_object"):
        for grade in GRADES_FOR[codec]:
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
        "tool_calling": "ready",
    }
    # The tools family ran, and ran LAST: five tasks, two turns each.
    assert profile.tools is not None
    assert profile.tools.supported is True
    assert profile.tools.composite == 1.0
    assert (profile.tools.n_tasks, profile.tools.n_turns) == (5, 10)
    assert profile.verdicts["tool_calling"]["lens"]["n_used"] == 5
    # ...and the loop family's error script is in the profile with a
    # visible denominator, not just in the rates.
    assert profile.loop.recovery_rate == 1.0
    assert profile.loop.doom_loop_rate == 0.0
    assert profile.loop.n_error_runs == 3
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
    # One line, and it is a MODE fact rather than a failure: quick does
    # not buy concurrency (v1.7), and an unmeasured family is named
    # whatever the reason it went unmeasured.
    assert profile.provenance["dropped"] == [_QUICK_PARALLEL_DROP]
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
        budget=Budget(max_calls=_CLEAN_FULL_RUN_HEADROOM,
                      max_prompt_tokens=2_000_000),
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
        budget=Budget(max_calls=_CLEAN_FULL_RUN_HEADROOM,
                      max_prompt_tokens=2_000_000),
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
    ("full", Budget(max_calls=_CLEAN_FULL_RUN_HEADROOM,
                    max_prompt_tokens=2_000_000), 3),
    ("thorough", Budget(max_calls=_CLEAN_FULL_RUN_HEADROOM,
                        max_prompt_tokens=2_000_000), 3),
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
    # UPDATED BY TASK 12: this pinned True while the floors were assumed
    # and the verdict layer capped them. The anchor capture derived
    # ZLIB_FLOOR, the cap released, and — under the completeness ruling
    # of 2026-08-15 — a ladder that climbed and scored all four rungs
    # with nothing skipped is exactly the case that reads settled. The
    # ceiling-truncated ladder below is the contrast.
    assert profile.long_output.skipped == ()
    assert all(rung.degenerate is not None for rung in profile.long_output.rungs)
    assert profile.verdicts["long_output"]["provisional"] is False
    # The verdict carries HOW FAR it was verified, not just that it was.
    lens = profile.verdicts["long_output"]["lens"]
    assert lens["rungs_scored"] == 4
    assert lens["deepest_scored_tokens"] == 4096
    # Quick names the parallel family it does not measure; nothing here
    # was dropped for want of budget.
    assert profile.provenance["dropped"] == [_QUICK_PARALLEL_DROP]


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
    # ...and "ready" here means ready THROUGH 1024, which the lens says
    # so a consumer cannot read it as the 4096-deep ready of a model
    # whose ceiling never truncated the ladder.
    lens = profile.verdicts["long_output"]["lens"]
    assert lens["rungs_scored"] == 2
    assert lens["deepest_scored_tokens"] == 1024
    # UPDATED BY TASK 12 (ruled 2026-08-15): the extent lives in the lens
    # for a reader who opens it, and in `provisional` for the one who
    # only sees the badge. This ladder did not finish — two rungs the
    # ceiling put out of reach — so the verdict is not settled.
    assert profile.verdicts["long_output"]["provisional"] is True


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
    through_speed = (_QUICK_CALLS_TOTAL - _QUICK_CALLS_LONG_OUTPUT
                     - _QUICK_CALLS_TOOLS)
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
    # tools runs after long_output on the same dry meter: it attempts
    # nothing, is None, and is NAMED.
    assert profile.tools is None
    assert "tools: budget exhausted before any turn completed" in (
        profile.provenance["dropped"])


# --- the tools family, last in the chain (v1.6) ----------------------------


def test_tools_runs_last_after_the_long_output_ladder():
    # Order is a budget fact, not a stylistic one: with exactly enough
    # calls for everything through the ladder, the ladder is COMPLETE and
    # tools is the family that goes hungry. Were the order reversed, the
    # same budget would buy ten tool turns and a truncated ladder.
    backend = ScriptedBackend()
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=_QUICK_CALLS_TOTAL - _QUICK_CALLS_TOOLS,
                      max_prompt_tokens=1_000_000),
        mode="quick", _backend_override=backend,
    )
    assert len(profile.long_output.rungs) == 4
    assert all(rung.degenerate is False for rung in profile.long_output.rungs)
    assert profile.tools is None
    assert "tools: budget exhausted before any turn completed" in (
        profile.provenance["dropped"])
    assert profile.verdicts["tool_calling"]["verdict"] == "unmeasured"


def test_tools_is_skipped_and_named_when_the_budget_died_earlier():
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=_CALLS_THROUGH_CEILING,
                      max_prompt_tokens=100_000),
        mode="quick", _backend_override=ScriptedBackend(),
    )
    assert profile.tools is None
    assert "tools: skipped, budget exhausted earlier" in (
        profile.provenance["dropped"])
    assert profile.verdicts["tool_calling"]["verdict"] == "unmeasured"


def test_a_tools_run_cut_off_mid_script_keeps_its_honest_partial():
    # Five extra calls buy T1+T2 for two tasks and T1 for a third: the
    # family is a MEASUREMENT at n_tasks=3, not None and not padded to 5.
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=_QUICK_CALLS_TOTAL - _QUICK_CALLS_TOOLS + 5,
                      max_prompt_tokens=1_000_000),
        mode="quick", _backend_override=ScriptedBackend(),
    )
    assert profile.tools is not None
    assert profile.tools.supported is True
    assert (profile.tools.n_tasks, profile.tools.n_turns) == (3, 5)
    assert profile.verdicts["tool_calling"]["lens"]["n_used"] == 3


def test_an_endpoint_that_refuses_tools_records_the_capability():
    # A ToolsUnsupported is handled INSIDE probe_tools: the orchestrator
    # sees an ordinary Tools value, keeps it (it is a measurement), and
    # never names it in dropped.
    backend = ToolsUnsupportedBackend()
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=1_000_000),
        mode="quick", _backend_override=backend,
    )
    assert profile.tools is not None
    assert profile.tools.supported is False
    assert profile.tools.composite is None
    assert profile.verdicts["tool_calling"]["verdict"] == "unsupported"
    assert profile.verdicts["tool_calling"]["interval95"] is None
    assert not any("tools" in entry for entry in profile.provenance["dropped"])
    # One refused call, not ten: nine more refusals measure nothing new.
    assert backend.calls == _QUICK_CALLS_TOTAL - _QUICK_CALLS_TOOLS + 1
    # The whole profile still serializes, refusal and all.
    assert Profile.from_json(json.loads(profile.to_json())) == profile


def test_the_tools_stopping_rule_travels_with_the_mode():
    # v1.7: the mode table owns the tools schedule exactly as it owns the
    # codec one. quick stays fixed at the v1 prefix (its numbers have to
    # compare across the version boundary); full and thorough spend the
    # pool sequentially — and thorough is full's alias, so it must not
    # drift into a third sampling rule.
    from assay.run import MODE_PARAMS
    from assay.tools import TOOLS_LOOK_SCHEDULE

    assert MODE_PARAMS["quick"].tools_look_schedule is None
    assert MODE_PARAMS["full"].tools_look_schedule == TOOLS_LOOK_SCHEDULE
    assert MODE_PARAMS["thorough"].tools_look_schedule == TOOLS_LOOK_SCHEDULE

    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=_CLEAN_FULL_RUN_HEADROOM,
                      max_prompt_tokens=2_000_000),
        mode="full", _backend_override=ScriptedBackend(),
    )
    # A perfect run cannot decide below 35 tasks, so it runs to the cap:
    # 20 tasks, both turns each, and the verdict is ready-but-provisional
    # at an interval n=5 could never have bought.
    assert (profile.tools.n_tasks, profile.tools.n_turns) == (20, 40)
    entry = profile.verdicts["tool_calling"]
    assert entry["lens"]["stopping_rule"] == "wilson95-looks-5-10-20"
    assert entry["lens"]["n_used"] == 20
    assert entry["verdict"] == "ready" and entry["provisional"] is True


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


# --- the parallel family (v1.7): full mode only, on speed's baseline -------


def _calls_before_speed() -> int:
    """The call index the full-mode speed family's first decode call
    falls on — MEASURED, not hand-counted.

    Every family before speed costs what it costs, and a literal here
    would rot the first time one of them changes (they all did, twice,
    in v1.7). The probe is run once with headroom against a backend that
    records where the decode prompt first appears; a second run capped at
    exactly that number starves the speed family and nothing else.
    """
    from assay.speed import DECODE_PROMPT

    class SpeedBoundary(ScriptedBackend):
        def __init__(self):
            super().__init__()
            self.before_speed = None

        def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
            if prompt == DECODE_PROMPT and self.before_speed is None:
                self.before_speed = self.calls  # calls spent BEFORE this one
            return super().generate(prompt, seed=seed, max_tokens=max_tokens,
                                    num_ctx=num_ctx)

    backend = SpeedBoundary()
    probe(_URL, "fake-model",
          budget=Budget(max_calls=_CLEAN_FULL_RUN_HEADROOM,
                        max_prompt_tokens=2_000_000),
          mode="full", _backend_override=backend)
    assert backend.before_speed is not None, "the speed family never ran"
    return backend.before_speed


def test_full_mode_measures_parallel_against_this_run_s_own_baseline():
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=_CLEAN_FULL_RUN_HEADROOM,
                      max_prompt_tokens=2_000_000),
        mode="full", _backend_override=ScriptedBackend(),
    )

    assert profile.parallel is not None
    assert [row.k for row in profile.parallel.rows] == [2, 4]
    # The ratio's denominator is the SAME RUN's single-lane decode rate,
    # never a number from another run or another instrument.
    assert profile.parallel.baseline_decode_tps == profile.speed.decode_tps
    for row in profile.parallel.rows:
        # The scripted endpoint reports the same server timings on every
        # lane, so a healthy k measures no degradation at all.
        assert row.per_lane_decode_tps == 16.0, row
        assert row.degradation_ratio == 1.0, row
        assert row.n_lanes_ok == row.k, row
        assert row.lane_errors == (), row
        assert row.evidence == "server_timings", row
    # Both configured k values were affordable, so no k is named skipped.
    assert profile.parallel.skipped == ()
    assert profile.provenance["dropped"] == []
    # The family survives the document contract with the rest of them.
    assert Profile.from_json(json.loads(profile.to_json())) == profile


def test_quick_mode_drops_the_parallel_family_by_name():
    # Quick is the mode an operator reaches for in a hurry; six extra
    # calls buy a concurrency finding it did not ask for. Dropped BY
    # NAME, never silently absent.
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000),
        mode="quick", _backend_override=ScriptedBackend(),
    )

    assert profile.parallel is None
    assert profile.provenance["dropped"] == [_QUICK_PARALLEL_DROP]


def test_parallel_is_skipped_and_named_when_the_budget_died_earlier():
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=_CALLS_THROUGH_CEILING,
                      max_prompt_tokens=100_000),
        mode="full", _backend_override=ScriptedBackend(),
    )

    assert profile.parallel is None
    dropped = profile.provenance["dropped"]
    assert "parallel: skipped, budget exhausted earlier" in dropped
    # Position is a measurement fact and a budget fact: the family needs
    # the same run's single-lane baseline (so it cannot precede speed)
    # and its six lanes are charged before the long ladder empties the
    # meter (so it cannot follow long_output).
    assert (dropped.index("speed: skipped, budget exhausted earlier")
            < dropped.index("parallel: skipped, budget exhausted earlier")
            < dropped.index("long_output: skipped, budget exhausted earlier"))


def test_a_full_run_that_never_measured_speed_drops_parallel_by_name():
    # `degradation_ratio` divides by the same run's single-lane decode
    # rate. A run that never measured one has nothing to divide by, and
    # the family is dropped by name rather than run against a baseline
    # from somewhere else.
    starved = _calls_before_speed()
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=starved, max_prompt_tokens=2_000_000),
        mode="full", _backend_override=ScriptedBackend(),
    )

    assert profile.speed is None
    assert profile.parallel is None
    assert ("parallel: no single-lane baseline (speed unmeasured)"
            in profile.provenance["dropped"])


def test_a_meter_that_refuses_the_first_k_names_parallel_and_stops_the_rest():
    # The one path where probe_parallel raises: the budget refused k=2
    # before any lane ran, so nothing was measured. That is a budget
    # death, and the families after it must be skipped and named rather
    # than quietly attempted on an exhausted meter.
    from assay.run import MODE_PARAMS

    through_speed = (_calls_before_speed()
                     + MODE_PARAMS["full"].speed_decode_calls + 1)  # +prefill
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=through_speed, max_prompt_tokens=2_000_000),
        mode="full", _backend_override=ScriptedBackend(),
    )

    assert profile.speed is not None and profile.speed.decode_tps == 16.0
    assert profile.parallel is None
    dropped = profile.provenance["dropped"]
    assert "parallel: budget exhausted before any lane ran" in dropped
    assert "long_output: skipped, budget exhausted earlier" in dropped
    assert "tools: skipped, budget exhausted earlier" in dropped


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
        budget=Budget(max_calls=_CLEAN_FULL_RUN_HEADROOM,
                      max_prompt_tokens=2_000_000),
        mode="thorough", _backend_override=ScriptedBackend(),
    )
    entry = profile.verdicts["structured_extraction"]
    assert entry["verdict"] == "ready"
    assert entry["provisional"] is False
    assert profile.codecs["json_object"]["small"].n == 35


# --- the wall-clock ceiling in provenance (v1.7) ---------------------------


def test_a_run_with_no_seconds_ceiling_writes_no_seconds_at_all():
    # `spent.seconds` is 0.0 on a clock-free run because the meter never
    # asked the clock what time it is — a floor, not a measurement. A
    # provenance that published it would report a run that took no time.
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000),
        mode="quick", _backend_override=ScriptedBackend(),
    )
    assert "max_seconds" not in profile.provenance["budget"]
    assert "seconds" not in profile.provenance["spent"]


def test_a_granted_seconds_ceiling_travels_with_what_the_clock_measured():
    backend = ScriptedBackend()
    # Time passes when the endpoint is called: a tenth of a second per
    # call, against a ceiling no quick run can reach.
    profile = probe(
        _URL, "fake-model",
        budget=Budget(max_calls=200, max_prompt_tokens=200_000,
                      max_seconds=10_000.0),
        mode="quick", _backend_override=backend,
        _clock=lambda: backend.calls * 0.1,
    )
    assert profile.provenance["budget"]["max_seconds"] == 10_000.0
    # The elapsed reading at the LAST admitted charge, which is one call
    # short of the run's total — the meter charges before it calls.
    assert profile.provenance["spent"]["seconds"] == pytest.approx(
        (_QUICK_CALLS_TOTAL - 1) * 0.1)


# --- the declared worst-case cost table (v1.7) ----------------------------

#: The clean full run tests/test_cli.py METERS end to end. The table is
#: checked against it rather than against a fresh hand count: a per-family
#: number that no longer sums to what the suite spends is a number that
#: has drifted from the instrument.
_MEASURED_CLEAN_FULL_RUN = 552


def _fat_meter():
    """A meter no family can exhaust: the ceiling under test is the
    TABLE's, not the budget's."""
    from assay.budget import BudgetMeter

    return BudgetMeter(Budget(max_calls=10_000, max_prompt_tokens=10**9))


def _family_runners():
    """family name -> the call the orchestrator makes for that family.

    Each entry takes the SAME arguments probe() passes, so a family whose
    parameters change breaks here rather than quietly costing something
    else than the table says.
    """
    from assay.ceiling import calibrate, probe_fixed_shapes
    from assay.codecs import JSON_CODECS, PATCH_CODECS, probe_codecs
    from assay.envelope import probe_envelope
    from assay.long_output import probe_long_output
    from assay.loop import probe_loop
    from assay.speed import probe_speed
    from assay.tools import probe_tools

    def calibrated(backend, params):
        """What probe() hands the families that size their own prompts.

        Not an ornament. Sized from the 3.0 fallback instead of this
        endpoint's measured chars-per-token, the fake's own token counts
        come back below the truncation floor and the shape family stops
        after ONE probe per shape — a starved run, not a maximizing one,
        and it would be the table this test blamed for the difference.
        The meter is a throwaway: the calibration calls belong to the
        calibration family, which is metered on its own row.
        """
        return calibrate(backend, _fat_meter(), seed=params.seeds[0])

    return {
        "calibration": lambda b, m, p: calibrate(b, m, seed=p.seeds[0]),
        "ceiling_shapes": lambda b, m, p: probe_fixed_shapes(
            b, m, calibration=calibrated(b, p), shapes=p.shape_probes),
        "envelope": lambda b, m, p: probe_envelope(b, m, n=p.envelope_n),
        "codecs-json": lambda b, m, p: probe_codecs(
            b, m, n_per_cell=p.codecs_n_per_cell,
            look_schedule=p.codec_look_schedule, only=JSON_CODECS),
        "codecs-patch": lambda b, m, p: probe_codecs(
            b, m, n_per_cell=p.codecs_n_per_cell,
            look_schedule=p.codec_look_schedule, only=PATCH_CODECS),
        "speed": lambda b, m, p: probe_speed(
            b, m, calibration=calibrated(b, p),
            decode_calls=p.speed_decode_calls),
        "loop": lambda b, m, p: probe_loop(b, m, runs=p.loop_runs),
        "long_output": lambda b, m, p: probe_long_output(b, m, ceiling_max=None),
        "tools": lambda b, m, p: probe_tools(
            b, m, look_schedule=p.tools_look_schedule),
    }


@pytest.mark.parametrize("mode", ("quick", "full"))
@pytest.mark.parametrize("family", sorted(_family_runners()))
def test_no_family_outspends_what_the_table_declares(family, mode):
    """The never-exceeds property, MEASURED against a maximizing endpoint.

    ScriptedBackend answers every probe and lands every codec cell, so
    nothing decides early anywhere: every family runs the longest run it
    has. The spend is metered and compared against the declaration — the
    test measures the instrument against the table, never the table
    against itself.
    """
    from assay.run import MODE_PARAMS, worst_case_calls

    params = MODE_PARAMS[mode]
    meter = _fat_meter()
    _family_runners()[family](ScriptedBackend(), meter, params)

    declared = worst_case_calls(family, params)
    assert meter.spent.calls <= declared
    # ...and TIGHT: a declaration comfortably above the real cost would
    # pass the line above while reserving budget nobody needs.
    assert meter.spent.calls == declared


def test_the_table_names_every_family_the_orchestrator_runs():
    """The keys are an interface: budget mode preflights BY these names,
    and the codec halves are split because it buys them separately."""
    from assay.run import WORST_CASE

    assert set(WORST_CASE) == {
        "geometry", "calibration", "ceiling", "ceiling_shapes", "envelope",
        "codecs-json", "codecs-patch", "speed", "loop", "long_output",
        "tools", "parallel",
    }


def test_the_table_refuses_a_family_it_does_not_know():
    """No guessed cost, ever: an unknown family is a KeyError, not a 0
    that would let a consumer run something it never budgeted for."""
    from assay.run import MODE_PARAMS, worst_case_calls

    with pytest.raises(KeyError):
        worst_case_calls("nonsense", MODE_PARAMS["quick"])


def test_geometry_costs_no_generative_calls():
    from assay.run import MODE_PARAMS, worst_case_calls

    for mode in ("quick", "full"):
        assert worst_case_calls("geometry", MODE_PARAMS[mode]) == 0


def test_the_ceiling_entry_is_the_ladder_enumeration_at_the_mode_cap():
    """Asserted against the ladder's own step function — running a
    maximizing ladder would measure the bisection too, and the bisection
    is declared separately (see the budget test below)."""
    from assay.ceiling import ladder_steps
    from assay.run import MODE_PARAMS, worst_case_calls

    for mode in ("quick", "full", "thorough"):
        params = MODE_PARAMS[mode]
        assert worst_case_calls("ceiling", params) == (
            len(ladder_steps(params.ceiling_cap)) * len(params.seeds))
    assert worst_case_calls("ceiling", MODE_PARAMS["quick"]) == 5
    assert worst_case_calls("ceiling", MODE_PARAMS["full"]) == 12


def test_the_parallel_entry_is_the_registered_lane_counts():
    from assay.parallel import DEFAULT_KS
    from assay.run import MODE_PARAMS, worst_case_calls

    assert worst_case_calls("parallel", MODE_PARAMS["full"]) == sum(DEFAULT_KS) == 6


def test_the_loop_and_long_output_entries_come_from_their_own_scripts():
    """Both are pure script lengths, and both are cross-checked against
    the hand count in cli.py's budget derivation (25 loop turns, 4 rungs)."""
    from assay.long_output import RUNGS
    from assay.loop import error_turn_prompts, turn_prompts
    from assay.run import MODE_PARAMS, worst_case_calls

    full = MODE_PARAMS["full"]
    assert worst_case_calls("loop", full) == 25 == full.loop_runs * (
        len(turn_prompts()) + len(error_turn_prompts()))
    assert worst_case_calls("long_output", full) == 4 == len(RUNGS)


def test_the_table_explains_the_measured_runs_and_fits_the_default_budget():
    """The sum is the whole point: it has to equal what a clean run
    SPENDS, and it has to fit the default budget with the one term the
    per-family numbers cannot carry — a failing ceiling's bisection —
    named rather than left as slack.
    """
    from assay.ceiling import bisection_worst_case_steps
    from assay.cli import DEFAULT_BUDGETS
    from assay.run import MODE_PARAMS, WORST_CASE, worst_case_calls

    full = MODE_PARAMS["full"]
    declared = sum(worst_case_calls(family, full) for family in WORST_CASE)
    assert declared == _MEASURED_CLEAN_FULL_RUN

    # A ladder that fails stops EARLY (it never reaches the cap) and
    # bisects instead; the extra is at most this, whatever it finds.
    bisection = bisection_worst_case_steps(full.ceiling_cap) * len(full.seeds)
    assert bisection == 8
    assert declared + bisection <= DEFAULT_BUDGETS["full"].max_calls

    quick = MODE_PARAMS["quick"]
    # Quick NAMES the parallel family instead of running it (run.py), so
    # its declared cost is not part of a quick run's bill.
    quick_declared = sum(worst_case_calls(family, quick)
                         for family in WORST_CASE if family != "parallel")
    assert quick_declared == _QUICK_CALLS_TOTAL
    quick_bisection = (bisection_worst_case_steps(quick.ceiling_cap)
                       * len(quick.seeds))
    assert (quick_declared + quick_bisection
            <= DEFAULT_BUDGETS["quick"].max_calls)


# --- budget mode: the priority-ordered consumer probe (v1.7) ---------------
#
# The consumer story: an application at settings-time wants the most
# load-bearing profile it can get in <=N calls and optionally <=S
# seconds. Every number below is DERIVED from the declared cost table
# above, because the whole mode is that table used as a preflight: a
# family that does not fit is dropped BY NAME and never started, and a
# cheaper family further down the priority may still fit.

#: What each family costs a budget-mode run, in priority order. The
#: ceiling term carries its bisection tail (see the preflight test).
_BUDGET_CALIBRATION = 2
_BUDGET_ENVELOPE = 10
_BUDGET_SPEED = 2
_BUDGET_CODECS_JSON = 30
_BUDGET_CODECS_PATCH = 30
_BUDGET_TOOLS = 10
_BUDGET_CEILING_LADDER = 5
_BUDGET_CEILING_BISECTION = 4
_BUDGET_LOOP = 15
_BUDGET_LONG_OUTPUT = 4

#: THE WORKED EXAMPLE (spec §4). The spec wrote 25 before calibration
#: was measured at two calls and before the deep json grades tripled the
#: json half; re-derived against the table this task consumes, the
#: budget that buys exactly {geometry, envelope, speed, json quick} is
#: 2 + 0 + 10 + 2 + 30 = 44 — and buys it exactly, with nothing left
#: over for the 4-call long-output ladder that follows the drops.
_WORKED_EXAMPLE_CALLS = (_BUDGET_CALIBRATION + _BUDGET_ENVELOPE
                         + _BUDGET_SPEED + _BUDGET_CODECS_JSON)

#: The two families budget mode does not run at all, named as MODE facts
#: rather than as budget failures — the reason is the priority, not the
#: meter, and a consumer reading `dropped` must be able to tell those
#: apart.
_BUDGET_MODE_FACTS = [
    "ceiling_shapes: budget mode — not in the registered priority",
    "parallel: budget mode — full mode measures concurrency",
]


def _budget_probe(calls, *, budget_kwargs=None, **kwargs):
    """One budget-mode run under a call ceiling and a token budget no
    family can reach — these tests are about the CALL preflight."""
    budget = Budget(max_calls=calls, max_prompt_tokens=10**9,
                    **(budget_kwargs or {}))
    return probe(_URL, "fake-model", budget=budget, mode="budget", **kwargs)


def test_the_budget_priority_order_is_the_registered_one():
    # Pinned as DATA, cheapest-and-most-load-bearing first. Reordering it
    # is a change to what a consumer's N calls buy, and it must fail a
    # test rather than quietly re-rank someone's settings-time probe.
    from assay.run import PRIORITY, WORST_CASE

    assert PRIORITY == ("geometry", "envelope", "speed", "codecs-json",
                        "codecs-patch", "tools", "ceiling", "loop",
                        "long_output")
    # Every name in it is priced by the declared table — an unpriced
    # family cannot be preflighted, and a preflight that guessed would be
    # exactly the mid-family death this mode exists to prevent.
    assert set(PRIORITY) <= set(WORST_CASE)
    # The families NOT in the priority are the two the mode names as mode
    # facts; calibration is not a family, it is the run's entry fee.
    assert set(WORST_CASE) - set(PRIORITY) == {
        "calibration", "ceiling_shapes", "parallel"}


def test_budget_mode_params_are_quicks_numbers():
    # Budget mode measures quick-SHAPED families under a consumer's
    # ceiling: the point is coverage order, not sequential depth. A
    # sequential codec schedule here would spend a whole budget deciding
    # one cell.
    from assay.run import MODE_PARAMS

    assert MODE_PARAMS["budget"] == MODE_PARAMS["quick"]
    assert MODE_PARAMS["budget"].codec_look_schedule is None
    assert MODE_PARAMS["budget"].tools_look_schedule is None


def test_the_worked_example_buys_geometry_envelope_speed_and_json():
    backend = ScriptedBackend()
    profile = _budget_probe(_WORKED_EXAMPLE_CALLS, _backend_override=backend)

    assert profile.provenance["mode"] == "budget"
    # What the priority bought, in order.
    assert profile.geometry is not None
    assert profile.envelope is not None and profile.envelope.n == 10
    assert profile.speed is not None and profile.speed.decode_tps == 16.0
    from assay.codecs import GRADES_FOR
    for grade in GRADES_FOR["json_object"]:
        assert profile.codecs["json_object"][grade].n == 5, grade
    # ...and what it did not buy: every patch cell unmeasured, and the
    # family that owns them named ONCE, by its priority name.
    for codec in ("search_replace", "whole_file"):
        for grade in GRADES_FOR[codec]:
            assert profile.codecs[codec][grade].n == 0, (codec, grade)
    assert profile.ceiling is None
    assert profile.loop is None
    assert profile.long_output is None
    assert profile.tools is None

    assert profile.provenance["dropped"] == [
        "codecs-patch: budget — would exceed remaining",
        "tools: budget — would exceed remaining",
        "ceiling: budget — would exceed remaining",
        "loop: budget — would exceed remaining",
        "long_output: budget — would exceed remaining",
        *_BUDGET_MODE_FACTS,
    ]
    # Nothing was started that could not finish: the spend is the
    # declared cost of the families that ran, to the call.
    assert profile.provenance["spent"]["calls"] == _WORKED_EXAMPLE_CALLS
    assert backend.calls == _WORKED_EXAMPLE_CALLS

    verdicts = {name: entry["verdict"]
                for name, entry in profile.verdicts.items()}
    assert verdicts["structured_extraction"] == "ready"
    assert verdicts["chat_speed"] == "ready"
    assert verdicts["agent_speed"] == "ready"
    for unmeasured in ("patch_editing", "long_context", "loop_discipline",
                       "long_output", "tool_calling"):
        assert verdicts[unmeasured] == "unmeasured", unmeasured

    # The budget-mode document survives its own serialization contract.
    assert Profile.from_json(json.loads(profile.to_json())) == profile


def test_one_call_short_of_the_worked_example_drops_the_json_half_whole():
    # The preflight is a REFUSAL, not a truncation: a budget one call
    # short of the json half buys none of it, rather than 29 probes and a
    # matrix nobody can read a verdict off.
    backend = ScriptedBackend()
    profile = _budget_probe(_WORKED_EXAMPLE_CALLS - 1,
                            _backend_override=backend)

    assert profile.codecs is None
    dropped = profile.provenance["dropped"]
    assert "codecs-json: budget — would exceed remaining" in dropped
    assert "codecs-patch: budget — would exceed remaining" in dropped
    # `codecs` is a v1 family: None here has to be named under its own
    # name too, or the schema guard would refuse the document.
    assert any(entry.startswith("codecs:") for entry in dropped)
    assert profile.verdicts["structured_extraction"]["verdict"] == "unmeasured"
    # The 29 calls the json half could not use are not lost: every
    # cheaper family below it in the priority that still fits runs —
    # tools (10), the ceiling with its bisection reserve (9, spending 5
    # on a clean ladder), and the long-output rungs (4). Only the loop
    # (15) is out of reach.
    assert profile.tools is not None
    assert profile.ceiling is not None
    assert profile.long_output is not None
    assert profile.loop is None
    assert "loop: budget — would exceed remaining" in dropped
    assert backend.calls == (_BUDGET_CALIBRATION + _BUDGET_ENVELOPE
                             + _BUDGET_SPEED + _BUDGET_TOOLS
                             + _BUDGET_CEILING_LADDER + _BUDGET_LONG_OUTPUT)


def test_the_priority_is_an_order_not_a_cliff():
    # The spec's original worked-example budget. Both codec halves are
    # unaffordable at 25, and the mode does NOT stop there: tools is
    # further down the priority and cheaper, so it still measures. A
    # first-refusal-ends-the-run reading would leave 11 calls unspent and
    # a consumer with less profile than their budget paid for.
    backend = ScriptedBackend()
    profile = _budget_probe(25, _backend_override=backend)

    assert profile.codecs is None
    assert profile.tools is not None and profile.tools.supported is True
    assert profile.verdicts["tool_calling"]["verdict"] == "ready"
    dropped = profile.provenance["dropped"]
    assert "codecs-json: budget — would exceed remaining" in dropped
    assert "codecs-patch: budget — would exceed remaining" in dropped
    assert backend.calls == (_BUDGET_CALIBRATION + _BUDGET_ENVELOPE
                             + _BUDGET_SPEED + _BUDGET_TOOLS)


def test_a_fat_budget_measures_every_family_in_the_priority():
    backend = ScriptedBackend()
    profile = _budget_probe(200, _backend_override=backend)

    for family in ("geometry", "envelope", "speed", "codecs", "tools",
                   "ceiling", "loop", "long_output"):
        assert getattr(profile, family) is not None, family
    # The two families outside the priority, named as mode facts — and
    # nothing else: no budget line at all on a run that could pay.
    assert profile.provenance["dropped"] == _BUDGET_MODE_FACTS
    assert profile.ceiling_shapes is None and profile.parallel is None
    # Quick's whole bill minus the two families budget mode does not run.
    assert backend.calls == _QUICK_CALLS_TOTAL - 9  # the nine shape probes


def test_the_ceiling_preflight_reserves_the_bisection_tail():
    # What starts, finishes — and a FAILING ladder does not stop at the
    # last rung, it bisects. The preflight reserves both, so a budget
    # with room for the five ladder calls and not for the bisection they
    # can force drops the family by name rather than dying inside it.
    room_for_the_ladder_only = _WORKED_EXAMPLE_CALLS + _BUDGET_CEILING_LADDER
    profile = _budget_probe(room_for_the_ladder_only,
                            _backend_override=ScriptedBackend())
    assert profile.ceiling is None
    assert ("ceiling: budget — would exceed remaining"
            in profile.provenance["dropped"])

    # Four calls more — the declared bisection worst case — and the same
    # ladder is affordable. This endpoint is clean, so it spends five.
    backend = ScriptedBackend()
    profile = _budget_probe(
        room_for_the_ladder_only + _BUDGET_CEILING_BISECTION,
        _backend_override=backend)
    assert profile.ceiling is not None
    assert profile.ceiling.max_verified == 16384
    assert profile.verdicts["long_context"]["verdict"] == "ready"


def test_the_ceiling_preflight_prices_the_cap_the_run_will_use():
    # The cost table prices the MODE cap (16384 here); the run's ladder
    # is bounded by the effective cap — the user's --window-cap or the
    # model's training_ctx, whichever is tighter. Reserving the mode
    # cap's price on a run that will never send those rungs over-reserves
    # by half and drops a family the budget could afford.
    room_for_the_ladder_only = _WORKED_EXAMPLE_CALLS + _BUDGET_CEILING_LADDER
    backend = ScriptedBackend()
    profile = _budget_probe(room_for_the_ladder_only, window_cap=2048,
                            _backend_override=backend)

    # Same budget as the drop above — capped at 2048 the ladder is two
    # rungs and its bisection two more, so the family fits and measures.
    assert profile.ceiling is not None
    assert profile.ceiling.max_verified == 2048
    assert not any(entry.startswith("ceiling:")
                   for entry in profile.provenance["dropped"])


def test_the_clock_never_cuts_a_family_in_half():
    # The seconds ceiling is checked at family boundaries and between
    # calls, never mid-call: the family that was running when the clock
    # ran out finishes, and every family after it is dropped by name with
    # the limit that stopped it.
    backend = ScriptedBackend()
    seconds_per_call = 1.0
    # Two calibration calls plus ten envelope probes = 12 calls, and the
    # meter's last admitted charge lands at 11 seconds.
    profile = _budget_probe(
        200,
        budget_kwargs={"max_seconds": 11.5},
        _backend_override=backend,
        _clock=lambda: backend.calls * seconds_per_call,
    )

    assert profile.geometry is not None
    # COMPLETE, not partial: ten of ten envelope probes.
    assert profile.envelope is not None and profile.envelope.n == 10
    assert profile.speed is None
    assert backend.calls == _BUDGET_CALIBRATION + _BUDGET_ENVELOPE

    dropped = profile.provenance["dropped"]
    for family in ("speed", "codecs-json", "codecs-patch", "tools",
                   "ceiling", "loop", "long_output"):
        assert f"{family}: budget — seconds" in dropped, family
    # The granted ceiling and what the clock read travel together.
    assert profile.provenance["budget"]["max_seconds"] == 11.5
    assert profile.provenance["spent"]["seconds"] == 11.0


def test_a_budget_too_small_for_calibration_still_reports_geometry():
    # Calibration is the run's entry fee (two calls). Below it nothing
    # generative can run — but geometry asks the model nothing, so it is
    # a measurement the meter has no business refusing.
    backend = ScriptedBackend()
    profile = _budget_probe(1, _backend_override=backend)

    assert backend.calls == 0
    assert profile.geometry is not None
    assert profile.geometry.usable_window > 0
    dropped = profile.provenance["dropped"]
    assert "calibration: budget — would exceed remaining" in dropped
    assert "envelope: budget — would exceed remaining" in dropped
    assert profile.provenance["calibration"] is None


def test_a_budget_that_measured_nothing_at_all_raises():
    # No metadata (geometry unmeasurable) and no room for calibration:
    # the run measured NOTHING, and a document full of Nones must not be
    # handed back as a result (the CLI maps this to exit 2).
    with pytest.raises(BudgetExhausted):
        _budget_probe(1, _backend_override=MetadataFreeBackend())


def test_an_unknown_mode_still_names_the_modes_that_exist():
    with pytest.raises(ValueError, match="budget"):
        probe(_URL, "fake-model",
              budget=Budget(max_calls=10, max_prompt_tokens=10**6),
              mode="nonsense", _backend_override=ScriptedBackend())
