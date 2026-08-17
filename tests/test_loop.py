"""Mini-loop discipline probe (v1.4): scripted three-turn repair.

v1.6 (`scripted-loop-v2`): the same golden script plus a two-turn ERROR
script — the model is shown its own patch failing to apply and scored on
what it does next (recover, or re-emit the same broken block).
"""

import json
import pathlib
from dataclasses import fields

from assay.backends.base import Reply
from assay.backends.ollama import OllamaNative
from assay.budget import Budget, BudgetMeter
from assay.codecs import _parse_blocks, apply_search_replace
from assay.loop import (LOOP_INSTRUMENT, Loop, broken_patch,
                        error_turn_prompts, probe_loop, turn_prompts)
from assay.replay import CallReplayer


def meter():
    return BudgetMeter(Budget(max_calls=99, max_prompt_tokens=10**9))


class TurnFake:
    """Replies keyed on the SCRIPTED prompt, never on turn arithmetic.

    `script` holds the three golden replies by turn index; `error` holds
    the reply to the error script's T2' and defaults to the golden turn-2
    reply (the same patch the model would have sent anyway). The error
    script's T1 prompt IS the golden T1, character for character, so one
    reply answers both: a model cannot tell those two turns apart, and
    neither may the fake.
    """

    def __init__(self, script, error=None):
        self.script = script
        self.error = script[1] if error is None else error
        self.calls = 0
        self.seeds = []

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
        self.calls += 1
        self.seeds.append(seed)
        golden = turn_prompts()
        if prompt == error_turn_prompts()[1]:
            text = self.error
        elif prompt in golden:
            text = self.script[golden.index(prompt)]
        else:
            raise AssertionError(f"unscripted loop prompt: {prompt[:60]!r}")
        return Reply(text=text, tokens_in=10, tokens_out=10,
                     stop_reason="stop", raw={})


def original_source():
    _, t2, _ = turn_prompts()
    return t2.split("Contents of `tiny.py`:\n", 1)[1].split(
        "\n\nYour next action:", 1)[0]


def good_patch():
    original = original_source()
    broken = "    subtotal * 1.08"
    fixed = "    return subtotal * 1.08"
    assert broken in original
    return (
        "patch tiny.py\n```\n<<<<<<< SEARCH\n" + broken + "\n=======\n"
        + fixed + "\n>>>>>>> REPLACE\n```"
    )


def test_disciplined_model_scores_clean():
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done fixed it"})
    loop = probe_loop(fake, meter(), runs=2)
    assert loop.action_fidelity == 1.0
    assert loop.patch_rate == 1.0
    assert loop.finish_rate == 1.0
    assert loop.repeat_rate == 0.0
    assert loop.anchor_violations == 0
    # n_turns 6 -> 10 BY DESIGN (v1.6): scripted-loop-v2 runs the 2-turn
    # error script alongside the 3-turn golden one, and every turn of
    # both is a scored turn of the same instrument. n_runs stays the
    # count of golden runs — it is patch_rate's and finish_rate's
    # denominator.
    assert loop.n_runs == 2 and loop.n_turns == 10
    assert loop.recovery_rate == 1.0
    assert loop.doom_loop_rate == 0.0


def test_looping_model_is_measured_not_excused():
    # Reads forever: every action is VALID but nothing advances — high
    # fidelity, zero patch/finish, repeats counted (the robigo 39%
    # repeat-rate shape at 14B).
    fake = TurnFake({0: "read tiny.py", 1: "read tiny.py", 2: "read tiny.py"})
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.action_fidelity == 1.0
    assert loop.patch_rate == 0.0
    assert loop.finish_rate == 0.0
    # repeat_rate 2/3 -> 3/5 BY DESIGN (v1.6): the error script adds two
    # more turns to the same run (one of them a repeat of its own T1),
    # so both numerator and denominator grow.
    assert loop.repeat_rate == 3 / 5
    assert loop.anchor_violations == 0
    # Reading again when told the patch failed is neither a recovery nor
    # a doom loop — it is the third thing, and it must not be forced
    # into either bucket.
    assert loop.recovery_rate == 0.0
    assert loop.doom_loop_rate == 0.0


def test_patching_the_readonly_tests_is_an_anchor_violation():
    # The error script's reply is kept benign so this test keeps
    # counting ONE thing: the golden turn-2 violation.
    fake = TurnFake({0: "read tiny.py",
                     1: "patch test_tiny.py\n```\nwhatever\n```",
                     2: "done"},
                    error="done")
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.anchor_violations == 1
    assert loop.patch_rate == 0.0


def test_the_error_turn_is_anchored_too():
    # Reaching for the read-only test file after a failed patch is the
    # same cardinal sin, and the error script is not a blind spot.
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error="patch test_tiny.py\n```\nwhatever\n```")
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.anchor_violations == 1
    assert loop.recovery_rate == 0.0 and loop.doom_loop_rate == 0.0


def test_prose_replies_fail_action_fidelity():
    fake = TurnFake({0: "Sure! I would love to help with this bug.",
                     1: good_patch(), 2: "done"})
    # 2/3 -> 3/5 BY DESIGN (v1.6): the error script's T1 prompt is the
    # golden T1, so the same prose comes back there too — one more
    # invalid reply, two more scored turns.
    assert probe_loop(fake, meter(), runs=1).action_fidelity == 3 / 5


def test_budget_death_before_any_turn_is_unmeasured():
    dead = BudgetMeter(Budget(max_calls=0, max_prompt_tokens=1))
    loop = probe_loop(TurnFake({0: "x", 1: "y", 2: "z"}), dead, runs=2)
    assert loop.action_fidelity is None
    assert loop.n_turns == 0
    assert loop.recovery_rate is None and loop.doom_loop_rate is None
    assert isinstance(loop, Loop)


def test_non_applying_patch_does_not_count_as_landed():
    # A patch to the RIGHT file whose SEARCH matches nothing must score
    # zero — counting intent instead of landing is the exact overclaim
    # the applies-and-parses lens exists to prevent.
    fake = TurnFake({0: "read tiny.py",
                     1: "patch tiny.py\n```\n<<<<<<< SEARCH\nnot in the file\n"
                        "=======\nstill not\n>>>>>>> REPLACE\n```",
                     2: "done"})
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.patch_rate == 0.0
    assert loop.action_fidelity == 1.0  # the ACTION was well-formed


def test_a_patch_that_applies_but_breaks_python_does_not_land_either():
    # The other half of applies-AND-parses on the golden turn: SEARCH
    # matched, the file changed, and the result is no longer Python.
    # Landing means the repair survives, not that the edit happened.
    reply = ("patch tiny.py\n```\n<<<<<<< SEARCH\n    subtotal * 1.08\n"
             "=======\n    return subtotal * (((\n>>>>>>> REPLACE\n```")
    assert apply_search_replace(original_source(), reply) is not None
    fake = TurnFake({0: "read tiny.py", 1: reply, 2: "done"}, error="done")
    assert probe_loop(fake, meter(), runs=1).patch_rate == 0.0


# --- the error script (v1.6, `scripted-loop-v2`) ----------------------

def doomed_reply():
    """`patch tiny.py` carrying the canned broken block back verbatim."""
    return "patch tiny.py\n" + broken_patch()


def test_the_instrument_is_named_v2():
    # The lens string IS the amendment: a profile scored under the error
    # script must never be readable as a scripted-loop-v1 measurement.
    assert LOOP_INSTRUMENT == "scripted-loop-v2"


def test_the_canned_patch_is_the_real_line_with_indentation_stripped():
    # The measured qwen signature: the right target line, dedented. Built
    # from the fixture's REAL lines, so it cannot drift from the file the
    # probe actually ships.
    original = original_source()
    assert "\n    subtotal * 1.08\n" in original
    assert "\nsubtotal * 1.08\n" in broken_patch()
    assert "<<<<<<< SEARCH" in broken_patch()
    # And it genuinely fails: a canned "failure" that would in fact
    # apply would make the whole error script a lie.
    assert apply_search_replace(original, broken_patch()) is None


def test_the_error_prompt_shows_the_failure_and_the_unchanged_file():
    t1, t2 = error_turn_prompts()
    assert t1 == turn_prompts()[0]          # T1 is the golden T1
    original = original_source()
    assert "> read tiny.py" in t2
    assert "> patch tiny.py" in t2
    assert broken_patch() in t2
    assert ("The patch FAILED to apply: SEARCH text not found in "
            "`tiny.py`.") in t2
    assert "Contents of `tiny.py` (unchanged):" in t2
    assert t2.count(original) == 2          # before the patch, and after
    assert t2.rstrip().endswith("Your next action:")


def test_correcting_the_indentation_is_a_recovery():
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=good_patch())
    loop = probe_loop(fake, meter(), runs=3)
    assert loop.recovery_rate == 1.0
    assert loop.doom_loop_rate == 0.0


def test_reemitting_the_broken_block_is_a_doom_loop():
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=doomed_reply())
    loop = probe_loop(fake, meter(), runs=3)
    assert loop.recovery_rate == 0.0
    assert loop.doom_loop_rate == 1.0


def test_a_doom_loop_survives_cosmetic_reformatting():
    # Re-emitting the same failing SEARCH with different spacing is the
    # same doom loop; whitespace normalization is what sees through it.
    reply = ("patch tiny.py\n```\n<<<<<<< SEARCH\n  subtotal  *  1.08   \n"
             "=======\nreturn subtotal * 1.08\n>>>>>>> REPLACE\n```")
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=reply)
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.doom_loop_rate == 1.0
    assert loop.recovery_rate == 0.0


def test_a_different_wrong_patch_is_not_a_doom_loop():
    # Wrong, but not the SAME wrong: doom loop names repetition of the
    # failure it was just shown, not failure in general.
    reply = ("patch tiny.py\n```\n<<<<<<< SEARCH\nreturn items[len(items)]\n"
             "=======\nreturn items[-1]\n>>>>>>> REPLACE\n```")
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=reply)
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.doom_loop_rate == 0.0
    assert loop.recovery_rate == 0.0


def test_a_landed_patch_is_never_scored_a_doom_loop():
    # The lens's sharp edge, pinned deliberately: normalization erases
    # exactly the indentation that separates the broken block from the
    # corrected one, so the SEARCH lines of a CORRECT patch normalize
    # equal to the canned broken one. Landing is the discriminator — a
    # block that applies is a recovery and cannot also be scored as
    # repeating the failure it just fixed.
    correct = _parse_blocks(good_patch())[0][0]
    canned = _parse_blocks(broken_patch())[0][0]
    assert correct != canned                                  # they differ...
    assert ([" ".join(line.split()) for line in correct]
            == [" ".join(line.split()) for line in canned])   # ...but not here
    loop = probe_loop(
        TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                 error=good_patch()),
        meter(), runs=1)
    assert loop.recovery_rate == 1.0 and loop.doom_loop_rate == 0.0


def test_a_patch_that_applies_but_breaks_python_is_neither():
    # SEARCH matched this time — the model DID escape the failure it was
    # shown — but the replacement is not Python, so it is no recovery
    # either. The applies-and-parses lens refuses to round either way.
    reply = ("patch tiny.py\n```\n<<<<<<< SEARCH\n    subtotal * 1.08\n"
             "=======\n    return subtotal * (((\n>>>>>>> REPLACE\n```")
    assert apply_search_replace(original_source(), reply) is not None
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=reply)
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.recovery_rate == 0.0
    assert loop.doom_loop_rate == 0.0


def test_a_correct_fix_framed_as_prose_is_not_a_doom_loop():
    # No action line, so no recovery — action fidelity is the thing that
    # failed. But the block is the CORRECT one and applies cleanly, so
    # it is emphatically not a re-emission of the failed SEARCH. The two
    # Nones must stay apart: "does not apply to the file" and "was not
    # offered as a patch action" are different facts, and collapsing
    # them would demote a model for the very fix it just produced.
    reply = "Here is the corrected block:\n" + good_patch().split("\n", 1)[1]
    assert _parse_action_is_invalid(reply)
    assert apply_search_replace(original_source(), reply) is not None
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=reply)
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.recovery_rate == 0.0
    assert loop.doom_loop_rate == 0.0


def _parse_action_is_invalid(reply):
    from assay.loop import _parse_action
    return _parse_action(reply)[0] is None


def test_two_blocks_are_not_one_doom_looping_action():
    # One action, one payload: the codec lens refuses a multi-block reply
    # and the doom lens refuses it identically, so a ragged reply cannot
    # be scored as the tidy failure it half-contains.
    block = broken_patch().strip("`").strip("\n")   # the fence removed
    reply = "patch tiny.py\n```\n" + block + "\n" + block + "\n```"
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=reply)
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.doom_loop_rate == 0.0
    assert loop.recovery_rate == 0.0


def test_error_turns_join_the_shared_denominators():
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=good_patch())
    loop = probe_loop(fake, meter(), runs=4)
    assert fake.calls == 4 * 3 + 4 * 2
    assert loop.n_turns == fake.calls  # every reply is a scored turn
    assert loop.n_runs == 4            # golden runs: patch/finish's n


def test_error_runs_get_their_own_seeds():
    # Distinct seeds keep the error script from being a re-roll of the
    # golden one on backends that cache by (prompt, seed).
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=good_patch())
    probe_loop(fake, meter(), runs=2, seed_base=800)
    assert fake.seeds == [800, 800, 800, 801, 801, 801, 850, 850, 851, 851]


def test_a_v1_5_loop_payload_still_constructs():
    # Profile.from_json does Loop(**payload); a profile written before
    # the error script existed has no recovery keys, and "the field was
    # never measured" must read as None, never as zero.
    v5_payload = {"action_fidelity": 1.0, "patch_rate": 0.0,
                  "finish_rate": 1.0, "repeat_rate": 0.0,
                  "anchor_violations": 0, "n_runs": 3, "n_turns": 9}
    loop = Loop(**v5_payload)
    assert loop.recovery_rate is None and loop.doom_loop_rate is None


def test_budget_death_before_the_error_script_keeps_the_golden_results():
    # Exactly the three golden calls of one run; the error script never
    # starts. Recovery is UNMEASURED, and the golden rates stand.
    tight = BudgetMeter(Budget(max_calls=3, max_prompt_tokens=10**9))
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=good_patch())
    loop = probe_loop(fake, tight, runs=1)
    assert loop.patch_rate == 1.0 and loop.finish_rate == 1.0
    assert loop.n_runs == 1 and loop.n_turns == 3
    assert loop.recovery_rate is None and loop.doom_loop_rate is None


def test_a_golden_run_cut_short_does_not_buy_a_doomed_error_run():
    # Room for the golden T1 twice over, but not for the long turns. The
    # error script's T1 is the SHORT golden T1, so a meter this dead
    # would still admit it — and spend a call on a script that provably
    # cannot reach the turn it exists to score. It must not be started.
    t1_tokens = max(1, len(turn_prompts()[0]) // 4)
    tight = BudgetMeter(Budget(max_calls=99, max_prompt_tokens=2 * t1_tokens))
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=good_patch())
    loop = probe_loop(fake, tight, runs=1)
    assert fake.calls == 1 and loop.n_turns == 1
    assert loop.n_runs == 0 and loop.patch_rate is None
    assert loop.recovery_rate is None and loop.doom_loop_rate is None


def test_budget_death_mid_error_script_reports_the_runs_that_finished():
    # Two golden runs (6 calls) + one whole error run (2) + one call of
    # the next, which dies. The completed error run is the honest
    # denominator; the half-run is not counted either way.
    tight = BudgetMeter(Budget(max_calls=9, max_prompt_tokens=10**9))
    fake = TurnFake({0: "read tiny.py", 1: good_patch(), 2: "done"},
                    error=doomed_reply())
    loop = probe_loop(fake, tight, runs=2)
    assert loop.n_runs == 2 and loop.patch_rate == 1.0
    assert loop.doom_loop_rate == 1.0   # 1 of 1 COMPLETED error run
    assert loop.recovery_rate == 0.0
    assert loop.n_turns == 9            # the ninth reply is still scored


# --- the live anchor (plan Task 10) ----------------------------------------
#
# Captured 2026-08-16 against ollama 0.32.13: one probe_loop(runs=3) run —
# both scripts, 15 calls — against qwen2.5-coder:7b-instruct-q8_0, committed
# under docs/superpowers/evidence/tools-anchor/. This test reads the
# COMMITTED transcript through the STRICT replayer: no daemon, no GPU, no
# network.

ANCHOR = pathlib.Path(__file__).resolve().parents[1] / (
    "docs/superpowers/evidence/tools-anchor")


def anchor_capture() -> dict:
    results = json.loads((ANCHOR / "results.json").read_text())
    assert results["loop"]["instrument"] == LOOP_INSTRUMENT
    return results["loop"]["captures"][0]


def anchor_rows(name: str) -> list[dict]:
    lines = (ANCHOR / name).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_the_committed_loop_transcript_replays_to_its_recorded_values():
    capture = anchor_capture()
    rows = anchor_rows(capture["transcript"])
    assert len(rows) == capture["rows"] == 15
    assert all(row["kind"] == "generate" for row in rows)

    replayed = probe_loop(
        CallReplayer(
            ANCHOR / capture["transcript"],
            model=capture["model"],
            caps=OllamaNative.caps,
        ),
        BudgetMeter(Budget(max_calls=30, max_prompt_tokens=200_000)),
        runs=3,
    )

    # Re-derived by the unmodified probe, never read out of results.json —
    # and every field of the measurement, not the subset someone chose to
    # write down.
    assert set(capture["result"]) == {f.name for f in fields(Loop)}
    for field, recorded in capture["result"].items():
        assert getattr(replayed, field) == recorded, field


def test_the_anchor_measured_a_real_doom_loop():
    """The failure the error script was written for, off a live endpoint.

    Shown its patch rejected with "SEARCH text not found" and the file
    unchanged, the model re-emitted the identical failing block on all
    three error runs. The rates are asserted through the probe above;
    this pins the BYTES they were derived from, so a transcript edited
    out from under the claim fails rather than quietly re-describing it.
    """
    capture = anchor_capture()
    assert capture["result"]["doom_loop_rate"] == 1.0
    assert capture["result"]["recovery_rate"] == 0.0
    assert capture["result"]["n_error_runs"] == 3

    # The error script's six rows are the last six: seeds 850-852, two
    # turns each. The second turn of each is the scored reply.
    error_rows = anchor_rows(capture["transcript"])[9:]
    assert [row["seed"] for row in error_rows] == [850, 850, 851, 851, 852, 852]
    replies = [row["text"] for row in error_rows[1::2]]

    canned = _parse_blocks(broken_patch())[0][0]
    for reply in replies:
        blocks = _parse_blocks(reply)
        assert len(blocks) == 1
        assert blocks[0][0] == canned          # the same SEARCH, verbatim
        # ...and it still does not apply, which is why it is a doom loop
        # and not a fix that happens to look like the block it followed.
        assert apply_search_replace(original_source(), reply) is None
