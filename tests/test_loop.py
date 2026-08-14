"""Mini-loop discipline probe (v1.4): scripted three-turn repair."""

from assay.backends.base import Reply
from assay.budget import Budget, BudgetMeter
from assay.loop import Loop, probe_loop, turn_prompts


def meter():
    return BudgetMeter(Budget(max_calls=99, max_prompt_tokens=10**9))


class TurnFake:
    """Replies per turn from a script: {turn_index: text}."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def generate(self, prompt, *, seed, max_tokens, num_ctx=None):
        turn = self.calls % 3
        self.calls += 1
        return Reply(text=self.script[turn], tokens_in=10, tokens_out=10,
                     stop_reason="stop", raw={})


def good_patch():
    t1, t2, _ = turn_prompts()
    original = t2.split("Contents of `tiny.py`:\n", 1)[1]
    original = original.split("\n\nYour next action:", 1)[0]
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
    assert loop.n_runs == 2 and loop.n_turns == 6


def test_looping_model_is_measured_not_excused():
    # Reads forever: every action is VALID but nothing advances — high
    # fidelity, zero patch/finish, repeats counted (the robigo 39%
    # repeat-rate shape at 14B).
    fake = TurnFake({0: "read tiny.py", 1: "read tiny.py", 2: "read tiny.py"})
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.action_fidelity == 1.0
    assert loop.patch_rate == 0.0
    assert loop.finish_rate == 0.0
    assert loop.repeat_rate == 2 / 3
    assert loop.anchor_violations == 0


def test_patching_the_readonly_tests_is_an_anchor_violation():
    fake = TurnFake({0: "read tiny.py",
                     1: "patch test_tiny.py\n```\nwhatever\n```",
                     2: "done"})
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.anchor_violations == 1
    assert loop.patch_rate == 0.0


def test_prose_replies_fail_action_fidelity():
    fake = TurnFake({0: "Sure! I would love to help with this bug.",
                     1: good_patch(), 2: "done"})
    loop = probe_loop(fake, meter(), runs=1)
    assert loop.action_fidelity == 2 / 3


def test_budget_death_before_any_turn_is_unmeasured():
    dead = BudgetMeter(Budget(max_calls=0, max_prompt_tokens=1))
    loop = probe_loop(TurnFake({0: "x", 1: "y", 2: "z"}), dead, runs=2)
    assert loop.action_fidelity is None
    assert loop.n_turns == 0
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
