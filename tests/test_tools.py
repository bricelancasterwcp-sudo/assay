"""Native tool-calling probe (v1.6): the scripted-tools-v1 instrument."""

import json
import pathlib
from dataclasses import fields
from typing import NamedTuple

import pytest
from fakes import ScriptedToolsBackend, ToolsUnsupportedBackend

from assay.backends.base import ToolCall, ToolReply, ToolsUnsupported
from assay.backends.ollama import OllamaNative
from assay.budget import Budget, BudgetMeter
from assay.errors import InfrastructureError
from assay.replay import CallReplayer, tools_key_material
from assay.tools import (
    TASKS,
    TOOLS_INSTRUMENT,
    TOOLSET,
    TOOLSET_NAME,
    Tools,
    probe_tools,
    t1_messages,
    t2_messages,
)


def meter(max_calls: int = 99):
    return BudgetMeter(Budget(max_calls=max_calls, max_prompt_tokens=10**9))


def _charge_for(messages) -> int:
    """What the probe pays for one call — the replay key material's size."""
    return max(1, len(tools_key_material(messages, TOOLSET)) // 4)


def task_index(messages) -> int:
    """Which of the five tasks this transcript is, by its user message."""
    user = next(m["content"] for m in messages if m["role"] == "user")
    return next(i for i, task in enumerate(TASKS) if task[0] == user)


def tool_result(messages) -> str | None:
    """The scripted tool result in a T2 transcript; None on a T1."""
    return next(
        (m["content"] for m in messages if m["role"] == "tool"), None
    )


class Sent(NamedTuple):
    """One call the probe made, exactly as the backend received it."""

    messages: list[dict]
    tools: list[dict]
    seed: int
    max_tokens: int


class ToolScriptFake:
    """chat_tools answers from injected per-turn scripts.

    Keyed on the SCRIPTED transcript (which task, which turn) and never
    on model output — there is nothing to branch on, which is what makes
    the instrument a script rather than a benchmark. Every call is
    recorded so a test can assert on what the probe actually sent.
    """

    model = "fake-model"

    def __init__(self, t1, t2=None, *, refuse_from=None):
        self._t1 = t1
        self._t2 = t2 if t2 is not None else golden_t2
        self._refuse_from = refuse_from  # 1-based call number, inclusive
        self.attempts = 0
        self.seen: list[Sent] = []

    def chat_tools(self, messages, tools, *, seed, max_tokens):
        self.attempts += 1
        if self._refuse_from is not None and self.attempts >= self._refuse_from:
            raise ToolsUnsupported(
                f"scripted refusal from call {self._refuse_from} onward"
            )
        self.seen.append(Sent(messages, tools, seed, max_tokens))
        index = task_index(messages)
        result = tool_result(messages)
        script = self._t1 if result is None else self._t2
        text, calls = script(index, messages, seed)
        return ToolReply(
            text=text,
            tool_calls=tuple(calls),
            tokens_in=10,
            tokens_out=5,
            stop_reason="stop",
            raw={"scripted": True},
        )


def golden_t1(index, messages, seed):
    _, name, arguments = TASKS[index]
    return "", (ToolCall(name=name, arguments=dict(arguments)),)


def golden_t2(index, messages, seed):
    return f"Done — the tool said: {tool_result(messages)}", ()


def wrong_tool_t1(index, messages, seed):
    """Calls a real tool, always the wrong one for this task."""
    _, expected, _ = TASKS[index]
    name = "run_tests" if expected != "run_tests" else "read_file"
    arguments = {} if name == "run_tests" else {"path": "somewhere.py"}
    return "", (ToolCall(name=name, arguments=arguments),)


def args_t1(override):
    """A right-tool call whose arguments are replaced by `override`."""

    def script(index, messages, seed):
        _, name, _ = TASKS[index]
        return "", (ToolCall(name=name, arguments=override),)

    return script


def prose_t1(index, messages, seed):
    return "I would open that file for you, but let me describe it instead.", ()


def two_calls_t1(index, messages, seed):
    _, name, arguments = TASKS[index]
    return "", (
        ToolCall(name=name, arguments=dict(arguments)),
        ToolCall(name="run_tests", arguments={}),
    )


def spurious_call_t2(index, messages, seed):
    """Echoes the canary but ALSO calls another tool: the job is not done."""
    _, name, arguments = TASKS[index]
    return (
        f"Let me double check: {tool_result(messages)}",
        (ToolCall(name=name, arguments=dict(arguments)),),
    )


def no_canary_t2(index, messages, seed):
    return "The tool ran and everything looks fine.", ()


# --- the registered instrument ----------------------------------------------


def test_instrument_and_toolset_are_named():
    assert TOOLS_INSTRUMENT == "scripted-tools-v1"
    assert TOOLSET_NAME == "toolset-v1"


def test_toolset_registers_the_three_schemas():
    names = [tool["function"]["name"] for tool in TOOLSET]
    assert names == ["read_file", "run_tests", "search_docs"]
    for tool in TOOLSET:
        assert tool["type"] == "function"
        parameters = tool["function"]["parameters"]
        assert parameters["type"] == "object"
        assert set(parameters["required"]) <= set(parameters["properties"])


def test_every_task_pins_a_registered_tool_and_valid_arguments():
    schemas = {
        t["function"]["name"]: t["function"]["parameters"]
        for t in TOOLSET
    }
    assert len(TASKS) == 5
    for message, name, arguments in TASKS:
        assert message and name in schemas
        assert set(arguments) == set(schemas[name]["required"])


# --- the golden path --------------------------------------------------------


def test_golden_model_scores_every_rate_one():
    fake = ScriptedToolsBackend()
    tools = probe_tools(fake, meter())
    assert tools.supported is True
    assert tools.call_rate == 1.0
    assert tools.right_tool_rate == 1.0
    assert tools.args_valid_rate == 1.0
    assert tools.result_use_rate == 1.0
    assert tools.composite == 1.0
    assert tools.n_tasks == 5
    assert tools.n_turns == 10
    assert isinstance(tools, Tools)


def test_seeds_are_distinct_and_deterministic():
    fake = ToolScriptFake(golden_t1)
    probe_tools(fake, meter(), seed_base=1400)
    seeds = [sent.seed for sent in fake.seen]
    assert seeds == [1400, 1500, 1401, 1501, 1402, 1502, 1403, 1503, 1404, 1504]


def test_t1_sends_a_system_line_and_the_task_with_the_toolset():
    fake = ToolScriptFake(golden_t1)
    probe_tools(fake, meter())
    messages, tools = fake.seen[0].messages, fake.seen[0].tools
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "tool" in messages[0]["content"].lower()
    assert messages[1]["content"] == TASKS[0][0]
    assert tools == TOOLSET


def test_t2_carries_the_golden_call_and_a_seeded_canary_result():
    # The wrong-tool fake proves the continuation is CANNED: the model
    # called something else, and T2 still carries the golden call.
    fake = ToolScriptFake(wrong_tool_t1)
    probe_tools(fake, meter(), seed_base=1400)
    messages, seed = fake.seen[1].messages, fake.seen[1].seed
    assert seed == 1500
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool"]
    emitted = messages[2]["tool_calls"][0]
    assert emitted["function"]["name"] == TASKS[0][1]
    assert emitted["function"]["arguments"] == TASKS[0][2]
    assert "CANARY-1500" in messages[3]["content"]
    # The result must answer THAT call: a tool message whose
    # tool_call_id matches nothing is what a strict OpenAI-compat
    # server 400s on, and the 400 would read as a tools refusal.
    assert messages[3]["tool_call_id"] == emitted["id"]
    assert messages[3]["name"] == emitted["function"]["name"]


def test_every_turn_asks_for_the_same_generation_headroom():
    # The generation cap reaches the wire, and is the same on both turns:
    # a T2 truncated before the canary would score a result-use miss the
    # model did not earn.
    from assay.tools import _MAX_TOKENS

    fake = ToolScriptFake(golden_t1)
    probe_tools(fake, meter())
    assert {sent.max_tokens for sent in fake.seen} == {_MAX_TOKENS}
    assert _MAX_TOKENS >= 64  # room for a call plus a sentence


def test_t1_messages_are_not_mutated_into_the_t2_transcript():
    fake = ToolScriptFake(golden_t1)
    probe_tools(fake, meter())
    sent_for_t1 = fake.seen[0].messages
    assert len(sent_for_t1) == 2  # still the two it was sent with


# --- T1 scoring -------------------------------------------------------------


def test_wrong_tool_keeps_call_rate_and_zeroes_the_rest():
    fake = ToolScriptFake(wrong_tool_t1)
    tools = probe_tools(fake, meter())
    assert tools.call_rate == 1.0
    assert tools.right_tool_rate == 0.0
    assert tools.args_valid_rate == 0.0  # the pinned value is the TASK's
    assert tools.composite == 0.0
    assert tools.result_use_rate == 1.0  # T2 is unaffected
    assert tools.n_tasks == 5


def test_unreadable_arguments_are_an_invalid_call_not_a_crash():
    # arguments=None is the malformed-JSON case the OpenAI wire produces.
    fake = ToolScriptFake(args_t1(None))
    tools = probe_tools(fake, meter())
    assert tools.call_rate == 1.0
    assert tools.right_tool_rate == 1.0
    assert tools.args_valid_rate == 0.0
    assert tools.composite == 0.0


def test_unknown_argument_key_is_schema_invalid():
    fake = ToolScriptFake(args_t1({"path": "config.yaml", "mode": "r"}))
    tools = probe_tools(fake, meter())
    assert tools.right_tool_rate == 1.0
    assert tools.args_valid_rate == 0.0


def test_missing_required_key_is_schema_invalid():
    fake = ToolScriptFake(args_t1({}))
    tools = probe_tools(fake, meter())
    assert tools.right_tool_rate == 1.0
    # The `run_tests` task pins {} and DOES score; the other four do not.
    assert tools.args_valid_rate == 1 / 5


def test_a_non_string_argument_never_scores_valid():
    # `{"path": 7}` reaches the declared-type rule on the two read_file
    # tasks (it is the right key) and the unknown-key rule on the rest.
    # Through the probe, equality with the pinned value rejects it either
    # way — the type rule itself is pinned in `test_schema_validation_rules`.
    fake = ToolScriptFake(args_t1({"path": 7}))
    tools = probe_tools(fake, meter())
    assert tools.args_valid_rate == 0.0
    assert tools.right_tool_rate == 1.0


def test_right_shape_wrong_value_fails_the_pinned_argument():
    fake = ToolScriptFake(args_t1({"path": "elsewhere.yaml"}))
    tools = probe_tools(fake, meter())
    assert tools.right_tool_rate == 1.0
    assert tools.args_valid_rate == 0.0
    assert tools.composite == 0.0


def test_prose_answer_scores_zero_calls_and_leaves_tool_rates_unmeasured():
    fake = ToolScriptFake(prose_t1)
    tools = probe_tools(fake, meter())
    assert tools.call_rate == 0.0
    assert tools.right_tool_rate is None  # nothing called: not measured, not 0
    assert tools.args_valid_rate is None
    assert tools.composite == 0.0
    assert tools.n_tasks == 5


def test_two_calls_fail_call_rate_while_the_first_is_still_scored():
    fake = ToolScriptFake(two_calls_t1)
    tools = probe_tools(fake, meter())
    assert tools.call_rate == 0.0
    assert tools.right_tool_rate == 1.0
    assert tools.args_valid_rate == 1.0
    assert tools.composite == 0.0


# --- T2 scoring -------------------------------------------------------------


def test_spurious_second_call_fails_only_result_use():
    fake = ToolScriptFake(golden_t1, spurious_call_t2)
    tools = probe_tools(fake, meter())
    assert tools.result_use_rate == 0.0
    assert tools.call_rate == 1.0
    assert tools.right_tool_rate == 1.0
    assert tools.args_valid_rate == 1.0
    assert tools.composite == 1.0


def test_missing_canary_fails_only_result_use():
    fake = ToolScriptFake(golden_t1, no_canary_t2)
    tools = probe_tools(fake, meter())
    assert tools.result_use_rate == 0.0
    assert tools.composite == 1.0
    assert tools.n_turns == 10


# --- unsupported, budget, infrastructure ------------------------------------


def test_unsupported_on_the_first_call_is_a_capability_fact():
    fake = ToolsUnsupportedBackend()
    tools = probe_tools(fake, meter())
    assert tools.supported is False
    assert tools.call_rate is None
    assert tools.right_tool_rate is None
    assert tools.args_valid_rate is None
    assert tools.result_use_rate is None
    assert tools.composite is None
    assert tools.n_tasks == 0 and tools.n_turns == 0
    assert fake.calls == 1  # it stopped; no burning nine more refusals


def test_refusal_on_the_canned_continuation_keeps_the_partial():
    # A server that will not accept the canned T2 message (the wire-shape
    # hazard in the module docstring) must not overwrite a capability it
    # already demonstrated on T1.
    fake = ToolScriptFake(golden_t1, refuse_from=2)
    tools = probe_tools(fake, meter())
    assert tools.supported is True  # it DID speak the protocol once
    assert tools.n_tasks == 1 and tools.n_turns == 1
    assert tools.call_rate == 1.0
    assert tools.result_use_rate is None  # no T2 was ever scored
    assert fake.attempts == 2  # and it stopped there


def test_a_later_task_refusing_does_not_erase_the_measured_tasks():
    # The refusal lands on task 1's T1, after task 0 scored two turns:
    # `supported=False` here would throw away a capability we measured.
    fake = ToolScriptFake(golden_t1, refuse_from=3)
    tools = probe_tools(fake, meter())
    assert tools.supported is True
    assert tools.n_tasks == 1 and tools.n_turns == 2
    assert tools.composite == 1.0
    assert tools.result_use_rate == 1.0
    assert fake.attempts == 3  # stopped, not eight more refusals


def test_budget_death_before_the_first_call_is_never_attempted():
    fake = ToolScriptFake(golden_t1)
    dead = BudgetMeter(Budget(max_calls=0, max_prompt_tokens=1))
    tools = probe_tools(fake, dead)
    assert tools.supported is None  # never attempted, not refused
    assert tools.call_rate is None and tools.composite is None
    assert tools.n_tasks == 0 and tools.n_turns == 0
    assert fake.seen == []


def test_budget_death_midway_yields_an_honest_partial_n():
    fake = ToolScriptFake(golden_t1)
    tools = probe_tools(fake, meter(max_calls=3))
    assert tools.supported is True
    assert tools.n_tasks == 2 and tools.n_turns == 3
    assert tools.composite == 1.0  # over the two tasks that ran
    assert tools.result_use_rate == 1.0  # over the one T2 that ran
    assert len(fake.seen) == 3


def test_budget_death_ends_the_probe_instead_of_skipping_to_cheaper_turns():
    # A budget with room for two T1s but not for a T2. Carrying on to the
    # affordable turns would grow the composite's n while `result_use`
    # stayed unmeasured — a run reported as fuller than it was.
    charge = _charge_for(t1_messages(0)) + _charge_for(t1_messages(1))
    assert _charge_for(t2_messages(0, 1500)) > _charge_for(t1_messages(1))
    fake = ToolScriptFake(golden_t1)
    tools = probe_tools(
        fake, BudgetMeter(Budget(max_calls=99, max_prompt_tokens=charge))
    )
    assert tools.n_tasks == 1 and tools.n_turns == 1
    assert fake.attempts == 1
    assert tools.result_use_rate is None


def test_budget_death_does_not_hunt_for_a_task_it_can_still_afford():
    # Enough left for task 3's (cheaper) first turn but not task 2's.
    # Skipping ahead would make partial n depend on which task happened
    # to be worded shortest — measurement decided by sentence length.
    full = sum(
        _charge_for(t1_messages(i)) + _charge_for(t2_messages(i, 1500 + i))
        for i in (0, 1)
    )
    spare = _charge_for(t1_messages(3))
    assert _charge_for(t1_messages(2)) > spare  # the ordering under test
    fake = ToolScriptFake(golden_t1)
    tools = probe_tools(
        fake, BudgetMeter(Budget(max_calls=99, max_prompt_tokens=full + spare))
    )
    assert tools.n_tasks == 2 and tools.n_turns == 4
    assert fake.attempts == 4


def test_infrastructure_errors_propagate():
    class Broken(ToolScriptFake):
        def chat_tools(self, messages, tools, *, seed, max_tokens):
            raise InfrastructureError("transport failure (scripted)")

    with pytest.raises(InfrastructureError):
        probe_tools(Broken(golden_t1), meter())


# --- the schema check, which today's tasks make a backstop ------------------
#
# Every registered task pins an exact argument value, and equality with a
# pinned value is STRICTER than the schema (it implies the keys, the
# types and the values). So these two suites cover the schema rules
# directly: through the probe they are redundant, and they stop being
# redundant the moment a task pins `None` instead of a value.


def test_schema_validation_rules():
    from assay.tools import _schema_valid

    assert _schema_valid("read_file", {"path": "a.py"}) is True
    assert _schema_valid("run_tests", {}) is True
    assert _schema_valid("read_file", {}) is False           # required missing
    assert _schema_valid("run_tests", {"path": "a"}) is False  # unknown key
    assert _schema_valid("read_file", {"path": 7}) is False  # declared string
    assert _schema_valid("read_file", None) is False         # unreadable
    assert _schema_valid("delete_everything", {}) is False   # not registered


def test_an_unpinned_task_still_scores_arguments_against_the_schema():
    from assay.tools import _score_t1

    def reply(arguments):
        return ToolReply(
            text="",
            tool_calls=(ToolCall(name="read_file", arguments=arguments),),
            tokens_in=1, tokens_out=1, stop_reason="stop", raw={},
        )

    # expected_args None: nothing to compare against, so schema validity
    # is the whole of the args verdict.
    assert _score_t1(reply({"path": "anything.py"}), "read_file", None) == (
        True, True, True, True
    )
    assert _score_t1(reply({}), "read_file", None)[3] is False
    assert _score_t1(reply({"path": 7}), "read_file", None)[3] is False
    assert _score_t1(reply(None), "read_file", None)[3] is False


# --- charging cannot drift from the replay key ------------------------------


def test_a_recorded_run_replays_call_for_call(tmp_path):
    # The instrument's whole claim to being replayable: identical
    # messages and seeds on the second pass, so every strict key hits.
    from assay.replay import CallRecorder, CallReplayer

    live = ScriptedToolsBackend()
    path = tmp_path / "tools.jsonl"
    recorded = probe_tools(CallRecorder(live, path), meter())
    replayed = probe_tools(
        CallReplayer(path, model=live.model, caps=live.caps), meter()
    )
    assert replayed == recorded
    assert recorded.n_turns == 10


def test_the_meter_is_charged_on_the_replay_key_material():
    fake = ToolScriptFake(golden_t1)
    m = meter()
    probe_tools(fake, m)
    expected = sum(
        max(1, len(tools_key_material(messages, tools)) // 4)
        for messages, tools, _, _ in fake.seen
    )
    assert m.spent.prompt_tokens == expected
    assert m.spent.calls == 10


# --- the live anchor (plan Task 10) ----------------------------------------
#
# Captured 2026-08-16 against ollama 0.32.13 on this box: four models, the
# probe's own prompts and seeds, one run each, committed under
# docs/superpowers/evidence/tools-anchor/ with the values they measured in
# results.json. These tests read the COMMITTED transcripts through the
# STRICT replayer — no daemon, no GPU, no network — and hold the probe to
# what the live endpoints actually did.

ANCHOR = pathlib.Path(__file__).resolve().parents[1] / (
    "docs/superpowers/evidence/tools-anchor")


def anchor_results() -> dict:
    return json.loads((ANCHOR / "results.json").read_text())


def anchor_rows(name: str) -> list[dict]:
    lines = (ANCHOR / name).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def anchor_replayer(capture: dict) -> CallReplayer:
    """The transcript, replayed under the caps that RECORDED it.

    Not a convenient stand-in: these rows came off `OllamaNative`, so its
    caps are the ones the run was made under, and a replay that quietly
    relaxed them would be measuring a different endpoint.
    """
    return CallReplayer(
        ANCHOR / capture["transcript"],
        model=capture["model"],
        caps=OllamaNative.caps,
    )


def anchor_captures() -> list[dict]:
    return anchor_results()["tools"]["captures"]


def test_the_tools_anchor_capture_is_committed_whole():
    results = anchor_results()
    assert results["daemon"]["version"] == "0.32.13"
    assert results["tools"]["instrument"] == TOOLS_INSTRUMENT
    assert results["tools"]["toolset"] == TOOLSET_NAME

    captures = anchor_captures()
    assert len(captures) == 4
    # One refusal and three endpoints that took the parameter — the
    # anchor is worthless if it only ever saw one outcome.
    supported = [c["result"]["supported"] for c in captures]
    assert supported.count(False) == 1 and supported.count(True) == 3
    for capture in captures:
        rows = anchor_rows(capture["transcript"])
        assert len(rows) == capture["rows"]
        assert all(row["model"] == capture["model"] for row in rows)
        assert all(row["kind"] == "chat_tools" for row in rows)


@pytest.mark.parametrize(
    "capture", anchor_captures(), ids=lambda c: c["model"]
)
def test_every_committed_capture_replays_to_its_recorded_values(capture):
    """The acceptance test the anchor exists to pass.

    Recorded rates are re-DERIVED by running the unmodified probe over
    the transcript, never read out of results.json — so a scoring change
    that would have moved a live number fails here instead of silently
    re-describing evidence that was measured under the old rules.
    """
    replayed = probe_tools(
        anchor_replayer(capture),
        BudgetMeter(Budget(max_calls=20, max_prompt_tokens=200_000)),
    )

    # Every field of the measurement, not the subset someone chose to
    # write down: a recorded value pruned out of results.json would
    # otherwise stop being checked without anything failing.
    assert set(capture["result"]) == {f.name for f in fields(Tools)}
    for field, recorded in capture["result"].items():
        assert getattr(replayed, field) == recorded, field


def test_the_committed_refusal_replays_as_a_capability_fact():
    """gemma2:9b: `supported=False`, and the endpoint's own words with it.

    The refusal body is the PRIMARY SOURCE behind the classification. A
    replay that restored the verdict but dropped the body would leave
    assay asserting a capability with the evidence discarded.
    """
    capture = next(c for c in anchor_captures() if c["model"] == "gemma2:9b")
    replayer = anchor_replayer(capture)

    with pytest.raises(ToolsUnsupported) as raised:
        replayer.chat_tools(
            t1_messages(0), TOOLSET, seed=1400, max_tokens=256
        )

    assert raised.value.raw == capture["error_raw"]
    body = raised.value.raw["error"]
    assert "does not support tools" in body
    # The classifier's live bet, restated over the real body: the rule is
    # "4xx and the complaint names tools", not Ollama's exact phrasing.
    assert "tool" in body.lower()


def test_the_daemon_corroborates_the_refusal_from_the_other_side():
    """The refusal is not just a 400: the model file has no tool template.

    `/api/show` for gemma2:9b lists `capabilities: ["completion"]` and no
    `tools`, which is the same fact the 400 reports — so the committed
    body is what that claim rests on rather than a remembered reading.
    """
    capture = next(c for c in anchor_captures() if c["model"] == "gemma2:9b")
    show = json.loads((ANCHOR / capture["show"]).read_text())

    assert show["capabilities"] == capture["capabilities"] == ["completion"]
    assert "tools" not in show["capabilities"]
    assert show["details"]["family"] == "gemma2"
    # The three models that DID speak the protocol are the control: this
    # is a property of gemma2's file, not of every model on the box.
    assert "tools" not in show.get("template", "")


def test_the_refusal_transcript_holds_one_row_and_no_reply():
    # The "stop on the first refusal" rule, pinned to the bytes: nine
    # further refusals would have measured nothing new, and a reply row
    # would mean the model was scored after being declared unsupported.
    rows = anchor_rows("tools-gemma2-9b.jsonl")

    assert len(rows) == 1
    assert rows[0]["outcome"] == "error"
    assert rows[0]["error_type"] == "ToolsUnsupported"
    assert rows[0]["error_raw"]
    assert rows[0]["text"] is None and rows[0]["tool_calls"] is None


def test_a_supported_endpoint_that_never_calls_is_not_an_unsupported_one():
    """The measured case the family was built to separate.

    qwen2.5-coder:7b took the `tools` parameter and then WROTE the right
    call as plain text five times over, emitting no native call at all.
    `supported` is about the endpoint, `call_rate` about the model, and
    this capture is the live proof they are different facts.
    """
    capture = next(
        c for c in anchor_captures()
        if c["model"] == "qwen2.5-coder:7b-instruct-q8_0"
    )
    replayed = probe_tools(
        anchor_replayer(capture),
        BudgetMeter(Budget(max_calls=20, max_prompt_tokens=200_000)),
    )

    assert replayed.supported is True
    assert replayed.call_rate == 0.0 and replayed.composite == 0.0
    # Nothing called at all => nothing to judge. None, never 0.0, or the
    # miss call_rate already carries would be counted twice.
    assert replayed.right_tool_rate is None
    assert replayed.args_valid_rate is None
    # ...and it reads a tool result perfectly well. The failure is the
    # protocol, not comprehension.
    assert replayed.result_use_rate == 0.8

    # The bytes behind the claim: every T1 row carries no tool call and a
    # text body that is itself a well-formed call to a registered tool.
    registered = {tool["function"]["name"] for tool in TOOLSET}
    t1_rows = anchor_rows(capture["transcript"])[0::2]
    assert len(t1_rows) == 5
    for row in t1_rows:
        assert row["tool_calls"] == []
        written = json.loads(row["text"])
        assert written["name"] in registered
